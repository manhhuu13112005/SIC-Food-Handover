"""Read-only source ingestion and conservative candidate audit; never train/merge.

All source formats are inventoried before exact-label mapping. Candidate manifests
and reports are atomic snapshots; generated image files are content addressed.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import stat
import sys
import tempfile
import warnings
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath

import numpy as np
import yaml
from PIL import Image, ImageOps, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
MANIFEST_FIELDS = ['source_dataset', 'source_image', 'source_split', 'original_label',
                   'target_class', 'bounding_box', 'image_group', 'crop_path', 'sha256',
                   'phash', 'source_sha256', 'group_method', 'record_id']
ISSUE_FIELDS = ['source_dataset', 'source_image', 'original_label', 'reason', 'detail', 'severity']


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def within(path, root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def overlaps(a, b):
    return within(a, b) or within(b, a)


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def write_csv(path, rows, fields):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, output.getvalue().encode('utf-8-sig'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def archive_snapshot(config):
    root = config['archives_dir']
    return {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() == '.zip'} if root.exists() else {}


def image_tree_snapshot(roots):
    """Fast freshness check; audit separately verifies candidate content hashes."""
    digest = hashlib.sha256()
    for root in roots:
        root = Path(root)
        digest.update(f'{root}:{root.exists()}\n'.encode())
        if root.is_dir():
            for path in sorted(root.rglob('*')):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    stats = path.stat()
                    digest.update(f'{path}:{stats.st_size}:{stats.st_mtime_ns}\n'.encode())
    return digest.hexdigest()


@contextmanager
def run_lock(config):
    config['work_dir'].mkdir(parents=True, exist_ok=True)
    path = config['work_dir'] / '.bulk_state.lock'
    try:
        handle = path.open('x', encoding='utf-8')
    except FileExistsError as error:
        raise ValueError(f'Another run may be active: {path}. If it crashed, verify no run is active before removing only this lock file.') from error
    try:
        with handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def load_config(path, project=PROJECT):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict) or config.get('version') != 1:
        raise ValueError('Configuration must use version: 1.')
    project = Path(project).resolve()
    for key in ('archives_dir', 'work_dir', 'output_dir', 'reports_dir'):
        config[key] = (project / config[key]).resolve()
        if not within(config[key], project):
            raise ValueError(f'{key} must stay inside the project.')
    references = [(project / p).resolve() for p in config.get('reference_datasets', [])]
    config['reference_datasets'] = references
    protected = [project / 'datasets/raw', project / 'datasets/efficientnet_v3',
                 project / 'datasets/efficientnet_v4_cabbage_candidates', project / 'models',
                 config['archives_dir'], *references]
    outputs = [config[k] for k in ('work_dir', 'output_dir', 'reports_dir')]
    for output in outputs:
        if any(overlaps(output, original) for original in protected):
            raise ValueError(f'Output overlaps a protected source/model: {output}')
    for index, output in enumerate(outputs):
        if any(overlaps(output, other) for other in outputs[index+1:]):
            raise ValueError('Work, output and reports directories must be separate.')
    ratios = config['split']['ratios']
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or not math.isclose(sum(ratios), 1):
        raise ValueError('split.ratios must be three positive numbers summing to 1.')
    if not 0 <= config['audit']['phash_distance'] <= 16:
        raise ValueError('phash_distance must be between 0 and 16.')
    for source in config['sources'].values():
        archive = source['archive']
        if Path(archive).name != archive or not archive.lower().endswith('.zip'):
            raise ValueError('Each archive must be a simple ZIP filename.')
        for label, target in source.get('approved_mappings', {}).items():
            validate_target(target, config)
    for pattern in config.get('group_patterns', []):
        if re.compile(pattern).groups < 1:
            raise ValueError('Each group pattern needs capture group 1.')
    config['_project'] = project
    return config


def validate_target(target, config):
    if not isinstance(target, str) or '__' not in target:
        raise ValueError(f'Invalid target class: {target}')
    ingredient, state = target.split('__', 1)
    if ingredient not in config['ingredients'] or state not in config['states']:
        raise ValueError(f'Target outside approved ingredient/state scope: {target}')


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def safe_zip_members(archive, limits):
    """Validate the entire archive before writing anything (Windows-aware)."""
    members = archive.infolist()
    if len(members) > limits['max_files']:
        raise ValueError('ZIP has too many members.')
    total, seen = 0, set()
    validated = []
    devices = re.compile(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)', re.I)
    files, directories = set(), set()
    for info in members:
        name = info.filename.replace('\\', '/')
        parts = PurePosixPath(name).parts
        if not parts or name.startswith('/') or any(
                p in ('.', '..') or ':' in p or '\x00' in p or
                p.endswith((' ', '.')) or devices.match(p) for p in parts):
            raise ValueError(f'Unsafe ZIP path: {info.filename}')
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
            raise ValueError(f'ZIP links/special files are forbidden: {name}')
        key = '/'.join(parts).casefold()
        if key in seen:
            raise ValueError(f'Duplicate/case-colliding ZIP path: {name}')
        seen.add(key)
        parents = {'/'.join(parts[:i]).casefold() for i in range(1, len(parts))}
        if key in files or parents & files or (not info.is_dir() and key in directories):
            raise ValueError(f'ZIP file/directory collision: {name}')
        directories.update(parents)
        (directories if info.is_dir() else files).add(key)
        if info.flag_bits & 1:
            raise ValueError('Encrypted ZIPs are not supported.')
        total += info.file_size
        if (info.file_size > limits['max_file_bytes'] or total > limits['max_total_bytes'] or
                info.file_size / max(info.compress_size, 1) > limits['max_compression_ratio']):
            raise ValueError('ZIP exceeds extraction size/compression limits.')
        validated.append((info, Path(*parts)))
    return validated


def extract_once(path, work_dir, limits):
    digest = sha256_file(path)
    destination = Path(work_dir) / 'extracted' / digest
    marker = destination / '_bulk_extraction.json'
    if marker.exists():
        metadata = json.loads(marker.read_text(encoding='utf-8'))
        if metadata.get('archive_sha256') == digest and all(
                within(destination / item['path'], destination) and
                (destination / item['path']).is_file() and
                (destination / item['path']).stat().st_size == item['size'] and
                sha256_file(destination / item['path']) == item.get('sha256')
                for item in metadata.get('files', [])):
            return destination, digest, 'cached'
        raise ValueError('Cached extraction is incomplete/modified; inspect work directory manually.')
    if destination.exists():
        raise ValueError('Unmarked extraction directory exists; refusing overwrite.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        members = safe_zip_members(archive, limits)
        if any(relative.as_posix().casefold() == '_bulk_extraction.json' for _, relative in members):
            raise ValueError('ZIP uses a reserved extraction marker name.')
        # Temporary directory is tool-owned; original source paths are never removed.
        with tempfile.TemporaryDirectory(prefix='extract-', dir=destination.parent) as staging:
            staging = Path(staging)
            files = []
            for info, relative in members:
                target = staging / relative
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                files.append({'path': relative.as_posix(), 'size': info.file_size,
                              'sha256': sha256_file(target)})
            write_json(staging / '_bulk_extraction.json', {'archive_sha256': digest, 'files': files})
            # Rename only children into a fresh final directory. No original files move.
            destination.mkdir()
            for child in staging.iterdir():
                child.rename(destination / child.name)
    return destination, digest, 'extracted'


@dataclass
class Sample:
    path: Path
    root: Path
    source_split: str
    label: str
    bbox: tuple | None = None  # normalized YOLO center x/y/width/height
    index: int = 0
    format: str = 'classification'
    acquisition: str = ''
    review_reason: str = ''
    annotation_source: str = ''


def issue(source, image, label, reason, detail='', severity='error'):
    return dict(source_dataset=source, source_image=str(image), original_label=label,
                reason=reason, detail=str(detail), severity=severity)


def read_names(data):
    names = data.get('names')
    if isinstance(names, list):
        result = dict(enumerate(names))
    elif isinstance(names, dict):
        result = {int(key): value for key, value in names.items()}
    else:
        raise ValueError('YOLO data.yaml requires names list or mapping.')
    if not result or any(not isinstance(label, str) or not label for label in result.values()):
        raise ValueError('Invalid YOLO class names.')
    if set(result) != set(range(len(result))) or len(set(result.values())) != len(result):
        raise ValueError('YOLO class IDs must be contiguous; names must be unique.')
    return result


def image_label_path(image):
    parts = list(image.parts)
    matches = [i for i, value in enumerate(parts) if value.lower() == 'images']
    if not matches:
        raise ValueError('YOLO image path has no images directory; annotation location is ambiguous.')
    parts[matches[-1]] = 'labels'
    return Path(*parts).with_suffix('.txt')


def yolo_image_paths(spec, yaml_dir, data_root, archive_root):
    entries = spec if isinstance(spec, list) else [spec]
    found = set()
    for entry in entries:
        if not isinstance(entry, str):
            raise ValueError('YOLO split entries must be paths.')
        # A narrowly recognized Roboflow export convention is relocatable.
        # Arbitrary traversal/absolute paths must never be repaired by stripping.
        normalized = entry.replace('\\', '/')
        relative = PurePosixPath(normalized)
        exported_split = re.fullmatch(r'\.\./(train|val|valid|test)/images/?', normalized)
        if relative.is_absolute() or ':' in normalized or ('..' in relative.parts and not exported_split):
            raise ValueError(f'Unsafe YOLO split path: {entry}')
        options = [data_root / normalized, yaml_dir / normalized]
        if exported_split:
            options.append(yaml_dir / exported_split.group(1) / 'images')
        candidate = next((p.resolve() for p in options if within(p, archive_root) and p.exists()), None)
        if candidate is None:
            raise FileNotFoundError(f'YOLO split path absent inside archive: {entry}')
        if candidate.is_dir():
            paths = [p for p in candidate.rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()]
        elif candidate.suffix.lower() == '.txt':
            paths = []
            for line in candidate.read_text(encoding='utf-8-sig').splitlines():
                if not line.strip():
                    continue
                possibilities = [candidate.parent / line.strip(), data_root / line.strip()]
                path = next((p.resolve() for p in possibilities if within(p, archive_root) and p.is_file()), None)
                if path is None:
                    raise ValueError(f'Invalid image list entry: {line}')
                paths.append(path)
        else:
            paths = [candidate]
        for path in paths:
            if not within(path, archive_root) or path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError('Invalid/outside-archive YOLO image path.')
            found.add(path)
    return sorted(found)


def scan_yolo(root, yamls, source, spec=None):
    samples, issues, labels, image_count = [], [], set(), set()
    optional = set((spec or {}).get('optional_missing_splits', []))
    seen = set()
    for yaml_path in yamls:
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding='utf-8-sig'))
            names = read_names(data)
            labels.update(names.values())
            data_root = (yaml_path.parent / str(data.get('path', '.'))).resolve()
            if not within(data_root, root):
                data_root = yaml_path.parent  # archived absolute training paths are not trusted
            for split in ('train', 'val', 'valid', 'test'):
                if not data.get(split):
                    continue
                try:
                    images = yolo_image_paths(data[split], yaml_path.parent, data_root, root)
                except FileNotFoundError as error:
                    if split in optional:
                        issues.append(issue(source, yaml_path.relative_to(root), '', 'split_absent_in_archive',
                                            f'{split}: {data[split]}; no holdout created', severity='warning'))
                    else:
                        issues.append(issue(source, yaml_path.relative_to(root), '', 'invalid_split_path', error))
                    continue
                except Exception as error:
                    issues.append(issue(source, yaml_path.relative_to(root), '', 'invalid_split_path', error))
                    continue
                for image in images:
                    identity = (image, split)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    image_count.add(image)
                    try:
                        label_path = image_label_path(image)
                        if not within(label_path, root):
                            raise ValueError('Annotation outside archive.')
                        lines = label_path.read_text(encoding='utf-8-sig').splitlines()
                    except Exception as error:
                        issues.append(issue(source, image.relative_to(root), '', 'missing_annotation', error))
                        continue
                    for index, line in enumerate(lines):
                        if not line.strip():
                            continue
                        try:
                            values = line.split()
                            if len(values) != 5:
                                raise ValueError('Expected detection TXT: class_id xc yc width height (5 fields).')
                            class_id = int(values[0])
                            label = names[class_id]
                            box = tuple(float(v) for v in values[1:])
                            if not all(math.isfinite(v) for v in box):
                                raise ValueError('Bounding box contains non-finite values.')
                            x, y, w, h = box
                            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                                raise ValueError('YOLO normalized coordinates are outside range.')
                            samples.append(Sample(image, root, split, label, box, index, 'yolo', yaml_path.parent.relative_to(root).as_posix()))
                        except Exception as error:
                            issues.append(issue(source, image.relative_to(root), '', 'invalid_annotation', f'line {index+1}: {error}'))
        except Exception as error:
            issues.append(issue(source, yaml_path.relative_to(root), '', 'invalid_data_yaml', error))
    return samples, issues, labels, len(image_count)


def scan_classification_csv(root, source, tables):
    """Roboflow multi-class/multi-label CSV; ambiguous rows remain review-only.

    Preserve every active source label for inspect counts. Never choose argmax,
    infer a class from filename, or convert a multi-hot row into one label.
    """
    samples, issues, labels, images = [], [], set(), set()
    referenced, path_rows = set(), defaultdict(list)
    for table in tables:
        try:
            with table.open(encoding='utf-8-sig', newline='') as handle:
                reader = csv.reader(handle)
                raw_headers = next(reader)
                headers = [h.strip() for h in raw_headers]
                if len(headers) < 2 or headers[0].casefold() != 'filename' or len(set(headers)) != len(headers):
                    raise ValueError('Expected filename followed by distinct class columns.')
                labels.update(headers[1:])
                for line, values in enumerate(reader, 2):
                    if not values or not any(v.strip() for v in values):
                        continue
                    try:
                        if len(values) != len(headers):
                            raise ValueError('CSV field count does not match header.')
                        relative = PurePosixPath(values[0].strip().replace('\\', '/'))
                        if relative.is_absolute() or '..' in relative.parts or ':' in str(relative):
                            raise ValueError('Unsafe CSV image path.')
                        image = (table.parent / Path(*relative.parts)).resolve()
                        if not within(image, root) or not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
                            raise ValueError('CSV image missing or outside archive.')
                        referenced.add(image)
                        images.add(image)
                        flags = [v.strip() for v in values[1:]]
                        if any(value not in ('0', '1', '0.0', '1.0') for value in flags):
                            raise ValueError('Classification flags must be binary 0/1.')
                        active = [label for label, value in zip(headers[1:], flags) if float(value) == 1]
                        reason = '' if len(active) == 1 else ('multilabel_classification' if active else 'unlabeled_classification')
                        if reason:
                            issues.append(issue(source, image.relative_to(root), json.dumps(active), reason,
                                                f'{table.relative_to(root)} line {line}; raw flags={flags}', severity='review'))
                        for label in active:
                            sample = Sample(image, root, table.parent.name, label, index=line,
                                            format='classification_csv', acquisition=table.parent.parent.relative_to(root).as_posix(),
                                            review_reason=reason, annotation_source=f'{table.relative_to(root).as_posix()}:{line}')
                            path_rows[image].append(sample)
                            samples.append(sample)
                    except Exception as error:
                        issues.append(issue(source, table.relative_to(root), '', 'invalid_classification_row', f'line {line}: {error}'))
        except Exception as error:
            issues.append(issue(source, table.relative_to(root), '', 'invalid_classification_csv', error))
    # Multiple annotation rows/tables for one image must not silently disagree.
    for image, records in path_rows.items():
        if len({r.annotation_source for r in records}) > 1:
            for sample in records:
                sample.review_reason = 'duplicate_classification_annotation'
            issues.append(issue(source, image.relative_to(root), '', 'duplicate_classification_annotation', severity='review'))
    for image in sorted(root.rglob('*')):
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
            images.add(image.resolve())
            if image.resolve() not in referenced:
                issues.append(issue(source, image.relative_to(root), '', 'image_missing_csv_annotation', severity='review'))
    return samples, issues, labels, len(images)


def scan_classification(root, source, label_mode='top'):
    samples, issues, labels = [], [], set()
    split_names = {'train', 'val', 'valid', 'validation', 'test'}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parts = path.relative_to(root).parts
        indices = [i for i, part in enumerate(parts[:-1]) if part.lower() in split_names]
        if indices:
            index = indices[-1]
            if index + 2 >= len(parts):
                issues.append(issue(source, path.relative_to(root), '', 'ambiguous_class_folder'))
                continue
            label = '/'.join(parts[index+1:-1]) if label_mode == 'relative' else parts[index+1]
            split = parts[index].lower()
            # All class folders under the same split-independent dataset root
            # are conservatively considered one acquisition until provenance is known.
            acquisition = '/'.join(parts[:index])
        elif len(parts) >= 2:
            label, split = path.parent.name, 'unspecified'
            acquisition = '/'.join(parts[:-2])
        else:
            issues.append(issue(source, path.relative_to(root), '', 'missing_class_folder'))
            continue
        labels.add(label)
        samples.append(Sample(path, root, split, label, acquisition=acquisition))
    return samples, issues, labels, len(samples)


def scan_source(root, source, spec):
    yamls = sorted(p for p in root.rglob('*') if p.name.lower() in ('data.yaml', 'data.yml'))
    mode = spec.get('format', 'auto')
    if mode not in ('auto', 'yolo', 'classification'):
        raise ValueError('Source format must be auto/yolo/classification.')
    if mode == 'yolo' or (mode == 'auto' and yamls):
        if not yamls:
            raise ValueError('YOLO source has no data.yaml.')
        return (*scan_yolo(root, yamls, source, spec), 'yolo')
    tables = sorted(root.rglob('_classes.csv'))
    if tables:
        return (*scan_classification_csv(root, source, tables), 'classification_csv')
    label_mode = spec.get('classification_label_mode', 'top')
    if label_mode not in ('top', 'relative'):
        raise ValueError('classification_label_mode must be top or relative.')
    return (*scan_classification(root, source, label_mode), 'classification')


def mapping_for(source, label, config):
    if label.strip().casefold() in {s.casefold() for s in config.get('blocked_labels', [])}:
        return '', '', 'blocked_label'
    spec = config['sources'][source]
    if label in spec.get('review_labels', []):
        return '', spec.get('proposed_mappings', {}).get(label, ''), 'needs_review'
    exact = config['sources'][source].get('approved_mappings', {}).get(label)
    if exact:
        validate_target(exact, config)
        return exact, exact, 'approved_exact'
    proposed = ''
    # Suggestions never authorize output. Fresh vs Spoiled must be explicit;
    # anatomical cut names alone do not encode food state.
    if source == 'pork':
        words = re.sub(r'[^a-z0-9]', '', label.casefold())
        cut = r'(?:kasim|liempo|porkchop|tenderloin)'
        if re.fullmatch(fr'(?:fresh{cut}|{cut}fresh)', words):
            proposed = 'pork__normal_appearance'
        elif re.fullmatch(fr'(?:spoiled{cut}|{cut}spoiled)', words):
            proposed = 'pork__visible_defect'
    return '', proposed, 'needs_approval'


def load_rgb(path, max_pixels, exif=True):
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(path) as image:
            if image.width * image.height > max_pixels:
                raise ValueError('Image exceeds max_pixels.')
            image.load()
            # YOLO coordinates refer to the stored pixel grid, not EXIF rotation.
            return (ImageOps.exif_transpose(image) if exif else image).convert('RGB')


def crop_yolo(image, box, minimum=16):
    x, y, width, height = box
    left = max(0, math.floor((x-width/2) * image.width))
    top = max(0, math.floor((y-height/2) * image.height))
    right = min(image.width, math.ceil((x+width/2) * image.width))
    bottom = min(image.height, math.ceil((y+height/2) * image.height))
    if right-left < minimum or bottom-top < minimum:
        raise ValueError('Bounding box is outside image or crop is too small.')
    return image.crop((left, top, right, bottom)), (left, top, right, bottom)


@lru_cache(maxsize=1)
def dct_basis():
    positions = np.arange(32)
    basis = np.cos(np.pi * np.arange(8)[:, None] * (2 * positions + 1) / 64)
    basis[0] *= 1 / math.sqrt(2)
    return basis * math.sqrt(2 / 32)


def phash(image):
    gray = np.asarray(image.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    low = dct_basis() @ gray @ dct_basis().T
    bits = low.ravel() > np.median(low.ravel()[1:])
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f'{value:016x}'


def original_stem(stem):
    previous = None
    while stem != previous:
        previous = stem
        stem = re.sub(r'^(?:rotated_by_[+-]?\d+(?:\.\d+)?_|(?:horizontal|vertical)_flip_|salt_?and_?pepper_|aug(?:mented)?[_-]\d*[_-]?)', '', stem, flags=re.I)
    stem = re.sub(r'\.rf\.[a-f0-9]+$', '', stem, flags=re.I)
    return stem.casefold()


def group_identity(sample, source, patterns):
    relative = sample.path.relative_to(sample.root).as_posix()
    for pattern in patterns:
        match = re.search(pattern, relative, flags=re.I)
        if match:
            return f'explicit:{match.group(1).casefold()}', 'explicit_pattern'
    stem = original_stem(sample.path.stem)
    # Match common Roboflow video exports: clip_mp4-123, clip_frame_001, etc.
    video = re.match(r'^(.*?)(?:[._-](?:mp4|avi|mov|mkv)[_-]?\d+|[_-](?:frame|frames)[_-]?\d+)', stem)
    if video:
        return f'video:{video.group(1)}', 'video_filename'
    screenshot = re.match(r'^(?:screen shot|screenshot)[ _-]*(\d{4}[-_]\d{2}[-_]\d{2})', stem)
    if screenshot:
        return f'{source}:capture_session:{screenshot.group(1)}', 'screenshot_session'
    # Unknown provenance is never assumed to represent independent objects.
    # This also keeps differently named augmentations from one export together.
    return f'{source}:acquisition:{sample.acquisition or "unknown"}', 'conservative_acquisition'


def review_contact_sheets(config, source, samples, spec):
    """Deterministic examples, full image + annotated box + crop; never approve."""
    wanted = spec.get('contact_sheet_labels', [])
    if not wanted:
        return [], []
    output = config['reports_dir'] / f'{source}_review'
    output.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
        title_font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 22)
    except OSError:
        font = title_font = ImageFont.load_default()
    rows, errors, sheet_links = [], [], []
    for label in wanted:
        pool = [s for s in samples if s.label == label]
        by_split = defaultdict(list)
        for sample in sorted(pool, key=lambda s: hashlib.sha256(f'{s.path.relative_to(s.root)}:{s.index}'.encode()).hexdigest()):
            by_split[sample.source_split].append(sample)
        selected, stems = [], set()
        limit = max(1, min(int(spec.get('review_images_per_label', 6)), 24))
        while len(selected) < limit and any(by_split.values()):
            for split in sorted(by_split):
                while by_split[split]:
                    sample = by_split[split].pop(0)
                    key = original_stem(sample.path.stem)
                    if key not in stems:
                        stems.add(key)
                        selected.append(sample)
                        break
                if len(selected) == limit:
                    break
        width, tile_h, header_h = 920, 290, 90
        canvas = Image.new('RGB', (width, header_h + max(1, math.ceil(len(selected)/2)) * tile_h), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 12), f'{source}: {label} | needs_approval', fill='black', font=title_font)
        draw.text((16, 46), 'Source labels only. Left: image + box. Right: crop. No label approval implied.', fill='black', font=font)
        slug = re.sub(r'[^a-z0-9]+', '_', label.casefold()).strip('_')
        sheet = output / f'{slug}.jpg'
        for index, sample in enumerate(selected):
            x, y = (index % 2) * 460, header_h + (index // 2) * tile_h
            try:
                image = load_rgb(sample.path, config['image_limits']['max_pixels'], exif=False)
                crop, box = crop_yolo(image, sample.bbox, 1) if sample.bbox else (image.copy(), None)
                full = image.copy()
                if box:
                    ImageDraw.Draw(full).rectangle(box, outline='red', width=max(2, image.width//200))
                for offset, panel in [(0, full), (225, crop)]:
                    panel.thumbnail((215, 210), Image.Resampling.LANCZOS)
                    canvas.paste(panel, (x+8+offset, y+28))
                draw.text((x+8, y+4), f'#{index+1} | {sample.source_split}', fill='black', font=font)
                draw.text((x+8, y+244), sample.path.name[:48], fill='black', font=font)
                rows.append({'label': label, 'sample_number': index+1, 'source_split': sample.source_split,
                             'source_image': sample.path.relative_to(sample.root).as_posix(),
                             'source_sha256': sha256_file(sample.path), 'bounding_box': json.dumps(box),
                             'sheet': sheet.relative_to(config['reports_dir']).as_posix(), 'approval': 'needs_approval'})
            except Exception as error:
                draw.text((x+8, y+28), f'Image unavailable: {type(error).__name__}', fill='red', font=font)
                errors.append(issue(source, sample.path.relative_to(sample.root), label, 'review_sample_error', error))
        buffer = io.BytesIO()
        canvas.save(buffer, format='JPEG', quality=90)
        atomic_bytes(sheet, buffer.getvalue())
        sheet_links.append((label, sheet.name, len(pool)))
    write_csv(output / 'contact_sheet_samples.csv', rows,
              ['label', 'sample_number', 'source_split', 'source_image', 'source_sha256', 'bounding_box', 'sheet', 'approval'])
    lines = [f'# {source} label review', '', 'All labels below remain **needs_approval**. These examples cannot establish actual spoilage or validate the whole source.',
             'Review crop correctness, source label semantics, sampling conditions and visual consistency before approving an exact label.',
             '', 'Full source metadata: ../source_metadata.json. Full label counts: ../inspect_labels.csv.', '']
    for label, name, count in sheet_links:
        lines += [f'## {label} ({count} bounding boxes)', '', f'![{label}]({name})', '']
    atomic_bytes(output / 'README.md', ('\n'.join(lines)+'\n').encode('utf-8'))
    return rows, errors


def source_structure(root, source):
    folders, metadata = Counter(), []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            folders[path.parent.relative_to(root).as_posix()] += 1
        if path.name.lower() in ('data.yaml', 'data.yml', 'readme.dataset.txt', 'readme.roboflow.txt', '_classes.csv'):
            content = path.read_text(encoding='utf-8-sig', errors='replace')
            metadata.append({'path': relative, 'sha256': sha256_file(path),
                             'content': content if path.name != '_classes.csv' else content.splitlines()[0],
                             'content_scope': 'header_only' if path.name == '_classes.csv' else 'full'})
    return [{'source_dataset': source, 'folder': folder, 'image_count': count} for folder, count in sorted(folders.items())], metadata


def inspect_summary(config, counts, statuses, source_samples):
    targets = []
    for ingredient in config['ingredients']:
        for state in config['states']:
            target = f'{ingredient}__{state}'
            accepted = [(source, sample) for source, samples in source_samples.items() for sample in samples
                        if mapping_for(source, sample.label, config)[0] == target and not sample.review_reason]
            groups = {group_identity(s, source, config.get('group_patterns', []))[0] for source, s in accepted}
            targets.append({'target_class': target, 'approved_source_samples': len(accepted),
                            'provisional_source_groups': len(groups), 'status': 'available_for_prepare' if accepted else 'missing_approved_samples'})
    write_csv(config['reports_dir'] / 'inspect_class_summary.csv', targets, ['target_class', 'approved_source_samples', 'provisional_source_groups', 'status'])
    write_json(config['reports_dir'] / 'inspect_summary.json', {
        'readable_sources': sum(s.get('readable', False) for s in statuses),
        'sources_without_blocking_errors': sum(s.get('readable', False) and not s.get('blocking_issue_count', 0) for s in statuses),
        'configured_sources': len(config['sources']), 'classes': targets,
        'missing_approved_classes': [r['target_class'] for r in targets if not r['approved_source_samples']],
        'note': 'Source groups are conservative hypotheses, not verified independent samples. Real crop/hash counts require prepare/audit.'})


def inventory(config):
    counts, issues, statuses, source_samples = [], [], [], {}
    folder_rows, metadata, classification_rows = [], {}, []
    archives = config['archives_dir']
    available = {p.name.casefold(): p for p in archives.iterdir() if p.is_file() and p.suffix.lower() == '.zip'} if archives.exists() else {}
    known = {s['archive'].casefold() for s in config['sources'].values()}
    for key, path in available.items():
        if key not in known:
            statuses.append(dict(source_dataset=path.stem, archive=path.name, status='unconfigured_archive', severity='info', detail='Informational only; not extracted or used by prepare.'))
    for source, spec in config['sources'].items():
        path = available.get(spec['archive'].casefold())
        if path is None:
            statuses.append(dict(source_dataset=source, archive=spec['archive'], status='missing_archive', detail=''))
            continue
        try:
            print(f'Inspecting {source}: verifying archive/cache...', flush=True)
            root, digest, extraction = extract_once(path, config['work_dir'], config['zip_limits'])
            samples, errors, labels, images, mode = scan_source(root, source, spec)
            folders, source_metadata = source_structure(root, source)
            folder_rows.extend(folders)
            metadata[source] = {'archive': path.name, 'archive_sha256': digest, 'files': source_metadata}
            _, sheet_errors = review_contact_sheets(config, source, samples, spec)
            errors.extend(sheet_errors)
            issues.extend(errors)
            source_samples[source] = samples
            box_counts = Counter(sample.label for sample in samples)
            unique_images = defaultdict(set)
            for sample in samples:
                unique_images[sample.label].add(sample.path)
            for label in sorted(labels):
                accepted, proposed, status = mapping_for(source, label, config)
                counts.append(dict(source_dataset=source, format=mode, original_label=label,
                                   image_count=len(unique_images[label]), bounding_box_count=box_counts[label] if mode == 'yolo' else 0,
                                   review_sample_count=sum(s.label == label and bool(s.review_reason) for s in samples),
                                   proposed_mapping=proposed, approved_mapping=accepted, mapping_status=status))
            for sample in samples:
                if sample.format == 'classification_csv':
                    classification_rows.append({'source_dataset': source, 'source_image': sample.path.relative_to(root).as_posix(),
                                                'source_split': sample.source_split, 'original_label': sample.label,
                                                'annotation_source': sample.annotation_source, 'review_reason': sample.review_reason,
                                                'mapping_status': mapping_for(source, sample.label, config)[2]})
            severities = Counter(e['severity'] for e in errors)
            statuses.append(dict(source_dataset=source, archive=path.name,
                                 status='inspected_with_issues' if severities['error'] else 'needs_review' if severities['review'] else 'inspected_with_warnings' if errors else 'inspected',
                                 archive_sha256=digest, extraction=extraction, images=images,
                                 samples=len(samples), issue_count=len(errors), readable=True,
                                 blocking_issue_count=severities['error'], warning_count=severities['warning'],
                                 review_issue_count=severities['review'], detail=''))
        except Exception as error:
            statuses.append(dict(source_dataset=source, archive=path.name, status='source_error', detail=str(error)))
            issues.append(issue(source, path.name, '', 'source_error', error))
    write_csv(config['reports_dir'] / 'inspect_labels.csv', counts,
              ['source_dataset', 'format', 'original_label', 'image_count', 'bounding_box_count', 'review_sample_count', 'proposed_mapping', 'approved_mapping', 'mapping_status'])
    write_csv(config['reports_dir'] / 'source_folders.csv', folder_rows, ['source_dataset', 'folder', 'image_count'])
    write_json(config['reports_dir'] / 'source_metadata.json', metadata)
    write_csv(config['reports_dir'] / 'classification_annotations.csv', classification_rows,
              ['source_dataset', 'source_image', 'source_split', 'original_label', 'annotation_source', 'review_reason', 'mapping_status'])
    inspect_summary(config, counts, statuses, source_samples)
    write_json(config['reports_dir'] / 'source_status.json', statuses)
    write_csv(config['reports_dir'] / 'inspect_issues.csv', issues, ISSUE_FIELDS)
    return source_samples, issues, statuses


def prepare(config):
    source_samples, issues, statuses = inventory(config)
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    manifest = []
    for source, samples in source_samples.items():
        accepted_count, rejected_count, invalid_count = 0, 0, 0
        last_path, last_image, last_source_sha = None, None, None
        for sample in samples:
            target, _, mapping_status = mapping_for(source, sample.label, config)
            if mapping_status != 'approved_exact' or not target or sample.review_reason:
                rejected_count += 1
                issues.append(issue(source, sample.path.relative_to(sample.root), sample.label, sample.review_reason or mapping_status, severity='review'))
                continue
            try:
                if sample.path != last_path:
                    last_image = load_rgb(sample.path, config['image_limits']['max_pixels'], exif=sample.format != 'yolo')
                    last_source_sha = sha256_file(sample.path)
                    last_path = sample.path
                image = last_image
                pixel_box = None
                if sample.bbox:
                    image, pixel_box = crop_yolo(image, sample.bbox, config['image_limits']['min_crop_size'])
                # Output pixels already have the intended orientation. Do not
                # retain a source EXIF orientation tag on a cropped PNG.
                image.info.clear()
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                payload = buffer.getvalue()
                digest = hashlib.sha256(payload).hexdigest()
                crop_path = config['output_dir'] / target / f'{digest}.png'
                if crop_path.exists():
                    if sha256_file(crop_path) != digest:
                        raise ValueError('Existing candidate content differs from its SHA filename.')
                else:
                    atomic_bytes(crop_path, payload)
                group, method = group_identity(sample, source, config.get('group_patterns', []))
                source_image = sample.path.relative_to(sample.root).as_posix()
                identity = f'{source}|{source_image}|{sample.source_split}|{sample.index}|{sample.label}|{target}'
                manifest.append(dict(source_dataset=source, source_image=source_image,
                                     source_split=sample.source_split, original_label=sample.label, target_class=target,
                                     bounding_box=json.dumps({'yolo_xywh': sample.bbox, 'pixel_xyxy': pixel_box}) if pixel_box else '',
                                     image_group=group, crop_path=crop_path.relative_to(config['_project']).as_posix(),
                                     sha256=digest, phash=phash(image), source_sha256=last_source_sha,
                                     group_method=method, record_id=hashlib.sha256(identity.encode()).hexdigest()[:24]))
                accepted_count += 1
            except Exception as error:
                invalid_count += 1
                issues.append(issue(source, sample.path.relative_to(sample.root), sample.label, 'invalid_image_or_crop', error))
        for status in statuses:
            if status['source_dataset'] == source and status.get('archive') == config['sources'][source]['archive']:
                status.update(status='prepared_with_issues' if status.get('issue_count') or invalid_count else 'prepared_candidates', accepted_records=accepted_count,
                              withheld_records=rejected_count, invalid_records=invalid_count)
    write_csv(config['reports_dir'] / 'manifest.csv', manifest, MANIFEST_FIELDS)
    write_csv(config['reports_dir'] / 'prepare_review.csv', issues, ISSUE_FIELDS)
    write_json(config['reports_dir'] / 'source_status.json', statuses)
    write_json(config['reports_dir'] / 'prepare_state.json', {
        'config_sha256': config_digest(config), 'records': len(manifest),
        'archive_snapshot': archive_snapshot(config)})
    # An audit is mandatory after every prepare; old split recommendations are
    # invalidated even if the current preparation produced zero candidates.
    audit(config)
    return manifest


class UnionFind:
    def __init__(self, count):
        self.parents = list(range(count))
        self.sizes = [1] * count

    def find(self, node):
        while self.parents[node] != node:
            self.parents[node] = self.parents[self.parents[node]]
            node = self.parents[node]
        return node

    def union(self, first, second):
        first, second = self.find(first), self.find(second)
        if first != second:
            if self.sizes[first] < self.sizes[second]:
                first, second = second, first
            self.parents[second] = first
            self.sizes[first] += self.sizes[second]


class BKTree:
    """Index distinct 64-bit hashes, retaining exact duplicates in separate lists."""
    def __init__(self):
        self.root = None

    def insert(self, value):
        if self.root is None:
            self.root = (value, {})
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                return
            if distance not in node[1]:
                node[1][distance] = (value, {})
                return
            node = node[1][distance]

    def query(self, value, radius):
        stack = [self.root] if self.root else []
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                yield node[0], distance
            stack.extend(child for edge, child in node[1].items() if distance-radius <= edge <= distance+radius)


def reference_class(path, root):
    parts = path.relative_to(root).parts
    for part in parts[:-1]:
        if '__' in part:
            return part
    return ''


def reference_records(config):
    records, issues, statuses = [], [], []
    cache_path = config['work_dir'] / 'reference_cache.json'
    cache = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else {}
    updated = {}
    for root in config['reference_datasets']:
        if not root.is_dir():
            statuses.append({'dataset': str(root), 'status': 'missing_reference', 'images': 0})
            continue
        count = 0
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if not within(path, root):
                issues.append(issue(root.name, path, '', 'reference_symlink_outside_root'))
                continue
            count += 1
            stats = path.stat()
            key = str(path.resolve())
            signature = [stats.st_size, stats.st_mtime_ns]
            cached = cache.get(key, {})
            try:
                if cached.get('signature') == signature:
                    record = cached['record']
                else:
                    image = load_rgb(path, config['image_limits']['max_pixels'])
                    record = {'source_dataset': root.name, 'source_image': path.relative_to(root).as_posix(),
                              'target_class': reference_class(path, root), 'crop_path': str(path),
                              'sha256': sha256_file(path), 'phash': phash(image), 'kind': 'reference'}
                updated[key] = {'signature': signature, 'record': record}
                records.append(record)
            except Exception as error:
                issues.append(issue(root.name, path, '', 'invalid_reference_image', error))
        statuses.append({'dataset': str(root), 'status': 'audited' if count else 'empty_reference', 'images': count})
    write_json(cache_path, updated)
    return records, issues, statuses


def plan_splits(rows, group_ids, review_reasons, config):
    """Plan only. Component-level allocation ensures no image group crosses splits."""
    classes = defaultdict(set)
    members = defaultdict(list)
    for i, (row, group) in enumerate(zip(rows, group_ids)):
        classes[row['target_class']].add(group)
        members[group].append(i)
    blocked = {group_ids[i] for i, reasons in review_reasons.items() if reasons}
    plans = {}
    # Connected class families share an allocation. If any class in a family
    # lacks independent groups, the whole family stays train_candidate.
    class_names = sorted(classes)
    family = UnionFind(len(class_names))
    owners = {}
    for index, name in enumerate(class_names):
        for group in classes[name]:
            if group in owners:
                family.union(index, owners[group])
            owners[group] = index
    families = defaultdict(list)
    for index, name in enumerate(class_names):
        families[family.find(index)].append(name)
    seed = config['split']['seed']
    for names in families.values():
        groups = sorted(set().union(*(classes[name] for name in names)) - blocked)
        eligible_counts = {name: len(classes[name] - blocked) for name in names}
        if any(count < config['split']['min_groups_per_class'] for count in eligible_counts.values()):
            plans.update({group: 'train_candidate' for group in groups})
            continue
        random.Random(seed).shuffle(groups)
        # Search deterministic proposals, checking actual per-class coverage.
        accepted = None
        for attempt in range(100):
            random.Random(seed + attempt).shuffle(groups)
            train_end = int(len(groups) * config['split']['ratios'][0])
            val_end = train_end + max(1, round(len(groups) * config['split']['ratios'][1]))
            candidate = {g: ('train' if i < train_end else 'val' if i < val_end else 'test') for i, g in enumerate(groups)}
            minimum = config['split']['min_groups_per_holdout']
            if all(sum(candidate[g] == split for g in classes[name] - blocked) >= (1 if split == 'train' else minimum)
                   for name in names for split in ('train', 'val', 'test')):
                accepted = candidate
                break
        plans.update(accepted or {group: 'train_candidate' for group in groups})
    result = []
    for i, row in enumerate(rows):
        group = group_ids[i]
        reasons = sorted(review_reasons.get(i, set()))
        if group in blocked and not reasons:
            reasons.append('linked_to_review_group')
        result.append({**row, 'audit_group': group,
                       'proposed_split': 'review' if group in blocked else plans.get(group, 'train_candidate'),
                       'review_reasons': ';'.join(reasons)})
    return result


def audit(config):
    manifest_path = config['reports_dir'] / 'manifest.csv'
    if not manifest_path.exists():
        raise ValueError('Run prepare before audit (manifest.csv is missing).')
    state_path = config['reports_dir'] / 'prepare_state.json'
    if not state_path.exists() or json.loads(state_path.read_text(encoding='utf-8'))['config_sha256'] != config_digest(config):
        raise ValueError('Configuration changed since prepare; rerun prepare before audit.')
    prepare_state = json.loads(state_path.read_text(encoding='utf-8'))
    if prepare_state.get('archive_snapshot') != archive_snapshot(config):
        raise ValueError('Archive inputs changed since prepare; rerun prepare before audit.')
    rows = read_csv(manifest_path)
    reasons = defaultdict(set)
    issues = []
    for i, row in enumerate(rows):
        try:
            path = (config['_project'] / row['crop_path']).resolve()
            if not within(path, config['output_dir']):
                raise ValueError('Candidate path escapes output directory.')
            if sha256_file(path) != row['sha256']:
                raise ValueError('Candidate SHA256 changed since prepare.')
            image = load_rgb(path, config['image_limits']['max_pixels'])
            if phash(image) != row['phash']:
                raise ValueError('Candidate pHash changed since prepare.')
        except Exception as error:
            reasons[i].add('invalid_or_modified_candidate')
            issues.append(issue(row['source_dataset'], row['source_image'], row['original_label'], 'invalid_or_modified_candidate', error))
    references, reference_issues, reference_status = reference_records(config) if rows else ([], [], [])
    issues.extend(reference_issues)
    references_ok = bool(reference_status) and all(r['status'] == 'audited' for r in reference_status) and not reference_issues
    if config['audit'].get('require_reference_audit') and not references_ok:
        for i in range(len(rows)):
            reasons[i].add('reference_audit_incomplete')
    all_rows = [{**r, 'kind': 'candidate'} for r in rows] + references
    uf = UnionFind(len(all_rows))
    group_first, parent_first, original_first, sha_nodes, hash_nodes = {}, {}, {}, defaultdict(list), defaultdict(list)
    for i, row in enumerate(all_rows):
        sha_nodes[row['sha256']].append(i)
        hash_nodes[int(row['phash'], 16)].append(i)
        if i < len(rows):
            for index, key in [(group_first, row['image_group']),
                               (parent_first, (row['source_dataset'], row['source_image']))]:
                if key in index:
                    uf.union(i, index[key])
                index[key] = i
            original_sha = row.get('source_sha256')
            if original_sha:
                if original_sha in original_first:
                    uf.union(i, original_first[original_sha])
                original_first[original_sha] = i
    pairs = []
    total_pairs = 0
    max_pairs = config['audit']['max_report_pairs']

    def register(first, second, kind, distance):
        nonlocal total_pairs
        uf.union(first, second)
        if first >= len(rows) and second >= len(rows):
            return
        total_pairs += 1
        a, b = all_rows[first], all_rows[second]
        conflict = bool(a['target_class'] and b['target_class'] and a['target_class'] != b['target_class'])
        # pHash is a review signal, never an automatic label decision.
        for i, other in [(first, b), (second, a)]:
            if i < len(rows):
                if conflict:
                    reasons[i].add('conflicting_labels')
                if other['kind'] == 'reference':
                    reasons[i].add('overlap_reference')
                if kind == 'near_phash':
                    reasons[i].add('near_duplicate_review')
                if kind == 'exact_phash' and a['sha256'] != b['sha256']:
                    reasons[i].add('phash_duplicate_review')
        if len(pairs) < max_pairs:
            pairs.append({'first': a['crop_path'], 'second': b['crop_path'], 'kind': kind,
                          'distance': distance, 'first_class': a['target_class'], 'second_class': b['target_class'],
                          'label_conflict': conflict, 'reference_overlap': a['kind'] != b['kind']})

    # Representative-star links are sufficient for exact duplicate components;
    # export group membership separately rather than quadratic all-pairs lists.
    for indices in sha_nodes.values():
        for i in indices[1:]:
            register(indices[0], i, 'sha256', 0)
    for indices in hash_nodes.values():
        representative = indices[0]
        for i in indices[1:]:
            register(representative, i, 'exact_phash', 0)
        classes = {all_rows[i]['target_class'] for i in indices if all_rows[i]['target_class']}
        has_reference = any(i >= len(rows) for i in indices)
        for i in indices:
            if i < len(rows):
                if len(classes) > 1:
                    reasons[i].add('conflicting_labels')
                if has_reference:
                    reasons[i].add('overlap_reference')
    tree = BKTree()
    radius = config['audit']['phash_distance']
    for value, indices in sorted(hash_nodes.items()):
        for other, distance in tree.query(value, radius):
            register(indices[0], hash_nodes[other][0], 'near_phash', distance)
            # Propagate review reasons to all exact-hash members of both nodes.
            for i in indices + hash_nodes[other]:
                if i < len(rows):
                    reasons[i].add('near_duplicate_review')
        tree.insert(value)
    components = defaultdict(list)
    for i in range(len(all_rows)):
        components[uf.find(i)].append(i)
    ids = {}
    for component in components.values():
        candidate_indices = [i for i in component if i < len(rows)]
        if not candidate_indices:
            continue
        key = '|'.join(sorted(rows[i]['record_id'] for i in candidate_indices))
        group = hashlib.sha256(key.encode()).hexdigest()[:24]
        for i in candidate_indices:
            ids[i] = group
    # Deduplication retains all provenance rows, but only one eligible row per
    # target + output image may be proposed for a split.
    seen_candidates = set()
    unique_indices = []
    for i, row in enumerate(rows):
        key = (row['target_class'], row['sha256'])
        if key not in seen_candidates:
            unique_indices.append(i)
            seen_candidates.add(key)
    # All provenance/parent links stay represented in component IDs.
    component_reasons = defaultdict(set)
    for i, values in reasons.items():
        component_reasons[ids[i]].update(values)
    selected_reasons = {j: component_reasons[ids[i]] for j, i in enumerate(unique_indices)}
    plan = plan_splits([rows[i] for i in unique_indices], [ids[i] for i in unique_indices], selected_reasons, config)
    write_csv(config['reports_dir'] / 'split_plan.csv', plan, MANIFEST_FIELDS + ['audit_group', 'proposed_split', 'review_reasons'])
    write_csv(config['reports_dir'] / 'audit_pairs.csv', pairs,
              ['first', 'second', 'kind', 'distance', 'first_class', 'second_class', 'label_conflict', 'reference_overlap'])
    duplicate_groups = []
    for kind, nodes in [('sha256', sha_nodes), ('phash', hash_nodes)]:
        for key, indices in nodes.items():
            if len(indices) > 1 and any(i < len(rows) for i in indices):
                duplicate_groups.append({'kind': kind, 'hash': str(key), 'count': len(indices),
                                         'members': json.dumps([all_rows[i]['crop_path'] for i in indices]),
                                         'classes': json.dumps(sorted({all_rows[i]['target_class'] for i in indices}))})
    write_csv(config['reports_dir'] / 'duplicate_groups.csv', duplicate_groups, ['kind', 'hash', 'count', 'members', 'classes'])
    write_csv(config['reports_dir'] / 'audit_issues.csv', issues, ISSUE_FIELDS)
    active_paths = {(config['_project'] / row['crop_path']).resolve() for row in rows}
    unreferenced = [{'crop_path': p.relative_to(config['_project']).as_posix(),
                     'reason': 'not_in_current_manifest_do_not_train'}
                    for p in config['output_dir'].rglob('*.png') if p.resolve() not in active_paths] if config['output_dir'].exists() else []
    write_csv(config['reports_dir'] / 'unreferenced_candidates.csv', unreferenced, ['crop_path', 'reason'])
    write_json(config['reports_dir'] / 'audit_state.json', {
        'config_sha256': config_digest(config), 'manifest_sha256': sha256_file(manifest_path),
        'candidate_records': len(rows), 'unique_candidate_images': len(plan),
        'crop_records': sum(bool(r['bounding_box']) for r in rows),
        'classification_records': sum(not r['bounding_box'] for r in rows),
        'source_images': len({(r['source_dataset'], r['source_image']) for r in rows}),
        'conservative_audit_groups': len({r['audit_group'] for r in plan}),
        'groups_without_review': len({r['audit_group'] for r in plan if r['proposed_split'] != 'review'}),
        'reference_status': reference_status, 'reference_audit_complete': references_ok,
        'pair_links_found': total_pairs, 'pair_links_reported': len(pairs),
        'pair_report_truncated': total_pairs > max_pairs,
        'archive_snapshot': archive_snapshot(config),
        'candidate_snapshot': image_tree_snapshot([config['output_dir']]),
        'reference_snapshot': image_tree_snapshot(config['reference_datasets']) if rows else None,
        'unreferenced_candidate_files': len(unreferenced),
        'proposed_splits': dict(Counter(p['proposed_split'] for p in plan)),
        'merge_status': 'NOT_MERGED_REVIEW_REQUIRED'})
    return plan


def summary(config):
    reports = config['reports_dir']
    manifest = read_csv(reports / 'manifest.csv') if (reports / 'manifest.csv').exists() else []
    plan = read_csv(reports / 'split_plan.csv') if (reports / 'split_plan.csv').exists() else []
    audit_state = json.loads((reports / 'audit_state.json').read_text(encoding='utf-8')) if (reports / 'audit_state.json').exists() else {}
    valid_audit = (bool(audit_state) and (reports / 'manifest.csv').exists()
                   and audit_state.get('config_sha256') == config_digest(config)
                   and audit_state.get('manifest_sha256') == sha256_file(reports / 'manifest.csv')
                   and audit_state.get('archive_snapshot') == archive_snapshot(config)
                   and audit_state.get('candidate_snapshot') == image_tree_snapshot([config['output_dir']])
                   and (not manifest or audit_state.get('reference_snapshot') == image_tree_snapshot(config['reference_datasets'])))
    counts = []
    for ingredient in config['ingredients']:
        for state in config['states']:
            target = f'{ingredient}__{state}'
            selected = [r for r in manifest if r['target_class'] == target]
            planned = [r for r in plan if r['target_class'] == target] if valid_audit else []
            counts.append({'ingredient': ingredient, 'state': state, 'target_class': target,
                           'records': len(selected), 'unique_images': len({r['sha256'] for r in selected}),
                           'independent_audit_groups': len({r['audit_group'] for r in planned}),
                           'review_images': sum(r['proposed_split'] == 'review' for r in planned),
                           'train_candidate': sum(r['proposed_split'] == 'train_candidate' for r in planned),
                           'status': 'missing' if not selected else 'candidate_only'})
    statuses = json.loads((reports / 'source_status.json').read_text(encoding='utf-8')) if (reports / 'source_status.json').exists() else []
    result = {'audit_current': valid_audit, 'merge_status': 'NOT_MERGED_REVIEW_REQUIRED',
              'classes': counts, 'sources': statuses,
              'totals': {key: audit_state.get(key) if valid_audit else None for key in
                         ('candidate_records', 'unique_candidate_images', 'crop_records', 'classification_records',
                          'source_images', 'conservative_audit_groups', 'groups_without_review')},
              'missing_classes': [r['target_class'] for r in counts if not r['unique_images']],
              'audit': audit_state}
    write_csv(reports / 'class_summary.csv', counts, list(counts[0]) if counts else [])
    write_json(reports / 'summary.json', result)
    lines = ['# State V4 candidate summary', '', 'No training or dataset merge has been performed.',
             f'Audit current: {valid_audit}',
             f'Counts: {json.dumps(result["totals"])}',
             'Group counts are conservative audit components, not proof of independent physical samples.',
             '', '| Class | Unique images | Groups | Review | Train candidate |',
             '| --- | ---: | ---: | ---: | ---: |']
    lines += [f'| {r["target_class"]} | {r["unique_images"]} | {r["independent_audit_groups"]} | {r["review_images"]} | {r["train_candidate"]} |' for r in counts]
    lines += ['', '## Source status', ''] + [f'- {r["source_dataset"]}: {r["status"]} {r.get("detail", "")}' for r in statuses]
    lines += ['', '## Limits', '', 'pHash similarity is a review signal, not proof of duplication. Unknown acquisition provenance is grouped conservatively. Split plans are proposals only; no files are copied into training splits. Existing V3/reference splits cannot be changed by this tool. Re-audit after any source/config/candidate changes.']
    atomic_bytes(reports / 'summary.md', ('\n'.join(lines) + '\n').encode('utf-8'))
    return result


def execute(command, config):
    if command == 'inspect':
        _, _, statuses = inventory(config)
        for status in statuses:
            print(f'{status["source_dataset"]}: {status["status"]} {status.get("detail", "")}')
        labels = read_csv(config['reports_dir'] / 'inspect_labels.csv')
        for row in labels:
            print(f'{row["source_dataset"]} | {row["original_label"]} | images={row["image_count"]} boxes={row["bounding_box_count"]} | {row["mapping_status"]} | proposal={row["proposed_mapping"]}')
    elif command == 'prepare':
        rows = prepare(config)
        print(f'Prepared {len(rows)} provenance records; audit completed. No merge/training.')
    elif command == 'audit':
        plan = audit(config)
        print(f'Audited {len(plan)} unique candidates: {dict(Counter(r["proposed_split"] for r in plan))}')
    else:
        result = summary(config)
        print(f'Audit current: {result["audit_current"]}. Missing classes: {len(result["missing_classes"])}. No merge/training.')
    print(f'Reports: {config["reports_dir"]}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inspect', 'prepare', 'audit', 'summary'])
    parser.add_argument('--config', type=Path, default=PROJECT / 'config/state_sources_v4.yaml')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    with run_lock(config):
        execute(args.command, config)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, zipfile.BadZipFile, yaml.YAMLError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(2)
