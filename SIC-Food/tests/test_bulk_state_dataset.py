"""Small synthetic fixtures only. Never reads/writes production dataset images."""
import copy
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import bulk_state_dataset as bulk


def png(seed=0, size=(64, 48)):
    data = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    output = io.BytesIO()
    Image.fromarray(data).save(output, format='PNG')
    return output.getvalue()


class BulkDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='bulk-fixture-', dir=ROOT / 'tests')
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to((ROOT / 'tests').resolve()))
        self.addCleanup(self.cleanup)
        raw = yaml.safe_load((ROOT / 'config/state_sources_v4.yaml').read_text(encoding='utf-8'))
        # Synthetic fixtures keep their own archive name, independent of the
        # reviewed ZIP filename selected for the real lettuce source.
        raw['sources']['lettuce']['archive'] = 'lettuce.zip'
        raw['reference_datasets'] = []
        raw['audit']['require_reference_audit'] = False
        self.config_path = self.root / 'config.yaml'
        self.config_path.write_text(yaml.safe_dump(raw), encoding='utf-8')
        self.config = bulk.load_config(self.config_path, self.root)
        self.config['archives_dir'].mkdir(parents=True)

    def cleanup(self):
        if not self.root.is_relative_to((ROOT / 'tests').resolve()):
            raise AssertionError('Unexpected cleanup target')
        self.temporary.cleanup()

    def archive(self, name, files):
        path = self.config['archives_dir'] / name
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for label, data in files.items():
                archive.writestr(label, data)
        return path

    def test_zip_slip_windows_paths_and_collisions(self):
        for name in ('../escape.png', '/absolute.png', 'C:/escape.png', 'ok/../../escape.png', 'ok\\..\\escape.png', 'NUL.png'):
            path = self.archive('lettuce.zip', {name: png()})
            with self.assertRaises(ValueError):
                bulk.extract_once(path, self.config['work_dir'], self.config['zip_limits'])
        path = self.archive('lettuce.zip', {'A.png': png(), 'a.png': png(1)})
        with self.assertRaises(ValueError):
            bulk.extract_once(path, self.config['work_dir'], self.config['zip_limits'])

    def test_zip_symlink_and_size_limit(self):
        path = self.config['archives_dir'] / 'lettuce.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            info = zipfile.ZipInfo('link')
            info.create_system = 3
            info.external_attr = 0o120777 << 16
            archive.writestr(info, '../outside')
        with self.assertRaises(ValueError):
            bulk.extract_once(path, self.config['work_dir'], self.config['zip_limits'])
        path = self.archive('lettuce.zip', {'image.png': png()})
        limits = {**self.config['zip_limits'], 'max_file_bytes': 2}
        with self.assertRaises(ValueError):
            bulk.extract_once(path, self.config['work_dir'], limits)

    def test_extract_once_across_archive_names(self):
        path = self.archive('lettuce.zip', {'train/Healthy/a.png': png()})
        before = bulk.sha256_file(path)
        first, digest, status = bulk.extract_once(path, self.config['work_dir'], self.config['zip_limits'])
        self.assertEqual(status, 'extracted')
        other = self.config['archives_dir'] / 'shrimp.zip'
        other.write_bytes(path.read_bytes())
        second, _, status = bulk.extract_once(other, self.config['work_dir'], self.config['zip_limits'])
        self.assertEqual(first, second)
        self.assertEqual(status, 'cached')
        self.assertEqual(before, bulk.sha256_file(path))
        extracted_image = first / 'train/Healthy/a.png'
        payload = extracted_image.read_bytes()
        extracted_image.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
        with self.assertRaisesRegex(ValueError, 'incomplete/modified'):
            bulk.extract_once(path, self.config['work_dir'], self.config['zip_limits'])

    def test_yolo_names_boxes_and_roboflow_paths(self):
        path = self.archive('lettuce.zip', {
            'data.yaml': 'train: ../train/images\nnames: [Healthy, Rotten Lettuce]\n',
            'train/images/a.png': png(),
            'train/labels/a.txt': '0 0.5 0.5 0.5 0.5\n1 0.5 0.5 0.8 0.8\n0 nan 0.5 0.2 0.2\n0 0.5 0.5 -1 1\n'})
        root, _, _ = bulk.extract_once(path, self.config['work_dir'], self.config['zip_limits'])
        samples, issues, labels, images, mode = bulk.scan_source(root, 'lettuce', {'format': 'auto'})
        self.assertEqual(mode, 'yolo')
        self.assertEqual(labels, {'Healthy', 'Rotten Lettuce'})
        self.assertEqual(len(samples), 2)
        self.assertEqual(images, 1)
        self.assertEqual(len(issues), 2)
        image = bulk.load_rgb(samples[0].path, 100000)
        crop, box = bulk.crop_yolo(image, samples[0].bbox)
        self.assertEqual(crop.size, (32, 24))
        self.assertEqual(box, (16, 12, 48, 36))

    def test_invalid_crop_and_missing_annotations(self):
        with self.assertRaises(ValueError):
            bulk.crop_yolo(Image.new('RGB', (40, 40)), (.5, .5, .01, .01))
        path = self.archive('lettuce.zip', {'data.yaml': 'train: train/images\nnames: [Healthy]\n', 'train/images/a.png': png()})
        root, _, _ = bulk.extract_once(path, self.config['work_dir'], self.config['zip_limits'])
        samples, issues, *_ = bulk.scan_source(root, 'lettuce', {'format': 'auto'})
        self.assertFalse(samples)
        self.assertEqual(issues[0]['reason'], 'missing_annotation')

    def test_mapping_exact_and_unapproved(self):
        self.assertEqual(bulk.mapping_for('lettuce', 'Healthy', self.config)[0], 'lettuce__normal_appearance')
        self.assertFalse(bulk.mapping_for('lettuce', 'healthy', self.config)[0])
        self.assertFalse(bulk.mapping_for('shrimp', 'Not', self.config)[0])
        self.assertFalse(bulk.mapping_for('smartveggies', 'Rotten broccoli', self.config)[0])
        self.assertEqual(bulk.mapping_for('pork', 'Fresh Kasim', self.config), ('', 'pork__normal_appearance', 'needs_approval'))
        self.assertFalse(bulk.mapping_for('pork', 'Kasim', self.config)[1])
        self.config['sources']['shrimp']['approved_mappings']['Not'] = 'shrimp__visible_defect'
        self.assertEqual(bulk.mapping_for('shrimp', 'Not', self.config)[2], 'blocked_label')

    def test_classification_prepare_idempotent_and_no_crop(self):
        self.archive('lettuce.zip', {'dataset/train/Healthy/a.png': png(),
                                     'dataset/train/Rotten Lettuce/b.png': png(1),
                                     'dataset/train/Larvae/c.png': png(2)})
        rows = bulk.prepare(self.config)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(not row['bounding_box'] for row in rows))
        output = self.root / rows[0]['crop_path']
        with Image.open(output) as image:
            self.assertEqual(image.size, (64, 48))
        again = bulk.prepare(self.config)
        self.assertEqual(rows, again)
        self.assertEqual(len(list(self.config['output_dir'].rglob('*.png'))), 2)
        plan = bulk.read_csv(self.config['reports_dir'] / 'split_plan.csv')
        self.assertEqual({row['proposed_split'] for row in plan}, {'train_candidate'})
        review = bulk.read_csv(self.config['reports_dir'] / 'prepare_review.csv')
        self.assertTrue(any(row['original_label'] == 'Larvae' for row in review))

    def test_yolo_prepare_two_crops(self):
        self.archive('lettuce.zip', {'data.yaml': 'train: train/images\nnames: [Healthy]\n',
                                     'train/images/a.png': png(),
                                     'train/labels/a.txt': '0 .25 .5 .5 1\n0 .75 .5 .5 1\n'})
        rows = bulk.prepare(self.config)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row['bounding_box'] for row in rows))
        self.assertEqual(rows[0]['image_group'], rows[1]['image_group'])
        self.assertNotEqual(rows[0]['crop_path'], rows[1]['crop_path'])

    def test_conflicting_labels_quarantined(self):
        self.archive('lettuce.zip', {'train/Healthy/a.png': png(), 'train/Rotten Lettuce/b.png': png()})
        bulk.prepare(self.config)
        rows = bulk.read_csv(self.config['reports_dir'] / 'split_plan.csv')
        self.assertEqual({r['proposed_split'] for r in rows}, {'review'})
        self.assertTrue(all('conflicting_labels' in r['review_reasons'] for r in rows))

    def test_exact_phash_without_exact_sha_is_reviewed(self):
        original = png()
        with Image.open(io.BytesIO(original)) as image:
            changed = image.copy()
        pixel = changed.getpixel((0, 0))
        changed.putpixel((0, 0), ((pixel[0] + 1) % 256, pixel[1], pixel[2]))
        buffer = io.BytesIO()
        changed.save(buffer, format='PNG')
        self.archive('lettuce.zip', {'train/Healthy/a.png': original, 'train/Healthy/b.png': buffer.getvalue()})
        bulk.prepare(self.config)
        rows = bulk.read_csv(self.config['reports_dir'] / 'split_plan.csv')
        self.assertEqual(len(rows), 2)
        self.assertEqual({r['proposed_split'] for r in rows}, {'review'})
        self.assertTrue(all('phash_duplicate_review' in r['review_reasons'] or 'near_duplicate_review' in r['review_reasons'] for r in rows))

    def test_nested_classification_pork_labels(self):
        path = self.archive('pork.zip', {'train/Kasim/Fresh/a.png': png(), 'train/Tenderloin/Spoiled/b.png': png(1)})
        root, _, _ = bulk.extract_once(path, self.config['work_dir'], self.config['zip_limits'])
        samples, _, labels, _, mode = bulk.scan_source(root, 'pork', self.config['sources']['pork'])
        self.assertEqual(labels, {'Kasim/Fresh', 'Tenderloin/Spoiled'})
        self.assertEqual(bulk.mapping_for('pork', 'Kasim/Fresh', self.config)[1], 'pork__normal_appearance')
        self.assertTrue(all(not sample.bbox for sample in samples))

    def test_reference_overlap_is_withheld(self):
        reference = self.root / 'reference' / 'train' / 'lettuce__normal_appearance'
        reference.mkdir(parents=True)
        (reference / 'a.png').write_bytes(png())
        self.config['reference_datasets'] = [self.root / 'reference']
        self.config['audit']['require_reference_audit'] = True
        self.archive('lettuce.zip', {'train/Healthy/a.png': png()})
        bulk.prepare(self.config)
        rows = bulk.read_csv(self.config['reports_dir'] / 'split_plan.csv')
        self.assertEqual(rows[0]['proposed_split'], 'review')
        self.assertIn('overlap_reference', rows[0]['review_reasons'])

    def test_split_does_not_cross_shared_source_group(self):
        rows, groups = [], []
        for i in range(20):
            for target in ['lettuce__normal_appearance', 'lettuce__visible_defect']:
                rows.append({'target_class': target})
                groups.append(f'source-{i}')
        plan = bulk.plan_splits(rows, groups, {}, self.config)
        assignments = {}
        for row in plan:
            assignments.setdefault(row['audit_group'], set()).add(row['proposed_split'])
        self.assertTrue(all(len(value) == 1 for value in assignments.values()))
        self.assertEqual({r['proposed_split'] for r in plan}, {'train', 'val', 'test'})

    def test_small_dataset_has_no_fake_holdout(self):
        rows = [{'target_class': 'pork__normal_appearance'} for _ in range(50)]
        plan = bulk.plan_splits(rows, ['one-video']*50, {}, self.config)
        self.assertEqual({r['proposed_split'] for r in plan}, {'train_candidate'})

    def test_video_augmentation_grouping(self):
        root = self.root
        samples = [bulk.Sample(root / name, root, 'train', 'Healthy') for name in
                   ['clip_mp4-001_jpg.rf.abcdef.png', 'rotated_by_15_clip_mp4-002_jpg.rf.123abc.png']]
        first = bulk.group_identity(samples[0], 'lettuce', [])
        second = bulk.group_identity(samples[1], 'lettuce', [])
        self.assertEqual(first, second)
        self.assertEqual(first[1], 'video_filename')

    def test_modified_candidate_and_stale_archives(self):
        self.archive('lettuce.zip', {'train/Healthy/a.png': png()})
        rows = bulk.prepare(self.config)
        (self.root / rows[0]['crop_path']).write_bytes(b'broken')
        plan = bulk.audit(self.config)
        self.assertIn('invalid_or_modified_candidate', plan[0]['review_reasons'])
        self.archive('shrimp.zip', {'train/Fresh/a.png': png(9)})
        with self.assertRaisesRegex(ValueError, 'Archive inputs changed'):
            bulk.audit(self.config)
        self.assertFalse(bulk.summary(self.config)['audit_current'])

    def test_missing_sources_and_missing_classes_reported(self):
        bulk.prepare(self.config)
        result = bulk.summary(self.config)
        self.assertEqual(len(result['missing_classes']), 12)
        self.assertEqual({r['status'] for r in result['sources']}, {'missing_archive'})

    def test_protected_outputs(self):
        raw = yaml.safe_load(self.config_path.read_text(encoding='utf-8'))
        raw['output_dir'] = 'datasets/efficientnet_v3'
        self.config_path.write_text(yaml.safe_dump(raw), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'protected'):
            bulk.load_config(self.config_path, self.root)

    def test_bktree_matches_bruteforce(self):
        values = [0, 1, 3, 15, 255, 2**63, 2**64-1]
        tree = bulk.BKTree()
        for value in values:
            tree.insert(value)
        for query in values:
            expected = {(v, (v ^ query).bit_count()) for v in values if (v ^ query).bit_count() <= 4}
            self.assertEqual(set(tree.query(query, 4)), expected)

    def test_optional_missing_yolo_splits_are_not_fabricated(self):
        path = self.archive('smartveggies.zip', {
            'data.yaml': 'train: ../train/images\nval: ../valid/images\ntest: ../test/images\nnames: [Fresh broccoli]\n',
            'train/images/a.png': png(), 'train/labels/a.txt': '0 .5 .5 .8 .8\n'})
        root, _, _ = bulk.extract_once(path, self.config['work_dir'], self.config['zip_limits'])
        samples, issues, _, images, _ = bulk.scan_source(root, 'smartveggies', self.config['sources']['smartveggies'])
        self.assertEqual(images, 1)
        self.assertEqual({s.source_split for s in samples}, {'train'})
        self.assertEqual(len(issues), 2)
        self.assertTrue(all(e['reason'] == 'split_absent_in_archive' and e['severity'] == 'warning' for e in issues))
        with self.assertRaises(ValueError):
            bulk.yolo_image_paths('../../outside/images', root, root, root)

    def test_csv_classification_preserves_multilabel_and_blocks_prepare(self):
        self.archive('shrimp.zip', {'valid/_classes.csv': 'filename, fresh, not\na.png, 1, 0\nb.png, 1, 1\nc.png, 0, 1\n',
                                    'valid/a.png': png(), 'valid/b.png': png(1), 'valid/c.png': png(2)})
        samples, issues, statuses = bulk.inventory(self.config)
        shrimp = samples['shrimp']
        self.assertEqual(len(shrimp), 4)
        self.assertEqual(sum(s.label == 'fresh' for s in shrimp), 2)
        self.assertTrue(all(s.review_reason for s in shrimp if s.path.name == 'b.png'))
        self.assertTrue(any(s.get('images') == 3 and s['status'] == 'needs_review' for s in statuses))
        self.assertEqual(bulk.prepare(self.config), [])
        # Even explicit future approval of single-label fresh cannot admit a
        # contradictory fresh+not image without correcting its source metadata.
        self.config['sources']['shrimp']['review_labels'] = []
        self.config['sources']['shrimp']['approved_mappings'] = {'fresh': 'shrimp__normal_appearance'}
        rows = bulk.prepare(self.config)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['source_image'], 'valid/a.png')
        self.assertFalse(rows[0]['bounding_box'])

    def test_csv_path_safety_bad_flags_missing_and_duplicate_rows(self):
        path = self.archive('shrimp.zip', {
            'valid/_classes.csv': 'filename,fresh,not\n../outside.png,1,0\nmissing.png,1,0\na.png,2,0\nb.png,1,0\nb.png,0,1\n',
            'outside.png': png(), 'valid/a.png': png(1), 'valid/b.png': png(2), 'valid/orphan.png': png(3)})
        root, _, _ = bulk.extract_once(path, self.config['work_dir'], self.config['zip_limits'])
        samples, issues, _, _, mode = bulk.scan_source(root, 'shrimp', self.config['sources']['shrimp'])
        self.assertEqual(mode, 'classification_csv')
        self.assertEqual(sum(e['reason'] == 'invalid_classification_row' for e in issues), 3)
        self.assertTrue(any(e['reason'] == 'image_missing_csv_annotation' for e in issues))
        self.assertTrue(all(s.review_reason == 'duplicate_classification_annotation' for s in samples))

    def test_pork_contact_sheets_do_not_approve_labels(self):
        # This scenario starts unapproved regardless of later human approvals
        # in the production config. Never change that production config here.
        self.config['sources']['pork']['approved_mappings'] = {}
        self.archive('pork.zip', {'data.yaml': 'train: train/images\nnames: [Kasim Fresh, Kasim Spoiled]\n',
                                  'train/images/a.png': png(), 'train/labels/a.txt': '0 .5 .5 .8 .8\n',
                                  'train/images/b.png': png(1), 'train/labels/b.txt': '1 .5 .5 .8 .8\n'})
        self.config['sources']['pork']['contact_sheet_labels'] = ['Kasim Fresh', 'Kasim Spoiled']
        bulk.inventory(self.config)
        sheets = list((self.config['reports_dir'] / 'pork_review').glob('*.jpg'))
        self.assertEqual(len(sheets), 2)
        with Image.open(sheets[0]) as image:
            self.assertEqual(image.width, 920)
        labels = bulk.read_csv(self.config['reports_dir'] / 'inspect_labels.csv')
        self.assertEqual({r['mapping_status'] for r in labels}, {'needs_approval'})
        self.assertEqual(bulk.prepare(self.config), [])

    def test_unused_archive_is_informational_during_prepare(self):
        self.archive('unused_original.zip', {'bad/../../escape.txt': 'unsafe archive must not be used'})
        self.archive('lettuce.zip', {'train/Healthy/a.png': png()})
        rows = bulk.prepare(self.config)
        self.assertEqual(len(rows), 1)
        statuses = json.loads((self.config['reports_dir'] / 'source_status.json').read_text(encoding='utf-8'))
        unused = next(s for s in statuses if s['archive'] == 'unused_original.zip')
        self.assertEqual(unused['status'], 'unconfigured_archive')
        self.assertEqual(unused['severity'], 'info')


if __name__ == '__main__':
    unittest.main()
