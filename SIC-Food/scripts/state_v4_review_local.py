"""Read-only contact sheets for SIC-Food V4 pork, shrimp and vegetables.
Copy to SIC-Food/scripts/ and run: %run scripts/state_v4_review_local.py
Requires Pillow and PyYAML. Never approves labels or changes source datasets.
"""
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from collections import defaultdict, Counter
from PIL import Image, ImageOps, ImageDraw
import yaml, io, csv, random, re, hashlib

PROJECT = Path(r'C:\Users\ASUS\SIC-Food')
ARCHIVES = PROJECT / 'datasets/raw/state_sources/extension_v4/archives'
OUT = PROJECT / 'datasets/reports/state_v4/review'
OUT.mkdir(parents=True, exist_ok=True)
SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}
PORK = {'Kasim Fresh', 'Kasim Spoiled', 'Liempo Fresh', 'Liempo Spoiled',
        'Pork chop Fresh', 'Pork chop Spoiled', 'Tenderloin Fresh', 'Tenderloin Spoiled'}
VEGGIES = {'Fresh Egg plant', 'RottenEgg plant', 'Fresh broccoli'}
summary = []


def csv_out(name, rows, fields):
    with (OUT/name).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def members_of(z):
    entries = {}
    for m in z.infolist():
        if m.is_dir():
            continue
        path = PurePosixPath(m.filename.replace('\\', '/'))
        if path.is_absolute() or '..' in path.parts or ':' in path.parts[0]:
            raise ValueError('Unsafe ZIP path: ' + m.filename)
        key = str(path).casefold()
        if key in entries:
            raise ValueError('Duplicate ZIP path: ' + m.filename)
        entries[key] = m
    return entries


def pil_image(z, info):
    with Image.open(io.BytesIO(z.read(info))) as im:
        return ImageOps.exif_transpose(im).convert('RGB')


def sheet(z, source, label, items):
    # Preview only. No claim that near-duplicate samples are independent.
    rng = random.Random(42)
    chosen = list(items)
    rng.shuffle(chosen)
    chosen = chosen[:12]
    page = Image.new('RGB', (4*350, 3*250), '#eeeeee')
    for k, item in enumerate(chosen):
        x, y = (k % 4)*350, (k//4)*250
        d = ImageDraw.Draw(page)
        d.rectangle((x,y,x+349,y+249), outline='#aaaaaa', fill='white')
        d.text((x+5,y+5), Path(item['path']).name[:42], fill='black')
        try:
            im = pil_image(z, item['info'])
            if item.get('bbox'):
                xc, yc, bw, bh = item['bbox']
                iw, ih = im.size
                box = (max(0,int((xc-bw/2)*iw)), max(0,int((yc-bh/2)*ih)),
                       min(iw,int((xc+bw/2)*iw)), min(ih,int((yc+bh/2)*ih)))
                original = im.copy()
                ImageDraw.Draw(original).rectangle(box, outline='red', width=max(2,iw//150))
                a = ImageOps.contain(original,(160,200))
                page.paste(a,(x+5,y+30))
                if box[2]>box[0] and box[3]>box[1]:
                    b = ImageOps.contain(im.crop(box),(165,200))
                    page.paste(b,(x+180,y+30))
            else:
                a = ImageOps.contain(im,(330,200))
                page.paste(a,(x+10,y+30))
        except Exception as ex:
            d.text((x+5,y+90),'Image read error',fill='red')
            print('Preview error:',item['path'],ex)
    fname = source + '_' + re.sub(r'[^a-z0-9]+','_',label.lower()).strip('_') + '.jpg'
    page.save(OUT/fname, quality=88)
    print(f'{source} | {label}: {len(items)} objects/images -> {fname}')
    summary.append({'source':source,'label':label,'count':len(items),'file':fname})


def yolo_review(source, wanted):
    path = ARCHIVES/(source+'.zip')
    if not path.exists():
        print('MISSING:',path)
        return
    with ZipFile(path) as z:
        entries = members_of(z)
        yaml_entries = [v for k,v in entries.items() if k=='data.yaml' or k.endswith('/data.yaml')]
        if not yaml_entries:
            print('NO data.yaml:',source)
            return
        configs = [yaml.safe_load(z.read(v)) for v in yaml_entries]
        def names(c):
            n = c.get('names',[])
            return [n.get(i,n.get(str(i))) for i in range(len(n))] if isinstance(n,dict) else n
        labels = names(configs[0])
        if any(names(c)!=labels for c in configs[1:]):
            raise ValueError('Conflicting class maps for '+source)
        images = {}
        for k,v in entries.items():
            p=PurePosixPath(k)
            if p.suffix.lower() not in SUFFIXES or 'images' not in p.parts:
                continue
            i=p.parts.index('images')
            images['/'.join(p.parts[:i]+('images',p.stem))]=v
        collected = defaultdict(list)
        warnings=Counter()
        for k,v in entries.items():
            p=PurePosixPath(k)
            if p.suffix.lower()!='.txt' or 'labels' not in p.parts:
                continue
            i=p.parts.index('labels')
            if i<1 or p.parts[i-1] not in {'train','test','valid','val'}:
                continue
            key='/'.join(p.parts[:i]+('images',p.stem))
            img=images.get(key)
            if img is None:
                warnings['unmatched_labels']+=1
                continue
            for line in z.read(v).decode('utf-8-sig').splitlines():
                tokens=line.split()
                if not tokens:
                    continue
                if len(tokens)!=5:
                    warnings['invalid_lines']+=1
                    continue
                try:
                    idx=int(tokens[0]); coords=tuple(map(float,tokens[1:]))
                    label=labels[idx]
                except (ValueError,IndexError,TypeError):
                    warnings['invalid_values']+=1
                    continue
                if label not in wanted:
                    continue
                if any(not 0<=v<=1 for v in coords) or coords[2]<=0 or coords[3]<=0:
                    warnings['invalid_bboxes']+=1
                    continue
                collected[label].append({'info':img,'path':img.filename,'bbox':coords})
        for label in sorted(wanted):
            sheet(z,source,label,collected[label])
        print('YOLO warnings for',source,dict(warnings))


def shrimp_review():
    path=ARCHIVES/'shrimp.zip'
    if not path.exists():
        print('MISSING:',path)
        return
    with ZipFile(path) as z:
        entries=members_of(z)
        classes=defaultdict(list); unknown=[]; hashes=defaultdict(set)
        duplicates=Counter()
        for key,info in entries.items():
            p=PurePosixPath(key)
            if p.suffix.lower() not in SUFFIXES:
                continue
            hits={part for part in p.parts[:-1] if part in {'fresh','not'}}
            if len(hits)!=1:
                unknown.append(info.filename)
                continue
            label=next(iter(hits))
            payload=z.read(info)
            digest=hashlib.sha256(payload).hexdigest()
            duplicates[digest]+=1
            hashes[digest].add(label)
            classes[label].append({'info':info,'path':info.filename})
        for label in ['fresh','not']:
            sheet(z,'shrimp',label,classes[label])
        conflicts=[{'sha256':h,'labels':' | '.join(sorted(labels))}
                   for h,labels in hashes.items() if len(labels)>1]
        csv_out('shrimp_sha_label_conflicts.csv',conflicts,['sha256','labels'])
        csv_out('shrimp_unknown_paths.csv',[{'path':p} for p in unknown],['path'])
        print('Shrimp duplicate files SHA256:',sum(n-1 for n in duplicates.values()))
        print('Shrimp conflicting SHA256 groups:',len(conflicts))
        print('Shrimp uncategorized images:',len(unknown))
        print('NOTICE: shrimp/not remains UNAPPROVED, not treated as spoiled.')


if __name__=='__main__':
    yolo_review('pork',PORK)
    yolo_review('smartveggies',VEGGIES)
    shrimp_review()
    csv_out('review_summary.csv',summary,['source','label','count','file'])
    print('COMPLETE. Review images at:',OUT)
    print('No training data or model checkpoint changed.')
