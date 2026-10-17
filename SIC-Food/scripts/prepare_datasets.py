"""Create empty, reproducible dataset layouts. No fabricated training images."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_config import ROOT, PIPELINE, TARGET_INGREDIENTS


def main():
    root = ROOT / 'datasets'
    for split in ('train', 'val', 'test'):
        for folder in ('images', 'labels'):
            (root / 'yolo' / split / folder).mkdir(parents=True, exist_ok=True)
        for ingredient in TARGET_INGREDIENTS:
            for state in PIPELINE['baseline_states']:
                (root / 'efficientnet' / split / f'{ingredient}_{state}').mkdir(parents=True, exist_ok=True)
    yaml = ['path: .', 'train: train/images', 'val: val/images', 'test: test/images', 'names:']
    yaml += [f'  {i}: {name}' for i, name in enumerate(TARGET_INGREDIENTS)]
    (root / 'yolo/data.yaml').write_text('\n'.join(yaml) + '\n', encoding='utf-8')
    (root / 'manifest.example.json').write_text(json.dumps([{
        'image': 'yolo/train/images/example.jpg', 'sample_id': 'sample-001',
        'batch_id': 'batch-001', 'source_id': 'capture-session-001', 'split': 'train',
        'parent_image': None, 'ingredient_id': 'tomato', 'visual_state': 'fresh',
        'preparation_state': 'unknown', 'license': 'record actual permission',
        'reviewed_by': 'reviewer-id'
    }], indent=2) + '\n', encoding='utf-8')
    print('Created empty datasets/yolo and datasets/efficientnet (20 ingredients, 40 baseline classification classes).')


if __name__ == '__main__':
    main()
