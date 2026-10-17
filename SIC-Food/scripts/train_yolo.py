"""Explicit training entrypoint. Run only after dataset review."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_config import ROOT, TARGET_INGREDIENTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'datasets/yolo/data.yaml')
    parser.add_argument('--weights', required=True, help='Local pretrained YOLO .pt file')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    import yaml
    from ultralytics import YOLO
    data = yaml.safe_load(args.data.read_text(encoding='utf-8'))
    names = data['names']
    names = [names[i] for i in range(len(names))] if isinstance(names, dict) else names
    if names != TARGET_INGREDIENTS:
        raise ValueError('data.yaml must match the 20 configured ingredients in order.')
    if not Path(args.weights).is_file():
        raise ValueError('--weights must reference an existing local checkpoint.')
    # Resolve '.' relative to data.yaml, not Ultralytics' global dataset directory.
    data['path'] = str(args.data.resolve().parent)
    resolved = args.data.parent / 'data.resolved.yaml'
    resolved.write_text(yaml.safe_dump(data, allow_unicode=True), encoding='utf-8')
    YOLO(args.weights).train(data=str(resolved.resolve()), epochs=args.epochs,
                             imgsz=640, seed=42, device=args.device,
                             project=str(ROOT / 'runs/yolo'), name='ingredients20')


if __name__ == '__main__':
    main()
