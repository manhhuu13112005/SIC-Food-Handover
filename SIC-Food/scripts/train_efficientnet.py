"""Train ingredient+condition classes and save a deployable EfficientNet checkpoint."""
import argparse
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_config import ROOT, PIPELINE, TARGET_INGREDIENTS, parse_condition_class
from condition_service import classification_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'datasets/efficientnet')
    parser.add_argument('--pretrained', type=Path, help='Optional local torchvision EfficientNet-B0 state_dict')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive')
    import torch
    from torchvision import datasets, transforms
    from torchvision.models import efficientnet_b0
    from torch.utils.data import DataLoader
    random.seed(42)
    torch.manual_seed(42)
    size = PIPELINE['image_size']
    train = datasets.ImageFolder(args.data / 'train', transform=transforms.Compose([
        transforms.Resize((size, size)), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    val = datasets.ImageFolder(args.data / 'val', transform=classification_transform(size))
    if train.class_to_idx != val.class_to_idx:
        raise ValueError('Train/val must contain exactly the same class folders.')
    parsed = [parse_condition_class(name) for name in train.classes]
    if {ingredient for ingredient, _ in parsed} != set(TARGET_INGREDIENTS):
        raise ValueError('Training data must cover all 20 configured ingredients.')
    for ingredient in TARGET_INGREDIENTS:
        if sum(i == ingredient for i, _ in parsed) < 2:
            raise ValueError(f'{ingredient} needs at least two condition classes.')
    model = efficientnet_b0(weights=None)
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu', weights_only=True))
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(train.classes))
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    counts = torch.bincount(torch.tensor(train.targets), minlength=len(train.classes)).float()
    loss_fn = torch.nn.CrossEntropyLoss(weight=(counts.sum() / (len(counts) * counts)).to(args.device))
    loaders = {name: DataLoader(ds, batch_size=args.batch_size, shuffle=name == 'train', num_workers=0)
               for name, ds in [('train', train), ('val', val)]}
    output = ROOT / 'models/efficientnet_condition.pt'
    output.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        scores = {}
        for phase, loader in loaders.items():
            model.train(phase == 'train')
            confusion = torch.zeros(len(train.classes), len(train.classes), dtype=torch.long)
            for images, labels in loader:
                images, labels = images.to(args.device), labels.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == 'train'):
                    logits = model(images)
                    loss = loss_fn(logits, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                predictions = logits.argmax(1)
                indices = (labels * len(train.classes) + predictions).detach().cpu()
                confusion += torch.bincount(indices, minlength=len(train.classes)**2).reshape(confusion.shape)
            tp = confusion.diag().float()
            f1 = 2 * tp / (confusion.sum(0) + confusion.sum(1)).clamp(min=1)
            scores[phase] = float(f1.mean())
        print(f'Epoch {epoch+1}: macro-F1 {scores}', flush=True)
        if scores['val'] > best:
            best = scores['val']
            torch.save({'architecture': 'efficientnet_b0', 'image_size': size,
                        'class_names': train.classes, 'state_dict': model.state_dict(),
                        'seed': 42, 'val_macro_f1': best,
                        'torch_version': str(torch.__version__)}, output)
    print(f'Saved {output}; test split was not used for checkpoint selection.')


if __name__ == '__main__':
    main()
