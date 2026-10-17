"""CPU EfficientNet inference; independent image classification, no downloads."""
import math
import time
from pathlib import Path
from pipeline_config import PIPELINE, TARGET_INGREDIENTS, parse_condition_class
from image_service import normalize_image


class ConditionClassifier:
    def __init__(self, model, transform, class_names, metadata=None):
        self.model = model
        self.transform = transform
        self.class_names = class_names
        self.metadata = metadata or {}

    def predict(self, image, ingredient=None, top_k=3):
        import torch
        started = time.perf_counter()
        image = normalize_image(image)
        with torch.inference_mode():
            logits = self.model(self.transform(image).unsqueeze(0).to('cpu'))
            if logits.shape != (1, len(self.class_names)) or not torch.isfinite(logits).all():
                raise ValueError('Đầu ra EfficientNet không hợp lệ.')
            scores = logits.softmax(1)[0]
        confidence, index = scores.max(0)
        label = self.class_names[int(index)]
        predicted_ingredient, condition = parse_condition_class(label)
        confidence = float(confidence)
        accepted = (ingredient is None or predicted_ingredient == ingredient) and confidence >= PIPELINE['classification_threshold']
        top_scores, indices = scores.topk(min(max(1, int(top_k)), len(self.class_names)))
        top = []
        for score, idx in zip(top_scores.tolist(), indices.tolist()):
            item_label = self.class_names[idx]
            name, state = parse_condition_class(item_label)
            top.append({'class_index': idx, 'label': item_label, 'ingredient': name,
                        'appearance': state, 'confidence': score})
        return {'label': label, 'confidence': confidence,
                'ingredient': predicted_ingredient, 'appearance': condition, 'class_index': int(index),
                'top3': top, 'probabilities': scores.tolist(), 'class_count': len(self.class_names),
                'elapsed_ms': (time.perf_counter()-started)*1000,
                'visual_condition': condition if accepted else 'unknown',
                'status': 'classified' if accepted else 'uncertain'}


def classification_transform(image_size):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((image_size, image_size)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_classifier_file(path, expected_classes=None):
    import torch
    from torchvision.models import efficientnet_b0
    if not Path(path).is_file():
        raise FileNotFoundError(f'Chưa có checkpoint: {path}')
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError('Checkpoint phải chứa state_dict và class_names.')
    names = checkpoint.get('class_names')
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names) or len(set(names)) != len(names):
        raise ValueError('class_names thiếu hoặc trùng lặp.')
    for name in names:
        parse_condition_class(name)
    if expected_classes is not None and len(names) != expected_classes:
        raise ValueError(f'Checkpoint cần có {expected_classes} lớp, nhận được {len(names)}.')
    if 'model_state_dict' in checkpoint:
        # Contract verified against efficientnet_37_fasttrack/train_efficientnet_37.py.
        if (checkpoint.get('num_classes') != len(names)
                or checkpoint.get('pretrained_weights') != 'IMAGENET1K_V1'
                or checkpoint.get('training_type') != 'prototype_frozen_backbone'):
            raise ValueError('Metadata checkpoint fast-track không tương thích.')
        from torchvision.models import EfficientNet_B0_Weights
        transform = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
        state = checkpoint['model_state_dict']
    else:
        # Preserve the explicitly versioned legacy training checkpoint format.
        if checkpoint.get('architecture') != 'efficientnet_b0':
            raise ValueError('Checkpoint phải dùng efficientnet_b0.')
        image_size = checkpoint.get('image_size')
        if not isinstance(image_size, int) or not 32 <= image_size <= 1024:
            raise ValueError('image_size không hợp lệ.')
        transform = classification_transform(image_size)
        state = checkpoint['state_dict']
    if tuple(state['classifier.1.weight'].shape) != (len(names), 1280):
        raise ValueError('Classifier không khớp class_names.')
    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(names))
    model.load_state_dict(state, strict=True)
    model.to('cpu').eval()
    metadata = {k: checkpoint.get(k) for k in ('epoch', 'training_type', 'validation_scope', 'pretrained_weights')}
    return ConditionClassifier(model, transform, names, metadata)


def classify_detections(detections, image, classifier=None):
    """Retain every detection, including unsupported objects and invalid crops."""
    width, height = image.size
    for detection in detections:
        ingredient = detection['prediction']['ingredient_id']
        detection['prediction']['visual_condition'] = 'unknown'
        assessment = {'status': 'model_unavailable', 'visual_condition': 'unknown'}
        if ingredient not in TARGET_INGREDIENTS:
            assessment['status'] = 'unsupported_ingredient'
        elif classifier is not None:
            coords = detection['bbox']
            if len(coords) != 4 or not all(math.isfinite(v) for v in coords):
                assessment['status'] = 'invalid_crop'
            else:
                x1, y1, x2, y2 = coords
                box = (max(0, math.floor(x1)), max(0, math.floor(y1)),
                       min(width, math.ceil(x2)), min(height, math.ceil(y2)))
                if box[2] <= box[0] or box[3] <= box[1]:
                    assessment['status'] = 'invalid_crop'
                else:
                    try:
                        assessment = classifier.predict(image.crop(box), ingredient)
                    except Exception:
                        assessment['status'] = 'inference_error'
        detection['condition_assessment'] = assessment
        detection['prediction']['visual_condition'] = assessment['visual_condition']
    return detections
