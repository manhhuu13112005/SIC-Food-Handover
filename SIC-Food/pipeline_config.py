"""Shared class contract for detection, classification and dataset preparation."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPELINE = json.loads((ROOT / 'config/ai_pipeline.json').read_text(encoding='utf-8'))
TARGET_INGREDIENTS = PIPELINE['ingredients']


def parse_condition_class(label):
    if not isinstance(label, str):
        raise ValueError('Nhãn EfficientNet phải là chuỗi.')
    if '__' in label:
        parts = label.split('__')
        if (len(parts) == 2 and re.fullmatch(r'[a-z][a-z0-9_]*', parts[0])
                and parts[1] in ('normal_appearance', 'visible_defect')):
            return tuple(parts)
        raise ValueError(f'Nhãn EfficientNet không hợp lệ: {label}')
    # Ingredient IDs may contain underscores (bell_pepper).
    for ingredient in sorted(TARGET_INGREDIENTS, key=len, reverse=True):
        prefix = ingredient + '_'
        if label.startswith(prefix):
            state = label[len(prefix):]
            if state in PIPELINE['state_mapping']:
                return ingredient, PIPELINE['state_mapping'][state]
    raise ValueError(f'Nhãn EfficientNet không hợp lệ: {label}')
