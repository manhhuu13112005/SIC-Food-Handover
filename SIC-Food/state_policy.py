"""Configuration-driven state mapping. Missing states stay unknown."""
import json
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'config/catalog.json').read_text(encoding='utf-8'))
INGREDIENTS = CONFIG['ingredients']
PREPARATIONS = {'unknown': 'Chưa xác định', 'raw': 'Sống / chưa chế biến nhiệt', 'cooked': 'Đã làm chín'}
CONDITIONS = {'unknown': 'Chưa xác định', 'fresh_appearing': 'Có vẻ tươi', 'normal_appearing': 'Chưa thấy dấu hiệu hỏng', 'visible_defect': 'Có khuyết tật bên ngoài', 'wilted': 'Héo', 'bruised': 'Dập', 'mold': 'Có dấu hiệu mốc', 'suspected_spoiled': 'Hỏng hoặc nghi hỏng'}
EXCLUDED_CONDITIONS = {'visible_defect', 'wilted', 'bruised', 'mold', 'suspected_spoiled'}

def map_label(label):
    value = CONFIG['class_mapping'].get(label.lower().strip(), {})
    # YOLO identifies ingredients only. Status comes exclusively from Model 2.
    return {'ingredient_id': value.get('ingredient_id', 'unknown'),
            'preparation_state': 'unknown', 'visual_condition': 'unknown'}

def allowed_conditions(ingredient):
    group = INGREDIENTS.get(ingredient, {}).get('group', 'other')
    base = ['unknown', 'suspected_spoiled', 'fresh_appearing', 'visible_defect', 'bruised', 'mold']
    if group == 'produce':
        base.append('wilted')
    if group not in ('meat', 'produce'):
        base.append('normal_appearing')
    return base

def new_item(detection=None):
    return {'id': detection['detection_id'] if detection else uuid4().hex,
            'detection': detection, 'prediction': dict(detection['prediction']) if detection else map_label(''),
            'source': 'model' if detection else 'manual'}

def reset_image_state(state, token):
    for key in list(state):
        if key.startswith('img_'):
            del state[key]
    state['img_token'] = token
    state['img_items'] = []
