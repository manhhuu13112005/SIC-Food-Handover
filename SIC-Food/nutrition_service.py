"""Explicit nutrition profiles; missing data is never zero-filled."""
import json
import math
from pathlib import Path

FIELDS = ('kcal', 'protein_g', 'carbohydrate_g', 'fat_g')

def load_profiles():
    return json.loads((Path(__file__).resolve().parent / 'config/nutrition_profiles.json').read_text(encoding='utf-8'))

def compatible(profile, ingredient, preparation):
    return profile['ingredient_id'] == ingredient and profile['preparation_state'] == preparation

def calculate(profile, grams):
    if isinstance(grams, bool) or not isinstance(grams, (int, float)) or not math.isfinite(grams) or grams <= 0:
        raise ValueError('Cần nhập khối lượng ăn được lớn hơn 0 g.')
    values = profile.get('per_100g', {})
    if any(isinstance(values.get(k), bool) or not isinstance(values.get(k), (int, float)) or not math.isfinite(values[k]) or values[k] < 0 for k in FIELDS):
        raise ValueError('Chưa có dữ liệu dinh dưỡng đầy đủ.')
    result = {k: values[k] * grams / 100 for k in FIELDS}
    # Keys carry units, e.g. vitamin_c_mg, vitamin_a_ug, calcium_mg.
    for key, value in values.items():
        if key not in FIELDS and key.endswith(('_mg', '_ug', '_g')) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
            result[key] = value * grams / 100
    return result

def summarize(rows):
    total = {k: 0.0 for k in FIELDS}
    calculated, missing = [], []
    included_count = sum(bool(row['include']) for row in rows)
    micronutrients = {}
    for row in rows:
        if not row['include']:
            continue
        profile = row.get('profile')
        reason = None
        if not row.get('confirmed') or row['ingredient_id'] == 'unknown':
            reason = 'Cần xác nhận nguyên liệu và thông tin trạng thái.'
        elif not profile:
            reason = 'Chưa có dữ liệu / chưa chọn hồ sơ dinh dưỡng.'
        elif not compatible(profile, row['ingredient_id'], row['preparation_state']):
            reason = 'Hồ sơ không khớp nguyên liệu hoặc trạng thái chế biến.'
        elif profile['verification_status'] != 'verified' and not row.get('accept_unverified'):
            reason = 'Chưa đồng ý dùng dữ liệu chưa kiểm chứng.'
        if reason is None:
            try:
                values = calculate(profile, row['grams'])
            except ValueError as error:
                reason = str(error)
        if reason:
            missing.append({'id': row['id'], 'reason': reason})
        else:
            calculated.append({'id': row['id'], **values})
            for k in FIELDS:
                total[k] += values[k]
            for key, value in values.items():
                if key in FIELDS:
                    continue
                entry = micronutrients.setdefault(key, {'nutrient': key, 'value': 0.0, 'covered_items': 0})
                entry['value'] += value
                entry['covered_items'] += 1
    for entry in micronutrients.values():
        entry['complete'] = entry['covered_items'] == included_count
    return {'total': total if calculated else None, 'calculated': calculated, 'missing': missing,
            'micronutrients': list(micronutrients.values()),
            'complete': bool(calculated) and not missing,
            'unverified': any(r['include'] and r.get('profile') and r['profile']['verification_status'] != 'verified' for r in rows)}
