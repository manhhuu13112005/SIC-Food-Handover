"""Synthetic fixtures only: not model predictions or verified nutrition data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from nutrition_service import calculate, summarize
from state_policy import map_label, reset_image_state
from detection_service import normalize_result

PROFILE = {'ingredient_id': 'banana', 'preparation_state': 'unknown', 'verification_status': 'verified',
           'per_100g': {'kcal': 100, 'protein_g': 2, 'carbohydrate_g': 20, 'fat_g': 1}}

class WorkflowTests(unittest.TestCase):
    def row(self, **changes):
        row = dict(id='1', include=True, confirmed=True, ingredient_id='banana',
                   preparation_state='unknown', grams=150, profile=PROFILE)
        return {**row, **changes}

    def test_portion(self):
        self.assertEqual(calculate(PROFILE, 150), {k: v*1.5 for k,v in PROFILE['per_100g'].items()})

    def test_unknown_and_cooked(self):
        self.assertEqual(map_label('chicken')['preparation_state'], 'unknown')
        self.assertEqual(map_label('chicken')['visual_condition'], 'unknown')
        self.assertEqual(map_label('chicken_cooked')['visual_condition'], 'unknown')
        self.assertEqual(map_label('unmapped')['ingredient_id'], 'unknown')

    def test_produce_policy(self):
        from state_policy import allowed_conditions
        for ingredient in ('tomato', 'banana', 'carrot', 'apple'):
            self.assertIn('fresh_appearing', allowed_conditions(ingredient))
        self.assertIn('normal_appearing', allowed_conditions('bread'))

    def test_instances(self):
        box = SimpleNamespace(cls=[0], conf=[0.8], xyxy=[[1,2,3,4]])
        ds = normalize_result(SimpleNamespace(names={0:'chicken'}, boxes=[box,box]))
        self.assertEqual(len(ds),2)
        self.assertNotEqual(ds[0]['detection_id'],ds[1]['detection_id'])

    def test_missing_and_partial(self):
        self.assertIsNone(summarize([self.row(profile=None)])['total'])
        result = summarize([self.row(),self.row(id='2',profile=None)])
        self.assertFalse(result['complete'])
        self.assertEqual(len(result['missing']),1)

    def test_no_stale_total(self):
        self.assertEqual(summarize([self.row()])['total']['kcal'],150)
        self.assertIsNone(summarize([self.row(grams=0)])['total'])
        self.assertEqual(summarize([self.row(grams=200)])['total']['kcal'],200)
        self.assertIsNone(summarize([self.row(confirmed=False)])['total'])
        self.assertIsNone(summarize([self.row(preparation_state='cooked')])['total'])

    def test_image_reset(self):
        state={'img_items':[1], 'img_old_grams':100,'img_annotated':'old','other':1}
        reset_image_state(state,'new')
        self.assertEqual(state,{'img_items':[],'img_token':'new','other':1})

    def test_model_failures_are_actionable(self):
        import model_loader
        with patch.object(model_loader, 'MODEL_PATH', Path(__file__).parent / 'missing-model-for-test.pt'):
            with self.assertRaisesRegex(RuntimeError, 'thủ công'):
                model_loader.load_model()
        with patch.object(model_loader, 'MODEL_PATH', Path(__file__)), patch.object(model_loader, '_load_cached', side_effect=ValueError('bad weights')):
            with self.assertRaisesRegex(RuntimeError, 'Không nạp được'):
                model_loader.load_model()

    def test_invalid_weights_and_missing_fields(self):
        for grams in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                calculate(PROFILE, grams)
        with self.assertRaises(ValueError):
            calculate({**PROFILE, 'per_100g': {'kcal':100}}, 150)

    def test_legacy_requires_opt_in(self):
        from nutrition_service import load_profiles
        profiles = load_profiles()
        self.assertEqual(len(profiles),47)
        self.assertTrue(all(p['verification_status']=='unverified' for p in profiles))
        profile = next(p for p in profiles if p['ingredient_id']=='banana')
        self.assertIsNone(summarize([self.row(profile=profile)])['total'])
        result = summarize([self.row(profile=profile,accept_unverified=True)])
        self.assertEqual(result['total']['kcal'],133.5)
        self.assertIsNone(summarize([self.row(profile=profile,accept_unverified=True,preparation_state='raw')])['total'])

if __name__ == '__main__':
    unittest.main()
