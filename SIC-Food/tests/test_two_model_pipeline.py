"""Synthetic crops, scores and nutrient values; no production data."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from condition_service import classify_detections, ConditionClassifier
from detection_service import normalize_result
from nutrition_service import summarize
from pipeline_config import TARGET_INGREDIENTS, parse_condition_class


class PipelineTests(unittest.TestCase):
    def detections(self, label='tomato', bbox=None):
        box = SimpleNamespace(cls=[0], conf=[0.9], xyxy=[bbox or [1, 2, 30, 40]])
        return normalize_result(SimpleNamespace(names={0: label}, boxes=[box, box]))

    def test_class_contract(self):
        self.assertEqual(len(set(TARGET_INGREDIENTS)), 20)
        self.assertEqual(parse_condition_class('bell_pepper_defect'), ('bell_pepper', 'visible_defect'))
        with self.assertRaises(ValueError):
            parse_condition_class('tomato_cooked')

    def test_per_box_classification(self):
        classifier = Mock()
        classifier.predict.return_value = {'visual_condition': 'mold', 'confidence': .9, 'label': 'tomato_mold', 'status': 'classified'}
        detections = self.detections()
        classify_detections(detections, Image.new('RGB', (50, 50)), classifier)
        self.assertEqual(classifier.predict.call_count, 2)
        self.assertEqual(classifier.predict.call_args.args[0].size, (29, 38))
        self.assertNotEqual(detections[0]['detection_id'], detections[1]['detection_id'])
        self.assertEqual(detections[0]['prediction']['visual_condition'], 'mold')
        self.assertEqual(detections[0]['prediction']['preparation_state'], 'unknown')

    def test_missing_unsupported_invalid_and_failure(self):
        image = Image.new('RGB', (50, 50))
        for label, bbox, classifier, expected in [
            ('tomato', None, None, 'model_unavailable'),
            ('unmapped', None, Mock(), 'unsupported_ingredient'),
            ('tomato', [60, 0, 70, 10], Mock(), 'invalid_crop'),
            ('tomato', None, Mock(predict=Mock(side_effect=RuntimeError())), 'inference_error')]:
            detections = classify_detections(self.detections(label, bbox), image, classifier)
            self.assertEqual(detections[0]['condition_assessment']['status'], expected)
            self.assertEqual(detections[0]['prediction']['visual_condition'], 'unknown')

    def test_wrong_ingredient_and_low_confidence(self):
        import torch
        def classifier(logits):
            return ConditionClassifier(lambda _: torch.tensor([logits]), lambda _: torch.zeros(3, 4, 4), ['apple_fresh', 'tomato_defect'])
        image = Image.new('RGB', (4, 4))
        self.assertEqual(classifier([10., 0.]).predict(image, 'tomato')['visual_condition'], 'unknown')
        self.assertEqual(classifier([0., 0.]).predict(image, 'tomato')['visual_condition'], 'unknown')
        self.assertEqual(classifier([0., 10.]).predict(image, 'tomato')['visual_condition'], 'visible_defect')

    def test_partial_micronutrients(self):
        profile = {'ingredient_id': 'tomato', 'preparation_state': 'unknown', 'verification_status': 'verified',
                   'per_100g': {'kcal': 20, 'protein_g': 1, 'carbohydrate_g': 3, 'fat_g': 0, 'vitamin_c_mg': 10}}
        row = {'id': '1', 'include': True, 'confirmed': True, 'ingredient_id': 'tomato',
               'preparation_state': 'unknown', 'grams': 150, 'profile': profile}
        other = {**row, 'id': '2', 'profile': {**profile, 'per_100g': {k: v for k, v in profile['per_100g'].items() if k != 'vitamin_c_mg'}}}
        result = summarize([row, other])
        self.assertEqual(result['total']['kcal'], 60)
        self.assertEqual(result['micronutrients'], [{'nutrient': 'vitamin_c_mg', 'value': 15., 'covered_items': 1, 'complete': False}])

    def test_real_efficientnet_checkpoint_roundtrip(self):
        import tempfile
        import torch
        from torchvision.models import efficientnet_b0
        from condition_service import load_classifier_file
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            model = efficientnet_b0(weights=None)
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'synthetic.pt'
                torch.save({'architecture': 'efficientnet_b0', 'image_size': 224,
                            'class_names': ['tomato_defect', 'tomato_fresh'],
                            'state_dict': model.state_dict()}, path)
                classifier = load_classifier_file(path)
                result = classifier.predict(Image.new('RGB', (40, 60)), 'tomato')
                self.assertTrue(0 <= result['confidence'] <= 1)
                self.assertIn(result['status'], ('classified', 'uncertain'))
                self.assertEqual(classifier.class_names, ['tomato_defect', 'tomato_fresh'])
        finally:
            torch.set_num_threads(previous_threads)

    def test_selected_video_frame(self):
        import cv2
        import numpy as np
        import tempfile
        from page2 import detect_video_frame
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 1, (64, 64))
            self.assertTrue(writer.isOpened())
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
            writer.write(np.full((64, 64, 3), 255, dtype=np.uint8))
            writer.release()
            box = SimpleNamespace(cls=[0], conf=[.9], xyxy=[[0, 0, 32, 32]])
            result = SimpleNamespace(names={0: 'tomato'}, boxes=[box], plot=lambda: np.zeros((64, 64, 3), dtype=np.uint8))
            detector = Mock()
            detector.predict.return_value = [result]
            detections, annotated = detect_video_frame(detector, path, 1)
            self.assertEqual(detections[0]['frame_index'], 1)
            self.assertNotIn('condition_assessment', detections[0])  # independent YOLO website path
            self.assertGreater(np.array(detector.predict.call_args.args[0]).mean(), 240)
            self.assertEqual(annotated.size, (64, 64))
            with self.assertRaises(RuntimeError):
                detect_video_frame(detector, path, 50)


if __name__ == '__main__':
    unittest.main()
