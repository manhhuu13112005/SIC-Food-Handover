"""Synthetic unit tests; real checkpoint checks live in check_website_models.py."""
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from image_service import decode_image, normalize_image
from pipeline_config import parse_condition_class
from condition_service import ConditionClassifier, load_classifier_file
from nutrition_service import calculate, load_profiles
from ui_components import food_name


def png(mode='RGB'):
    b=io.BytesIO()
    Image.new(mode,(300,270)).save(b,format='PNG')
    return b.getvalue()


class WebsiteUnitTests(unittest.TestCase):
    def test_canonical_labels_and_vietnamese_fallback(self):
        self.assertEqual(parse_condition_class('bell_pepper__normal_appearance'), ('bell_pepper','normal_appearance'))
        self.assertEqual(parse_condition_class('eggplant__visible_defect'), ('eggplant','visible_defect'))
        for label in ('apple__fresh','apple__normal_appearance__extra','__visible_defect',None):
            with self.assertRaises(ValueError): parse_condition_class(label)
        self.assertEqual(food_name('Apple'),'Táo')
        self.assertEqual(food_name('unlisted food'),'unlisted food')

    def test_images_modes_formats_and_exif(self):
        for mode in ('RGB','RGBA','L'):
            image=decode_image(png(mode))
            self.assertEqual(image.mode,'RGB')
        for fmt in ('JPEG','WEBP'):
            b=io.BytesIO();Image.new('RGB',(90,40)).save(b,format=fmt)
            self.assertEqual(decode_image(b.getvalue()).size,(90,40))
        image=Image.new('RGB',(90,40));image.getexif()[274]=6
        self.assertEqual(normalize_image(image).size,(40,90))

    def test_corrupt_nonimage_and_limits(self):
        for data in (b'',b'not image',png()[:45]):
            with self.assertRaises(ValueError): decode_image(data)
        with patch('image_service.MAX_IMAGE_BYTES',10):
            with self.assertRaises(ValueError): decode_image(png())
        with patch('image_service.MAX_PIXELS',10):
            with self.assertRaises(ValueError): decode_image(png())

    def test_preprocessing_matches_feature_cache(self):
        from torchvision.models import EfficientNet_B0_Weights
        transform=EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
        self.assertEqual(transform.resize_size,[256])
        self.assertEqual(transform.crop_size,[224])
        self.assertEqual(transform.interpolation.value,'bicubic')
        self.assertEqual(tuple(transform(decode_image(png())).shape),(3,224,224))
        self.assertEqual(transform.mean,[0.485,0.456,0.406])

    def test_output_top3_low_confidence_and_invalid_logits(self):
        import torch
        names=['apple__normal_appearance','apple__visible_defect','banana__normal_appearance']
        classifier=ConditionClassifier(lambda x:torch.tensor([[0.,1.,6.]]),lambda x:torch.zeros(3,4,4),names)
        result=classifier.predict(Image.new('RGB',(20,20)))
        self.assertEqual(result['label'],names[2])
        self.assertEqual(result['top3'][0]['class_index'],2)
        self.assertAlmostEqual(sum(result['probabilities']),1,places=6)
        classifier.model=lambda x:torch.zeros(1,3)
        self.assertEqual(classifier.predict(Image.new('RGB',(20,20)))['status'],'uncertain')
        for logits in (torch.zeros(1,2),torch.full((1,3),float('nan')),torch.full((1,3),float('inf'))):
            classifier.model=lambda x:logits
            with self.assertRaises(ValueError): classifier.predict(Image.new('RGB',(20,20)))

    def test_missing_checkpoint_and_no_cross_module_dependency(self):
        import model_loader
        absent=Path(__file__).parent/'no_such_model.pth'
        with self.assertRaises(FileNotFoundError): load_classifier_file(absent)
        with patch.object(model_loader,'CONDITION_MODEL_PATH',absent):
            with self.assertRaisesRegex(RuntimeError,'37 lớp'): model_loader.load_condition_model()
        import page1, page2
        self.assertFalse(hasattr(page1,'load_condition_model'))
        self.assertFalse(hasattr(page2,'load_condition_model'))

    def test_nutrition_original_data_and_input_errors(self):
        profiles=load_profiles()
        banana=next(p for p in profiles if p['ingredient_id']=='banana')
        self.assertEqual(calculate(banana,100)['kcal'],89)
        self.assertEqual(calculate(banana,150)['kcal'],133.5)
        self.assertAlmostEqual(calculate(banana,150)['protein_g'],1.635)
        for grams in (0,-1,'150',None,True,float('inf'),float('nan')):
            with self.assertRaises(ValueError): calculate(banana,grams)
        with self.assertRaises(ValueError): calculate({'per_100g':{'kcal':89}},100)
        self.assertFalse(any(p['ingredient_id']=='not_in_database' for p in profiles))

if __name__=='__main__': unittest.main()

