"""Real local checkpoint smoke check; no download or training."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['SIC_MODEL_PATH'] = str(ROOT/'best.pt')
from PIL import Image
from model_loader import load_model
from detection_service import detect_image
model = load_model()
assert load_model() is model, 'Model cache did not reuse the resource'
detections, image = detect_image(model, Image.open(ROOT/'images/test image.png').convert('RGB'))
assert len({d['detection_id'] for d in detections}) == len(detections)
assert all(d['prediction']['preparation_state']=='unknown' for d in detections)
print('Real model:',len(model.names),'classes;',len(detections),'boxes; labels:', [d['original_label'] for d in detections])
print('PASS: model load/cache and real-image inference; no accuracy claim')
