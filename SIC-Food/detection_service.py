"""Keep every bounding box, even when the label is not in the catalog."""
from uuid import uuid4
from threading import RLock
from state_policy import map_label
_INFERENCE_LOCK = RLock()

def normalize_result(result):
    detections = []
    if result.boxes is None:
        return detections
    for box in result.boxes:
        class_id = int(box.cls[0])
        label = result.names[class_id]
        detections.append({'detection_id': uuid4().hex, 'original_label': label,
                           'class_id': class_id, 'confidence': float(box.conf[0]),
                           'bbox': [float(x) for x in box.xyxy[0]], 'prediction': map_label(label)})
    return detections

def detect_image(model, image):
    with _INFERENCE_LOCK:
        result = model.predict(image, conf=0.5, device='cpu', verbose=False)[0]
    detections = normalize_result(result)
    # Number each box so duplicate ingredients can be matched to editable rows.
    from PIL import Image, ImageDraw
    import numpy as np
    annotated = Image.fromarray(result.plot()[:, :, ::-1])
    draw = ImageDraw.Draw(annotated)
    for index, detection in enumerate(detections, 1):
        x, y = detection['bbox'][:2]
        draw.rectangle((x, y, x + 34, y + 22), fill='black')
        draw.text((x + 4, y + 4), str(index), fill='white')
    return detections, np.asarray(annotated)
