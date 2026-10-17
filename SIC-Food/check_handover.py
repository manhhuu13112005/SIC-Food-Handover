"""Verify delivered files and run both real models without dataset/network."""
import hashlib
import json
import os
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def blocked(*args, **kwargs):
    raise RuntimeError('Network access is disabled for the handover check')


def main():
    manifest = json.loads((ROOT / 'HANDOVER_MANIFEST.json').read_text(encoding='utf-8'))
    for name, metadata in manifest['files'].items():
        path = ROOT / name
        assert path.is_file(), f'Missing: {name}'
        assert path.stat().st_size == metadata['bytes'], f'Size changed: {name}'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata['sha256'], f'Changed: {name}'
    assert not (ROOT / 'datasets').exists(), 'Handover should not contain datasets'
    os.environ['SIC_MODEL_PATH'] = str(ROOT / 'best.pt')
    os.environ['SIC_CONDITION_MODEL_PATH'] = str(ROOT / 'models/efficientnet_37_fasttrack/efficientnet_b0_37_prototype.pth')
    os.environ['YOLO_OFFLINE'] = 'true'
    original_connect, original_connect_ex = socket.socket.connect, socket.socket.connect_ex
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    try:
        from PIL import Image
        from model_loader import load_model, load_condition_model
        from detection_service import detect_image
        from image_service import normalize_image
        from page2 import detect_video_frame
        detector, classifier = load_model(), load_condition_model()
        assert detector is load_model() and classifier is load_condition_model()
        assert detector.task == 'detect' and len(detector.names) == 214
        assert len(classifier.class_names) == 37
        assert not classifier.model.training
        assert next(classifier.model.parameters()).device.type == 'cpu'
        with Image.open(ROOT / 'images/test image.png') as image:
            detections, _ = detect_image(detector, normalize_image(image))
        assert detections, 'Expected detections in sample image'
        video_detections, _ = detect_video_frame(detector, ROOT / 'images/test video.mp4', 0)
        results = []
        for sample in sorted((ROOT / 'images/real_test').glob('*.jpg')):
            with Image.open(sample) as image:
                result = classifier.predict(image)
            assert result['class_count'] == 37
            assert len(result['probabilities']) == 37
            assert all(0 <= p <= 1 for p in result['probabilities'])
            assert abs(sum(result['probabilities']) - 1) < 1e-5
            assert result['label'] == classifier.class_names[result['class_index']]
            results.append({'sample': sample.name, 'prediction': result['label'], 'confidence': result['confidence']})
        assert len(results) == 3
        print(json.dumps({'verified_files': len(manifest['files']), 'yolo_classes': len(detector.names),
                          'image_boxes': len(detections), 'video_boxes': len(video_detections),
                          'efficientnet_classes': 37, 'samples': results}, ensure_ascii=False, indent=2))
        print('PASS: both real models run without dataset or network. No accuracy claim.')
    finally:
        socket.socket.connect, socket.socket.connect_ex = original_connect, original_connect_ex


if __name__ == '__main__':
    main()
