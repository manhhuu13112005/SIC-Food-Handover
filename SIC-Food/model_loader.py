"""Load local models lazily; manual mode never requires YOLO."""
from pathlib import Path
import os
import urllib.request
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get('SIC_MODEL_PATH', str(ROOT / 'best.pt'))).expanduser()
if not MODEL_PATH.is_absolute():
    MODEL_PATH = ROOT / MODEL_PATH
MODEL_PATH = MODEL_PATH.resolve()
MODEL_URL = 'https://huggingface.co/bandarai/food-reco-sic-2050/resolve/main/best.pt'
CONDITION_MODEL_PATH = Path(os.environ.get('SIC_CONDITION_MODEL_PATH', str(ROOT / 'models/efficientnet_37_fasttrack/efficientnet_b0_37_prototype.pth'))).expanduser()
if not CONDITION_MODEL_PATH.is_absolute():
    CONDITION_MODEL_PATH = ROOT / CONDITION_MODEL_PATH
CONDITION_MODEL_PATH = CONDITION_MODEL_PATH.resolve()


def configure_cpu():
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))


@st.cache_resource
def _load_condition_cached(path, modified_ns, size):
    from condition_service import load_classifier_file
    configure_cpu()
    return load_classifier_file(path, expected_classes=37)


def load_condition_model():
    if not CONDITION_MODEL_PATH.is_file():
        raise RuntimeError(f'Chưa tìm thấy EfficientNet 37 lớp. Bổ sung checkpoint tại: {CONDITION_MODEL_PATH}')
    try:
        stat = CONDITION_MODEL_PATH.stat()
        return _load_condition_cached(str(CONDITION_MODEL_PATH), stat.st_mtime_ns, stat.st_size)
    except Exception as error:
        raise RuntimeError(f'Không nạp được EfficientNet 37 lớp. Kiểm tra checkpoint và class_names tại: {CONDITION_MODEL_PATH}') from error

def download_model():
    from tempfile import NamedTemporaryFile
    from shutil import copyfileobj
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=MODEL_PATH.parent, suffix='.download', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, temporary.open('wb') as target:
            copyfileobj(response, target)
        if temporary.stat().st_size == 0:
            raise RuntimeError('Empty download')
        temporary.replace(MODEL_PATH)
    except Exception as error:
        raise RuntimeError('Tải model thất bại. Kiểm tra mạng và quyền ghi hoặc chọn model cục bộ.') from error
    finally:
        temporary.unlink(missing_ok=True)

@st.cache_resource

def _load_cached(path, modified_ns, size):
    from ultralytics import YOLO
    configure_cpu()
    model = YOLO(path)
    if model.task != 'detect':
        raise ValueError('Trang nhận diện cần checkpoint YOLO Detection.')
    return model

def load_model():
    if not MODEL_PATH.is_file():
        raise RuntimeError(f'Chưa có YOLO tại: {MODEL_PATH}. Bạn vẫn có thể thêm món ăn thủ công.')
    try:
        stat = MODEL_PATH.stat()
        return _load_cached(str(MODEL_PATH), stat.st_mtime_ns, stat.st_size)
    except Exception as error:
        raise RuntimeError('Không nạp được model: file không tương thích, hỏng hoặc thiếu Ultralytics/PyTorch. Có thể tiếp tục nhập thủ công.') from error
