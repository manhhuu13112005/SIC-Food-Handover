"""Bounded image decoding shared by the two independent website modules."""
from io import BytesIO
import warnings
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 25_000_000


def normalize_image(image):
    if not isinstance(image, Image.Image):
        raise ValueError('Đầu vào cần là một ảnh hợp lệ.')
    if image.width * image.height > MAX_PIXELS:
        raise ValueError('Ảnh quá lớn. Vui lòng dùng ảnh tối đa 25 triệu pixel.')
    # Match feature-cache training: EXIF transpose, then RGB (including alpha).
    return ImageOps.exif_transpose(image).convert('RGB')


def decode_image(payload):
    if not isinstance(payload, bytes) or not payload:
        raise ValueError('File ảnh rỗng hoặc không hợp lệ.')
    if len(payload) > MAX_IMAGE_BYTES:
        raise ValueError('Ảnh vượt quá 20 MB. Hãy chọn ảnh nhỏ hơn.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as image:
                if image.format not in ('JPEG', 'PNG', 'WEBP'):
                    raise ValueError('Chỉ hỗ trợ nội dung ảnh JPEG, PNG và WEBP.')
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError('Ảnh quá lớn. Vui lòng dùng ảnh tối đa 25 triệu pixel.')
                image.load()
                return normalize_image(image)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError('Không đọc được ảnh. File có thể hỏng hoặc không phải ảnh hợp lệ.') from error
