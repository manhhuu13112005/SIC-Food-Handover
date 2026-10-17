"""Selected-frame YOLO video support; no automatic ingredient classification."""
import tempfile
import logging
from pathlib import Path
import streamlit as st
from detection_service import detect_image
from model_loader import load_model
from state_policy import new_item
from app_state import upload, reset_results, clear_input
from page1 import render_items, show_model_scope
from ui_components import empty_state
LOG = logging.getLogger(__name__)


def detect_video_frame(model, video_path, frame_index, classifier=None):
    import cv2
    from PIL import Image
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError('Chỉ số khung hình phải là số nguyên không âm.')
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError('Không mở được video.')
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count > 0 and frame_index >= count:
            raise RuntimeError(f'Video chỉ có {count} khung hình. Chọn chỉ số nhỏ hơn {count}.')
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('Không đọc được khung hình đã chọn.')
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections, annotated = detect_image(model, image)
        # Compatibility for offline legacy callers, not used by the website.
        if classifier is not None:
            from condition_service import classify_detections
            classify_detections(detections, image, classifier)
        for detection in detections:
            detection['frame_index'] = frame_index
        return detections, Image.fromarray(annotated)
    finally:
        cap.release()


def app():
    left, right = st.columns([1, 1.12], gap='large')
    with left:
        with st.container(border=True):
            st.subheader('Video món ăn')
            st.caption('Phân tích một khung hình để tránh cộng lặp khẩu phần. Chưa có tracking.')
            payload = upload('vid', 'Chọn video MP4, MOV hoặc AVI', ['mp4', 'mov', 'avi'])
            if payload:
                if len(payload) > 64*1024*1024:
                    st.error('Video vượt quá 64 MB.')
                    payload = None
                    reset_results('vid')
                else:
                    st.video(payload)
                    st.button('Xóa video', on_click=clear_input, args=('vid',), key='vid_clear')
            else:
                empty_state('Chọn video có sẵn', 'Dung lượng tối đa 64 MB; chọn một khung hình để nhận diện.')
            index = int(st.number_input('Khung hình (bắt đầu từ 0)', min_value=0,
                                        value=st.session_state.get('vid_frame', 0), step=1, key='video_frame_choice'))
            if index != st.session_state.get('vid_frame'):
                reset_results('vid')
                st.session_state['vid_frame'] = index
            if st.button('Nhận diện khung hình', type='primary', disabled=payload is None, width='stretch'):
                reset_results('vid')
                path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as handle:
                        path = Path(handle.name)
                        handle.write(payload)
                    with st.spinner('YOLO đang xử lý khung hình…'):
                        model = load_model()
                        detections, annotated = detect_video_frame(model, path, index)
                    st.session_state['vid_items'] = [new_item(d) for d in detections]
                    st.session_state['vid_annotated'] = annotated
                    st.session_state['vid_model_names'] = model.names
                    if not detections:
                        st.info('Không có đối tượng đạt ngưỡng trong khung hình này.')
                except Exception:
                    LOG.exception('Video inference failed')
                    st.error('Không đọc hoặc nhận diện được khung hình. Kiểm tra định dạng video, chỉ số khung hình và checkpoint YOLO.')
                finally:
                    if path:
                        path.unlink(missing_ok=True)
        if 'vid_annotated' in st.session_state:
            st.image(st.session_state['vid_annotated'], caption=f'Kết quả tại khung hình {index}', width='stretch')
        if 'vid_model_names' in st.session_state:
            from types import SimpleNamespace
            show_model_scope(SimpleNamespace(names=st.session_state['vid_model_names']))
    with right:
        st.subheader('Khẩu phần từ khung hình')
        if st.button('Thêm món ăn thủ công', key='vid_add'):
            st.session_state.setdefault('vid_items', []).append(new_item())
        render_items('vid_items', 'vid_')

