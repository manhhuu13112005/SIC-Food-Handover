"""Independent whole-image/crop EfficientNet page."""
import logging
import streamlit as st
from app_state import upload, reset_results, clear_input
from model_loader import load_condition_model, CONDITION_MODEL_PATH
from image_service import decode_image
from ui_components import food_name, APPEARANCES, page_intro, empty_state
LOG = logging.getLogger(__name__)


def app():
    page_intro('02 / QUAN SÁT NGUYÊN LIỆU', 'Trạng thái nguyên liệu',
               'EfficientNet-B0 phân loại một nguyên liệu hoặc một vùng crop; không định vị nhiều nguyên liệu.')
    left, right = st.columns([1, 1.12], gap='large')
    with left:
        with st.container(border=True):
            st.subheader('Ảnh nguyên liệu')
            payload = upload('cond', 'Chọn ảnh một nguyên liệu', ['jpg', 'jpeg', 'png', 'webp'])
            image = None
            if payload:
                try:
                    image = decode_image(payload)
                    st.image(image, caption=st.session_state.get('cond_filename'), width='stretch')
                except ValueError as error:
                    reset_results('cond')
                    st.error(str(error))
                st.button('Xóa ảnh', on_click=clear_input, args=('cond',), key='cond_clear')
            else:
                empty_state('Quan sát một nguyên liệu', 'Chọn ảnh rõ nét, gần vật thể; cắt riêng nguyên liệu nếu ảnh có nhiều loại.')
            if not CONDITION_MODEL_PATH.is_file():
                st.warning(f'Chưa có EfficientNet. Bổ sung checkpoint tại: {CONDITION_MODEL_PATH}')
            if st.button('Phân tích trạng thái', type='primary', disabled=image is None, width='stretch'):
                reset_results('cond')
                try:
                    with st.spinner('EfficientNet đang phân tích…'):
                        classifier = load_condition_model()
                        result = classifier.predict(image)
                    st.session_state['cond_result'] = result
                    st.session_state['cond_model_names'] = classifier.class_names
                except Exception:
                    LOG.exception('EfficientNet inference failed')
                    st.error('Không phân tích được ảnh. Kiểm tra checkpoint EfficientNet 37 lớp hoặc thử ảnh khác.')
        st.caption('Ảnh chỉ cho biết dấu hiệu bên ngoài, không chứng nhận thực phẩm an toàn để ăn.')
    with right:
        result = st.session_state.get('cond_result')
        if result:
            with st.container(border=True):
                st.subheader(food_name(result['ingredient']))
                state = APPEARANCES.get(result['appearance'], result['appearance'])
                st.write('Dự đoán: **' + state + '**')
                st.metric('Điểm softmax', f'{result["confidence"]:.1%}')
                if result['status'] != 'classified':
                    st.warning('Điểm dự đoán thấp: chưa đủ căn cứ kết luận. Thử ảnh rõ hơn, chỉ chứa một nguyên liệu.')
                else:
                    st.caption('Đây là dự đoán hình ảnh; điểm cao vẫn có thể sai với ảnh ngoài tập nhãn.')
                st.subheader('Ba lớp có điểm cao nhất')
                for rank, item in enumerate(result['top3'], 1):
                    label = f'{rank}. {food_name(item["ingredient"])} · {APPEARANCES.get(item["appearance"], item["appearance"])}'
                    st.progress(float(item['confidence']), text=f'{label} — {item["confidence"]:.1%}')
                st.caption(f'CPU · {result["elapsed_ms"]:.0f} ms · {result["class_count"]} lớp')
        else:
            empty_state('Kết quả quan sát', 'Tên nguyên liệu, trạng thái và ba dự đoán hàng đầu sẽ xuất hiện ở đây.')
        with st.expander('Phạm vi và giới hạn của mô hình', expanded=True):
            st.write('Mô hình thử nghiệm 37 lớp. Softmax là điểm dự đoán, không phải độ chính xác đã được kiểm định.')
            st.write('Validation 97,01% chỉ trên dữ liệu V3 gồm 26/37 lớp. Chưa có đánh giá độc lập bao phủ toàn bộ 37 lớp.')
            st.caption('Mô hình hiện không có lớp tôm hoặc broccoli bất thường. Ảnh không thuộc tập nhãn vẫn có thể nhận điểm cao; chưa có bộ phát hiện ngoài phân phối.')
            if 'cond_model_names' in st.session_state:
                with st.expander('Danh sách lớp đọc từ checkpoint'):
                    st.write(', '.join(st.session_state['cond_model_names']))

