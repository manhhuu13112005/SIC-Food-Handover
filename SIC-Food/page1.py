"""YOLO food recognition and nutrition; EfficientNet is a separate page."""
import json
import time
import logging
import streamlit as st
from detection_service import detect_image
from model_loader import load_model, MODEL_PATH
from image_service import decode_image
from nutrition_service import load_profiles, summarize
from state_policy import INGREDIENTS, PREPARATIONS, CONDITIONS, allowed_conditions, new_item
from app_state import upload, reset_results, clear_input, edit_widget
from ui_components import food_name, page_intro, empty_state, nutrition_cards

LOG = logging.getLogger(__name__)


def show_model_scope(model):
    names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    st.caption(f'YOLO Detection · {len(names)} lớp thực tế · ngưỡng nhận diện 50% · CPU')
    with st.expander('Xem phạm vi nhận diện của YOLO'):
        st.write('Chỉ nhận diện các nhãn đã huấn luyện. Một số món chưa có dữ liệu dinh dưỡng.')
        st.write(', '.join(names))


def app():
    page_intro('01 / NHẬN DIỆN & KHẨU PHẦN', 'Món ăn & dinh dưỡng',
               'Tải ảnh hoặc chọn khung hình video → nhận diện → xác nhận món ăn → nhập gram.')
    mode = st.radio('Loại đầu vào', ['Ảnh', 'Video'], horizontal=True, key='food_media_type')
    if mode == 'Video':
        import page2
        page2.app()
        return
    left, right = st.columns([1, 1.12], gap='large')
    with left:
        with st.container(border=True):
            st.subheader('Ảnh món ăn')
            payload = upload('img', 'Chọn ảnh JPG, PNG hoặc WEBP', ['jpg', 'jpeg', 'png', 'webp'])
            image = None
            if payload:
                try:
                    image = decode_image(payload)
                    st.image(image, caption=st.session_state.get('img_filename'), width='stretch')
                except ValueError as error:
                    reset_results('img')
                    st.error(str(error))
                st.button('Xóa ảnh', on_click=clear_input, args=('img',), key='img_clear')
            else:
                empty_state('Bắt đầu với một bức ảnh', 'Ảnh rõ nét giúp nhận diện tốt hơn. Tối đa 20 MB / 25 triệu pixel.')
            if not MODEL_PATH.is_file():
                st.warning(f'Chưa có YOLO tại {MODEL_PATH}. Bạn vẫn có thể nhập món ăn thủ công.')
            if st.button('Nhận diện ảnh', type='primary', disabled=image is None, width='stretch'):
                reset_results('img')
                try:
                    with st.spinner('YOLO đang nhận diện món ăn…'):
                        started = time.perf_counter()
                        model = load_model()
                        detections, annotated = detect_image(model, image)
                    st.session_state['img_items'] = [new_item(d) for d in detections]
                    st.session_state['img_annotated'] = annotated
                    st.session_state['img_model_names'] = model.names
                    st.session_state['img_elapsed'] = time.perf_counter()-started
                    if not detections:
                        st.info('Không tìm thấy đối tượng đạt ngưỡng 50%. Có thể thêm món ăn thủ công.')
                except Exception:
                    LOG.exception('YOLO image inference failed')
                    st.error('Không nhận diện được ảnh. Kiểm tra checkpoint YOLO hoặc thử ảnh khác; vẫn có thể nhập thủ công.')
        if 'img_annotated' in st.session_state:
            st.image(st.session_state['img_annotated'], caption='Các khung được đánh số tương ứng với món ăn', width='stretch')
            st.caption(f'Thời gian xử lý: {st.session_state["img_elapsed"]:.2f} giây')
        if 'img_model_names' in st.session_state:
            from types import SimpleNamespace
            show_model_scope(SimpleNamespace(names=st.session_state['img_model_names']))
    with right:
        st.subheader('Xác nhận khẩu phần')
        if st.button('Thêm món ăn thủ công', key='img_add'):
            st.session_state.setdefault('img_items', []).append(new_item())
        render_items()


def render_items(items_key='img_items', prefix_root='img_'):
    namespace = prefix_root.rstrip('_')
    profiles = load_profiles()
    rows = []
    for index, item in enumerate(st.session_state.get(items_key, []), 1):
        detection = item['detection']
        predicted = item['prediction']['ingredient_id']
        item_id = item['id']
        label = food_name(detection['original_label']) if detection else 'Món nhập thủ công'
        with st.expander(f'{index:02d} · {label}', expanded=True):
            if detection:
                st.caption(f'YOLO: {detection["original_label"]} · độ tin cậy {detection["confidence"]:.1%}')
                if predicted == 'unknown':
                    st.info('Nhãn này chưa có ánh xạ dinh dưỡng. Hãy chọn đúng món nếu có dữ liệu; không thay bằng món tương tự.')
            def widget(field, factory, text, default, options=None, **kwargs):
                return edit_widget(namespace, item_id, field, factory, text, default, options, **kwargs)
            ingredient = widget('ingredient', st.selectbox, 'Món ăn / nguyên liệu', predicted,
                                list(INGREDIENTS), format_func=food_name)
            grams = widget('grams', st.number_input, 'Khối lượng phần ăn được (g)', 0.0,
                           min_value=0.0, max_value=100000.0, step=10.0)
            with st.expander('Thông tin chế biến và quan sát thủ công'):
                prep = widget('prep', st.selectbox, 'Trạng thái chế biến', 'unknown',
                              list(PREPARATIONS), format_func=PREPARATIONS.get)
                condition = widget('condition', st.selectbox, 'Tình trạng quan sát', 'unknown',
                                   allowed_conditions(ingredient), format_func=CONDITIONS.get)
                st.caption('YOLO không đánh giá độ tươi. Module trạng thái nguyên liệu hoạt động riêng.')
            confirmed = widget('confirmed', st.checkbox, 'Tôi xác nhận món ăn và thông tin đã chọn', False)
            include = widget('include', st.checkbox, 'Đưa vào khẩu phần', True)
            candidates = {p['profile_id']: p for p in profiles
                          if p['ingredient_id'] == ingredient and p['preparation_state'] == prep}
            profile_id = widget('profile', st.selectbox, 'Dữ liệu dinh dưỡng', '',
                                [''] + list(candidates),
                                format_func=lambda k: f'{food_name(candidates[k]["ingredient_id"])} · trên 100 g' if k else 'Chọn hồ sơ phù hợp')
            profile = candidates.get(profile_id)
            accepted = False
            if profile:
                st.caption('Nguồn: food_items.json · dữ liệu gốc chưa kiểm chứng, trạng thái chế biến chưa xác định.')
                accepted = widget('accept', st.checkbox, 'Dùng dữ liệu chưa kiểm chứng để ước tính tham khảo', False)
            elif not candidates:
                st.info('Chưa có dữ liệu dinh dưỡng cho lựa chọn này. Không tự điền giá trị còn thiếu.')
            rows.append({'id': str(index), 'ingredient_id': ingredient, 'preparation_state': prep,
                         'visual_condition': condition, 'grams': grams, 'include': include,
                         'confirmed': confirmed, 'profile': profile, 'accept_unverified': accepted})
    if not rows:
        empty_state('Khẩu phần của bạn', 'Kết quả nhận diện và phần điều chỉnh món ăn sẽ xuất hiện ở đây.')
        return
    result = summarize(rows)
    for missing in result['missing']:
        st.caption(f'Mục {missing["id"]}: {missing["reason"]}')
    nutrition_cards(result)
    st.download_button('Tải bảng khẩu phần (JSON)',
                       json.dumps({'items': rows, 'result': result}, ensure_ascii=False, indent=2),
                       'khau_phan.json', 'application/json', key=namespace+'_export')

