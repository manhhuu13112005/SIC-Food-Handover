"""Food AI: two independent model workflows on one Streamlit website."""
import streamlit as st
st.set_page_config(page_title='Food AI · SIC-Food', page_icon='🌿', layout='wide', initial_sidebar_state='expanded')
from ui_components import PAGES, setup_style, hero
setup_style()


def navigate(page):
    st.session_state['navigation'] = page


with st.sidebar:
    st.title('🌿 Food AI')
    st.caption('SIC–FOOD / KHÔNG GIAN THỰC PHẨM')
    page = st.radio('Khám phá', PAGES, key='navigation')
    st.divider()
    st.markdown('**Hai công cụ độc lập**')
    st.caption('YOLO · nhận diện món ăn\n\nEfficientNet-B0 · quan sát nguyên liệu')
    st.info('Khối lượng do bạn nhập. Trạng thái hình ảnh không chứng nhận an toàn thực phẩm.')

if page == PAGES[0]:
    hero()
    st.subheader('Bạn muốn khám phá điều gì?')
    first, second = st.columns(2, gap='large')
    with first:
        with st.container(border=True):
            st.caption('01 / MÓN ĂN & KHẨU PHẦN')
            st.subheader('Nhận diện món ăn & dinh dưỡng')
            st.write('Tải ảnh hoặc video, xác nhận món được YOLO nhận diện và tính năng lượng theo khẩu phần của bạn.')
            st.caption('Bounding box · nhiều đối tượng · gram do người dùng nhập')
            st.button('Khám phá món ăn →', type='primary', on_click=navigate, args=(PAGES[1],), width='stretch')
    with second:
        with st.container(border=True):
            st.caption('02 / NGUYÊN LIỆU & QUAN SÁT')
            st.subheader('Nhận diện trạng thái nguyên liệu')
            st.write('Phân tích một nguyên liệu bằng EfficientNet-B0 và xem các dấu hiệu bất thường có thể nhìn thấy.')
            st.caption('Ảnh đơn / crop · 37 lớp thử nghiệm · ba dự đoán hàng đầu')
            st.button('Quan sát nguyên liệu →', on_click=navigate, args=(PAGES[2],), width='stretch')
    st.divider()
    st.subheader('Bắt đầu trong ba bước')
    for col, title, body in zip(st.columns(3),
        ['01 · Chọn công cụ', '02 · Tải hình ảnh', '03 · Đọc & xác nhận'],
        ['Món ăn và dinh dưỡng, hoặc trạng thái nguyên liệu.',
         'Dùng ảnh rõ nét. Với trạng thái, chỉ chọn một nguyên liệu.',
         'Kiểm tra dự đoán; nhập gram nếu bạn muốn tính khẩu phần.']):
        col.markdown('**' + title + '**')
        col.caption(body)
    st.caption('SIC-Food · Công cụ nghiên cứu hỗ trợ quan sát và ước tính. Không thay thế kiểm tra thực phẩm thực tế.')
elif page == PAGES[1]:
    import page1
    page1.app()
else:
    import condition_page
    condition_page.app()

