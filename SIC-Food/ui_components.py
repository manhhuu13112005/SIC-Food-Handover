"""Shared Vietnamese copy and scoped visual components; no model imports."""
from html import escape
import streamlit as st

NAMES = {
    'unknown': 'Chưa xác định', 'apple': 'Táo', 'banana': 'Chuối', 'orange': 'Cam',
    'beef': 'Thịt bò', 'pork': 'Thịt lợn', 'chicken': 'Thịt gà', 'fish': 'Cá',
    'shrimp': 'Tôm', 'bell_pepper': 'Ớt chuông', 'bell pepper': 'Ớt chuông',
    'broccoli': 'Bông cải xanh', 'cabbage': 'Bắp cải', 'carrot': 'Cà rốt',
    'cucumber': 'Dưa chuột', 'eggplant': 'Cà tím', 'lettuce': 'Xà lách',
    'mango': 'Xoài', 'onion': 'Hành tây', 'potato': 'Khoai tây', 'tomato': 'Cà chua',
    'strawberry': 'Dâu tây', 'grapes': 'Nho', 'grape': 'Nho', 'pancake': 'Bánh kếp',
    'fries': 'Khoai tây chiên', 'french fries': 'Khoai tây chiên', 'zucchini': 'Bí ngòi',
    'pizza': 'Pizza', 'pineapple': 'Dứa', 'watermelon': 'Dưa hấu', 'guacamole': 'Sốt bơ',
    'lemon': 'Chanh vàng', 'lentil': 'Đậu lăng', 'peas': 'Đậu Hà Lan', 'pear': 'Lê',
    'waffle': 'Bánh waffle', 'spinach': 'Rau chân vịt', 'garlic': 'Tỏi', 'avocado': 'Bơ',
    'cake': 'Bánh ngọt', 'coconut': 'Dừa', 'cheeseburger': 'Burger phô mai',
    'muffin': 'Bánh muffin', 'bread': 'Bánh mì', 'corn': 'Ngô', 'rice': 'Cơm / gạo',
    'egg': 'Trứng', 'cheese': 'Phô mai', 'milk': 'Sữa', 'yogurt': 'Sữa chua',
    'butter': 'Bơ sữa', 'olive oil': 'Dầu ô liu', 'pasta': 'Mì Ý', 'sauce': 'Nước sốt',
    'salad': 'Salad', 'mushroom': 'Nấm'
}
APPEARANCES = {'normal_appearance': 'Bề ngoài bình thường',
               'visible_defect': 'Có dấu hiệu bất thường',
               'unknown': 'Chưa đủ căn cứ kết luận'}
PAGES = ['Trang chủ', 'Món ăn & dinh dưỡng', 'Trạng thái nguyên liệu']


def food_name(value):
    return NAMES.get(value.casefold(), value)


def setup_style():
    # Only style our own HTML classes. Native widgets use Streamlit's theme.
    st.markdown("""<style>
    .food-hero{background:linear-gradient(125deg,#123d2c,#256649);padding:2.6rem;
      border-radius:24px;color:#fff;margin:0 0 1.6rem;position:relative;overflow:hidden}
    .food-hero h1{color:#fff;font-size:clamp(2rem,4vw,3.5rem);line-height:1.14;
      letter-spacing:-.04em;max-width:760px;margin:12px 0 18px}
    .food-hero p{color:#d8e9df;max-width:680px;font-size:1.03rem;line-height:1.7}
    .food-eyebrow{font-size:.75rem;letter-spacing:.14em;font-weight:700;text-transform:uppercase}
    .food-tag{display:inline-block;padding:5px 12px;border:1px solid #80ae91;
      border-radius:20px;font-size:.78rem;margin:12px 8px 0 0;color:#e9f5ed}
    .food-empty{border:1px dashed #bdd3c4;border-radius:16px;background:#f6faf7;
      padding:3rem 1.5rem;text-align:center;color:#597364;margin:1rem 0}
    .food-kicker{font-size:.78rem;letter-spacing:.1em;color:#377855;font-weight:700;
      text-transform:uppercase;margin-bottom:.6rem}
    @media(max-width:640px){.food-hero{padding:1.5rem}.food-empty{padding:2rem 1rem}}
    </style>""", unsafe_allow_html=True)


def hero():
    st.markdown("""<div class="food-hero">
    <div class="food-eyebrow">SIC–FOOD / TRỢ LÝ HÌNH ẢNH THỰC PHẨM</div>
    <h1>Hiểu món ăn.<br>Quan sát nguyên liệu.</h1>
    <p>Nhận diện thực phẩm, ước tính khẩu phần và quan sát trạng thái bên ngoài.
    Hai công cụ độc lập, một không gian làm việc.</p>
    <span class="food-tag">Nhận diện bằng YOLO</span>
    <span class="food-tag">Phân loại bằng EfficientNet-B0</span>
    </div>""", unsafe_allow_html=True)


def empty_state(title, body):
    st.markdown(f'<div class="food-empty"><strong>{escape(title)}</strong><br><br>{escape(body)}</div>',
                unsafe_allow_html=True)


def page_intro(kicker, title, body):
    st.markdown(f'<div class="food-kicker">{escape(kicker)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(body)


def nutrition_cards(result):
    values = result['total']
    if values is None:
        st.info('Xác nhận món ăn, chọn dữ liệu và nhập khối lượng để tính khẩu phần.')
        return
    st.subheader('Dinh dưỡng khẩu phần')
    if not result['complete']:
        st.warning('Tổng chưa đầy đủ: chỉ bao gồm các mục tính được.')
    columns = st.columns(2)
    for i, (key, label, unit) in enumerate([
        ('kcal', 'Năng lượng', 'kcal'), ('protein_g', 'Chất đạm', 'g'),
        ('carbohydrate_g', 'Carbohydrate', 'g'), ('fat_g', 'Chất béo', 'g')]):
        columns[i % 2].metric(label, f'{values[key]:.2f} {unit}')
    st.bar_chart({'Chất dinh dưỡng': ['Chất đạm', 'Carbohydrate', 'Chất béo'],
                  'Khối lượng (g)': [values['protein_g'], values['carbohydrate_g'], values['fat_g']]},
                 x='Chất dinh dưỡng', y='Khối lượng (g)', color='#30845a', height=210)
    if result['unverified']:
        st.caption('Ước tính sử dụng dữ liệu gốc chưa kiểm chứng; không suy khối lượng từ ảnh.')
    if result['micronutrients']:
        st.write('Vi chất có dữ liệu (không coi dữ liệu thiếu là 0)')
        st.dataframe(result['micronutrients'], hide_index=True)

