"""AppTest integration: UI is real, model/upload fixtures explicitly mocked."""
import io
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from streamlit.testing.v1 import AppTest
from state_policy import map_label
ROOT=Path(__file__).resolve().parents[1]


def picture(color):
    b=io.BytesIO();Image.new('RGB',(32,32),color).save(b,format='PNG');return b.getvalue()


def button(app,label):
    next(b for b in app.button if b.label==label).click().run(timeout=30)
    assert not app.exception,app.exception


def control(app,kind,label):
    return next(x for x in getattr(app,kind) if x.label==label)


class NavigationUITests(unittest.TestCase):
    def test_actual_main_navigation_and_manual_nutrition(self):
        app=AppTest.from_file(str(ROOT/'main.py'), default_timeout=30).run(timeout=30)
        self.assertFalse(app.exception)
        button(app,'Khám phá món ăn →')
        button(app,'Thêm món ăn thủ công')
        control(app,'selectbox','Món ăn / nguyên liệu').select('banana').run()
        control(app,'number_input','Khối lượng phần ăn được (g)').set_value(150.).run()
        control(app,'checkbox','Tôi xác nhận món ăn và thông tin đã chọn').check().run()
        control(app,'selectbox','Dữ liệu dinh dưỡng').select('legacy_banana').run()
        control(app,'checkbox','Dùng dữ liệu chưa kiểm chứng để ước tính tham khảo').check().run()
        self.assertEqual(app.metric[0].value,'133.50 kcal')
        app.radio(key='navigation').set_value('Trạng thái nguyên liệu').run()
        self.assertFalse(app.exception)
        app.radio(key='navigation').set_value('Món ăn & dinh dưỡng').run()
        self.assertEqual(control(app,'number_input','Khối lượng phần ăn được (g)').value,150)
        self.assertEqual(app.metric[0].value,'133.50 kcal')
        control(app,'number_input','Khối lượng phần ăn được (g)').set_value(0.).run()
        self.assertFalse(app.metric)
        control(app,'selectbox','Món ăn / nguyên liệu').select('apple').run()
        self.assertFalse(control(app,'checkbox','Tôi xác nhận món ăn và thông tin đã chọn').value)
        self.assertFalse(app.exception)


class FoodStateUITests(unittest.TestCase):
    def test_upload_inference_navigation_reset_and_module_independence(self):
        app=AppTest.from_file(str(ROOT/'tests/ui_harness.py'), default_timeout=30)
        app.session_state['fixture_img']=picture('red')
        app.session_state['fixture_cond']=picture('blue')
        app.session_state['fixture_detections']=[
            {'detection_id':'one','original_label':'Banana','class_id':0,'confidence':.9,
             'bbox':[0,0,20,20],'prediction':map_label('banana')}]
        app.run(timeout=30)
        button(app,'Khám phá món ăn →')
        button(app,'Nhận diện ảnh')
        self.assertEqual(app.session_state['detection_calls'],1)
        control(app,'number_input','Khối lượng phần ăn được (g)').set_value(150.).run()
        self.assertEqual(app.session_state['detection_calls'],1)
        app.radio(key='navigation').set_value('Trạng thái nguyên liệu').run()
        button(app,'Phân tích trạng thái')
        self.assertEqual(app.session_state['condition_calls'],1)
        self.assertEqual(len(app.get('progress')),3)
        app.radio(key='navigation').set_value('Món ăn & dinh dưỡng').run()
        self.assertEqual(len(app.session_state['img_items']),1)
        self.assertEqual(control(app,'number_input','Khối lượng phần ăn được (g)').value,150)
        self.assertEqual(app.session_state['detection_calls'],1)
        app.session_state['fixture_img']=picture('green')
        app.run()
        self.assertEqual(app.session_state['img_items'],[])
        self.assertFalse(app.metric)
        self.assertIn('label',app.session_state['cond_result'])
        button(app,'Nhận diện ảnh')
        app.session_state['fixture_fail']=True
        button(app,'Nhận diện ảnh')
        self.assertEqual(app.session_state['img_items'],[])
        self.assertFalse(app.exception)
        self.assertFalse(any('Internal synthetic' in e.value for e in app.error))
        app.session_state['fixture_fail']=False
        app.session_state['fixture_detections']=[]
        button(app,'Nhận diện ảnh')
        self.assertEqual(app.session_state['img_items'],[])
        app.session_state['fixture_img']=b'invalid file'
        app.run()
        self.assertTrue(app.error)
        self.assertFalse(app.exception)


class VideoUITests(unittest.TestCase):
    def test_video_frame_and_upload_replacement(self):
        app=AppTest.from_file(str(ROOT/'tests/ui_harness.py'), default_timeout=30)
        app.session_state['fixture_vid']=b'synthetic video'
        app.session_state['fixture_detections']=[
            {'detection_id':'v1','original_label':'Banana','class_id':0,'confidence':.9,
             'bbox':[0,0,20,20],'prediction':map_label('banana')}]
        app.run(timeout=30)
        button(app,'Khám phá món ăn →')
        app.radio(key='food_media_type').set_value('Video').run()
        button(app,'Nhận diện khung hình')
        self.assertEqual(len(app.session_state['vid_items']),1)
        control(app,'number_input','Khung hình (bắt đầu từ 0)').set_value(1).run()
        self.assertEqual(app.session_state['vid_items'],[])
        button(app,'Nhận diện khung hình')
        app.session_state['fixture_vid']=b'new synthetic video'
        app.run()
        self.assertEqual(app.session_state['vid_items'],[])
        self.assertFalse(app.exception)

if __name__=='__main__': unittest.main()

