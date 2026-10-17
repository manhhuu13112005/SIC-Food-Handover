"""Mock uploads and models, real main.py/widgets. Never imported in production."""
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import runpy
import streamlit as st
from PIL import Image
import page1, page2, condition_page


def uploader(*args, **kwargs):
    key=kwargs.get('key','')
    namespace=key.split('_')[0]
    payload=st.session_state.get('fixture_'+namespace)
    return BytesIO(payload) if payload else None


def infer(*args):
    st.session_state['detection_calls']=st.session_state.get('detection_calls',0)+1
    if st.session_state.get('fixture_fail'):
        raise RuntimeError('Internal synthetic failure must not leak')
    return st.session_state.get('fixture_detections',[]), Image.new('RGB',(32,32))


class Classifier:
    class_names=['apple__normal_appearance','apple__visible_defect','banana__normal_appearance']
    def predict(self,image):
        st.session_state['condition_calls']=st.session_state.get('condition_calls',0)+1
        return {'ingredient':'apple','appearance':'normal_appearance','confidence':.8,'status':'classified',
                'elapsed_ms':10,'class_count':37,'label':self.class_names[0],
                'top3':[{'ingredient':'apple','appearance':'normal_appearance','confidence':.8},
                        {'ingredient':'apple','appearance':'visible_defect','confidence':.15},
                        {'ingredient':'banana','appearance':'normal_appearance','confidence':.05}]}


with patch('streamlit.file_uploader',side_effect=uploader), \
     patch.object(page1,'load_model',return_value=SimpleNamespace(names={0:'Banana'})), \
     patch.object(page1,'detect_image',side_effect=infer), \
     patch.object(page2,'load_model',return_value=SimpleNamespace(names={0:'Banana'})), \
     patch.object(page2,'detect_video_frame',side_effect=infer), \
     patch.object(condition_page,'load_condition_model',return_value=Classifier()):
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'main.py'),run_name='__main__')

