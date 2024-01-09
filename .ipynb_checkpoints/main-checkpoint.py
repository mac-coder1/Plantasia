import streamlit as st
from fastai.vision.all import *

st.title("🌳 Plantasia Tree Classifier 🌳")

st.write('Shoutout to the haters who say there arent enough trees on this app yet, watch this space!!!')

#image upload

uploaded_file = st.file_uploader("Choose an image...", type="jpeg")

if uploaded_file is not None:
    im = PILImage.create(uploaded_file)
    learn = load_learner('model.pkl')
    x = learn.predict(im)
    
    st.write(f'This is an {x[0]} tree.')

