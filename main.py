import streamlit as st
import timm
from fastai.vision.all import *

st.title("🌳 Plantasia Tree Classifier 🌳")

# image upload
picture_choice = st.radio(
    "How do you want to upload the tree image?", ("Take a picture", "Upload an image"), horizontal=True,
    help = "The picture should be from close up to the tree so that the leaves are visible."
)

uploaded_file = None
if picture_choice == "Upload an image":
    uploaded_file = st.file_uploader("Choose an image...", type="jpeg")
elif picture_choice == "Take a picture":
    uploaded_file = st.camera_input("Take a picture of your the tree")

#make prediction
if uploaded_file is not None:
    im = PILImage.create(uploaded_file)
    learn = load_learner('model.pkl')
    x = learn.predict(im)
    
    st.title(f'{x[0]} Tree')

