


import streamlit as st
from PIL import Image
import cv2 
import numpy as np

import torch
from realesrgan import RealESRGAN

def main():

    selected_box = st.sidebar.selectbox(
    'Select',
    ('About','Image Enhancement')
    )
    
    if selected_box == 'About':
        about() 
    if selected_box == 'Image Enhancement':
        photo()
    

def about():
    
    st.title("Image Enhancement using GAN's")
    
    st.subheader('A web app that converts low resolution CCTV footages to high resolution images.You can give either an input image in .jpeg format or a video.Note that video enhancement feature(not added yet) takes a bit more time, since we need to work with each frames.')
    
    st.image('cctv.jpg',use_column_width=True)


def load_image(filename):
    image = cv2.imread(filename)
    return image
def convert(img):
    
    with st.spinner('Working on it...'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = RealESRGAN(device, scale=4)
        model.load_weights('weights/RealESRGAN_x4.pth')

   
        image = img.convert('RGB')

        sr_image = model.predict(image)
    st.success('Done!')
    
    st.image(sr_image, caption='Super Resolution image', use_column_width=True)

def photo():

    st.header("Image Enhancement")
    uploaded_file = st.file_uploader("Choose an image...", type="jpeg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        convert(image) 
    
    
    

    

    
if __name__ == "__main__":
    main()
