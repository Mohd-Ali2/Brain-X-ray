import streamlit as st
from keras.preprocessing import image
import numpy as np
from keras.models import model_from_json
from PIL import Image

with open("cnn.json", "r") as json_file:
    loaded_model_json = json_file.read()
    
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model_weights.weights.h5')

def predict(image_file):
    test_image = Image.open(image_file)
    test_image = test_image.resize((128, 128))
    test_image = np.array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    
    result = loaded_model.predict(test_image)
    return result




st.sidebar.title('Author :memo:')
st.sidebar.subheader('Mohammad Ali :name_badge:')
st.title("Deep Learning Model for Brain X-Ray ")
st.sidebar.title('Description :scroll:')
st.sidebar.write("A brain deep learning model is a sophisticated artificial intelligence (AI) system designed to analyze and interpret data related to the brain, often in the form of medical images such as MRI scans, CT scans, or fMRI scans. These models leverage deep learning techniques, particularly convolutional neural networks (CNNs) and recurrent neural networks (RNNs), to extract meaningful features from brain images and make predictions or classifications based on those features.")
st.sidebar.title('Connect :link:')
st.sidebar.link_button('Linkdin :large_blue_diamond:', url='https://www.linkedin.com/in/mohdali02/')
st.sidebar.link_button('Github  :black_large_square:', url='github.com/Mohd-Ali2')



upload_file = st.file_uploader('Choose an image..', type=['jpeg', 'jpg', 'webp'])

if upload_file is not None:
    upload_image = Image.open(upload_file)
    st.image(upload_file, caption='Uploaded Image', use_column_width=True)
if st.button('Check'):
    result = predict(upload_file)
    

    if result[0][0] == 0:

        st.write('Normal:white_check_mark:')
    else:
        st.write('Not Normal:heavy_exclamation_mark:')
