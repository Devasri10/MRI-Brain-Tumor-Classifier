import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

@st.cache_resource
def load_model():
    model = keras.models.load_model('model.keras')
    return model

model = load_model()

Classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.title('MRI Brain Tumor Classifier')
st.write('Upload an MRI image to detect the tumor type')

uploaded_image = st.file_uploader('Upload', type = ['jpg', 'png', 'jpeg'])

if uploaded_image is not None:

    image = Image.open(uploaded_image).convert('RGB')

    st.image(image, caption='Uploaded MRI Image', use_container_width=True)

    image = image.resize((256,256))

    img_array = np.array(image) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    try:
        if st.button('Predict tumor type'):
            prediction = model.predict(img_array)

            predicted_class_index = np.argmax(prediction, axis=1)[0]
            confidence_score = np.max(prediction, axis=1)[0]

            st.subheader('Prediction')

            if Classes[predicted_class_index] == 'notumor':
                st.success('No Tumor Detected')
            else:
                st.error(f"Tumor Type: {Classes[predicted_class_index]}")

            st.write(f"Confidence: {confidence_score * 100:.2f}%")

    except Exception as e:
        print("Error in processing the image:", str(e))

