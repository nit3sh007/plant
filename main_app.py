# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

try:
  # Loading the Model
  model = load_model('plant1.h5')
except:
  st.error("Error loading the model. Please check the model file and ensure it is valid and compatible with the current environment.")

# Name of Classes
CLASS_NAMES = ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy','Corn_(maize)___Common_rust_']

# Setting Title of App
st.title("Plant Disease Detection")
st.markdown("Upload an image of the plant leaf")
link='sample data [link](https://drive.google.com/drive/folders/1jmDFUaKS-MEmOtBWnIoPJZtLOUFW_xmd?usp=sharing)'
st.markdown(link,unsafe_allow_html=True)

# Uploading the plant image
plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict')

# On predict button click
if submit:
    if plant_image is not None:
        try:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            # Check if the image size is greater than 256x256
            if opencv_image.shape[0] > 256 or opencv_image.shape[1] > 256:
                st.warning("Image size is greater than 256x256. Resizing...")
                opencv_image = cv2.resize(opencv_image, (256, 256))

            # Displaying the image
            st.image(opencv_image, channels="BGR")
            st.write(opencv_image.shape)

            # Resizing the image
            opencv_image = cv2.resize(opencv_image, (256, 256))

            # Convert image to 4 Dimension
            opencv_image = np.expand_dims(opencv_image, axis=0)

            # Make Prediction
            Y_pred = model.predict(opencv_image)
            result_index = np.argmax(Y_pred)

            # Check if the result index is within bounds
            if 0 <= result_index < len(CLASS_NAMES):
                result = CLASS_NAMES[result_index]
                st.title(f"This is {result} leaf")
            else:
                st.warning("Invalid result index.")
        except Exception as e:
            st.error("Error during prediction:", e)
    else:
        st.warning("Please upload an image before clicking Predict.")
