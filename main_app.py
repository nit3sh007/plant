import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load the trained model
model_path = "plant1.h5"
loaded_model = tf.keras.models.load_model(model_path)

# Load the class dictionary
class_dict_path = "Disease-class_dict.csv"
class_dict = {}
with open(class_dict_path, 'r') as file:
    lines = file.readlines()[1:]
    for line in lines:
        index, plant_class, _, _ = line.strip().split(',')
        class_dict[int(index)] = plant_class

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_disease(img):
    img_array = preprocess_image(img)
    predictions = loaded_model.predict(img_array)
    predicted_class = class_dict[np.argmax(predictions)]
    return predicted_class

def main():
    st.title("Plant Disease Classification App")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded image
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption="Uploaded Image.", use_column_width=True)

        # Make predictions
        if st.button("Predict Disease"):
            disease = predict_disease(image_display)
            st.success(f"Predicted Disease: {disease}")

if __name__ == "__main__":
    main()
