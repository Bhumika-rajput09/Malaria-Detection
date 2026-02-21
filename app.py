import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("malaria_model.h5")

st.title("🦟 Malaria Detection App")
st.write("Upload a blood cell image to check for malaria.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = img.reshape(1, 64, 64, 3)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        st.error("🟥 Parasitized (Malaria Detected)")
    else:
        st.success("🟢 Uninfected (Healthy)")