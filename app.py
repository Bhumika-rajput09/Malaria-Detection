import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.title("🦟 Malaria Detection App")
st.write("Upload a blood cell image")

uploaded_file = st.file_uploader("Choose image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)
    img = cv2.resize(img, (64,64))
    img = img / 255.0

    # Simple dummy logic for demo
    avg_pixel = np.mean(img)

    if avg_pixel > 0.5:
        st.error("🟥 Parasitized (Malaria Detected)")
    else:
        st.success("🟢 Uninfected (Healthy)")