import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tempfile
from PIL import Image

# Load the model
model = load_model("deepfake_detection_model.h5")

# Set page config
st.set_page_config(page_title="Deepfake Detection App", page_icon="🧠", layout="centered")

# Load external CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3342/3342137.png", width=120)
st.sidebar.title("About the Project")
st.sidebar.info(
    "🔎 This Deepfake Detection app was built using TensorFlow and Streamlit.\n\n"
    "Upload a face image, and the model will predict if it's Real ✅ or Fake ❌.\n\n"
    "Project by [Dipika Bangera]"
)

# Main title
st.title("🔍 Deepfake Detection System")
st.caption("AI-based image classification | Built with 💙 using TensorFlow")

# Upload
st.subheader("📤 Upload an Image for Analysis")
uploaded_file = st.file_uploader("Choose a .jpg or .png image", type=["jpg", "jpeg", "png"])

def preprocess_image(img):
    img = img.resize((256, 256))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

if uploaded_file is not None:
    with st.spinner('Analyzing the image... 🔍'):
        # Save the uploaded file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())

        # Open image
        image = Image.open(temp_file.name)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        # Preprocess
        processed_img = preprocess_image(image)

        # Prediction
        prediction = model.predict(processed_img)
        predicted_class = "Real ✅" if prediction[0][0] < 0.5 else "Fake ❌"

        # Display result
        if predicted_class == "Real ✅":
            st.markdown('<div class="result real">Result: Real ✅</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result fake">Result: Fake ❌</div>', unsafe_allow_html=True)
