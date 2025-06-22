import streamlit as st
import tempfile
import os
import sys

# Add scripts/ directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from inference import predict, preprocess_video, model, labels, device
import torch

st.set_page_config(page_title="Gym Exercise Classifier", layout="centered")
st.title(" Gym Exercise Video Classifier")
st.write("Upload a short video (MP4, 2–5 seconds, showing a single exercise).")

uploaded_file = st.file_uploader(" Upload your test video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        temp_path = temp_video.name

    input_tensor = preprocess_video(temp_path)

    if input_tensor is not None:
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1).squeeze()[prediction].item()

        st.success(f" **Prediction:** {labels[prediction]}  \n **Confidence:** {confidence * 100:.2f}%")
    else:
        st.error(" Could not process this video. Try a different clip with more frames.")
