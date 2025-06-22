import torch
import cv2
import numpy as np
import os
from train import Simple3DCNN

labels = ['bicep_curl', 'lateral_raise', 'squat']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = Simple3DCNN()
model.load_state_dict(torch.load("models/exercise_classifier.pth", map_location=device))
model.to(device)
model.eval()

def preprocess_video(video_path):
    if not os.path.isfile(video_path):
        print(f" File not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(" Failed to open video file. Please check the file path and format.")
        return None

    frames = []
    while len(frames) < 16:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (112, 112))
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        print(" No readable frames found in the video.")
        return None

    if len(frames) < 16:
        print(f" Only {len(frames)} frames found. Padding with last frame...")
        while len(frames) < 16:
            frames.append(frames[-1])

    frames = np.array(frames)
    tensor = torch.tensor(frames).permute(3, 0, 1, 2).unsqueeze(0) / 255.0
    return tensor.float().to(device)

def predict(video_path):
    print(f"\n Processing: {video_path}")
    input_tensor = preprocess_video(video_path)
    if input_tensor is None:
        return

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1).squeeze()[prediction].item()

    print(f" Prediction: {labels[prediction]} ({confidence * 100:.2f}%)\n")

if __name__ == "__main__":
  
    video_rel_path = "sample_videos/converted/fixed_test.mp4"
    video_abs_path = os.path.abspath(video_rel_path)
    predict(video_abs_path)
