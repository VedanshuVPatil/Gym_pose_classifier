import os
import cv2
from moviepy.editor import VideoFileClip

VIDEO_DIR = "sample_videos"
FIXED_DIR = "sample_videos/converted"
os.makedirs(FIXED_DIR, exist_ok=True)

def inspect_and_convert(video_name):
    path = os.path.join(VIDEO_DIR, video_name)
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print(f" Cannot open: {video_name}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps else 0
    cap.release()

    print(f" {video_name} — {frame_count} frames, {fps:.2f} FPS, {duration:.2f} sec")

    if frame_count < 16 or fps == 0:
        print("  Re-encoding for compatibility...")
        clip = VideoFileClip(path).set_fps(25)
        fixed_path = os.path.join(FIXED_DIR, f"fixed_{video_name}")
        clip.write_videofile(fixed_path, codec="libx264", audio=False)
        print(f" Saved fixed video: {fixed_path}\n")
    else:
        print(" No fix needed.\n")

def main():
    for file in os.listdir(VIDEO_DIR):
        if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            inspect_and_convert(file)

if __name__ == "__main__":
    main()
