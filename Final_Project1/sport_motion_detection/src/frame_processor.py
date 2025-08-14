# frame_processor.py
"""
Frame processing functions for the motion detection project.
"""

import cv2
import numpy as np


def process_video(video_path, target_fps=5, resize_dim=(1280, 720)):
    """
    Extract frames from a video at a specified frame rate.

    Args:
        video_path: Path to the video file
        target_fps: Target frames per second to extract
        resize_dim: Dimensions to resize frames to (width, height)

    Returns:
        List of extracted frames
    """

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video")
        exit()

    # Get original FPS
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Original FPS: {original_fps}")

    # We calculate the frame skip interval
    frame_interval = int(original_fps / target_fps)
    print(f"Frame interval: {frame_interval}")

    frame_count = 0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process only every frame interval frame
        if frame_count % frame_interval == 0:
            # Resize the frame
            frame = cv2.resize(frame, resize_dim)
            # append frame
            frames.append(frame)

        frame += 1

    cap.release()
    return frames
