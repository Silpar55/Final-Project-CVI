# visualizer.py
"""
Visualization functions for displaying motion detection and viewport tracking results.
"""

import os
import cv2
import numpy as np


def visualize_results(frames, motion_results, viewport_positions, viewport_size, output_dir):
    """
    Create visualization of motion detection and viewport tracking results.

    Args:
        frames: List of video frames
        motion_results: List of motion detection results for each frame
        viewport_positions: List of viewport center positions for each frame
        viewport_size: Tuple (width, height) of the viewport
        output_dir: Directory to save visualization results
    """
    # Create output directory for frames
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    viewport_dir = os.path.join(output_dir, "viewport")
    os.makedirs(viewport_dir, exist_ok=True)

    # Get dimensions for the output video
    height, width = frames[0].shape[:2]

    # Create video writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(output_dir, "motion_detection.mp4")
    video_writer = cv2.VideoWriter(video_path, fourcc, 5, (width, height))

    viewport_video_path = os.path.join(output_dir, "viewport_tracking.mp4")
    vp_width, vp_height = viewport_size
    viewport_writer = cv2.VideoWriter(
        viewport_video_path, fourcc, 5, (vp_width, vp_height)
    )

    # TODO: Implement visualization
    # 1. Process each frame
    #    a. Create a copy of the frame for visualization
    #    b. Draw bounding boxes around motion regions
    #       (hint: cv2.rectangle with green color (0, 255, 0))
    #    c. Draw the viewport rectangle
    #       (hint: cv2.rectangle with blue color (255, 0, 0))
    #    d. Extract the viewport content (the area inside the viewport)
    #    e. Add frame number to the visualization (hint: cv2.putText)
    #    f. Save visualization frames and viewport frames as images
    #    g. Write frames to both video writers
    # 2. Release the video writers when done

    # Example starter code:
    for i, frame in enumerate(frames):
        vis_frame = frame.copy()

        # Draw boxes
        for (x, y, w, h) in motion_results[i]:
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw viewport rectangle
        cx, cy = viewport_positions[i]
        x1 = cx - vp_width // 2
        y1 = cy - vp_height // 2
        x2 = cx + vp_width // 2
        y2 = cy + vp_height // 2
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Add frame number
        cv2.putText(vis_frame, f"Frame: {i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Extract viewport content
        viewport_crop = frame[y1:y2, x1:x2]

        # Save images
        cv2.imwrite(os.path.join(frames_dir, f"frame_{i:04d}.jpg"), vis_frame)
        if viewport_crop.size > 0:
            cv2.imwrite(os.path.join(viewport_dir, f"viewport_{i:04d}.jpg"), viewport_crop)

        # Write to videos
        video_writer.write(vis_frame)
        if viewport_crop.shape[0] > 0 and viewport_crop.shape[1] > 0:
            viewport_writer.write(viewport_crop)

    # Release writers
    video_writer.release()
    viewport_writer.release()

    print(f"Visualization saved to {video_path}")
    print(f"Viewport video saved to {viewport_video_path}")
    print(f"Individual frames saved to {frames_dir} and {viewport_dir}")