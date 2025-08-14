# viewport_tracker.py
"""
Viewport tracking functions for creating a smooth "virtual camera".
"""

import cv2
import numpy as np


def calculate_region_of_interest(motion_boxes, frame_shape):
    """
    Calculate the primary region of interest based on motion boxes.

    Args:
        motion_boxes: List of motion detection bounding boxes
        frame_shape: Shape of the video frame (height, width)

    Returns:
        Tuple (x, y, w, h) representing the region of interest center point and dimensions
    """
    # TODO: Implement region of interest calculation
    # 1. Choose a strategy for determining the main area of interest
    #    - You could use the largest motion box
    #    - Or combine nearby boxes
    #    - Or use a weighted average of all motion boxes
    # 2. Return the coordinates of the chosen region

    # Example starter code:
    if not motion_boxes:
        # If no motion is detected, use the center of the frame
        height, width = frame_shape[:2]
        return (width // 2, height // 2, 0, 0)

    # Your implementation here
    # using largest motion box by area
    largest_box = max(motion_boxes, key=lambda b: b[2] * b[3])  # b = (x, y, w, h) -> w * h

    return largest_box


def track_viewport(frames, motion_results, viewport_size, smoothing_factor=0.3):
    """
    Track viewport position across frames with smoothing.

    Args:
        frames: List of video frames
        motion_results: List of motion detection results for each frame
        viewport_size: Tuple (width, height) of the viewport
        smoothing_factor: Factor for smoothing viewport movement (0-1)
                          Lower values create smoother movement

    Returns:
        List of viewport positions for each frame as (x, y) center coordinates
    """
    # TODO: Implement viewport tracking with smoothing
    # 1. For each frame, determine the region of interest based on motion_results
    # 2. Apply smoothing to avoid jerky movements
    #    - Use previous viewport positions to smooth the movement
    #    - Consider implementing a simple exponential moving average
    #    - Or a more advanced approach like Kalman filtering
    # 3. Ensure the viewport stays within the frame boundaries
    # 4. Return the list of viewport positions for all frames

    # Example starter code:
    viewport_positions = []

    # Initialize with center of first frame if available
    if frames:
        height, width = frames[0].shape[:2]
        prev_x, prev_y = width // 2, height // 2
    else:
        return []

    # Your implementation here

    vp_w, vp_h = viewport_size

    for frame_idx, frame in enumerate(frames):
        # Get the motion results for this frame
        motion_boxes = motion_results[frame_idx] if frame_idx < len(motion_results) else []

        # Get ROI
        roi_x, roi_y, _, _ = calculate_region_of_interest(motion_boxes, frame.shape)

        # Smooth the movement (exponential moving average)
        smoothed_x = int(prev_x + smoothing_factor * (roi_x - prev_x))
        smoothed_y = int(prev_y + smoothing_factor * (roi_y - prev_y))

        # Ensure viewport stays within frame boundaries
        smoothed_x = max(vp_w // 2, min(width - vp_w // 2, smoothed_x))
        smoothed_y = max(vp_h // 2, min(height - vp_h // 2, smoothed_y))

        viewport_positions.append((smoothed_x, smoothed_y))

        # Update previous position for next smoothing step
        prev_x, prev_y = smoothed_x, smoothed_y

    return viewport_positions
