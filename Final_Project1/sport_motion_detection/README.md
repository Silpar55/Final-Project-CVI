### 🏃 Sports Motion Detection & Viewport Tracking

This project is a computer vision application that processes sports video to detect motion and create a dynamic "virtual camera" that follows the action. The core idea is to automatically pan and zoom a viewport to keep the most significant motion in the center of the frame, similar to a human camera operator.

-----

### 🚀 How It Works

The application processes a video file through a pipeline of computer vision techniques:

1.  **Frame Extraction**: The video is sampled at a reduced frame rate to optimize processing.
2.  **Motion Detection**: Motion is detected by comparing consecutive frames. The areas with the most significant changes are identified as "motion regions."
3.  **Viewport Tracking**: The primary region of interest is determined from the detected motion, and a virtual camera viewport is smoothly tracked across the video to follow this region. A smoothing algorithm is used to prevent jerky movements.
4.  **Visualization**: The final output is a new video with the tracked viewport and motion regions highlighted.

-----

### ⚙️ Getting Started

#### **Prerequisites**

You'll need Python and the necessary libraries to run this project. The primary dependencies are **OpenCV** and **NumPy**. You can install them using pip:

```bash
pip install opencv-python numpy
```

#### **Usage**

To run the application, use the following command-line arguments:

```bash
python main.py --video <path_to_video_file> --output <output_directory>
```

**Optional Arguments:**

  - `--fps`: The target frames per second to process (default is 5).
  - `--viewport_size`: The size of the virtual camera viewport in `WIDTHxHEIGHT` format (default is `720x480`).

**Example:**

```bash
python main.py --video my_sports_clip.mp4 --output results --fps 10 --viewport_size 1080x720
```

-----

### 📂 Project Structure

  - `main.py`: The main script that orchestrates the entire pipeline. It handles argument parsing, calls the other modules, and saves the final output.
  - `frame_processor.py`: Contains functions to handle video file I/O and frame extraction.
  - `motion_detector.py`: Implements the motion detection logic using frame differencing, blurring, and contour detection.
  - `viewport_tracker.py`: Houses the logic for tracking the viewport across the frames, including smoothing the movement to create a cinematic effect.
  - `visualizer.py`: Handles the output, creating videos and images that visualize the motion detection and the tracked viewport.
