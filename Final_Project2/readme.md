This project is a computer vision-based self-driving car application. It uses a Convolutional Neural Network (CNN) to predict steering angles based on images captured from the car's perspective. The project includes a training script to create the model and a simulation script to apply the trained model in a virtual environment.

-----

### 🤖 Project Overview

This project implements a self-driving car using a deep learning approach. The core of the system is a **Convolutional Neural Network (CNN)** that is trained to predict the correct steering angle for a vehicle based on a single front-facing camera image.

  - `main.py`: This is the training script. It loads a dataset, balances the data, preprocesses images, and trains a CNN model. The trained model is then saved as `model.h5`.
  - `utils.py`: Contains utility functions for data loading, data balancing, image augmentation, preprocessing, and the definition of the CNN model architecture.
  - `TestSimulation.py`: This script runs the simulation. It connects to a simulator via `socketio`, receives camera images and vehicle speed, uses the pre-trained `model.h5` to predict the steering angle, and sends control commands (steering and throttle) back to the simulator.
  - `environment.yml`: Specifies the project's dependencies and their versions, ensuring a reproducible environment.

-----

### 🚀 Getting Started

#### **Prerequisites**

To run this project, you need to set up the Python environment using `conda`. The `environment.yml` file lists all necessary packages.

#### **Installation**

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-project.git
    cd your-project
    ```
2.  Create and activate the `conda` environment:
    ```bash
    conda env create -f environment.yml
    conda activate self-driving-car
    ```

-----

### ⚙️ How to Use

#### **Training the Model**

To train the model, you need a dataset of camera images and corresponding steering angles (e.g., from a simulator). After placing the dataset in the correct directory, run the `main.py` script:

```bash
python main.py
```

This will generate a `model.h5` file, which is the trained neural network.

#### **Running the Simulation**

To run the simulation, first, ensure you have a compatible simulator running and connected. Then execute `TestSimulation.py`:

```bash
python TestSimulation.py
```

This script will connect to the simulator and use the `model.h5` file to autonomously drive the car.

-----

### 📺 Demos

The following videos demonstrate the application in action.


### [Driving Test 1](https://youtu.be/72kGig6Tx5U)

[![Watch the video](https://img.youtube.com/vi/72kGig6Tx5U/maxresdefault.jpg)](https://youtu.be/72kGig6Tx5U)

### [Driving Test 2](https://youtu.be/GVDVLXQCeTQ)

[![Watch the video](https://img.youtube.com/vi/GVDVLXQCeTQ/maxresdefault.jpg)](https://youtu.be/GVDVLXQCeTQ)

