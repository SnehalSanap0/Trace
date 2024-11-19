# Real-Time Object Detection with TensorFlow and OpenCV

This repository contains Python scripts for implementing object detection using TensorFlow and OpenCV. The project uses a pre-trained SSD Mobilenet V2 model from TensorFlow Hub for detecting objects in real-time from either video files or webcam streams.

---

## Features

- **Pre-trained Model:** Utilizes TensorFlow Hub's SSD Mobilenet V2 for efficient object detection.
- **Real-Time Processing:** Supports both video files (`real.py`) and live webcam streams (`real2.py`).
- **Visualization:** Draws bounding boxes with class names and confidence scores on detected objects.

---

## Requirements

- Python 3.7 or later
- TensorFlow
- TensorFlow Hub
- OpenCV
- NumPy
- ScreenInfo

---

## Installation

### Step 1: Clone the Repository
Use the following commands to clone this repository and navigate to the project directory:
```bash
git clone https://github.com/SnehalSanap0/Trace.git
cd Trace
```

### Step 2: Install Dependencies
Install the necessary dependencies using the requirements.txt file. 
Run this command:

```bash
pip install tensorflow tensorflow-hub opencv-python numpy screeninfo
```

### Step 3: Run the Scripts
Once the dependencies are installed, you can execute the scripts:

To process video files (real.py):

```bash
python real.py
```

To detect objects using your webcam (real2.py):

```bash
python real2.py
```

### Usage
Running real.py
Place your video file (e.g., classroom.mp4) in the project directory.
Run the script:
```bash
python real.py
```
The output window will display detected objects with bounding boxes.
Running real2.py
Connect your webcam.
Run the script:
```bash
python real2.py
```
Press q to close the detection window.

### Customization
Video Input for real.py: Update the video file path in the script real.py to use a different file:

```python
video = cv2.VideoCapture("your_video.mp4")
```
Resolution for real2.py: Modify the resolution settings in real2.py:

```python
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

### Troubleshooting
TensorFlow Hub Loading Issues:

Ensure your internet connection is active when running the script for the first time.
Webcam Not Working:

Check that the correct webcam index (0 for default camera) is used in real2.py:
```python
video = cv2.VideoCapture(0)
```

### Performance Issues:

Reduce video resolution to improve performance in real2.py.

### Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request to improve the project.
