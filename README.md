# Real-Time Object Detection with TensorFlow and OpenCV

This repository contains Python scripts for implementing object detection using TensorFlow and OpenCV. The project uses a pre-trained SSD Mobilenet V2 model from TensorFlow Hub for detecting objects in real-time from either image or video files or webcam streams.

## Features

- **Pre-trained Model:** Utilizes TensorFlow Hub's SSD Mobilenet V2 for efficient object detection.
- **Real-Time Processing:** Supports image(`real3.py`) as well as video files (`real.py`) and live webcam streams (`real2.py`).
- **Visualization:** Draws bounding boxes with class names and confidence scores on detected objects.

## Requirements

- Python 3.7 or later
- TensorFlow
- TensorFlow Hub
- OpenCV
- NumPy
- ScreenInfo

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

To process image files (real3.py):

```bash
python real3.py
```

## Usage

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

Running real3.py
Place your image file (e.g., image3.jpg) in the project directory.
Update the image file path in the script:
```python
image_path = "image3.jpg"
```
Run the script:
```bash
python real3.py
```
The script will display the image with detected objects highlighted. Press any key to close the window.

## Customization

Video Input for real.py: Update the video file path in the script real.py to use a different file:

```python
video = cv2.VideoCapture("your_video.mp4")
```

Resolution for real2.py: Modify the resolution settings in real2.py:

```python
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

Image Path: Update the image_path variable to specify the file you want to process:

```python
image_path = "your_image.jpg"
```

Detection Threshold: Change the score_threshold in the draw function to filter detections by confidence level:

```python
score_threshold=0.1  # Increase to show fewer, higher-confidence detections
```

## Troubleshooting

TensorFlow Hub Loading Issues:

Ensure your internet connection is active when running the script for the first time.

Webcam Not Working:

Check that the correct webcam index (0 for default camera) is used in real2.py:
```python
video = cv2.VideoCapture(0)
```

## Performance Issues

Reduce video resolution to improve performance in real2.py.
Use smaller input images to speed up processing.
Adjust the detection threshold to reduce the number of displayed objects.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request to improve the project.
