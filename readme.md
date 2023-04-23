# Gesture Recognition with Hand Tracking
This repository contains code for recognizing American Sign Language (ASL) gestures using hand tracking and a convolutional neural network (CNN).

## Getting Started
### Prerequisites
+ Python 3.6 or higher
+ OpenCV 2 or higher
+ NumPy
+ TensorFlow
+ scikit-learn
+ mediapipe
+ Installing

1. Clone this repository:

```
git clone https://github.com/UPavankumar/sign-language-interpreter.git
cd sign-language-interpreter
---
2. Install the required packages:

```
pip install -r requirements.txt
```
3. Download the dataset:

```
wget https://storage.googleapis.com/wandb_datasets/nih/dataset.zip
unzip dataset.zip
```
## Running the Code
To run the gesture recognition program, execute the following command:

```
python gesture_recognition.py
```

This will launch the program and begin capturing video from your default camera.

To exit the program, press q.

## Acknowledgments
This project was inspired by the [ ASL Recognition with MediaPipe](https://github.com/linghduo/mediapipe_asl_recognition) project by Linghao Du.

The dataset used in this project was created by the [National Institutes of Health](https://www.nih.gov/).

The ```gesture_recognition.py ``` script was adapted from the [Gesture Recognition with MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands) tutorial on the MediaPipe website.
