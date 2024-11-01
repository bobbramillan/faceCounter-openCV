# Real-Time Face Counter

This project is a real-time face detection web application built with Flask, OpenCV, and JavaScript. The app captures video from the user's webcam, detects faces in each frame, and overlays a bounding box and face count directly on the video feed. 

## Features
- **Real-Time Face Detection**: Detects and counts faces in each frame of the video feed.
- **Bounding Boxes**: Draws green bounding boxes around detected faces.
- **Face Counter Overlay**: Displays the number of faces detected in real-time.

## Technologies Used
- **Flask**: Backend web framework for routing and video streaming.
- **OpenCV**: Library used for face detection with Haar Cascade Classifiers.
- **JavaScript & HTML**: Frontend for accessing the webcam and displaying the video feed with overlays.

## Setup (Click on http://127.0.0.1:5000)

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/bobbramillan/faceCounter.git
   pip install flask opencv-python

   cd faceCounter
   python app.py
