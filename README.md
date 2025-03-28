# Computer_vision_Tracking_-_Detection
# People Counter using YOLOv8 Object Detection

A computer vision system that counts people crossing a defined line in video streams using YOLOv8 object detection and SORT tracking algorithm.

## Features
- Real-time people detection with YOLOv8-nano model
- Mask-based ROI (Region of Interest) filtering
- SORT (Simple Online and Realtime Tracking) implementation
- People counting with crossing line logic
- Visual tracking IDs and bounding boxes
- Confidence threshold filtering (0.3)

## Requirements
- Python 3.8+
- Ultralytics YOLOv8
- OpenCV (cv2)
- cvzone
- numpy
- filterpy (for SORT implementation)

## Installation
```bash
pip install ultralytics opencv-python cvzone numpy filterpy
