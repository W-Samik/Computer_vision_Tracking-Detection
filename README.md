
# People Counter using YOLOv8 Object Detection

<div align="center">
  <img src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-yolov8.png" alt="YOLOv8 Logo" width="800">
</div>

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
