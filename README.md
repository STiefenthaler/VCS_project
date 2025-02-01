# Vehicle Detection and Tracking System

This project is developed as part of a university course to demonstrate advanced vehicle detection and tracking techniques under various weather conditions. Using the YOLO object detection model combined with several tracking algorithms, the system accurately tracks vehicle movements in real-time.

## **Key Features**
- **Object Detection:** Utilizes YOLO for fast and accurate vehicle detection.
- **Multi-Object Tracking:** Implements DeepSort for reliable vehicle tracking across frames.
- **Weather Adaptability:** Handles videos captured in foggy, rainy, and sunny conditions.
- **Lane-Specific Analysis:** Assigns vehicles to specific lanes with predefined directions (incoming/outgoing).
- **Custom Polygon Zones:** Considers only vehicles within defined lane polygons for precise analysis.

## **Project Structure**
- **Lane Assignment:** Maps vehicles to incoming or outgoing lanes based on the video.
- **Video Analysis:** Processes different road conditions with lane-specific configurations.
- **Tracking Logic:** Maintains vehicle counts and tracks movements across designated lines.

## **Technologies Used**
- **Python** (OpenCV, NumPy, argparse)
- **Ultralytics YOLO** for object detection
- **DeepSort** for real-time object tracking

## **Purpose**
This project is part of a university coursework designed to apply machine learning techniques to real-world traffic monitoring problems, focusing on environmental adaptability and tracking accuracy.

---

*Developed for academic purposes.*

