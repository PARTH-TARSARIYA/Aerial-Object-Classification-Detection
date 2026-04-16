# 📌 Aerial Object Detection (Bird vs Drone)
### 🚀 Overview

This project focuses on real-time aerial object detection using YOLOv8, capable of distinguishing between birds and drones from images, videos, and live camera feeds.

The system is designed for safety-critical and environmental applications, where accurate differentiation between natural and artificial flying objects is essential.

### 🎯 Objectives

Detect and classify Birds and Drones

Enable real-time monitoring

Reduce false positives in aerial detection systems

Provide a deployable Streamlit-based interface

### 📊 Dataset

The dataset is sourced from Roboflow:

Project: drones-and-birds

Classes: 2 (Bird, Drone)

Format: YOLOv8 compatible

License: CC BY 4.0

```
Dataset Link:
https://universe.roboflow.com/new-workspace-x00wt/drones-and-birds-0muie/dataset/1
```

### 🧠 Model Architecture
Model: YOLOv8n (Nano)

Framework: Ultralytics YOLO

Task: Object Detection

Classes: 2

# ⚙️ Training Details

model training : https://www.kaggle.com/code/partar/aerial-object-classification-and-detection?scriptVersionId=312077539

Input size: 480

Epochs: 100

Optimizer: Default (SGD/Adam based on YOLO config)

📈 Sample Results

mAP@50: ~0.80

mAP@50-95: ~0.50

Precision: ~0.80

Recall: ~0.75

### 🛠️ Project Pipeline
```
Dataset (Roboflow)
        ↓
Data Preparation (YOLO format)
        ↓
Model Training (YOLOv8)
        ↓
Evaluation (mAP, Precision, Recall)
        ↓
Model Export (best.pt)
        ↓
Deployment (Streamlit App)
```

### 💻 Features

✅ Image Detection

✅ Video Detection

✅ Live Camera Detection

✅ Real-time bounding box visualization

✅ External OpenCV window support

✅ Lightweight model (fast inference)

### 🖥️ Deployment

The application is built using Streamlit.

Run Locally:
pip install -r requirements.txt
streamlit run app.py
📂 Project Structure
├── model/
│   └── best.pt
├── app.py
├── requirements.txt
└── README.md

## 🎯 Use Cases
### 🐦 Wildlife Protection

Detect birds near wind farms or airports to prevent collisions.

### 🛡️ Security & Defense Surveillance :

Identify unauthorized drones in restricted airspace.

### ✈️ Airport Bird-Strike Prevention :

Monitor runway zones to reduce bird-strike risks.

### 🌍 Environmental Research :

Track bird populations using aerial imagery without misclassification.

### ⚠️ Challenges & Limitations :

Class imbalance may affect recall (bird vs drone)

Small object detection remains challenging

Performance depends heavily on dataset quality

Real-time processing speed varies with hardware

### 🔮 Future Improvements :

Improve dataset diversity and balance

Use larger models (YOLOv8m/l) for higher accuracy

Add tracking (e.g., DeepSORT)

Deploy on edge devices (Jetson, Raspberry Pi)

Integrate alert/notification system

### 📜 License

Dataset: CC BY 4.0 (via Roboflow)

Project: Open-source for educational/research purposes

### 🙌 Acknowledgements :

Roboflow for dataset hosting

Ultralytics for YOLOv8

OpenCV & Streamlit communities

### 🎯 Final Note

This project demonstrates an end-to-end computer vision pipeline:

Data → Model → Evaluation → Deployment → Real-world application
