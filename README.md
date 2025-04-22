# Face Recognition Suite

A modular toolkit for face detection, embedding generation, and facial feature analysis using OpenCV, TensorFlow, and OpenVINO.

## 📁 Structure

```
face-recognition-suite/
│
├── utils/
│   ├── feature_utils.py
│   ├── feature_utils_extra.py
│   ├── regex_feature_match.py
│   ├── simple_feature_check.py
│   └── string_matching.py
│
├── embeddings/
│   ├── face_embedding_with_custom_layer.py
│   └── generate_face_embeddings.py
│
├── detection/
│   ├── realtime_face_detection_openvino.py
│   ├── realtime_face_detection_output.py
│   ├── video_face_detection_openvino.py
│   └── face_detection_fps_display.py
│
└── controller.py
```

## 🚀 Features

- Real-time face detection (OpenVINO & OpenCV)
- Face embedding generation with customizable layers
- Facial feature utilities and matching
- Modular script design for ease of integration

## 🛠️ Requirements

Install dependencies using:

```
pip install -r requirements.txt
```

