# 🎥 DeskVision Pro — Real-Time Computer Vision Analytics

![Streamlit App](https://img.shields.io/badge/Framework-Streamlit-ff4b4b?logo=streamlit)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-blue?logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)

---

## 🧩 Overview

**DeskVision Pro** is a **real-time computer vision application** combining:

- Object detection using **YOLOv8**
- Multi-object tracking with **IoU Tracker**
- Zone and line-based analytics
- **Heatmap visualization** of motion
- **Hand and finger detection** using **MediaPipe**
- An interactive **Streamlit** dashboard

It demonstrates a complete AI → Tracking → Analytics pipeline, perfect for intelligent monitoring, research, or portfolio projects.

---

## 🚀 Features

✅ Real-time **object detection** (YOLOv8)  
✅ Multi-object **tracking with persistent IDs**  
✅ Dynamic **heatmap visualization**  
✅ **Zone and line counting** analytics  
✅ **Hand & finger landmark detection** (MediaPipe)  
✅ **Streamlit interface** for control and display  
✅ Automatic **CSV export** of all tracked objects  

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/DeskVision-Pro.git
cd DeskVision-Pro
```

### 2️⃣ Create a virtual environment
```bash 
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

## 🧠 Usage 
### Run the app
```bash
streamlit run app.py
```

### In the Streamlit interface:
- Choose between Webcam or Video File
- Adjust model confidence threshold
- Enable or disable:
    - 🔲 Heatmap
    - 🔲 Face blurring
    - 🔲 Hand / finger tracking

## 📊 Data Logging
Tracking data is automatically saved to:
```bash
data/outputs/tracks.csv
```

## Tech Stack
| Category     | Technology                                                         |
| ------------ | ------------------------------------------------------------------ |
| Detection    | [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) |
| Vision       | [OpenCV](https://opencv.org/)                                      |
| Tracking     | Custom IoU-based tracker                                           |
| Hands & Pose | [MediaPipe Hands](https://developers.google.com/mediapipe)         |
| UI           | [Streamlit](https://streamlit.io/)                                 |
| Analytics    | Zone / Line Counting, Heatmap                                      |
| Export       | CSV logging with Polars                                            |


## 💡 Future Improvements
- 🧍‍♂️ Combine YOLO + MediaPipe for per-hand tracking
- ✋ Count raised fingers
- 📈 Add interactive dashboard (Plotly / Polars)
- 😄 Integrate facial emotion detection

## 🧾 License
This project is released under the MIT License — free for personal, educational, and commercial use.