import streamlit as st
import cv2
from ultralytics import YOLO
from src.tracker import IouTracker
from src.zones import Zone, CountLine
from src.analytics import ZoneCounter, LineCounter, CsvLogger
from src.privacy import blur_bbox
from src.viz import draw_tracks, draw_zones, draw_lines, Heatmap
import mediapipe as mp

st.set_page_config(page_title="DeskVision Pro", layout="wide")
st.title("🎥 DeskVision – Detection, tracking and real-time analytics")

conf = st.sidebar.slider("Model confidence", 0.2, 0.9, 0.5, 0.05)
max_age = st.sidebar.number_input("Max age (tracker)", 5, 100, 30)
blur_faces = st.sidebar.checkbox("Blur faces", False)
show_heatmap = st.sidebar.checkbox("Show heatmap", True)
show_hands = st.sidebar.checkbox("Show hands / fingers", True)

source = st.selectbox("Video source", ["Webcam", "File"])
video_file = None
if source == "File":
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

model = YOLO("yolov8n.pt")
tracker = IouTracker(max_age=max_age)

zones = [Zone([(50, 50), (600, 50), (600, 400), (50, 400)], "Zone A")]
lines = [CountLine((100, 200), (500, 200), "Line 1")]
zc = ZoneCounter(zones)
lc = LineCounter(lines)
logger = CsvLogger("data/outputs/tracks.csv")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

if source == "Webcam":
    cap = cv2.VideoCapture(0)
elif video_file:
    tmp_path = f"/tmp/{video_file.name}"
    with open(tmp_path, "wb") as f:
        f.write(video_file.read())
    cap = cv2.VideoCapture(tmp_path)
else:
    st.warning("Please choose a video source.")
    st.stop()

frame_placeholder = st.empty()
metrics_zone = st.empty()
metrics_line = st.empty()

frame_idx = 0
heatmap = Heatmap(int(cap.get(3)), int(cap.get(4)))

while True:
    ok, frame = cap.read()
    if not ok:
        break

    res = model.predict(source=frame, conf=conf, verbose=False)
    detections = []
    for r in res:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            score = float(b.conf[0])
            cls = int(b.cls[0])
            detections.append((x1, y1, x2, y2, score, cls))

    tracks = tracker.update(detections)

    zc.update(tracks)
    lc.update(tracks)
    logger.log_tracks(frame_idx, tracks)

    draw_tracks(frame, tracks)
    draw_zones(frame, zones)
    draw_lines(frame, lines)

    if show_hands:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                )

    if blur_faces:
        for t in tracks:
            if t.get("cls") == 0:
                frame = blur_bbox(frame, t["bbox"])

    if show_heatmap:
        heatmap.add_tracks(tracks)
        frame = heatmap.render(frame)

    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    metrics_zone.write(f"Zones : {zc.snapshot()}")
    metrics_line.write(f"Lines : {lc.counts}")
    frame_idx += 1

cap.release()
st.success("✅ Video finished")
