from ultralytics import YOLO

class YoloDetector:
    def __init__(self, weights:"yolov8n.pt", conf=0.5, device=None):
        self.model = YOLO(weights)
        self.conf = conf
        self.device = device

    def predict(self, frame):
        res = self.model.predict(source=frame, conf=self.conf, device=self.device, verbose=False)
        boxes = []
        for r in res:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cls = int(b.cls[0])
                score = float(b.conf[0])
                boxes.append({"xyxy": (x1, y1, x2, y2), "class": cls, "score": score})
            return boxes