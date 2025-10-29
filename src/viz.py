import cv2
import numpy as np
from typing import List, Tuple, Dict
from .zones import Zone, CountLine

def draw_tracks(frame, tracks: List[Dict]):
    """Draw bounding boxes and trajectories for tracked objects."""
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        track_id = t["id"]
        color = (0, 255, 0) if t["confirmed"] else (100, 100, 100)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        history = t.get("history", [])
        if len(history) > 1:
            for i in range(1, len(history)):
                p1 = history[i - 1]
                p2 = history[i]
                cv2.line(frame, p1, p2, color, 2, lineType=cv2.LINE_AA)
            cx, cy = history[-1]
            cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)


def draw_zones(frame, zones: List[Zone]):
    """Draw defined polygonal zones on the scene."""
    for z in zones:
        pts = np.array(z.polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        x, y = np.mean(pts[:, 0, 0]), np.mean(pts[:, 0, 1])
        cv2.putText(frame, z.name, (int(x) - 20, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


def draw_lines(frame, lines: List[CountLine]):
    """Draw counting lines."""
    for ln in lines:
        cv2.line(frame, ln.p1, ln.p2, (0, 0, 255), 2)
        mx = (ln.p1[0] + ln.p2[0]) // 2
        my = (ln.p1[1] + ln.p2[1]) // 2
        cv2.putText(frame, ln.name, (mx + 10, my),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


class Heatmap:
    def __init__(self, width: int, height: int):
        self.map = np.zeros((height, width), dtype=np.float32)
        self.decay = 0.95

    def add_tracks(self, tracks: List[Dict]):
        """Add current track positions to the heatmap."""
        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            if 0 <= cx < self.map.shape[1] and 0 <= cy < self.map.shape[0]:
                self.map[cy, cx] += 2.0
        self.map *= self.decay

    def render(self, frame):
        """Overlay the heatmap onto the frame."""
        hm = np.clip(self.map / self.map.max() * 255, 0, 255).astype(np.uint8) if self.map.max() > 0 else self.map
        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 0.7, hm_color, 0.5, 0)
        return blended
