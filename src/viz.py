# src/viz.py
import cv2
import numpy as np
from typing import List, Tuple, Dict
from .zones import Zone, CountLine

# -------- Tracks (boxes + smooth trails) --------
def draw_tracks(frame, tracks: List[Dict]):
    """
    Draw detection boxes + smooth trajectory lines for each track.
    Expects track dicts with keys: id, bbox, confirmed, history.
    """
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        color = (0, 255, 0) if t.get("confirmed", False) else (120, 120, 120)

        # box + id label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID {t['id']}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

        # smooth trajectory (history of centers)
        hist = t.get("history", [])
        if len(hist) > 1:
            for p1, p2 in zip(hist[:-1], hist[1:]):
                cv2.line(frame, p1, p2, color, 2, lineType=cv2.LINE_AA)
            cx, cy = hist[-1]
            cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)

# -------- Zones (polygons) --------
def draw_zones(frame, zones: List[Zone]):
    for z in zones:
        pts = np.array(z.polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        centroid = pts[:, 0, :].mean(axis=0).astype(int)
        cv2.putText(
            frame,
            z.name,
            (int(centroid[0]), int(centroid[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

# -------- Count lines --------
def draw_lines(frame, lines: List[CountLine]):
    for ln in lines:
        cv2.line(frame, ln.p1, ln.p2, (0, 0, 255), 2)
        mid = ((ln.p1[0] + ln.p2[0]) // 2, (ln.p1[1] + ln.p2[1]) // 2)
        cv2.putText(
            frame,
            ln.name,
            mid,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

# -------- Heatmap --------
class Heatmap:
    def __init__(self, width: int, height: int):
        self.map = np.zeros((height, width), dtype=np.float32)
        self.decay = 0.95  # fading per frame

    def add_tracks(self, tracks: List[Dict], value: float = 2.0):
        h, w = self.map.shape
        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            if 0 <= cx < w and 0 <= cy < h:
                self.map[cy, cx] += value
        self.map *= self.decay

    def render(self, frame, alpha: float = 0.5):
        # normalize -> uint8 (CV_8UC1) to satisfy applyColorMap
        if self.map.max() > 0:
            norm = (self.map / self.map.max()) * 255.0
        else:
            norm = np.zeros_like(self.map, dtype=np.float32)
        hm = np.clip(norm, 0, 255).astype(np.uint8)  # CV_8UC1
        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)  # CV_8UC3

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        return cv2.addWeighted(frame, 1 - alpha, hm_color, alpha, 0)

# Optional explicit export (helps static analyzers & autocomplete)
__all__ = ["draw_tracks", "draw_zones", "draw_lines", "Heatmap"]
