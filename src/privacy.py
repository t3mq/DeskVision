import cv2
from typing import Tuple

BBox = Tuple[int, int, int, int]

def _clip_bbox(bbox: BBox, w: int, h: int) -> BBox:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def blur_bbox(frame, bbox: BBox, ksize: int = 25):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(bbox, w, h)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame
    blurred = cv2.GaussianBlur(roi, (ksize|1, ksize|1), 0)
    frame[y1:y2, x1:x2] = blurred
    return frame

def mosaic_bbox(frame, bbox: BBox, tiles: int = 20):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(bbox, w, h)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame
    sh, sw = roi.shape[:2]
    ph = max(1, sh // tiles)
    pw = max(1, sw // tiles)
    small = cv2.resize(roi, (pw, ph), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (sw, sh), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = mosaic
    return frame

def blur_polygon(frame, polygon):
    import numpy as np
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    blurred = cv2.GaussianBlur(frame, (25, 25), 0)
    frame[mask == 255] = blurred[mask == 255]
    return frame