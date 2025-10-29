from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import math

BBox = Tuple[int, int, int, int]
Detection = Tuple[int, int, int, int, float, Optional[int]]


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


@dataclass
class Track:
    id: int
    bbox: BBox
    cls: Optional[int] = None
    score: float = 1.0
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    history: List[Tuple[int, int]] = field(default_factory=list)

    def update(self, bbox: BBox, score: float):
        self.bbox = bbox
        self.score = score
        self.hits += 1
        self.time_since_update = 0
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        self.history.append((cx, cy))

    def predict(self):
        self.age += 1
        self.time_since_update += 1
        return self.bbox


class IouTracker:
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_thresh: float = 0.3,
        class_agnostic: bool = True,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thresh = iou_thresh
        self.class_agnostic = class_agnostic
        self._next_id = 1
        self.tracks: List[Track] = []

    def _assign(self, detections: List[Detection]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not self.tracks or not detections:
            return [], list(range(len(self.tracks))), list(range(len(detections)))

        cost = []
        for t in self.tracks:
            row = []
            for d in detections:
                dbbox = (d[0], d[1], d[2], d[3])
                if (not self.class_agnostic) and (t.cls is not None) and (t.cls != d[5]):
                    row.append(1.0)
                else:
                    row.append(1 - iou(t.bbox, dbbox))
            cost.append(row)

        pairs = []
        used_t, used_d = set(), set()
        candidates = [(c, ti, di) for ti, row in enumerate(cost) for di, c in enumerate(row)]
        candidates.sort(key=lambda x: x[0])

        for c, ti, di in candidates:
            if ti in used_t or di in used_d:
                continue
            iou_val = 1 - c
            if iou_val < self.iou_thresh:
                continue
            pairs.append((ti, di))
            used_t.add(ti)
            used_d.add(di)

        unmatched_t = [i for i in range(len(self.tracks)) if i not in used_t]
        unmatched_d = [i for i in range(len(detections)) if i not in used_d]
        return pairs, unmatched_t, unmatched_d

    def update(self, detections: List[Detection]) -> List[Dict]:
        for t in self.tracks:
            t.predict()

        matches, unmatched_t, unmatched_d = self._assign(detections)

        for ti, di in matches:
            d = detections[di]
            bbox = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
            score = float(d[4])
            self.tracks[ti].update(bbox, score)
            if d[5] is not None:
                self.tracks[ti].cls = int(d[5])

        for di in unmatched_d:
            d = detections[di]
            bbox = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
            score = float(d[4])
            cls = int(d[5]) if d[5] is not None else None
            tr = Track(id=self._next_id, bbox=bbox, score=score, cls=cls)
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            tr.history.append((cx, cy))
            self.tracks.append(tr)
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        outputs = []
        for t in self.tracks:
            confirmed = t.hits >= self.min_hits
            outputs.append({
                "id": t.id,
                "bbox": t.bbox,
                "cls": t.cls,
                "score": t.score,
                "age": t.age,
                "hits": t.hits,
                "time_since_update": t.time_since_update,
                "confirmed": confirmed,
                "history": t.history[-20:],
            })

        return outputs
