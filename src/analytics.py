from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import csv
from datetime import datetime

from .zones import Zone, CountLine, bbox_center

class ZoneCounter:
    def __init__(self, zones: List[Zone]):
        self.zones = zones
        self._present = {z.name: set() for z in zones}
        self._unique = defaultdict(int)
    
    def update(self, tracks: List[Dict]):
        for z in self.zones:
            zset = self._present[z.name]
            current = set()
            for t in tracks:
                c = bbox_center(t["bbox"])
                if z.contains(c):
                    current.add(t["id"])
                    if t["id"] not in zset:
                        self._unique[z.name] += 1
            self._present[z.name] = current

    def snapshot(self) -> Dict:
        return {
            "present": {k: len(v) for k, v in self._present.items()},
            "unique": dict(self._unique),
        }
    
class LineCounter:

    def __init__(self, lines: List[CountLine]):
        self.lines = lines
        self.counts = {ln.name: {"+": 0, "-": 0} for ln in lines}
        self._seen = defaultdict(dict)

    def update(self, tracks: List[Dict]):
        for t in tracks:
            path = t.get("history", [])
            for ln in self.lines:
                crossed, sign = ln.crossed(path)
                if crossed:
                    last = self._seen[t["id"]].get(ln.name)
                    if last is None or sign != last:
                        if sign >= 0:
                            self.counts[ln.name]["+"] += 1
                        else:
                            self.counts[ln.name]["-"] += 1
                        self._seen[t["id"]][ln.name] = sign

class CsvLogger:
    def __init__(self, csv_path: str, fieldnames: Optional[List[str]] = None):
        self.csv_path = csv_path
        self.fieldnames = fieldnames or [
            "timestamp", "frame", "track_id", "cls", "x1", "y1", "x2", "y2"
        ]
        try:
            with open(self.csv_path, "x", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
        except FileExistsError:
            pass

    def log_tracks(self, frame_idx: int, tracks: List[Dict]):
        ts = datetime.utcnow().isoformat()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            for t in tracks:
                x1, y1, x2, y2 = t["bbox"]
                writer.writerow({
                    "timestamp": ts,
                    "frame": frame_idx,
                    "track_id": t["id"],
                    "cls": t.get("cls"),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })