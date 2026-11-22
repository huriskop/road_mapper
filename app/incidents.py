"""Defects shown on the map, kept in one JSON file on the Pi.

Reports less than 5 m from an unrepaired incident are added to it instead of making a new pin.
Type: a camera pothole or speed bump beats the jolt's direction; camera cracks (image rules, less
reliable) count only when there was no jolt. A change made on the map page beats both."""
from __future__ import annotations

import json
import math
import threading
import uuid
from collections import Counter
from pathlib import Path

from .sensors import now_iso

MERGE_M = 5.0
CHOICES = {
    "defect_type": ("pothole", "bump", "crack", "unclassified"),
    "severity": ("false_positive", "no_repair_needed", "low", "medium", "high", "critical"),
    "repair_status": ("awaiting_repair", "in_progress", "completed"),
}
SENSOR_TYPE = {"down": "pothole", "up": "bump"}
CAMERA_TYPE = {"pothole": "pothole", "speed_bump": "bump"}
SEVERITY_BY_PEAK = ((0.30, "no_repair_needed"), (0.60, "low"), (1.00, "medium"), (1.60, "high"))   # below this peak, g


def severity_for(peak_g: float | None) -> str:
    if peak_g is None:
        return "low"
    return next((s for limit, s in SEVERITY_BY_PEAK if peak_g < limit), "critical")


def distance_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    x = math.radians(lng2 - lng1) * math.cos(math.radians((lat1 + lat2) / 2))
    return 6_371_000 * math.hypot(x, math.radians(lat2 - lat1))


class Incidents:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.rev = 1                                 # goes up with every change; pages poll it
        self.items: list[dict] = json.loads(path.read_text()) if path.exists() else []

    def _save(self) -> None:
        self.rev += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")          # replace in one step: a power cut never leaves half a file
        tmp.write_text(json.dumps(self.items, indent=1))
        tmp.replace(self.path)

    def all(self) -> dict:
        with self.lock:
            return {"rev": self.rev, "items": json.loads(json.dumps(self.items))}

    def add(self, *, lat: float, lng: float, time: str, image: str | None, reason: str,
            peak_g: float | None, analysis: dict | None, location: str) -> dict:
        """Put one report on the map. reason: down | up (jolt) or upload. Returns its incident."""
        done = analysis if analysis and analysis.get("status") == "done" else None
        report = {"time": time, "lat": lat, "lng": lng, "image": image, "reason": reason, "location": location,
                  "peak_g": abs(peak_g) if peak_g is not None else None, "camera": done}
        with self.lock:
            near = [(distance_m(lat, lng, i["lat"], i["lng"]), i) for i in self.items if i["repair_status"] != "completed"]
            dist, inc = min(near, key=lambda p: p[0], default=(None, None))
            if inc is None or dist > MERGE_M:
                inc = {"id": str(uuid.uuid4()), "repair_status": "awaiting_repair", "edited": [], "reports": []}
                self.items.append(inc)
            inc["reports"].append(report)
            self._classify(inc)
            self._save()
            return dict(inc)

    @staticmethod
    def _classify(inc: dict) -> None:
        reports = inc["reports"]
        n = len(reports)
        times = [r["time"] for r in reports]
        peaks = [r["peak_g"] for r in reports if r["peak_g"] is not None]
        inc.update(lat=sum(r["lat"] for r in reports) / n, lng=sum(r["lng"] for r in reports) / n,
                   created_at=min(times), last_report_at=max(times), report_count=n,
                   peak_g=max(peaks, default=None))
        if "severity" not in inc["edited"]:
            inc["severity"] = severity_for(inc["peak_g"])
        if "defect_type" in inc["edited"]:
            return
        seen = Counter(CAMERA_TYPE.get(r["camera"]["verdict"]) for r in reports if r["camera"])
        jolts = Counter(SENSOR_TYPE.get(r["reason"]) for r in reports)
        seen.pop(None, None)
        jolts.pop(None, None)
        if seen:
            inc["defect_type"], inc["type_source"] = seen.most_common(1)[0][0], "camera"
        elif jolts:
            inc["defect_type"], inc["type_source"] = jolts.most_common(1)[0][0], "sensor"
        elif any(r["camera"] and r["camera"]["cracks"] for r in reports):
            inc["defect_type"], inc["type_source"] = "crack", "camera"
        else:
            inc["defect_type"], inc["type_source"] = "unclassified", None

    def edit(self, incident_id: str, changes: dict) -> dict | None:
        for k, v in changes.items():
            if v not in CHOICES.get(k, ()):
                raise ValueError(f"{k} cannot be {v!r}")
        with self.lock:
            inc = next((i for i in self.items if i["id"] == incident_id), None)
            if inc is None:
                return None
            inc.update(changes, updated_at=now_iso())
            inc["edited"] = sorted({*inc["edited"], *changes} - {"repair_status"})
            if "defect_type" in changes:
                inc["type_source"] = "operator"
            self._save()
            return dict(inc)

    def delete(self, incident_id: str) -> bool:
        """Remove it from the map. The photos stay on the Pi."""
        with self.lock:
            left = [i for i in self.items if i["id"] != incident_id]
            if len(left) == len(self.items):
                return False
            self.items = left
            self._save()
            return True
