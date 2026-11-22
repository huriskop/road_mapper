"""Photo log and the jolt trigger that takes the photos."""
from __future__ import annotations

import collections
import threading
from datetime import datetime
from pathlib import Path

from .sensors import Camera, Gps, Imu, now_iso


class CaptureStore:
    """Photos taken by the trigger (or by hand), kept on disk plus a short in-memory log.

    Every add or update bumps `seq`, so the live stream can send changed entries
    (for example when the detector finishes) as well as new ones."""

    def __init__(self, directory: Path, keep: int = 200) -> None:
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.items: collections.deque[dict] = collections.deque(maxlen=keep)
        self.last_id = 0
        self.seq = 0
        self.on_add: list = []          # called as f(item, frame, file_name) after a photo is saved

    def add(self, frame: bytes | None, reason: str, peak_g: float | None, gps: dict, when: str) -> dict:
        name = None
        if frame:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
            name = f"{stamp}_{reason}.jpg"
            (self.dir / name).write_bytes(frame)
        with self.lock:
            self.last_id += 1
            item = {
                "id": self.last_id, "time": when, "reason": reason,
                "guess": {"down": "pothole?", "up": "bump?"}.get(reason),
                "peak_g": round(peak_g, 2) if peak_g is not None else None,
                "image": f"/captures/{name}" if name else None,
                "lat": gps.get("lat"), "lng": gps.get("lng"), "gps_fix": gps.get("fix"),
                "speed_kmh": gps.get("speed_kmh"),
                "annotated": None, "analysis": None,
            }
            self.seq += 1
            item["seq"] = self.seq
            self.items.append(item)
        for f in self.on_add:
            f(item, frame, name)
        return dict(item)

    def update(self, item_id: int, **fields) -> None:
        with self.lock:
            for item in self.items:
                if item["id"] == item_id:
                    item.update(fields)
                    self.seq += 1
                    item["seq"] = self.seq
                    return

    def since(self, last_seq: int) -> list[dict]:
        """New or changed entries, oldest first."""
        with self.lock:
            return sorted((dict(i) for i in self.items if i["seq"] > last_seq), key=lambda i: i["seq"])

    def get(self, item_id: int) -> dict | None:
        with self.lock:
            return next((dict(i) for i in self.items if i["id"] == item_id), None)

    def all(self) -> list[dict]:
        with self.lock:
            return [dict(i) for i in self.items]


class Trigger:
    """Takes a photo when the vertical acceleration passes the threshold.

    The photo is grabbed at the first sample over the threshold (the moment of the jolt);
    the peak is then measured over the next 0.3 s, and nothing fires during the refractory
    time, so the front and rear wheels on the same hole give one photo."""

    WINDOW_S = 0.3

    def __init__(self, imu: Imu, camera: Camera, gps: Gps, store: CaptureStore,
                 threshold_g: float = 0.4, refractory_s: float = 1.5) -> None:
        self.imu, self.camera, self.gps, self.store = imu, camera, gps, store
        self.threshold, self.refractory, self.enabled = threshold_g, refractory_s, True
        self._active = False
        self._quiet_until = 0.0
        self._pending: dict = {}
        imu.listeners.append(self.feed)

    def info(self) -> dict:
        return {"enabled": self.enabled, "threshold_g": self.threshold, "refractory_s": self.refractory}

    def feed(self, t: float, dyn: float) -> None:
        if self._active:
            p = self._pending
            if abs(dyn) > abs(p["peak"]):
                p["peak"] = dyn
            if t - p["t0"] >= self.WINDOW_S:
                self._active = False
                self._quiet_until = t + self.refractory
                threading.Thread(target=self.store.add, daemon=True, args=(
                    p["frame"], p["reason"], p["peak"], p["gps"], p["when"])).start()
            return
        if not self.enabled or self.imu.calibrating or t < self._quiet_until or abs(dyn) < self.threshold:
            return
        self._pending = {"t0": t, "peak": dyn, "reason": "up" if dyn > 0 else "down", "frame": self.camera.latest(),
                         "gps": self.gps.snapshot(), "when": now_iso()}
        self._active = True

    def manual(self) -> dict:
        return self.store.add(self.camera.latest(), "manual", None, self.gps.snapshot(), now_iso())
