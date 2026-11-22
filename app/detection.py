"""Runs the camera detector (vision_detector.py) on photos, one at a time.

Potholes: trained YOLOv8 model (models/pothole_1class.onnx). Speed bumps:
models/road_anomaly_7class.onnx plus the painted-marking rules. Cracks: image rules.
Without a model file, vision_detector falls back to its rules for that type.

A photo takes seconds on a Pi, so captures are analyzed in one background worker: the
100 Hz sensor loop and the photo trigger never wait for it.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import Counter

from .capture import CaptureStore

TYPES = ("pothole", "crack", "speed_bump")
MAX_QUEUE = 5                                       # photos waiting; more than this are skipped
Roi = tuple[float, float, float, float]


class Analyzer:
    """on_result(item) is called once for every capture, when its analysis has ended in any way."""

    def __init__(self, store: CaptureStore, detect: tuple[str, ...] = TYPES, pothole_conf: float = 0.5,
                 threads: int = 3, roi: Roi | None = None, on_result=None) -> None:
        self.store, self.detect, self.conf, self.roi, self.on_result = store, detect, pothole_conf, roi, on_result
        self.error: str | None = None
        self.ready = False
        self.models = {"pothole": False, "speed_bump": False}
        self.last_ms: int | None = None
        self.queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()                # the OpenCV networks are not thread-safe
        self.vd = None
        try:
            import cv2
            import numpy as np

            import vision_detector
            cv2.setNumThreads(threads)               # leave a core for the sensor loop and the camera
            self.cv2, self.np, self.vd = cv2, np, vision_detector
        except ImportError as exc:
            self.error = f"detector not available ({exc})"
        store.on_add.append(self.submit)
        if self.vd:
            threading.Thread(target=self._run, daemon=True, name="detector").start()

    def info(self) -> dict:
        return {"available": self.vd is not None, "ready": self.ready, "error": self.error,
                "detect": list(self.detect), "roi": self.roi, "models": self.models, "queued": self.queue.qsize(),
                "last_ms": self.last_ms}

    def submit(self, item: dict, frame: bytes | None, name: str | None) -> None:
        if not frame or not name:
            self._finish(item["id"], analysis={"status": "skipped", "message": "no photo: the camera sent no frame"})
        elif self.vd is None:
            self._finish(item["id"], analysis={"status": "skipped", "message": self.error})
        elif self.queue.qsize() >= MAX_QUEUE:
            self._finish(item["id"], analysis={"status": "skipped", "message": "detector busy, photo skipped"})
        else:
            self.store.update(item["id"], analysis={"status": "queued"})
            self.queue.put((item["id"], frame, name))

    def _finish(self, item_id: int, **fields) -> None:
        self.store.update(item_id, **fields)
        item = self.store.get(item_id)
        if item and self.on_result:
            self.on_result(item)

    def _run(self) -> None:
        t0 = time.perf_counter()
        with self._lock:                             # load the models once, up front
            self.models["pothole"] = self.vd.pothole_model() is not None
            if "speed_bump" in self.detect:
                self.models["speed_bump"] = self.vd.speed_bump_model() is not None
        self.ready = True
        print(f"detector ready in {time.perf_counter() - t0:.1f} s; models: {self.models}", flush=True)
        while True:
            item_id, frame, name = self.queue.get()
            self.store.update(item_id, analysis={"status": "running"})
            try:
                labelled, result = self.analyze(frame, self.detect, self.conf, self.roi)
                out = name.rsplit(".", 1)[0] + "_labelled.jpg"
                (self.store.dir / out).write_bytes(labelled)
                self.last_ms = result["time_ms"]
                fields = {"annotated": f"/captures/{out}", "analysis": result}
            except Exception as exc:                  # a bad photo must not stop the worker
                fields = {"analysis": {"status": "error", "message": str(exc)}}
            try:
                self._finish(item_id, **fields)
            except Exception as exc:
                print(f"could not record capture {item_id}: {exc}", flush=True)

    def analyze(self, photo: bytes, detect: tuple[str, ...], conf: float, roi: Roi | None = None) -> tuple[bytes, dict]:
        """(labelled JPEG, result) for one photo."""
        cv2, np = self.cv2, self.np
        bgr = cv2.imdecode(np.frombuffer(photo, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("could not read the photo (use JPEG, PNG or WebP)")
        h, w = bgr.shape[:2]
        x0, y0, x1, y1 = (int(f * n) for f, n in zip(roi or (0, 0, 1, 1), (w, h, w, h)))
        with self._lock:
            t0 = time.perf_counter()
            # The detector takes the bottom-centre of its input as the road surface. With --roi only
            # the chosen area is analyzed (a picture on a screen, without its bezel and the desk).
            defects, crop = self.vd.analyze(bgr[y0:y1, x0:x1], detect=detect, pothole_conf=conf)
            ms = round(1000 * (time.perf_counter() - t0))
        vis = bgr.copy()
        vis[y0:y1, x0:x1] = crop
        if roi:
            cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 1)
        jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        return jpg, {**summarize(defects), "time_ms": ms, "width": w, "height": h}


def summarize(defects) -> dict:
    """verdict: pothole > speed_bump > crack > none (the most serious thing the camera saw)."""
    kinds = Counter(d.kind for d in defects)
    cracks = sum(v for k, v in kinds.items() if k.endswith("_crack"))

    def best(kind: str):
        return max((d.confidence or 0 for d in defects if d.kind == kind), default=None)

    verdict = ("pothole" if kinds["pothole"] else "speed_bump" if kinds["speed_bump"]
               else "crack" if cracks else "none")
    return {
        "status": "done", "verdict": verdict,
        "potholes": kinds["pothole"], "speed_bumps": kinds["speed_bump"], "cracks": cracks,
        "pothole_confidence": best("pothole"), "speed_bump_confidence": best("speed_bump"),
        "defects": [{"kind": d.kind, "severity": d.severity, "confidence": d.confidence} for d in defects],
    }
