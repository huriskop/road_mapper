"""Runs the project's camera detector (vision_detector.py) on every photo the bench takes.

Potholes come from the trained YOLOv8 model (models/pothole_1class.onnx), speed bumps from
models/road_anomaly_7class.onnx plus the painted-marking rules, and cracks from the image rules.
Without a model file, vision_detector falls back to its rules for that type.

Detection is slow on a Pi (seconds per photo with the speed-bump model), so it runs in one
background worker: the 100 Hz sensor loop and the photo trigger never wait for it.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from sensors import CaptureStore

ROOT = Path(__file__).resolve().parents[1]          # repository root: vision_detector.py, models/
MAX_QUEUE = 5                                       # photos waiting; more than this are skipped


class Analyzer:
    def __init__(self, store: CaptureStore, detect: tuple[str, ...] = ("pothole", "crack", "speed_bump"),
                 pothole_conf: float = 0.5, threads: int = 3,
                 roi: tuple[float, float, float, float] | None = None) -> None:
        self.store, self.detect, self.conf, self.roi = store, detect, pothole_conf, roi
        self.error: str | None = None
        self.ready = False
        self.models = {"pothole": False, "speed_bump": False}
        self.last_ms: int | None = None
        self.queue: queue.Queue = queue.Queue()
        try:
            import cv2
            import numpy as np
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            import vision_detector
            cv2.setNumThreads(threads)               # leave a core for the sensor loop and the camera
            self.cv2, self.np, self.vd = cv2, np, vision_detector
        except ImportError as exc:
            self.vd = None
            self.error = f"detector not available on this computer ({exc})"
        store.on_add.append(self.submit)
        threading.Thread(target=self._run, daemon=True, name="detector").start()

    def info(self) -> dict:
        return {"available": self.vd is not None, "ready": self.ready, "error": self.error,
                "detect": list(self.detect), "roi": self.roi, "models": self.models, "queued": self.queue.qsize(),
                "last_ms": self.last_ms}

    def submit(self, item: dict, frame: bytes | None, mime: str, name: str | None) -> None:
        if not frame or not name:
            status = {"status": "skipped", "message": "no photo"}
        elif mime != "image/jpeg":
            status = {"status": "skipped", "message": "simulated camera frame, nothing to analyze"}
        elif self.vd is None:
            status = {"status": "skipped", "message": self.error}
        elif self.queue.qsize() >= MAX_QUEUE:
            status = {"status": "skipped", "message": "detector busy, photo skipped"}
        else:
            self.queue.put((item["id"], frame, name))
            status = {"status": "queued"}
        self.store.update(item["id"], analysis=status)

    def _run(self) -> None:
        if self.vd is None:
            return
        t0 = time.perf_counter()
        self.models["pothole"] = self.vd.pothole_model() is not None          # load once, up front
        if "speed_bump" in self.detect:
            self.models["speed_bump"] = self.vd.speed_bump_model() is not None
        self.ready = True
        print(f"detector ready in {time.perf_counter() - t0:.1f} s; models: {self.models}", flush=True)
        while True:
            item_id, frame, name = self.queue.get()
            self.store.update(item_id, analysis={"status": "running"})
            try:
                self.store.update(item_id, **self._analyze(frame, name))
            except Exception as exc:                  # a bad photo must not stop the worker
                self.store.update(item_id, analysis={"status": "error", "message": str(exc)})

    def _analyze(self, frame: bytes, name: str) -> dict:
        cv2, np = self.cv2, self.np
        bgr = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("could not decode the photo")
        t0 = time.perf_counter()
        if self.roi:
            # The detector takes the bottom-centre of its input as the road surface, so a screen's
            # bezel or the desk in that spot hides everything. Analyse only the chosen area.
            h, w = bgr.shape[:2]
            x0, y0, x1, y1 = (int(self.roi[0] * w), int(self.roi[1] * h), int(self.roi[2] * w), int(self.roi[3] * h))
            defects, crop_vis = self.vd.analyze(bgr[y0:y1, x0:x1], detect=self.detect, pothole_conf=self.conf)
            vis = bgr.copy()
            vis[y0:y1, x0:x1] = crop_vis
            cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 1)
        else:
            defects, vis = self.vd.analyze(bgr, detect=self.detect, pothole_conf=self.conf)
        ms = round(1000 * (time.perf_counter() - t0))
        self.last_ms = ms
        out = name.rsplit(".", 1)[0] + "_labelled.jpg"
        cv2.imwrite(str(self.store.dir / out), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return {"annotated": f"/captures/{out}", "analysis": summarize(defects, ms)}


def summarize(defects, ms: int) -> dict:
    """verdict: pothole > speed_bump > crack > none (the most serious thing the camera saw)."""
    kinds = Counter(d.kind for d in defects)
    cracks = sum(v for k, v in kinds.items() if k.endswith("_crack"))
    best = lambda k: max((d.confidence or 0 for d in defects if d.kind == k), default=None)  # noqa: E731
    verdict = ("pothole" if kinds["pothole"] else "speed_bump" if kinds["speed_bump"]
               else "crack" if cracks else "none")
    return {
        "status": "done", "time_ms": ms, "verdict": verdict,
        "potholes": kinds["pothole"], "speed_bumps": kinds["speed_bump"], "cracks": cracks,
        "pothole_confidence": best("pothole"), "speed_bump_confidence": best("speed_bump"),
        "defects": [{"kind": d.kind, "severity": d.severity, "confidence": d.confidence} for d in defects],
    }
