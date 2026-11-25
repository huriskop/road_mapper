"""Runs the camera detector (vision_detector.py) on photos, one at a time.

One trained YOLOv8n model (models/road_damage_5class_v8n.onnx) finds potholes, speed bumps and cracks.
Without it, vision_detector falls back to the older models and image rules.

A photo takes seconds on a Pi, so captures are analyzed in one background worker: the
100 Hz sensor loop and the photo trigger never wait for it.

A jolt comes with the 6 s of video before it (the camera has already passed the hole when the wheel
hits it). One frame every SCAN_STEP_S is checked (13 frames, about 6 s on the Pi 4); the frame that
shows the defect best becomes the capture's photo, and the clip is saved as a video.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import Counter

from .capture import CaptureStore

TYPES = ("pothole", "crack", "speed_bump")
MAX_QUEUE = 5                                       # photos waiting; more than this are skipped
SCAN_STEP_S = 0.5                                   # look-back: check one frame every this many seconds
RANK = {"pothole": 3, "speed_bump": 3, "crack": 1, "none": 0}
EXPECTED = {"down": "pothole", "up": "speed_bump"}  # what the jolt's direction suggests
Roi = tuple[float, float, float, float]


class Analyzer:
    """on_result(item) is called once for every capture, when its analysis has ended in any way."""

    def __init__(self, store: CaptureStore, detect: tuple[str, ...] = TYPES, pothole_conf: float | None = None,
                 threads: int = 3, roi: Roi | None = None, on_result=None) -> None:
        self.store, self.detect, self.conf, self.roi, self.on_result = store, detect, pothole_conf, roi, on_result
        self.error: str | None = None
        self.ready = False
        self.models = {"pothole": False, "crack": False, "speed_bump": False}
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

    def submit(self, item: dict, frame: bytes | None, name: str | None, clip: list | None = None) -> None:
        if not frame or not name:
            self._finish(item["id"], analysis={"status": "skipped", "message": "no photo: the camera sent no frame"})
        elif self.vd is None:
            self._finish(item["id"], analysis={"status": "skipped", "message": self.error})
        elif self.queue.qsize() >= MAX_QUEUE:
            self._finish(item["id"], analysis={"status": "skipped", "message": "detector busy, photo skipped"})
        else:
            self.store.update(item["id"], analysis={"status": "queued"})
            self.queue.put((item["id"], frame, name, clip))

    def _finish(self, item_id: int, **fields) -> None:
        self.store.update(item_id, **fields)
        item = self.store.get(item_id)
        if item and self.on_result:
            self.on_result(item)

    def _run(self) -> None:
        t0 = time.perf_counter()
        with self._lock:                             # load the models once, up front
            if self.vd.road_model() is not None:
                self.models = dict.fromkeys(self.models, True)
            else:
                self.models["pothole"] = self.vd.pothole_model() is not None
                if "speed_bump" in self.detect:
                    self.models["speed_bump"] = self.vd.speed_bump_model() is not None
        self.ready = True
        print(f"detector ready in {time.perf_counter() - t0:.1f} s; models: {self.models}", flush=True)
        while True:
            item_id, frame, name, clip = self.queue.get()
            self.store.update(item_id, analysis={"status": "running"})
            stem = name.rsplit(".", 1)[0]
            try:
                fields = {}
                if clip:
                    try:
                        self.save_video(clip, self.store.dir / f"{stem}.avi")
                        fields["video"] = f"/captures/{stem}.avi"
                    except Exception as exc:          # no video, but still look for the defect
                        print(f"could not save the video of capture {item_id}: {exc}", flush=True)
                    best, labelled, result = self.scan(clip, EXPECTED.get((self.store.get(item_id) or {}).get("reason")))
                    (self.store.dir / f"{stem}_best.jpg").write_bytes(best)
                    fields["image"] = f"/captures/{stem}_best.jpg"
                else:
                    labelled, result = self.analyze(frame, self.detect, self.conf, self.roi)
                out = stem + "_labelled.jpg"
                (self.store.dir / out).write_bytes(labelled)
                self.last_ms = result["time_ms"]
                fields.update(annotated=f"/captures/{out}", analysis=result)
            except Exception as exc:                  # a bad photo must not stop the worker
                fields = {"analysis": {"status": "error", "message": str(exc)}}
            try:
                self._finish(item_id, **fields)
            except Exception as exc:
                print(f"could not record capture {item_id}: {exc}", flush=True)

    def scan(self, clip: list[tuple[float, bytes]], expected: str | None) -> tuple[bytes, bytes, dict]:
        """Check one frame every SCAN_STEP_S of the clip and keep the one that shows the defect best:
        the type the jolt suggests first, then any pothole or bump, then cracks; the most confident wins.
        Returns (that frame, its labelled copy, its result with seconds_before and frames_checked)."""
        picks, last = [], None
        for ago, jpg in clip:                         # oldest first
            if last is None or last - ago >= SCAN_STEP_S - 0.01:     # 0.01: float rounding
                picks.append((ago, jpg))
                last = ago
        if picks[-1][1] is not clip[-1][1]:
            picks.append(clip[-1])                    # always include the moment of the jolt
        best, total_ms = None, 0
        for ago, jpg in picks:
            try:
                labelled, result = self.analyze(jpg, self.detect, self.conf, self.roi)
            except ValueError:                        # a broken frame: skip it
                continue
            total_ms += result["time_ms"]
            v = result["verdict"]
            conf = result.get(f"{v}_confidence") or max((d["confidence"] or 0 for d in result["defects"]), default=0)
            score = (v == expected and v != "none", RANK[v], conf, -ago)   # ties: the frame closest to the jolt
            if best is None or score > best[0]:
                best = (score, ago, jpg, labelled, result)
        if best is None:
            raise ValueError("no readable frame in the video")
        _, ago, jpg, labelled, result = best
        return jpg, labelled, {**result, "time_ms": total_ms, "seconds_before": round(ago, 1),
                               "frames_checked": len(picks)}

    def save_video(self, clip: list[tuple[float, bytes]], path) -> None:
        """The clip as an MJPEG .avi, at the rate it was filmed."""
        cv2, np = self.cv2, self.np
        fps = (len(clip) - 1) / max(clip[0][0] - clip[-1][0], 1e-3) if len(clip) > 1 else 1.0
        writer = None
        try:
            for _, jpg in clip:
                bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                if writer is None:
                    h, w = bgr.shape[:2]
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
                writer.write(bgr)
        finally:
            if writer is not None:
                writer.release()

    def analyze(self, photo: bytes, detect: tuple[str, ...], conf: float | None, roi: Roi | None = None) -> tuple[bytes, dict]:
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
