"""Run the camera detector (vision_detector.py) on uploaded photos.

The stored photo is replaced by the annotated version: labelled boxes for every
pothole and crack the detector finds, so the web console shows what was found.
If OpenCV is missing or the photo cannot be decoded, the original is kept.
"""
from __future__ import annotations

import base64
import logging
import sys
import time
from collections import Counter
from typing import Any

from .config import PROJECT_ROOT

log = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import vision_detector
except ImportError:  # detector is optional: the API still works without OpenCV
    vision_detector = None


def available() -> bool:
    return vision_detector is not None


def analyze_photo(image: bytes, detect: tuple[str, ...], pothole_conf: float = 0.5) -> dict[str, Any]:
    """Run the detector on one photo for the test page. Raises ValueError for unreadable images."""
    if vision_detector is None:
        raise RuntimeError("OpenCV is not installed on the server")
    bgr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not read the image (use JPEG, PNG or WebP)")
    t0 = time.perf_counter()
    defects, vis = vision_detector.analyze(bgr, detect=detect, pothole_conf=pothole_conf)
    ms = round(1000 * (time.perf_counter() - t0))
    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    h, w = bgr.shape[:2]
    return {
        "width": w,
        "height": h,
        "time_ms": ms,
        "camera_type": summarize(defects)["camera_type"],
        "gps": photo_gps(image),
        "pothole_model": vision_detector.pothole_model() is not None,
        "speed_bump_model": vision_detector.speed_bump_model() is not None,
        "detect": list(detect),
        "pothole_conf": pothole_conf,
        "defects": [
            {"kind": d.kind, "severity": d.severity, "confidence": d.confidence,
             "bbox": {"x": d.bbox[0], "y": d.bbox[1], "w": d.bbox[2], "h": d.bbox[3]},
             "length_px": d.length_px or None, "width_px": d.width_px or None}
            for d in defects
        ],
        "image": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(),
    }


def annotate(image: bytes, detect: tuple[str, ...] = ("pothole", "crack", "speed_bump")) -> tuple[bytes, dict[str, Any] | None]:
    """Return (annotated JPEG bytes, detection summary). Falls back to the original image."""
    if vision_detector is None:
        return image, None
    try:
        bgr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return image, None
        defects, vis = vision_detector.analyze(bgr, detect=detect)
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            return image, None
    except Exception:  # never lose a report because the detector failed
        log.exception("vision detector failed; keeping original image")
        return image, None
    return buf.tobytes(), summarize(defects)


def summarize(defects) -> dict[str, Any]:
    """Compact record of what the camera saw, stored with each report.

    camera_type is the camera's verdict for the incident type: "pothole" or "bump" when the
    trained models / marking detector found one, else None. Cracks come from image rules
    that also fire on pothole rims and painted lines, so they never set camera_type."""
    kinds = Counter(d.kind for d in defects)
    best = lambda k: max((d.confidence or 0 for d in defects if d.kind == k), default=None)  # noqa: E731
    camera_type = "pothole" if kinds["pothole"] else "bump" if kinds["speed_bump"] else None
    return {
        "camera_type": camera_type,
        "potholes": kinds["pothole"],
        "speed_bumps": kinds["speed_bump"],
        "cracks": sum(v for k, v in kinds.items() if k.endswith("_crack")),
        "pothole_confidence": best("pothole"),
        "speed_bump_confidence": best("speed_bump"),
        "by_type": dict(kinds),
        "pothole_model": vision_detector.pothole_model() is not None,
    }


def photo_gps(image: bytes) -> dict[str, Any] | None:
    """GPS position and capture time from a photo's EXIF data, if the camera recorded them."""
    try:
        import io
        from PIL import Image
        exif = Image.open(io.BytesIO(image)).getexif()
        gps = exif.get_ifd(0x8825)
        if not gps or 2 not in gps or 4 not in gps:
            return None
        dms = lambda v: float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600  # noqa: E731
        lat = dms(gps[2]) * (-1 if gps.get(1) == "S" else 1)
        lng = dms(gps[4]) * (-1 if gps.get(3) == "W" else 1)
        taken = exif.get_ifd(0x8769).get(36867) or exif.get(306)       # DateTimeOriginal / DateTime
        return {"lat": round(lat, 7), "lng": round(lng, 7), "taken": str(taken) if taken else None}
    except Exception:
        return None
