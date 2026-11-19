"""Classification worker: turns queued observations into map defects.

Upload returns in milliseconds; this runs in the background. For each leased job:
  1. image quality gate (blur = Laplacian variance, too dark / too bright) -> bad images go to review
  2. classify(image) -> (label, confidence) through the existing camera detector
  3. decide: >= confirm_confidence CONFIRMED, between review and confirm NEEDS_REVIEW,
     "nothing" with high confidence REJECTED (kept to measure false triggers)
  4. CONFIRMED -> join a defect with the same label within dedupe_radius_m and a similar
     heading, or create one (report_packets.id = observation_id, so a retry never counts twice)
Observations without an image skip the classifier: NEEDS_REVIEW if the IMU score is high,
else REJECTED. A crash leaves the job leased; it is retried after the lease expires, up to
job_max_attempts, then FAILED.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from . import vision
from .config import Settings
from .db import Database
from .models import DefectType, ObservationStatus
from .storage import LocalImageStorage

log = logging.getLogger(__name__)
G = 9.80665


def image_quality(image: bytes, settings: Settings) -> dict[str, Any]:
    if not vision.available():
        return {"ok": True, "reason": "not checked (OpenCV missing)"}
    import cv2
    import numpy as np

    gray = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return {"ok": False, "reason": "unreadable"}
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    lo, hi = settings.brightness_range
    reason = None
    if blur < settings.blur_min_laplacian_var:
        reason = "blurred"
    elif bright < lo:
        reason = "too dark"
    elif bright > hi:
        reason = "too bright"
    return {"ok": reason is None, "reason": reason, "blur": round(blur, 1), "brightness": round(bright, 1)}


def classify(image: bytes, settings: Settings) -> tuple[str, float, bytes, dict | None]:
    """Adapter around the existing detector: (label, confidence, annotated JPEG, summary).
    label is one of pothole, crack, bump, nothing - or unknown if the detector is unavailable."""
    annotated, found = vision.annotate(image, settings.detect)
    if found is None:
        return "unknown", 0.0, image, None
    if found.get("camera_type") == "pothole":
        return "pothole", float(found.get("pothole_confidence") or 0.5), annotated, found
    if found.get("camera_type") == "bump":
        return "bump", float(found.get("speed_bump_confidence") or 0.6), annotated, found
    if found.get("cracks"):
        return "crack", settings.crack_confidence, annotated, found
    # Nothing found. Only trust "nothing" when the trained pothole model actually looked.
    return "nothing", (0.75 if found.get("pothole_model") else 0.5), annotated, found


def decide(label: str, conf: float, quality_ok: bool, s: Settings) -> str:
    if not quality_ok or label == "unknown":
        return ObservationStatus.NEEDS_REVIEW.value
    if label == "nothing":
        return ObservationStatus.REJECTED.value if conf >= s.confirm_confidence else ObservationStatus.NEEDS_REVIEW.value
    if conf >= s.confirm_confidence:
        return ObservationStatus.CONFIRMED.value
    return ObservationStatus.NEEDS_REVIEW.value


class ReviewConflict(Exception):
    """The observation is not waiting for review (already decided)."""


class Worker:
    def __init__(self, db: Database, storage: LocalImageStorage, settings: Settings) -> None:
        self.db, self.storage, self.settings = db, storage, settings

    def run_once(self) -> dict[str, Any] | None:
        """Process one job. Returns {"observation_id", "status", "incident"} or None if idle."""
        s = self.settings
        obs = self.db.claim_job(s.job_lease_s, s.job_max_attempts)
        if obs is None:
            return None
        obs_id = obs["id"]
        try:
            return self._process(obs)
        except Exception as exc:
            log.exception("classification failed for %s", obs_id)
            self.db.fail_job(obs_id, f"{type(exc).__name__}: {exc}", s.job_max_attempts)
            return {"observation_id": obs_id, "status": "RETRY", "incident": None}

    def _process(self, obs) -> dict[str, Any]:
        s = self.settings
        obs_id = obs["id"]
        if not obs["image_path"]:
            score = obs["imu_score"] or 0.0
            status = (ObservationStatus.NEEDS_REVIEW if score >= s.review_imu_score else ObservationStatus.REJECTED).value
            self.db.finish_observation(obs_id, status=status, label=None, confidence=None, quality=None, vision=None,
                                       annotated_path=None, decided_by="worker")
            return {"observation_id": obs_id, "status": status, "incident": None}

        image = Path(self.storage.root / obs["image_path"]).read_bytes()
        quality = image_quality(image, s)
        label, conf, annotated, found = classify(image, s)
        status = decide(label, conf, quality["ok"], s)
        annotated_key = self.storage.save(annotated) if annotated is not image else obs["image_path"]
        attach = None
        if status == ObservationStatus.CONFIRMED.value:
            attach = self._attach_args(obs, DefectType(label), annotated_key, found, "camera", conf)
        result = self.db.finish_observation(
            obs_id, status=status, label=label, confidence=round(conf, 3), quality=quality, vision=found,
            annotated_path=annotated_key, decided_by="worker", attach=attach,
            radius_m=s.dedupe_radius_m, heading_tol=s.heading_tolerance_deg)
        return {"observation_id": obs_id, "status": status, "incident": result}

    def _attach_args(self, obs, label: DefectType, image_key: str, found, source: str, conf) -> dict[str, Any]:

        meta = json.loads(obs["metadata"])
        sensor = {"observation_id": obs["id"], "evidence_status": obs["evidence_status"],
                  "imu": meta.get("imu"), "speed_mps": obs["speed_mps"], "heading_deg": obs["heading_deg"],
                  "label": label.value, "confidence": conf, "decided_by": source}
        if found is not None:
            sensor["vision"] = found
        peak = obs["peak_az_mps2"] / G if obs["peak_az_mps2"] is not None else None
        sensor["peak_g"] = peak
        return {"defect_type": label, "image_path": image_key, "sensor_metadata": sensor, "peak": peak,
                "source": source}

    def review(self, obs_id: str, decision: str, label: DefectType | None) -> dict[str, Any]:
        """Operator verdict on a NEEDS_REVIEW observation. Raises LookupError / ValueError."""
        with self.db.connect() as conn:
            obs = conn.execute("SELECT * FROM observations WHERE id = ?", (obs_id,)).fetchone()
        if obs is None:
            raise LookupError("observation not found")
        if obs["status"] != ObservationStatus.NEEDS_REVIEW.value:
            raise ReviewConflict(f"observation is {obs['status']}, not NEEDS_REVIEW")
        if decision == "reject":
            self.db.finish_observation(obs_id, status=ObservationStatus.REJECTED.value, label=obs["label"],
                                       confidence=obs["confidence"], quality=None, vision=None,
                                       annotated_path=None, decided_by="operator")
            return {"observation_id": obs_id, "status": "REJECTED", "incident": None}
        if label is None and obs["label"] in ("pothole", "crack", "bump"):
            label = DefectType(obs["label"])
        if label is None or label == DefectType.unclassified:
            raise ValueError("label (pothole, crack or bump) is required to confirm")
        if obs["lat"] is None:
            raise ValueError("observation has no location (no GPS fix); it cannot go on the map")
        image_key = obs["annotated_path"] or obs["image_path"]
        if image_key is None:
            raise ValueError("observation has no photo; confirm needs one")
        found = json.loads(obs["vision"]) if obs["vision"] else None
        attach = self._attach_args(obs, label, image_key, found, "operator", obs["confidence"])
        result = self.db.finish_observation(
            obs_id, status=ObservationStatus.CONFIRMED.value, label=label.value, confidence=obs["confidence"],
            quality=None, vision=None, annotated_path=None, decided_by="operator", attach=attach,
            radius_m=self.settings.dedupe_radius_m, heading_tol=self.settings.heading_tolerance_deg)
        return {"observation_id": obs_id, "status": "CONFIRMED", "incident": result}
