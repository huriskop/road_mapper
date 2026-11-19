"""Candidate -> observation. Shared by the live app and tools/replay.py so both behave the same."""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

from . import __version__
from .clock import Clock, iso_utc
from .config import Config
from .detect.detector import Candidate
from .detect.localize import ImuRing, localize
from .evidence.select import Evidence, EvidenceStatus, select_evidence
from .gps.trajectory import Trajectory

SEC = 1_000_000_000


@dataclass
class Built:
    cand: Candidate
    evidence: Evidence
    loc: dict | None
    snippet: dict[str, np.ndarray]


def localize_candidate(cand: Candidate, imu: ImuRing, fs: float) -> dict | None:
    w = imu.window(cand.t_d - SEC // 2, cand.t_d + SEC // 2)
    if len(w["t"]) == 0:
        return None
    # never filter across an IMU gap: keep the segment the trigger came from
    k = int(np.argmin(np.abs(w["t"] - cand.t_d)))
    keep = w["seg"] == w["seg"][k]
    loc = localize(w["t"][keep], w["az"][keep], cand.t_d, fs=fs)
    cand.gyro_peak = float(np.max(w["gyro"][keep])) if keep.any() else 0.0
    cand.horiz_peak = float(np.max(w["horiz"][keep])) if keep.any() else 0.0
    return loc


def build(cand: Candidate, imu: ImuRing, traj: Trajectory, frames, cfg: Config,
          loc: dict | None = None) -> Built:
    """frames: a FrameRing (live) or anything with span()/nearest() (replay)."""
    if loc is None:
        loc = localize_candidate(cand, imu, cfg.detector.fs)
    cand.t_i = loc["t_i"] if loc else cand.t_d
    ev = select_evidence(cand.t_i, traj, frames, cfg.evidence)
    half = int(cfg.snippet_s * SEC)
    return Built(cand, ev, loc, imu.window(cand.t_i - half, cand.t_i + half))


def metadata(obs_id: str, b: Built, clock: Clock, cfg: Config) -> dict:
    c, ev = b.cand, b.evidence
    t_imp = clock.mono_to_utc(c.t_i)
    t_frame = clock.mono_to_utc(ev.t_f) if ev.t_f is not None else None
    r = lambda v, n=2: None if v is None else round(float(v), n)  # noqa: E731
    return {
        "schema_version": 1,
        "observation_id": obs_id,
        "device_id": cfg.device_id,
        "evidence_status": ev.status.value,
        "evidence_detail": ev.detail or None,
        "t_impact_mono_ns": int(c.t_i),
        "t_impact_utc": iso_utc(t_imp),
        "t_frame_utc": iso_utc(t_frame) if ev.status == EvidenceStatus.AVAILABLE else None,
        "utc_estimated": t_imp is not None,
        "alignment_error_ms": r(ev.alignment_error_ms, 1),
        "speed_before_mps": r(ev.speed),
        "lookahead_m": cfg.evidence.lookahead_m,
        "defect": {"lat": round(ev.defect_latlon[0], 7), "lon": round(ev.defect_latlon[1], 7)}
        if ev.defect_latlon else None,
        "vehicle_at_frame": {"lat": round(ev.vehicle_latlon_at_tf[0], 7), "lon": round(ev.vehicle_latlon_at_tf[1], 7)}
        if ev.vehicle_latlon_at_tf else None,
        "heading_deg": r(ev.heading, 1),
        "hdop": r(ev.hdop),
        "imu": {
            "peak_az_mps2": r(b.loc["peak_a"] if b.loc else c.peak_a),
            "peak_jz_mps3": r(b.loc["peak_j"] if b.loc else c.peak_j, 0),
            "trigger_a_mps2": r(c.peak_a),
            "trigger_j_mps3": r(c.peak_j, 0),
            "score": r(c.score),
            "axle2_seen": bool(c.axle2_seen),
            "gyro_peak_dps": r(np.degrees(c.gyro_peak), 1),
            "horiz_peak_mps2": r(c.horiz_peak),
            "detect_delay_ms": r(b.loc["delay_ms"], 1) if b.loc else None,
        },
        "firmware": __version__,
    }


def snippet_bytes(snippet: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **snippet)
    return buf.getvalue()
