"""E6 - pick the frame that showed the defect before the car reached it.

    t_i     the wheel hit the defect (from localize)
    t_f     the camera saw that road patch:  integral_{t_f}^{t_i} v dt = d_l
The defect's coordinates are the vehicle position at t_i, moved forward by the GPS antenna
to front-wheel offset. Missing evidence is a status, never a guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..camera.ring import Frame, FrameRing
from ..gps.trajectory import InsufficientTrajectory, NoFix, Trajectory, offset_latlon


class EvidenceStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    LOW_SPEED = "LOW_SPEED"
    INSUFFICIENT_TRAJECTORY = "INSUFFICIENT_TRAJECTORY"
    BUFFER_MISS = "BUFFER_MISS"
    ALIGNMENT_REJECTED = "ALIGNMENT_REJECTED"
    NO_GPS_FIX = "NO_GPS_FIX"
    NO_CAMERA_DATA = "NO_CAMERA_DATA"


@dataclass
class EvidenceConfig:
    lookahead_m: float = 4.0
    antenna_to_front_wheel_m: float = 1.5   # positive: wheel is ahead of the antenna
    min_speed_mps: float = 2.0
    tolerance_ms: float = 30.0
    neighbours: int = 1


@dataclass
class Evidence:
    status: EvidenceStatus
    t_i: int
    t_f: int | None = None
    frame: Frame | None = None
    neighbours: list[Frame] = field(default_factory=list)
    alignment_error_ms: float | None = None
    speed: float | None = None
    heading: float | None = None
    hdop: float | None = None
    defect_latlon: tuple[float, float] | None = None
    vehicle_latlon_at_tf: tuple[float, float] | None = None
    detail: str = ""


def select_evidence(t_i: int, traj: Trajectory, ring: FrameRing | None, cfg: EvidenceConfig) -> Evidence:
    ev = Evidence(EvidenceStatus.AVAILABLE, t_i)
    # 1. a good fix around the impact, and where it was
    try:
        traj.fix_quality_at(t_i)
        st = traj.state_at(t_i)
    except (NoFix, InsufficientTrajectory) as exc:
        ev.status, ev.detail = EvidenceStatus.NO_GPS_FIX, str(exc)
        return ev
    ev.speed, ev.heading, ev.hdop = st["speed"], st["heading"], st["hdop"]
    if st["heading"] is not None:
        ev.defect_latlon = offset_latlon(st["lat"], st["lon"], st["heading"], cfg.antenna_to_front_wheel_m)
    else:
        ev.defect_latlon = (st["lat"], st["lon"])

    # 2. fast enough that timing errors stay small (error in metres grows as 1/v)
    if st["speed"] < cfg.min_speed_mps:
        ev.status, ev.detail = EvidenceStatus.LOW_SPEED, f"{st['speed']:.2f} m/s"
        return ev

    # 3. when the camera saw the patch: integrate speed backward over d_l
    try:
        ev.t_f = traj.time_before_distance(t_i, cfg.lookahead_m)
        vf = traj.state_at(ev.t_f)
        ev.vehicle_latlon_at_tf = (vf["lat"], vf["lon"])
    except InsufficientTrajectory as exc:
        ev.status, ev.detail = EvidenceStatus.INSUFFICIENT_TRAJECTORY, str(exc)
        return ev

    # 4. the frame at t_f, copied out of the ring now, before it is overwritten
    span = ring.span() if ring is not None else None
    if span is None:
        ev.status, ev.detail = EvidenceStatus.NO_CAMERA_DATA, "no frames"
        return ev
    if ev.t_f < span[0]:
        ev.status, ev.detail = EvidenceStatus.BUFFER_MISS, f"{(span[0] - ev.t_f) / 1e6:.0f} ms older than ring"
        return ev
    got = ring.nearest(ev.t_f, cfg.neighbours)
    frame, nb, err_ns, at_edge = got
    ev.alignment_error_ms = err_ns / 1e6
    if at_edge:
        ev.status, ev.detail = EvidenceStatus.BUFFER_MISS, "best frame is at the oldest edge of the ring"
        return ev
    if abs(ev.alignment_error_ms) > cfg.tolerance_ms:
        ev.status = EvidenceStatus.ALIGNMENT_REJECTED
        ev.detail = f"nearest frame {ev.alignment_error_ms:+.1f} ms away (dropped frames?)"
        return ev
    ev.frame, ev.neighbours = frame, nb
    return ev
