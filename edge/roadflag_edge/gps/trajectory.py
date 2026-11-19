"""E3 - "where was the car, how fast, and when was it d metres back?"

Speed is piecewise-linear between fixes, so distance over a segment is the trapezoid area.
Position is interpolated along the same segments. Nothing is ever extrapolated: a question
outside the covered span, or across a gap > max_gap_s, raises InsufficientTrajectory.
"""
from __future__ import annotations

import bisect
import math
import threading
from collections import deque

from .neo6m import Fix

EARTH_R = 6_371_008.8


class InsufficientTrajectory(Exception):
    pass


class NoFix(Exception):
    pass


def offset_latlon(lat: float, lon: float, heading_deg: float, metres: float) -> tuple[float, float]:
    """Move a point `metres` along heading (flat-earth; fine for a few metres)."""
    h = math.radians(heading_deg)
    dlat = metres * math.cos(h) / EARTH_R
    dlon = metres * math.sin(h) / (EARTH_R * math.cos(math.radians(lat)))
    return lat + math.degrees(dlat), lon + math.degrees(dlon)


def _lerp_angle(a: float, b: float, w: float) -> float:
    d = (b - a + 540) % 360 - 180
    return (a + w * d) % 360


class Trajectory:
    def __init__(self, keep_s: float = 30.0, max_gap_s: float = 1.0, max_hdop: float = 5.0) -> None:
        self.keep_ns = int(keep_s * 1e9)
        self.max_gap_ns = int(max_gap_s * 1e9)
        self.max_hdop = max_hdop
        self._fixes: deque[Fix] = deque()
        self._t: list[int] = []
        self._lock = threading.Lock()
        self.last_fix: Fix | None = None

    def add(self, fix: Fix) -> None:
        with self._lock:
            self.last_fix = fix
            if not fix.valid or fix.speed_mps is None:
                return
            if self._fixes and fix.t_ns <= self._fixes[-1].t_ns:
                return
            self._fixes.append(fix)
            while self._fixes and self._fixes[0].t_ns < fix.t_ns - self.keep_ns:
                self._fixes.popleft()
            self._t = [f.t_ns for f in self._fixes]

    # ------------------------------------------------------------- queries
    def newest_ns(self) -> int | None:
        with self._lock:
            return self._fixes[-1].t_ns if self._fixes else None

    def current_speed(self, max_age_s: float = 2.0, now_ns: int | None = None) -> float | None:
        import time

        now_ns = now_ns or time.monotonic_ns()
        f = self.last_fix
        if f is None or not f.valid or f.speed_mps is None or now_ns - f.t_ns > max_age_s * 1e9:
            return None
        return f.speed_mps

    def _segment(self, t: int) -> tuple[Fix, Fix, float]:
        """Bracketing fixes and the interpolation weight; caller holds the lock."""
        if len(self._fixes) < 2 or t < self._t[0] or t > self._t[-1]:
            raise InsufficientTrajectory(f"t outside trajectory ({len(self._fixes)} fixes)")
        i = bisect.bisect_right(self._t, t)
        i = min(max(i, 1), len(self._t) - 1)
        a, b = self._fixes[i - 1], self._fixes[i]
        if b.t_ns - a.t_ns > self.max_gap_ns:
            raise InsufficientTrajectory("GPS gap around this time")
        w = (t - a.t_ns) / (b.t_ns - a.t_ns)
        return a, b, w

    def fix_quality_at(self, t: int) -> None:
        """Raises NoFix if the fixes around t are missing or too imprecise."""
        with self._lock:
            try:
                a, b, _ = self._segment(t)
            except InsufficientTrajectory as exc:
                raise NoFix(str(exc)) from exc
            for f in (a, b):
                if f.hdop is not None and f.hdop > self.max_hdop:
                    raise NoFix(f"HDOP {f.hdop} > {self.max_hdop}")

    def speed_at(self, t: int) -> float:
        with self._lock:
            a, b, w = self._segment(t)
            return a.speed_mps + w * (b.speed_mps - a.speed_mps)

    def state_at(self, t: int) -> dict:
        with self._lock:
            a, b, w = self._segment(t)
            heading = None
            if a.heading is not None and b.heading is not None:
                heading = _lerp_angle(a.heading, b.heading, w)
            else:
                heading = a.heading if a.heading is not None else b.heading
            return {
                "lat": a.lat + w * (b.lat - a.lat),
                "lon": a.lon + w * (b.lon - a.lon),
                "speed": a.speed_mps + w * (b.speed_mps - a.speed_mps),
                "heading": heading,
                "hdop": max((f.hdop for f in (a, b) if f.hdop is not None), default=None),
            }

    def time_before_distance(self, end_t: int, d: float) -> int:
        """The time t_f < end_t such that the car travelled exactly d metres in [t_f, end_t]."""
        if d <= 0:
            return end_t
        with self._lock:
            a, b, w = self._segment(end_t)
            i = self._t.index(b.t_ns)
            # walk backward segment by segment
            t_hi = end_t
            v_hi = a.speed_mps + w * (b.speed_mps - a.speed_mps)
            remaining = d
            j = i - 1                                   # index of the segment's start fix
            while True:
                fa = self._fixes[j]
                if j + 1 < len(self._fixes) and self._fixes[j + 1].t_ns - fa.t_ns > self.max_gap_ns:
                    raise InsufficientTrajectory("GPS gap inside look-back")
                t_lo, v_lo = fa.t_ns, fa.speed_mps
                dt = (t_hi - t_lo) / 1e9
                seg = 0.5 * (v_lo + v_hi) * dt
                if seg >= remaining:
                    # solve for x in [0, dt] (seconds back from t_hi), speed linear in x
                    slope = (v_lo - v_hi) / dt if dt > 0 else 0.0
                    if abs(slope) < 1e-9:
                        x = remaining / v_hi
                    else:          # v_hi*x + slope*x^2/2 = remaining
                        disc = v_hi * v_hi + 2 * slope * remaining
                        x = (-v_hi + math.sqrt(max(disc, 0.0))) / slope
                    return int(t_hi - x * 1e9)
                remaining -= seg
                t_hi, v_hi = t_lo, v_lo
                j -= 1
                if j < 0:
                    raise InsufficientTrajectory("trajectory too short for look-back")
