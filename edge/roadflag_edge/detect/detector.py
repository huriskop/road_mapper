"""E5 - causal trigger: decide that a hit happened.

Per 100 ms window k on the 1 Hz high-passed vertical acceleration:
    A_k = max |a|        (m/s^2)        J_k = max |da/dt|   (m/s^3)
    candidate_k = A_k > tau_a(v)  or  J_k > tau_j(v),   tau(v) = theta0 + theta1 * v
Hysteresis: arm above tau, release when both fall under release_ratio * tau. After release,
a refractory window lasts until the car has moved one wheelbase + 1 m; a second hit inside
it is the rear axle (axle2_seen) and is not a new event. The candidate is returned at
release, so evidence can be copied out of the rings right away; `done` is set when the
refractory window ends and axle2_seen is final.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
from scipy import signal


@dataclass
class DetectorConfig:
    fs: float = 200.0
    window_s: float = 0.1
    highpass_hz: float = 1.0
    tau_a0: float = 6.0          # m/s^2 at standstill
    tau_a1: float = 0.10         # m/s^2 per m/s
    tau_j0: float = 600.0        # m/s^3 (a 15 m/s^2 hit ringing at 12 Hz is ~1100)
    tau_j1: float = 5.0          # m/s^3 per m/s
    release_ratio: float = 0.6
    wheelbase_m: float = 2.7
    refractory_extra_m: float = 1.0
    refractory_max_s: float = 2.0
    refractory_no_speed_s: float = 1.0
    park_speed_mps: float = 0.5  # below this (and known) nothing is reported: door slams

    def tau_a(self, v: float | None) -> float:
        return self.tau_a0 + self.tau_a1 * (v or 0.0)

    def tau_j(self, v: float | None) -> float:
        return self.tau_j0 + self.tau_j1 * (v or 0.0)


@dataclass
class Candidate:
    t_d: int                      # detection time (end of the window that armed), monotonic ns
    peak_a: float
    peak_j: float
    score: float
    speed: float | None
    t_start: int                  # first sample of the event
    t_end: int                    # release time
    gyro_peak: float = 0.0        # rad/s, filled by the app from the IMU ring
    horiz_peak: float = 0.0
    axle2_seen: bool = False
    t_axle2: int | None = None
    t_i: int | None = None        # impact time, set by localize
    extra: dict = field(default_factory=dict)
    done: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)


class Detector:
    def __init__(self, cfg: DetectorConfig | None = None) -> None:
        self.cfg = cfg or DetectorConfig()
        self.sos = signal.butter(2, self.cfg.highpass_hz, "highpass", fs=self.cfg.fs, output="sos")
        self.win = max(2, int(round(self.cfg.window_s * self.cfg.fs)))
        self.reset()

    def reset(self) -> None:
        """Forget filter state and any open event (new IMU segment, or between test cases)."""
        self.zi = None
        self._prev = None
        self._buf_t: list[int] = []
        self._buf_a: list[float] = []
        self._buf_j: list[float] = []
        self._event: Candidate | None = None
        self._armed = False
        self._refractory: tuple[Candidate, float, float] | None = None   # cand, metres left, seconds left
        self.dropped_parked = 0

    def process(self, t_ns: np.ndarray, az: np.ndarray, speed: float | None) -> list[Candidate]:
        """Feed consecutive samples of one segment. Returns candidates released in this batch."""
        if len(az) == 0:
            return []
        if self.zi is None:
            self.zi = signal.sosfilt_zi(self.sos) * az[0]
        hp, self.zi = signal.sosfilt(self.sos, az, zi=self.zi)
        prev = self._prev if self._prev is not None else hp[0]
        jerk = np.abs(np.diff(np.concatenate(([prev], hp)))) * self.cfg.fs
        self._prev = hp[-1]
        self._buf_t.extend(t_ns.tolist())
        self._buf_a.extend(np.abs(hp).tolist())
        self._buf_j.extend(jerk.tolist())
        out: list[Candidate] = []
        while len(self._buf_t) >= self.win:
            t = self._buf_t[: self.win]
            a = max(self._buf_a[: self.win])
            j = max(self._buf_j[: self.win])
            del self._buf_t[: self.win], self._buf_a[: self.win], self._buf_j[: self.win]
            rel = self._window(t[0], t[-1], a, j, speed)
            if rel is not None:
                out.append(rel)
        return out

    def _window(self, t0: int, t1: int, a: float, j: float, v: float | None) -> Candidate | None:
        c = self.cfg
        ta, tj = c.tau_a(v), c.tau_j(v)
        dt = (t1 - t0) / 1e9 + 1 / c.fs
        released = None

        # refractory bookkeeping (counts down in metres, or seconds without speed)
        if self._refractory is not None:
            cand, metres, secs = self._refractory
            if a > ta or j > tj:
                if not cand.axle2_seen:
                    cand.axle2_seen, cand.t_axle2 = True, t1
            metres -= (v or 0.0) * dt
            secs -= dt
            if (v is not None and v > 0 and metres <= 0) or secs <= 0:
                self._refractory = None
                cand.done.set()
            else:
                self._refractory = (cand, metres, secs)
                return None

        if not self._armed:
            if a > ta or j > tj:
                if v is not None and v < c.park_speed_mps:
                    self.dropped_parked += 1
                    return None
                self._armed = True
                self._event = Candidate(t_d=t1, peak_a=a, peak_j=j, score=max(a / ta, j / tj),
                                        speed=v, t_start=t0, t_end=t1)
        else:
            e = self._event
            e.peak_a, e.peak_j = max(e.peak_a, a), max(e.peak_j, j)
            e.score = max(e.score, a / ta, j / tj)
            e.t_end = t1
            if a < c.release_ratio * ta and j < c.release_ratio * tj:
                self._armed = False
                if v is not None and v > 0:
                    secs = c.refractory_max_s
                else:
                    secs = c.refractory_no_speed_s
                self._refractory = (e, c.wheelbase_m + c.refractory_extra_m, secs)
                self._event = None
                released = e
        return released

    def flush(self) -> list[Candidate]:
        """End of data (replay): finish the refractory window, emit an unreleased event."""
        out = []
        if self._refractory is not None:
            self._refractory[0].done.set()
            self._refractory = None
        if self._event is not None:
            self._event.done.set()
            out.append(self._event)
            self._event = None
            self._armed = False
        return out
