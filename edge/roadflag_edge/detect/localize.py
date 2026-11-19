"""E5 - find the impact time t_i exactly.

The causal trigger fires late by an unknown filter delay. Instead of modelling it, re-run a
zero-phase band-pass (sosfiltfilt: no delay) on the raw ring +-0.5 s around t_d and take the
first strong peak as the moment the wheel hit.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


class ImuRing:
    """Last N seconds of (t, az_w, horiz, |gyro|) for localisation and evidence snippets."""

    def __init__(self, seconds: float = 10.0, fs: float = 200.0) -> None:
        import threading

        self.n = int(seconds * fs)
        self.t = np.zeros(self.n, dtype=np.int64)
        self.az = np.zeros(self.n)
        self.horiz = np.zeros(self.n)
        self.gyro = np.zeros(self.n)
        self.seg = np.zeros(self.n, dtype=np.int64)
        self.count = 0
        self._lock = threading.Lock()

    def push(self, t, az, horiz, gyro_mag, segment: int) -> None:
        with self._lock:
            for arr, val in ((self.t, t), (self.az, az), (self.horiz, horiz), (self.gyro, gyro_mag)):
                self._put(arr, np.asarray(val))
            self._put(self.seg, np.full(len(t), segment))
            self.count += len(t)

    def _put(self, arr, vals) -> None:
        start = self.count % self.n
        k = len(vals)
        if k >= self.n:
            arr[:] = vals[-self.n:]
            return
        end = start + k
        if end <= self.n:
            arr[start:end] = vals
        else:
            cut = self.n - start
            arr[start:] = vals[:cut]
            arr[: end - self.n] = vals[cut:]

    def newest_ns(self) -> int | None:
        with self._lock:
            return int(self.t[(self.count - 1) % self.n]) if self.count else None

    def window(self, t0: int, t1: int) -> dict[str, np.ndarray]:
        """Samples with t0 <= t <= t1, in time order (copies)."""
        with self._lock:
            m = min(self.count, self.n)
            start = (self.count - m) % self.n
            order = (np.arange(m) + start) % self.n
            t = self.t[order]
            sel = order[(t >= t0) & (t <= t1)]
            return {"t": self.t[sel].copy(), "az": self.az[sel].copy(), "horiz": self.horiz[sel].copy(),
                    "gyro": self.gyro[sel].copy(), "seg": self.seg[sel].copy()}


def localize(t: np.ndarray, az: np.ndarray, t_d: int, fs: float = 200.0,
             band: tuple[float, float] = (1.0, 30.0), first_peak_ratio: float = 0.5) -> dict | None:
    """Return {'t_i', 'peak_a', 'peak_j'} from samples around t_d, or None if too few."""
    if len(t) < int(0.3 * fs):
        return None
    hi = min(band[1], 0.45 * fs)
    sos = signal.butter(2, (band[0], hi), "bandpass", fs=fs, output="sos")
    padlen = min(len(az) - 1, 3 * (2 * len(sos) + 1))
    y = signal.sosfiltfilt(sos, az - np.mean(az), padlen=padlen)
    mag = np.abs(y)
    peak = float(mag.max())
    if peak <= 0:
        return None
    # first sample above half the max, then climb to that local maximum
    k = int(np.argmax(mag >= first_peak_ratio * peak))
    while k + 1 < len(mag) and mag[k + 1] >= mag[k]:
        k += 1
    jerk = np.abs(np.diff(y)) * fs
    return {"t_i": int(t[k]), "peak_a": peak, "peak_j": float(jerk.max()) if len(jerk) else 0.0,
            "delay_ms": (t_d - int(t[k])) / 1e6}
