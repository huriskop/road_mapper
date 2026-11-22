"""
Pothole / speed-bump / rough-road detection from accelerometer data
(smartphone, dashcam IMU, Arduino + MPU6050, etc.)

Algorithm
---------
1. Orientation-independent vertical axis:
   low-pass the raw 3-axis signal (< 0.3 Hz) -> gravity vector per sample,
   project the raw acceleration on it -> "vertical" specific force.
   This works no matter how the phone sits in the car.
2. Remove gravity and slow vehicle dynamics, keep the band where wheel
   impacts live (1 - 15 Hz Butterworth band-pass, zero phase).
3. Adaptive threshold: k * robust sigma, where sigma is a rolling MAD
   (median absolute deviation) over ~10 s, so smooth and rough roads
   both work without retuning.
4. Samples above threshold are grouped into events (gaps < merge_gap are
   merged so front + rear wheel hits become one event).
5. Classification by signature of the FIRST strong excursion:
     pothole : wheel drops first  -> vertical acc goes NEGATIVE (free fall),
               then a sharp positive impact spike.
     bump    : wheel is pushed up -> POSITIVE first, then negative.
     rough   : long event with many zero crossings (gravel, broken asphalt).
6. Severity from peak-to-peak amplitude (in g), GPS / speed attached at peak.

Input CSV columns (header required):
    time, ax, ay, az [, lat, lon, speed]
    time in seconds or milliseconds, acceleration in m/s^2 (or g with --units g)

Usage:
    python sensor_detector.py trip.csv
    python sensor_detector.py --demo          # synthetic data self-test
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage, signal

G = 9.80665


@dataclass
class RoadEvent:
    kind: str            # 'pothole' | 'bump' | 'rough'
    severity: str        # 'low' | 'medium' | 'high'
    t_start: float
    t_end: float
    t_peak: float
    peak_to_peak_g: float
    min_g: float
    max_g: float
    lat: float | None = None
    lon: float | None = None
    speed: float | None = None


# --------------------------------------------------------------------------- #
# Signal processing
# --------------------------------------------------------------------------- #
def vertical_acceleration(acc: np.ndarray, fs: float, gravity_cutoff: float = 0.3) -> np.ndarray:
    """Dynamic vertical acceleration (m/s^2), independent of device orientation.

    acc: (N, 3) raw accelerometer incl. gravity, m/s^2.
    Positive = pushed up, negative = falling (wheel drops into a hole).
    """
    b, a = signal.butter(2, gravity_cutoff / (fs / 2), "low")
    g = signal.filtfilt(b, a, acc, axis=0)
    g_norm = np.linalg.norm(g, axis=1, keepdims=True)
    g_unit = g / np.maximum(g_norm, 1e-6)
    return np.sum(acc * g_unit, axis=1) - g_norm[:, 0]


def bandpass(x: np.ndarray, fs: float, lo: float = 1.0, hi: float = 15.0) -> np.ndarray:
    hi = min(hi, 0.45 * fs)
    b, a = signal.butter(3, [lo / (fs / 2), hi / (fs / 2)], "band")
    return signal.filtfilt(b, a, x)


def rolling_sigma(x: np.ndarray, fs: float, window_s: float = 10.0) -> np.ndarray:
    """Robust rolling standard deviation (1.4826 * rolling MAD)."""
    w = max(3, int(window_s * fs) | 1)
    return 1.4826 * ndimage.median_filter(np.abs(x), size=w, mode="nearest")


def _regions(mask: np.ndarray, merge_gap: int, min_len: int) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > merge_gap:
            out.append((start, prev))
            start = i
        prev = i
    out.append((start, prev))
    return [(s, e) for s, e in out if e - s + 1 >= min_len]


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
def detect_events(
    t: np.ndarray,
    acc: np.ndarray,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    speed: np.ndarray | None = None,
    k: float = 5.0,              # threshold in robust sigmas
    min_abs_g: float = 0.25,     # never trigger below this amplitude
    merge_gap_s: float = 0.5,    # merge front/rear wheel hits
    min_speed: float = 1.5,      # m/s, ignore events while (almost) standing
    severity_g: tuple[float, float] = (0.5, 1.0),
) -> tuple[list[RoadEvent], dict]:
    t = np.asarray(t, float)
    fs = 1.0 / np.median(np.diff(t))
    if fs < 30:
        raise ValueError(f"Sample rate {fs:.1f} Hz too low; use >= 50 Hz.")

    av = bandpass(vertical_acceleration(acc, fs), fs)
    thr = np.maximum(k * rolling_sigma(av, fs), min_abs_g * G)
    above = np.abs(av) > thr

    events: list[RoadEvent] = []
    pad = int(0.1 * fs)
    for s, e in _regions(above, int(merge_gap_s * fs), min_len=2):
        s0, e0 = max(0, s - pad), min(len(av) - 1, e + pad)
        seg = av[s0:e0 + 1]
        p = s0 + int(np.argmax(np.abs(seg)))
        if speed is not None and speed[p] < min_speed:
            continue

        # First strong excursion decides pothole (down first) vs bump (up first).
        # (full threshold: the band-pass creates small opposite-sign side lobes
        #  before every pulse; those must not decide the class)
        first = s0 + int(np.flatnonzero(np.abs(seg) > thr[s0:e0 + 1])[0])
        # walk to that lobe's extremum
        j = first
        while j + 1 <= e0 and np.sign(av[j + 1]) == np.sign(av[first]) and abs(av[j + 1]) >= abs(av[j]):
            j += 1
        first_sign = np.sign(av[j])

        duration = t[e0] - t[s0]
        zero_cross = int(np.sum(np.diff(np.sign(seg)) != 0))
        ptp_g = float((seg.max() - seg.min()) / G)

        if duration > 2.0 and zero_cross / duration > 8:
            kind = "rough"
        elif first_sign < 0:
            kind = "pothole"
        else:
            kind = "bump"

        sev = "low" if ptp_g < severity_g[0] else "medium" if ptp_g < severity_g[1] else "high"
        events.append(RoadEvent(
            kind=kind, severity=sev,
            t_start=round(float(t[s0]), 3), t_end=round(float(t[e0]), 3), t_peak=round(float(t[p]), 3),
            peak_to_peak_g=round(ptp_g, 3),
            min_g=round(float(seg.min() / G), 3), max_g=round(float(seg.max() / G), 3),
            lat=None if lat is None else float(lat[p]),
            lon=None if lon is None else float(lon[p]),
            speed=None if speed is None else round(float(speed[p]), 2),
        ))
    return events, {"fs": fs, "vertical": av, "threshold": thr}


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_csv(path: str, units: str = "ms2"):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = {c.strip().lower(): c for c in rows[0].keys()}

    def col(name):
        return np.array([float(r[cols[name]]) for r in rows]) if name in cols else None

    t = col("time")
    if np.median(np.diff(t)) > 1.0:      # milliseconds
        t = t / 1000.0
    t = t - t[0]
    acc = np.column_stack([col("ax"), col("ay"), col("az")])
    if units == "g":
        acc *= G
    extra = {n: col(n) for n in ("lat", "lon", "speed")}

    # resample to a uniform grid (phones deliver jittery timestamps)
    fs = round(1.0 / np.median(np.diff(t)))
    tu = np.arange(0, t[-1], 1.0 / fs)
    acc = np.column_stack([np.interp(tu, t, acc[:, i]) for i in range(3)])
    extra = {n: (None if v is None else np.interp(tu, t, v)) for n, v in extra.items()}
    return tu, acc, extra


def synthetic_trip(fs: float = 100.0, seed: int = 0):
    """60 s drive with known potholes and speed bumps; phone mounted at a random tilt."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 60, 1 / fs)
    az = rng.normal(0, 0.35, t.size) + 0.3 * np.sin(2 * np.pi * 1.3 * t)  # road noise + body bounce
    truth = []

    def pothole(t0, depth=6.0):
        i = int(t0 * fs)
        n = int(0.08 * fs)
        az[i:i + n] -= depth * np.sin(np.linspace(0, np.pi, n))            # drop (free fall)
        k = np.arange(int(0.4 * fs))
        az[i + n:i + n + k.size] += 2.5 * depth * np.exp(-k / (0.05 * fs)) * np.cos(2 * np.pi * 12 * k / fs)
        truth.append(("pothole", t0))

    def bump(t0, h=5.0, wheelbase_dt=0.35):
        for dt in (0.0, wheelbase_dt):                                     # front + rear wheel
            i = int((t0 + dt) * fs)
            n = int(0.25 * fs)
            az[i:i + n] += h * np.sin(np.linspace(0, np.pi, n))             # pushed up
            az[i + n:i + 2 * n] -= 0.7 * h * np.sin(np.linspace(0, np.pi, n))  # comes down
        truth.append(("bump", t0))

    pothole(8.0); bump(17.0); pothole(26.0, 9.0); pothole(35.0, 4.0); bump(45.0, 7.0); pothole(53.0)

    # rough-road segment (gravel) — deliberately long and oscillatory
    seg = (t > 39) & (t < 42)
    az[seg] += rng.normal(0, 3.5, seg.sum())

    acc_world = np.column_stack([rng.normal(0, 0.3, t.size), rng.normal(0, 0.3, t.size), G + az])
    # rotate into a tilted phone frame
    a, b = np.deg2rad(35), np.deg2rad(-20)
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    acc = acc_world @ (ry @ rx).T
    speed = np.full(t.size, 12.0)
    return t, acc, speed, truth


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--units", choices=["ms2", "g"], default="ms2")
    ap.add_argument("--k", type=float, default=5.0, help="threshold in robust sigmas (lower = more sensitive)")
    ap.add_argument("--json", help="write events to this JSON file")
    ap.add_argument("--plot", help="save a diagnostic plot (needs matplotlib)")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        t, acc, speed, truth = synthetic_trip()
        events, dbg = detect_events(t, acc, speed=speed, k=args.k)
        print("Ground truth:", truth)
    elif args.csv:
        t, acc, extra = load_csv(args.csv, args.units)
        events, dbg = detect_events(t, acc, k=args.k, **extra)
    else:
        ap.error("give a CSV file or --demo")

    print(f"fs = {dbg['fs']:.0f} Hz, {len(events)} events")
    for e in events:
        loc = f"  @ {e.lat:.6f},{e.lon:.6f}" if e.lat is not None else ""
        print(f"  t={e.t_peak:7.2f}s  {e.kind:8s} {e.severity:6s} ptp={e.peak_to_peak_g:.2f} g{loc}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(e) for e in events], f, indent=2)
    if args.plot:
        import matplotlib.pyplot as plt
        tt = np.arange(len(dbg["vertical"])) / dbg["fs"]
        plt.figure(figsize=(14, 4))
        plt.plot(tt, dbg["vertical"] / G, lw=0.6, label="vertical acc (g)")
        plt.plot(tt, dbg["threshold"] / G, "r--", lw=0.6)
        plt.plot(tt, -dbg["threshold"] / G, "r--", lw=0.6)
        colors = {"pothole": "tab:red", "bump": "tab:green", "rough": "tab:orange"}
        for e in events:
            plt.axvspan(e.t_start, e.t_end, color=colors[e.kind], alpha=0.25)
            plt.text(e.t_peak, e.max_g, e.kind, fontsize=8)
        plt.xlabel("time (s)"); plt.legend(); plt.tight_layout(); plt.savefig(args.plot, dpi=120)


if __name__ == "__main__":
    main()
