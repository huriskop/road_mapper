#!/usr/bin/env python3
"""M6 - run the detector and frame selector on a recorded drive, at the desk.

    python tools/replay.py DRIVE_DIR [--set detector.tau_a0=5 --set evidence.lookahead_m=4.3]

Prints every detection with its evidence status, compares them with your marks, reports
false triggers per km, suggests thresholds per speed band (99.9th percentile of normal road),
and writes DRIVE_DIR/replay/events.json + contact_sheet.jpg (the chosen frame for each
AVAILABLE event; with the recorder's JPEG log it is the nearest logged image).
"""
import argparse
import bisect
import csv
import json
import math
from pathlib import Path

import _common  # noqa: F401
import numpy as np
from scipy import signal

from roadflag_edge.camera.ring import Frame
from roadflag_edge.clock import Clock
from roadflag_edge.config import Config, load
from roadflag_edge.detect.detector import Detector
from roadflag_edge.detect.localize import ImuRing
from roadflag_edge.gps.neo6m import NmeaAssembler
from roadflag_edge.gps.trajectory import Trajectory
from roadflag_edge.imu.orientation import GravityTracker, MountCalibration
from roadflag_edge import pipeline

SEC = 1_000_000_000


class FrameIndex:
    """Frame timestamps from frames.csv, seen as the live 5 s ring would hold them at `now`."""

    def __init__(self, times: list[int], seqs: list[int], ring_ns: int) -> None:
        self.t, self.seq, self.ring_ns = times, seqs, ring_ns
        self.now = 0

    def _range(self):
        hi = bisect.bisect_right(self.t, self.now)
        lo = bisect.bisect_left(self.t, self.now - self.ring_ns)
        return lo, hi

    def span(self):
        lo, hi = self._range()
        return (self.t[lo], self.t[hi - 1]) if hi > lo else None

    def nearest(self, t_ns: int, neighbours: int = 1):
        lo, hi = self._range()
        if hi <= lo:
            return None
        i = bisect.bisect_left(self.t, t_ns, lo, hi)
        if i == hi or (i > lo and t_ns - self.t[i - 1] <= self.t[i] - t_ns):
            i -= 1
        mk = lambda j: Frame(self.t[j], self.seq[j], None)  # noqa: E731
        nb = [mk(j) for j in range(i - neighbours, i + neighbours + 1) if j != i and lo <= j < hi]
        full = (hi - lo) >= int(self.ring_ns / SEC * 23.5)
        return mk(i), nb, self.t[i] - t_ns, i == lo and full


def load_imu(path: Path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2:5], data[:, 5:8]


def load_gps(path: Path, latency_ms: float):
    asm = NmeaAssembler(int(latency_ms * 1e6))
    fixes = []
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            t, _, s = line.partition("\t")
            fix = asm.feed(s, int(t))
            if fix is not None:
                fixes.append(fix)
    return fixes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("drive")
    p.add_argument("--config", help="default: the config.yaml saved with the drive")
    p.add_argument("--set", action="append", default=[], metavar="section.key=value")
    p.add_argument("--match-s", type=float, default=3.0, help="a detection within this of a mark is a hit")
    a = p.parse_args()
    d = Path(a.drive).expanduser()
    cfg: Config = load(a.config or d / "config.yaml")
    for kv in a.set:
        key, val = kv.split("=", 1)
        sec, k = key.split(".")
        setattr(getattr(cfg, sec), k, type(getattr(getattr(cfg, sec), k))(float(val)))
    fs = cfg.imu.rate_hz

    t, seg, acc, gyro = load_imu(d / "imu.csv")
    fixes = load_gps(d / "gps.nmea", cfg.gps.nmea_latency_ms)
    frames_t, frames_seq = [], []
    if (d / "frames.csv").exists():
        with open(d / "frames.csv") as fh:
            for row in csv.DictReader(fh):
                frames_t.append(int(row["t_ns"]))
                frames_seq.append(int(row["seq"]))
    images = sorted(int(p.stem) for p in (d / "images").glob("*.jpg")) if (d / "images").exists() else []
    marks = []
    if (d / "marks.csv").exists():
        with open(d / "marks.csv") as fh:
            marks = [(int(r["t_ns"]), r["label"]) for r in csv.DictReader(fh)]
    cal = MountCalibration.load(d / "mount.json")
    print(f"{len(t)} IMU samples ({(t[-1] - t[0]) / SEC / 60:.1f} min), {len(fixes)} GPS epochs "
          f"({sum(f.valid for f in fixes)} with fix), {len(frames_t)} frames, {len(images)} images, {len(marks)} marks")

    # full trajectory (evidence waits for the fix after t_i, so using all fixes is what live sees)
    traj = Trajectory(keep_s=1e9, max_gap_s=cfg.gps.max_gap_s, max_hdop=cfg.gps.max_hdop)
    clock = Clock()
    for f in fixes:
        traj.add(f)
        if f.valid and f.utc:
            clock.add_gps_fix(f.t_ns, f.utc)
    fix_t = [f.t_ns for f in fixes]

    def speed_asof(tn):          # what the live detector knew: last fix at or before tn
        i = bisect.bisect_right(fix_t, tn) - 1
        if i < 0 or tn - fix_t[i] > 2 * SEC or not fixes[i].valid:
            return None
        return fixes[i].speed_mps

    tracker = GravityTracker(cal, fs, cfg.imu.attitude_tau_s)
    ring = ImuRing(seconds=(t[-1] - t[0]) / SEC + 5, fs=fs)
    det = Detector(cfg.detector)
    frames = FrameIndex(frames_t, frames_seq, int(cfg.camera.ring_seconds * SEC))
    cands = []
    batch = int(cfg.imu.read_interval_ms / 1000 * fs)
    az_all = np.empty(len(t))
    for s in np.unique(seg):
        idx = np.flatnonzero(seg == s)
        det.reset()
        for i in range(0, len(idx), batch):
            j = idx[i:i + batch]
            az, horiz = tracker.update(acc[j], gyro[j])
            az_all[j] = az
            ring.push(t[j], az, horiz, np.linalg.norm(gyro[j], axis=1), int(s))
            cands += det.process(t[j], az, speed_asof(int(t[j][-1])))
        cands += det.flush()

    events = []
    for c in cands:
        frames.now = c.t_end + SEC // 2      # when the live evidence worker would look
        b = pipeline.build(c, ring, traj, frames, cfg)
        meta = pipeline.metadata("replay", b, clock, cfg)
        ev = b.evidence
        img = None
        if ev.frame is not None and images:
            k = min(range(len(images)), key=lambda i: abs(images[i] - ev.frame.t_ns))
            img = (images[k], (images[k] - ev.frame.t_ns) / 1e6)
        events.append({"t_i": c.t_i, "status": ev.status.value, "detail": ev.detail, "score": c.score,
                       "speed": ev.speed, "defect": meta["defect"], "axle2": c.axle2_seen,
                       "peak_a": meta["imu"]["peak_az_mps2"], "peak_j": meta["imu"]["peak_jz_mps3"],
                       "t_f": ev.t_f, "frame_t": ev.frame.t_ns if ev.frame else None,
                       "align_ms": ev.alignment_error_ms, "image": img})

    t0 = t[0]
    print(f"\n{'time':>8} {'status':<24} {'score':>5} {'peak':>6} {'jerk':>6} {'km/h':>5} {'axle2':>5}  where")
    for e in events:
        kmh = f"{e['speed'] * 3.6:5.1f}" if e["speed"] is not None else "    -"
        where = f"{e['defect']['lat']:.6f},{e['defect']['lon']:.6f}" if e["defect"] else "-"
        print(f"{(e['t_i'] - t0) / SEC:8.2f} {e['status']:<24} {e['score']:5.2f} {e['peak_a']:6.1f} "
              f"{e['peak_j']:6.0f} {kmh} {str(e['axle2']):>5}  {where}")

    # marks vs detections
    hit_marks, used = 0, set()
    for tm, label in marks:
        best = min(((abs(e["t_i"] - tm), i) for i, e in enumerate(events)), default=None)
        ok = best is not None and best[0] <= a.match_s * SEC
        if ok:
            hit_marks += 1
            used.add(best[1])
        print(f"mark {(tm - t0) / SEC:8.2f} {label or '':<10} -> "
              f"{'found ' + format(best[0] / 1e6, '.0f') + ' ms away' if ok else 'MISSED'}")
    km = 0.0
    for f1, f2 in zip(fixes, fixes[1:]):
        if f1.valid and f2.valid and f2.t_ns - f1.t_ns < 2 * SEC:
            km += 0.5 * (f1.speed_mps + f2.speed_mps) * (f2.t_ns - f1.t_ns) / SEC / 1000
    unmarked = len(events) - len(used)
    print(f"\n{len(events)} detections; marks found {hit_marks}/{len(marks)}; unmarked detections {unmarked}"
          + (f" over {km:.2f} km = {unmarked / km:.1f} per km" if km > 0.05 else ""))
    by = {}
    for e in events:
        by[e["status"]] = by.get(e["status"], 0) + 1
    print("evidence:", by)

    # threshold suggestion: 99.9th percentile of A and J per speed band, away from marks
    hp = signal.sosfilt(signal.butter(2, cfg.detector.highpass_hz, "highpass", fs=fs, output="sos"), az_all)
    w = int(cfg.detector.window_s * fs)
    n = len(hp) // w
    A = np.abs(hp[: n * w]).reshape(n, w).max(axis=1)
    J = (np.abs(np.diff(hp, prepend=hp[0]))[: n * w] * fs).reshape(n, w).max(axis=1)
    tw = t[: n * w : w]
    away = np.ones(n, bool)
    for tm, _ in marks:
        away &= np.abs(tw - tm) > 3 * SEC
    v = np.array([speed_asof(int(x)) or np.nan for x in tw])
    print("\nnormal-road percentiles (suggested tau = 99.9th):")
    for lo, hi in ((0, 20), (20, 40), (40, 200)):
        m = away & (v * 3.6 >= lo) & (v * 3.6 < hi)
        if m.sum() > 50:
            print(f"  {lo:>3}-{hi:<3} km/h  n={m.sum():6d}  A99.9={np.percentile(A[m], 99.9):6.2f} m/s^2  "
                  f"J99.9={np.percentile(J[m], 99.9):6.0f} m/s^3")
    if np.isnan(v).all() and away.sum() > 50:
        print(f"  (no GPS speed) n={away.sum()}  A99.9={np.percentile(A[away], 99.9):.2f}  "
              f"J99.9={np.percentile(J[away], 99.9):.0f}")

    out = d / "replay"
    out.mkdir(exist_ok=True)
    (out / "events.json").write_text(json.dumps(events, indent=1, default=str))
    sheet(d, out / "contact_sheet.jpg", events, t0)
    print(f"\nwrote {out / 'events.json'}")


def sheet(d: Path, path: Path, events, t0) -> None:
    import cv2

    tiles = []
    for e in events:
        if not e["image"]:
            continue
        im = cv2.imread(str(d / "images" / f"{e['image'][0]}.jpg"))
        if im is None:
            continue
        im = cv2.resize(im, (480, int(480 * im.shape[0] / im.shape[1])))
        h = im.shape[0]
        cv2.line(im, (0, int(h * 0.7)), (479, int(h * 0.7)), (0, 255, 255), 1)
        txt = f"{(e['t_i'] - t0) / 1e9:.1f}s s={e['score']:.1f} img {e['image'][1]:+.0f}ms"
        cv2.putText(im, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(im, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tiles.append(im)
    if not tiles:
        return
    cols = 4
    h, w = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / cols)
    grid = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, im in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = im[:h, :w]
    cv2.imwrite(str(path), grid)
    print(f"wrote {path} ({len(tiles)} frames)")


if __name__ == "__main__":
    main()
