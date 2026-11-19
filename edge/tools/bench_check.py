#!/usr/bin/env python3
"""Bench checks for milestones M1-M3 (IMU rate, camera ring, GPS) without driving.

    python tools/bench_check.py imu    --seconds 3600   # M1: 200.0 +- 0.5 Hz, no FIFO overflow
    python tools/bench_check.py camera --seconds 3600   # M3: 24 FPS, flat memory, 41.7 +- 1 ms
    python tools/bench_check.py gps    --configure      # M2: 5 Hz fixes at 38400 baud
    python tools/bench_check.py tap                     # tap the sensor: one sharp peak each time
"""
import argparse
import resource
import time

import _common  # noqa: F401
import numpy as np

from roadflag_edge.config import load
from roadflag_edge.detect.detector import Detector
from roadflag_edge.imu.orientation import GravityTracker, MountCalibration


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def imu(cfg, seconds: float, tap: bool = False) -> None:
    from roadflag_edge.imu.mpu6050 import MPU6050

    mpu = MPU6050(cfg.imu.bus, cfg.imu.addr, cfg.imu.rate_hz, cfg.imu.dlpf)
    tracker = GravityTracker(MountCalibration.load(cfg.path(cfg.mount_file)))
    det = Detector(cfg.detector)
    n, first_t, last_t, jit = 0, None, None, []
    mags = []
    start = time.monotonic()
    nxt = start
    last_print = start
    seg = None
    try:
        while time.monotonic() - start < seconds:
            nxt += cfg.imu.read_interval_ms / 1000
            time.sleep(max(0.0, nxt - time.monotonic()))
            t_call = time.monotonic()
            b = mpu.read()
            jit.append((time.monotonic() - t_call) * 1000)
            if b is None:
                continue
            if b.segment != seg:
                seg, first_t, n = b.segment, b.t_ns[0], 0
                det.reset()
            n += len(b)
            last_t = b.t_ns[-1]
            az, _ = tracker.update(b.acc, b.gyro)
            mags.append(np.linalg.norm(b.acc, axis=1))
            if tap:
                for c in det.process(b.t_ns, az, None):
                    print(f"  TAP  peak {c.peak_a:5.1f} m/s^2  jerk {c.peak_j:6.0f} m/s^3  score {c.score:.1f}")
            if time.monotonic() - last_print > (2 if tap else 10):
                last_print = time.monotonic()
                rate = (n - 1) / ((last_t - first_t) / 1e9) if n > 1 else 0
                m = np.concatenate(mags[-200:])
                print(f"{time.monotonic() - start:6.0f}s  rate {rate:7.2f} Hz  overflows {mpu.overflows}  "
                      f"i2c errors {mpu.i2c_errors}  |a| {m.mean():.3f} +- {m.std():.3f}  az {az.mean():+.3f}  "
                      f"read {np.mean(jit[-200:]):.1f} ms  rss {rss_mb():.0f} MB", flush=True)
    except KeyboardInterrupt:
        pass
    rate = (n - 1) / ((last_t - first_t) / 1e9) if n > 1 else 0
    m = np.concatenate(mags)
    ok = abs(rate - cfg.imu.rate_hz) <= 0.5 and mpu.overflows == 0
    print(f"\nRESULT imu: {n} samples, effective rate {rate:.2f} Hz, overflows {mpu.overflows}, "
          f"|a| {m.mean():.3f} m/s^2 (noise {m.std():.3f}) -> {'PASS' if ok else 'CHECK'}")
    mpu.close()


def camera(cfg, seconds: float) -> None:
    import threading

    from roadflag_edge.camera.ring import FrameRing, PiCameraSource

    c = cfg.camera
    ring = FrameRing(int(c.ring_seconds * c.fps), c.width, c.height)
    src = PiCameraSource(ring, c.width, c.height, c.fps, c.exposure_us, c.frame_time_offset_ms)
    threading.Thread(target=src.run, daemon=True).start()
    start = time.monotonic()
    pushed0, t0 = 0, start
    worst = 0.0
    try:
        while time.monotonic() - start < seconds:
            time.sleep(10)
            sp = ring.frame_spacing_ms()
            now = time.monotonic()
            fps = (ring.pushed - pushed0) / (now - t0)
            pushed0, t0 = ring.pushed, now
            if sp:
                worst = max(worst, sp[1])
            print(f"{now - start:6.0f}s  fps {fps:5.2f}  spacing {sp[0]:.2f} +- {sp[1]:.2f} ms  dropped {ring.dropped}  "
                  f"restarts {src.restarts}  rss {rss_mb():.0f} MB", flush=True)
    except KeyboardInterrupt:
        pass
    src.stop()
    sp = ring.frame_spacing_ms()
    ok = sp and abs(sp[0] - 1000 / c.fps) < 1 and worst < 1 and ring.dropped == 0
    print(f"\nRESULT camera: {ring.pushed} frames, spacing {sp[0]:.2f} ms (worst std {worst:.2f}), dropped "
          f"{ring.dropped}, restarts {src.restarts} -> {'PASS' if ok else 'CHECK'}")
    if len(ring):
        f, *_ = ring.nearest(ring.span()[1])
        out = cfg.path(cfg.data_dir) / "bench_frame.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(f.to_jpeg())
        print(f"last frame saved to {out}")


def gps(cfg, seconds: float, configure: bool) -> None:
    from roadflag_edge.gps.neo6m import Neo6m

    g = Neo6m(cfg.gps.port, cfg.gps.baud, cfg.gps.rate_hz, cfg.gps.nmea_latency_ms)
    if configure:
        g.configure()
    else:
        print(f"GPS at {g.open_port()} baud")
    start = time.monotonic()
    times, valid = [], 0
    for fix in g.fixes():
        times.append(fix.t_ns)
        valid += fix.valid
        if len(times) % max(1, cfg.gps.rate_hz) == 0:
            dt = np.diff(times[-20:]) / 1e6 if len(times) > 2 else [0]
            print(f"{time.monotonic() - start:5.0f}s  q={fix.quality} sats={fix.sats} hdop={fix.hdop} "
                  f"lat={fix.lat} lon={fix.lon} v={fix.speed_mps} hdg={fix.heading}  "
                  f"epoch {np.mean(dt):.0f} ms", flush=True)
        if time.monotonic() - start > seconds:
            break
    rate = (len(times) - 1) / ((times[-1] - times[0]) / 1e9) if len(times) > 1 else 0
    print(f"\nRESULT gps: {len(times)} epochs at {rate:.2f} Hz, {valid} with a fix, {g.bad_lines} bad lines")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["imu", "camera", "gps", "tap"])
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--configure", action="store_true", help="gps: send the UBX setup first")
    p.add_argument("--config", default=str(_common.DEFAULT_CONFIG))
    a = p.parse_args()
    _common.setup_logging("WARNING")
    cfg = load(a.config)
    if a.what == "imu":
        imu(cfg, a.seconds)
    elif a.what == "tap":
        imu(cfg, a.seconds, tap=True)
    elif a.what == "camera":
        camera(cfg, a.seconds)
    else:
        gps(cfg, a.seconds, a.configure)


if __name__ == "__main__":
    main()
