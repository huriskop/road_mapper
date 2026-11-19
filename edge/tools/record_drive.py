#!/usr/bin/env python3
"""M5 - record a drive: raw IMU, raw GPS, every frame's timestamp, and a JPEG log.

    python tools/record_drive.py [--out ~/roadflag/drives] [--image-fps 4] [--button-pin 27]

Press Enter (or the button) when the car hits a pothole/bump you want to mark; type a word
before Enter to label it (e.g. "pothole", "bump"). Ctrl+C stops. Output folder:

    imu.csv      t_ns,seg,ax,ay,az,gx,gy,gz       (m/s^2, rad/s, raw sensor frame)
    gps.nmea     t_ns<TAB>sentence                (arrival time, monotonic)
    frames.csv   t_ns,seq                         (every frame's mid-exposure time)
    images/      <t_ns>.jpg at --image-fps        (for replay's contact sheet)
    marks.csv    t_ns,label
    mount.json, config.yaml                       (copied, so replay uses the same setup)
"""
import argparse
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import _common  # noqa: F401
import numpy as np

from roadflag_edge.camera.ring import FrameRing, PiCameraSource
from roadflag_edge.config import load
from roadflag_edge.imu.mpu6050 import MPU6050


class Recorder:
    def __init__(self, cfg, out: Path, image_fps: float) -> None:
        self.cfg, self.out = cfg, out
        (out / "images").mkdir(parents=True, exist_ok=True)
        self.stop = threading.Event()
        self.image_period = 1e9 / image_fps if image_fps > 0 else None
        self.marks = open(out / "marks.csv", "a", buffering=1)
        if self.marks.tell() == 0:
            self.marks.write("t_ns,label\n")
        self.n_imu = self.n_gps = self.n_frames = self.n_images = 0

    def mark(self, label: str = "") -> None:
        t = time.monotonic_ns()
        self.marks.write(f"{t},{label.replace(',', ' ')}\n")
        print(f"  MARK {label or '(no label)'}", flush=True)

    def imu(self) -> None:
        c = self.cfg.imu
        mpu = MPU6050(c.bus, c.addr, c.rate_hz, c.dlpf)
        with open(self.out / "imu.csv", "w", buffering=1 << 16) as fh:
            fh.write("t_ns,seg,ax,ay,az,gx,gy,gz\n")
            nxt = time.monotonic()
            while not self.stop.is_set():
                nxt += c.read_interval_ms / 1000
                time.sleep(max(0.0, nxt - time.monotonic()))
                b = mpu.read()
                if b is None:
                    continue
                for t, a, g in zip(b.t_ns.tolist(), b.acc.tolist(), b.gyro.tolist()):
                    fh.write(f"{t},{b.segment},{a[0]:.4f},{a[1]:.4f},{a[2]:.4f},{g[0]:.5f},{g[1]:.5f},{g[2]:.5f}\n")
                self.n_imu += len(b)
        self.overflows = mpu.overflows
        mpu.close()

    def gps(self) -> None:
        from roadflag_edge.gps.neo6m import Neo6m

        c = self.cfg.gps
        g = Neo6m(c.port, c.baud, c.rate_hz, c.nmea_latency_ms)
        g.open_port()
        with open(self.out / "gps.nmea", "w", buffering=1) as fh:
            for line, t in g.lines():
                if self.stop.is_set():
                    break
                fh.write(f"{t}\t{line.strip()}\n")
                self.n_gps += 1

    def camera(self) -> None:
        c = self.cfg.camera
        ring = FrameRing(int(c.ring_seconds * c.fps), c.width, c.height)
        src = PiCameraSource(ring, c.width, c.height, c.fps, c.exposure_us, c.frame_time_offset_ms)
        threading.Thread(target=src.run, daemon=True).start()
        last_seen, last_img = 0, 0
        with open(self.out / "frames.csv", "w", buffering=1 << 14) as fh:
            fh.write("t_ns,seq\n")
            while not self.stop.is_set():
                time.sleep(0.1)
                frames = ring.frames_between(last_seen + 1, 1 << 62)
                for f in frames:
                    fh.write(f"{f.t_ns},{f.seq}\n")
                    self.n_frames += 1
                    if self.image_period and f.t_ns - last_img >= self.image_period:
                        (self.out / "images" / f"{f.t_ns}.jpg").write_bytes(f.to_jpeg(80))
                        last_img = f.t_ns
                        self.n_images += 1
                if frames:
                    last_seen = frames[-1].t_ns
        src.stop()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(_common.DEFAULT_CONFIG))
    p.add_argument("--out", default="~/roadflag/drives")
    p.add_argument("--image-fps", type=float, default=4.0)
    p.add_argument("--button-pin", type=int, help="GPIO (BCM) of a push button to GND for marks")
    p.add_argument("--no-camera", action="store_true")
    p.add_argument("--no-gps", action="store_true")
    a = p.parse_args()
    cfg = load(a.config)
    out = Path(a.out).expanduser() / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    shutil.copy(a.config, out / "config.yaml")
    mount = cfg.path(cfg.mount_file)
    if mount.exists():
        shutil.copy(mount, out / "mount.json")
    else:
        print("WARNING: no mount.json - run tools/calibrate_mount.py first", file=sys.stderr)
    rec = Recorder(cfg, out, a.image_fps)
    workers = [rec.imu] + ([] if a.no_gps else [rec.gps]) + ([] if a.no_camera else [rec.camera])
    for fn in workers:
        threading.Thread(target=fn, daemon=True, name=fn.__name__).start()
    if a.button_pin is not None:
        from gpiozero import Button

        btn = Button(a.button_pin, bounce_time=0.2)
        btn.when_pressed = lambda: rec.mark("button")
    print(f"Recording to {out}\nEnter = mark (type a label first if you like), Ctrl+C = stop")

    def status() -> None:
        while not rec.stop.wait(10):
            print(f"  imu {rec.n_imu}  gps lines {rec.n_gps}  frames {rec.n_frames}  images {rec.n_images}", flush=True)

    threading.Thread(target=status, daemon=True).start()
    try:
        if sys.stdin.isatty():
            for line in sys.stdin:
                rec.mark(line.strip())
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    rec.stop.set()
    time.sleep(0.5)
    print(f"\nSaved {out}: imu {rec.n_imu}, gps lines {rec.n_gps}, frames {rec.n_frames}, images {rec.n_images}")
    print(f"Replay with: python tools/replay.py {out}")


if __name__ == "__main__":
    main()
