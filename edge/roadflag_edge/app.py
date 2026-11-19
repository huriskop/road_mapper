"""E8 - supervisor: start every thread, print a health line, run under systemd.

    IMU reader (200 Hz, FIFO drained every 50 ms) -> IMU ring + detector queue
    GPS reader (5 Hz)                             -> trajectory + clock
    camera callback (24 FPS)                      -> frame ring (5 s)
    detector                                      -> candidate queue
    evidence worker                               -> observation store (SQLite + JPEG)
    uploader                                      -> server

All queues are bounded. The IMU reader never blocks: if the detector falls behind its batch
is dropped and counted; if the evidence worker falls behind, the lowest-score candidate is.

    python -m roadflag_edge.app --config config.yaml [--no-camera] [--no-gps] [--no-upload]
"""
from __future__ import annotations

import argparse
import json
import logging
import queue
import signal
import threading
import time
from pathlib import Path

import numpy as np

from . import __version__, pipeline
from .camera.ring import FrameRing, PiCameraSource
from .clock import Clock
from .config import Config, load
from .detect.detector import Candidate, Detector
from .detect.localize import ImuRing
from .gps.neo6m import Neo6m
from .gps.trajectory import Trajectory
from .imu.orientation import GravityTracker, MountCalibration
from .store.observations import ObservationStore, new_observation_id
from .uplink.uploader import Uploader

log = logging.getLogger("roadflag")
SEC = 1_000_000_000


class CandidateQueue:
    """Bounded; when full, the lowest-score candidate (new or queued) is dropped and counted."""

    def __init__(self, maxsize: int = 8) -> None:
        self.maxsize = maxsize
        self.items: list[Candidate] = []
        self.cv = threading.Condition()
        self.dropped = 0

    def offer(self, c: Candidate) -> None:
        with self.cv:
            self.items.append(c)
            if len(self.items) > self.maxsize:
                worst = min(self.items, key=lambda x: x.score)
                self.items.remove(worst)
                self.dropped += 1
                log.warning("evidence worker behind: dropped candidate with score %.2f", worst.score)
            self.cv.notify()

    def get(self, timeout: float) -> Candidate | None:
        with self.cv:
            if not self.items:
                self.cv.wait(timeout)
            return self.items.pop(0) if self.items else None

    def __len__(self) -> int:
        return len(self.items)


class EdgeApp:
    def __init__(self, cfg: Config, camera: bool = True, gps: bool = True, upload: bool = True) -> None:
        self.cfg = cfg
        self.stop = threading.Event()
        data = cfg.path(cfg.data_dir)
        data.mkdir(parents=True, exist_ok=True)
        self.data = data
        self.clock = Clock()
        self.traj = Trajectory(cfg.gps.keep_s, cfg.gps.max_gap_s, cfg.gps.max_hdop)
        self.imu_ring = ImuRing(cfg.imu.ring_seconds, cfg.imu.rate_hz)
        self.cal = MountCalibration.load(cfg.path(cfg.mount_file))
        if not self.cal.samples:
            log.warning("no mount calibration (%s): assuming sensor z axis is up; run tools/calibrate_mount.py",
                        cfg.mount_file)
        self.det_q: queue.Queue = queue.Queue(maxsize=40)      # ~2 s of 50 ms batches
        self.cand_q = CandidateQueue()
        self.detector = Detector(cfg.detector)
        self.store = ObservationStore(data / "store", int(cfg.budget_mb * 1024 * 1024))
        self.use_camera, self.use_gps, self.use_upload = camera and cfg.camera.enabled, gps and cfg.gps.enabled, \
            upload and cfg.uplink.enabled
        self.ring = None
        self.cam = None
        if self.use_camera:
            cap = int(round(cfg.camera.ring_seconds * cfg.camera.fps))
            self.ring = FrameRing(cap, cfg.camera.width, cfg.camera.height)
            self.cam = PiCameraSource(self.ring, cfg.camera.width, cfg.camera.height, cfg.camera.fps,
                                      cfg.camera.exposure_us, cfg.camera.frame_time_offset_ms)
        self.uploader = None
        if self.use_upload:
            token = cfg.device_token()
            if not token:
                log.error("no device token (uplink.token / uplink.token_file): uploads will be rejected")
            self.uploader = Uploader(self.store, cfg.uplink.server_url, token, cfg.uplink.timeout_s,
                                     cfg.uplink.lease_s)
        # counters for the health line
        self.imu_samples = 0
        self.imu_dropped_batches = 0
        self.mpu = None
        self.gps = None
        self.candidates = 0
        self.observations = 0
        self.by_status: dict[str, int] = {}

    # ---------------------------------------------------------------- threads
    def imu_loop(self) -> None:
        from .imu.mpu6050 import MPU6050

        c = self.cfg.imu
        self.mpu = MPU6050(c.bus, c.addr, c.rate_hz, c.dlpf)
        tracker = GravityTracker(self.cal, c.rate_hz, c.attitude_tau_s)
        period = c.read_interval_ms / 1000
        nxt = time.monotonic()
        while not self.stop.is_set():
            batch = self.mpu.read()
            if batch is not None and len(batch):
                az, horiz = tracker.update(batch.acc, batch.gyro)
                gyro_mag = np.linalg.norm(batch.gyro, axis=1)
                self.imu_ring.push(batch.t_ns, az, horiz, gyro_mag, batch.segment)
                self.imu_samples += len(batch)
                try:
                    self.det_q.put_nowait((batch.segment, batch.t_ns, az))
                except queue.Full:
                    self.imu_dropped_batches += 1
            nxt += period
            delay = nxt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.monotonic()
        self.mpu.close()

    def detector_loop(self) -> None:
        seg = None
        while not self.stop.is_set():
            try:
                s, t, az = self.det_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if s != seg:
                self.detector.reset()
                seg = s
            for cand in self.detector.process(t, az, self.traj.current_speed()):
                self.candidates += 1
                self.cand_q.offer(cand)

    def _wait(self, cond, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while not cond():
            if time.monotonic() > end or self.stop.is_set():
                return False
            time.sleep(0.02)
        return True

    def evidence_loop(self) -> None:
        cfg = self.cfg
        while not self.stop.is_set():
            cand = self.cand_q.get(timeout=0.5)
            if cand is None:
                continue
            try:
                self._wait(lambda: (self.imu_ring.newest_ns() or 0) >= cand.t_d + SEC // 2, 2.0)
                loc = pipeline.localize_candidate(cand, self.imu_ring, cfg.detector.fs)
                t_i = loc["t_i"] if loc else cand.t_d
                if self.use_gps:   # position at t_i needs the fix after it
                    self._wait(lambda: (self.traj.newest_ns() or 0) >= t_i, 1.5)
                built = pipeline.build(cand, self.imu_ring, self.traj, self.ring, cfg, loc=loc)
                cand.done.wait(3.0)                       # rear-axle verdict is final
                obs_id = new_observation_id()
                meta = pipeline.metadata(obs_id, built, self.clock, cfg)
                ev = built.evidence
                image = ev.frame.to_jpeg(cfg.camera.jpeg_quality) if ev.frame is not None else None
                extras = {f"n{i + 1}.jpg": f.to_jpeg(80) for i, f in enumerate(ev.neighbours)}
                extras["imu.npz"] = pipeline.snippet_bytes(built.snippet)
                self.store.persist(obs_id, meta, image, extras, score=cand.score)
                self.observations += 1
                self.by_status[ev.status.value] = self.by_status.get(ev.status.value, 0) + 1
                log.info("observation %s: %s score=%.2f peak=%.1f m/s^2 v=%s %s", obs_id[:8], ev.status.value,
                         cand.score, meta["imu"]["peak_az_mps2"] or 0, meta["speed_before_mps"], ev.detail)
                if self.uploader:
                    self.uploader.wake.set()
            except Exception:
                log.exception("evidence worker failed on a candidate")

    def gps_loop(self) -> None:
        c = self.cfg.gps
        while not self.stop.is_set():
            try:
                self.gps = Neo6m(c.port, c.baud, c.rate_hz, c.nmea_latency_ms)
                if c.configure_on_start:
                    try:
                        self.gps.configure()
                    except RuntimeError as exc:
                        log.warning("GPS configure failed (%s); using it as it is", exc)
                        self.gps.open_port()
                else:
                    self.gps.open_port()
                for fix in self.gps.fixes():
                    self.traj.add(fix)
                    if fix.valid and fix.utc is not None:
                        self.clock.add_gps_fix(fix.t_ns, fix.utc)
                    if self.stop.is_set():
                        break
            except Exception as exc:
                log.warning("GPS reader error (%s); retrying in 3 s", exc)
                time.sleep(3)

    def health(self) -> dict:
        now = time.monotonic()
        dt = now - self._h_last
        f = self.traj.last_fix
        h = {
            "imu_hz": round((self.imu_samples - self._h_imu) / dt, 1),
            "fifo_overflows": self.mpu.overflows if self.mpu else None,
            "i2c_errors": self.mpu.i2c_errors if self.mpu else None,
            "imu_dropped_batches": self.imu_dropped_batches,
            "camera_fps": round((self.ring.pushed - self._h_frames) / dt, 1) if self.ring else None,
            "frames_dropped": self.ring.dropped if self.ring else None,
            "frame_spacing_ms": self.ring.frame_spacing_ms() if self.ring else None,
            "camera_restarts": self.cam.restarts if self.cam else None,
            "gps_fix": (f.quality if f else None), "sats": (f.sats if f else None),
            "hdop": (f.hdop if f else None), "speed_mps": self.traj.current_speed(),
            "clock_spread_ms": self.clock.spread_ms(),
            "candidates": self.candidates, "candidates_dropped": self.cand_q.dropped,
            "parked_ignored": self.detector.dropped_parked,
            "observations": self.observations, "by_status": self.by_status,
            "store": self.store.counts(),
            "uploaded": self.uploader.sent if self.uploader else None,
            "upload_error": self.uploader.last_error if self.uploader else None,
            "auth_alarm": self.uploader.auth_alarm if self.uploader else None,
            "free_disk_mb": round(self.store.free_disk_mb()),
            "temp_c": _cpu_temp(),
        }
        self._h_last, self._h_imu = now, self.imu_samples
        self._h_frames = self.ring.pushed if self.ring else 0
        return h

    def health_loop(self) -> None:
        self._h_last, self._h_imu, self._h_frames = time.monotonic(), 0, 0
        while not self.stop.wait(self.cfg.health_interval_s):
            h = self.health()
            log.info("health %s", json.dumps(h, separators=(",", ":")))
            try:
                (self.data / "health.json").write_text(json.dumps({"t": time.time(), **h}, indent=1))
            except OSError:
                pass

    # ------------------------------------------------------------------- main
    def run(self, duration_s: float | None = None) -> None:
        rec = self.store.recover()
        if any(rec.values()):
            log.info("store recovery: %s", rec)
        threads = [("imu", self.imu_loop), ("detector", self.detector_loop), ("evidence", self.evidence_loop),
                   ("health", self.health_loop)]
        if self.use_gps:
            threads.append(("gps", self.gps_loop))
        if self.cam:
            threads.append(("camera", self.cam.run))
        if self.uploader:
            threads.append(("uploader", self.uploader.run))
        for name, fn in threads:
            threading.Thread(target=fn, name=name, daemon=True).start()
        log.info("roadflag edge %s running: %s", __version__, ", ".join(n for n, _ in threads))
        end = time.monotonic() + duration_s if duration_s else None
        while not self.stop.wait(0.5):
            if end and time.monotonic() > end:
                break
        self.shutdown()

    def shutdown(self) -> None:
        self.stop.set()
        if self.cam:
            self.cam.stop()
        if self.uploader:
            self.uploader.stop()
        time.sleep(0.5)
        log.info("stopped")


def _cpu_temp() -> float | None:
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except (OSError, ValueError):
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    p.add_argument("--no-camera", action="store_true")
    p.add_argument("--no-gps", action="store_true")
    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--duration", type=float, help="stop after N seconds (bench tests)")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(threadName)s: %(message)s")
    app = EdgeApp(load(a.config), camera=not a.no_camera, gps=not a.no_gps, upload=not a.no_upload)
    signal.signal(signal.SIGTERM, lambda *_: app.stop.set())
    try:
        app.run(a.duration)
    except KeyboardInterrupt:
        app.shutdown()


if __name__ == "__main__":
    main()
