"""E4 - the last 5 s of camera frames, in RAM, with exposure-time stamps.

Raw YUV420 slots (not an H.264 circular buffer) so one exact frame can be pulled out
without decoding. Slots are preallocated; a sorted index of exposure times is kept
alongside, so a lookup is a bisect, not a rebuild. Nothing goes to disk unless asked.
"""
from __future__ import annotations

import bisect
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Frame:
    t_ns: int                 # mid-exposure, monotonic
    seq: int
    yuv: np.ndarray           # (h*3/2, w) uint8, I420

    def to_jpeg(self, quality: int = 90) -> bytes:
        import cv2

        bgr = cv2.cvtColor(self.yuv, cv2.COLOR_YUV2BGR_I420)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return buf.tobytes()


class FrameRing:
    def __init__(self, capacity: int, width: int, height: int) -> None:
        self.capacity = capacity
        self.shape = (height * 3 // 2, width)
        self.slots = np.zeros((capacity, *self.shape), dtype=np.uint8)
        self._t: list[int] = []          # sorted exposure times of the frames held
        self._slot: list[int] = []       # slot index for each entry of _t
        self._seq: list[int] = []
        self._next = 0
        self._lock = threading.Lock()
        self.pushed = 0
        self.dropped = 0                 # gaps in sequence numbers (camera dropped frames)
        self._last_seq: int | None = None

    def push(self, t_ns: int, seq: int, yuv: np.ndarray) -> None:
        with self._lock:
            if self._last_seq is not None and seq > self._last_seq + 1:
                self.dropped += seq - self._last_seq - 1
            self._last_seq = seq
            slot = self._next
            self._next = (self._next + 1) % self.capacity
            if len(self._t) == self.capacity:      # overwrite the oldest
                self._t.pop(0)
                self._slot.pop(0)
                self._seq.pop(0)
            h, w = self.shape
            self.slots[slot] = yuv[:h, :w]
            if self._t and t_ns < self._t[-1]:    # out of order (should not happen): keep sorted
                i = bisect.bisect_left(self._t, t_ns)
            else:
                i = len(self._t)
            self._t.insert(i, t_ns)
            self._slot.insert(i, slot)
            self._seq.insert(i, seq)
            self.pushed += 1

    def span(self) -> tuple[int, int] | None:
        with self._lock:
            return (self._t[0], self._t[-1]) if self._t else None

    def __len__(self) -> int:
        return len(self._t)

    def nearest(self, t_ns: int, neighbours: int = 1) -> tuple[Frame, list[Frame], int, bool] | None:
        """Copy out the frame closest to t_ns and `neighbours` on each side.

        Returns (frame, neighbour frames, error ns, at_oldest_edge) or None if empty.
        at_oldest_edge is True when the best match is the oldest frame held: it may show a
        different place than asked for (the right one was already overwritten)."""
        with self._lock:
            if not self._t:
                return None
            i = bisect.bisect_left(self._t, t_ns)
            if i == len(self._t) or (i > 0 and t_ns - self._t[i - 1] <= self._t[i] - t_ns):
                i -= 1
            best = self._copy(i)
            nb = [self._copy(j) for j in range(i - neighbours, i + neighbours + 1)
                  if j != i and 0 <= j < len(self._t)]
            return best, nb, self._t[i] - t_ns, i == 0 and len(self._t) == self.capacity

    def frames_between(self, t0: int, t1: int) -> list[Frame]:
        """Copies of all frames with t0 <= t <= t1."""
        with self._lock:
            lo, hi = bisect.bisect_left(self._t, t0), bisect.bisect_right(self._t, t1)
            return [self._copy(i) for i in range(lo, hi)]

    def _copy(self, i: int) -> Frame:
        return Frame(self._t[i], self._seq[i], self.slots[self._slot[i]].copy())

    def frame_spacing_ms(self) -> tuple[float, float] | None:
        with self._lock:
            if len(self._t) < 3:
                return None
            d = np.diff(np.asarray(self._t)) / 1e6
            return float(d.mean()), float(d.std())


class PiCameraSource:
    """picamera2 capture loop feeding a FrameRing. Frame time = SensorTimestamp + exposure/2."""

    def __init__(self, ring: FrameRing, width: int, height: int, fps: float = 24.0,
                 exposure_us: int | None = 1500, frame_time_offset_ms: float = 0.0,
                 stall_s: float = 3.0) -> None:
        self.ring = ring
        self.size = (width, height)
        self.fps = fps
        self.exposure_us = exposure_us
        self.offset_ns = int(frame_time_offset_ms * 1e6)
        self.stall_s = stall_s
        self.cam = None
        self.restarts = 0
        self.last_frame_mono = 0.0
        self._stop = threading.Event()
        self.running = False

    def _open(self) -> None:
        from picamera2 import Picamera2

        self.cam = Picamera2()
        frame_us = int(round(1e6 / self.fps))
        controls = {"FrameDurationLimits": (frame_us, frame_us)}
        if self.exposure_us:
            controls["ExposureTime"] = int(self.exposure_us)   # gain stays automatic
        cfg = self.cam.create_video_configuration(
            main={"size": self.size, "format": "YUV420"}, buffer_count=6, controls=controls)
        self.cam.configure(cfg)
        self.cam.start()
        self.last_frame_mono = time.monotonic() + 3.0     # grace period while the sensor starts

    def _close(self) -> None:
        if self.cam is not None:
            try:
                self.cam.stop()
                self.cam.close()
            except Exception:
                pass
            self.cam = None

    def run(self) -> None:
        seq = 0
        while not self._stop.is_set():
            try:
                if self.cam is None:
                    self._open()
                    self.running = True
                req = self.cam.capture_request(wait=self.stall_s)
                try:
                    md = req.get_metadata()
                    yuv = req.make_array("main")
                finally:
                    req.release()
                exp_ns = int(md.get("ExposureTime", self.exposure_us or 0)) * 1000
                t = int(md["SensorTimestamp"]) + exp_ns // 2 + self.offset_ns
                seq = int(md.get("FrameSequence", seq + 1))
                self.ring.push(t, seq, yuv)
                self.last_frame_mono = time.monotonic()
            except Exception as exc:      # stall / timeout / libcamera error -> restart
                if self._stop.is_set():
                    break
                self.running = False
                self.restarts += 1
                log.warning("camera problem (%s); restarting (#%d)", exc, self.restarts)
                self._close()
                time.sleep(1.0)
        self._close()

    def stop(self) -> None:
        self._stop.set()
