"""E2 - MPU6050 driver: 200 Hz accel + gyro through the FIFO, with accurate per-sample times.

Polling registers from Python jitters, so the chip samples on its own clock into its
1024-byte FIFO and we drain it every ~50 ms. Each sample's time is its index on a fitted
time base (see SampleClock), because the chip's oscillator drifts about 1 %.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

G = 9.80665

# registers
SMPLRT_DIV, CONFIG, GYRO_CONFIG, ACCEL_CONFIG = 0x19, 0x1A, 0x1B, 0x1C
FIFO_EN, INT_STATUS, USER_CTRL, PWR_MGMT_1 = 0x23, 0x3A, 0x6A, 0x6B
FIFO_COUNT_H, FIFO_R_W, WHO_AM_I = 0x72, 0x74, 0x75

SAMPLE_BYTES = 12            # accel xyz + gyro xyz, int16 big-endian each (temp not in FIFO)
FIFO_SIZE = 1024
ACCEL_LSB_PER_G = 4096.0     # +-8 g
GYRO_LSB_PER_DPS = 65.5      # +-500 dps
OFLOW_BIT = 0x10


@dataclass
class ImuBatch:
    """Consecutive samples of one gap-free segment. Units: s-time ns, m/s^2, rad/s."""
    t_ns: np.ndarray          # int64, shape (n,)
    acc: np.ndarray           # float64, (n, 3) specific force
    gyro: np.ndarray          # float64, (n, 3)
    segment: int              # increments after every FIFO reset; never stitch across segments

    def __len__(self) -> int:
        return len(self.t_ns)


class SampleClock:
    """Maps sample index -> monotonic time.

    Each drain gives (index of the last sample read, time of the read). The read always
    happens after the sample was taken, so the true time base is the *lower envelope* of
    those points: slope from a least-squares fit over the last minute, intercept from the
    minimum residual. This absorbs Python scheduling jitter (always late, never early).
    """

    def __init__(self, nominal_period_ns: int, window: int = 1200) -> None:
        self.nominal = nominal_period_ns
        self.points: deque[tuple[int, int]] = deque(maxlen=window)  # (index, read time)

    def reset(self) -> None:
        self.points.clear()

    def add(self, last_index: int, read_ns: int) -> None:
        self.points.append((last_index, read_ns))

    def period_ns(self) -> float:
        if len(self.points) < 20:
            return float(self.nominal)
        k = np.fromiter((p[0] for p in self.points), dtype=np.float64)
        t = np.fromiter((p[1] for p in self.points), dtype=np.float64)
        if k[-1] - k[0] < 400:  # under 2 s of data: the fit is noisier than the chip
            return float(self.nominal)
        slope = np.polyfit(k - k[0], t - t[0], 1)[0]
        # the chip is within a few % of nominal; anything else is a fitting artefact
        return float(np.clip(slope, self.nominal * 0.95, self.nominal * 1.05))

    def times(self, first_index: int, n: int) -> np.ndarray:
        period = self.period_ns()
        k0, t0 = self.points[0]
        base = min(t - (k - k0) * period for k, t in self.points)  # lower envelope at k0
        idx = np.arange(first_index, first_index + n, dtype=np.float64)
        return (base + (idx - k0) * period).astype(np.int64)


class MPU6050:
    def __init__(self, bus: int = 1, addr: int = 0x68, rate_hz: int = 200, dlpf: int = 2) -> None:
        from smbus2 import SMBus  # imported here so tests run without the library

        self.addr = addr
        self.bus = SMBus(bus)
        self.rate_hz = rate_hz
        self.dlpf = dlpf
        self.clock = SampleClock(int(1e9 / rate_hz))
        self.segment = 0
        self.index = 0             # samples read in the current segment
        self.overflows = 0
        self.i2c_errors = 0
        self._partial = b""
        self.init_chip()

    # ------------------------------------------------------------------ setup
    def _w(self, reg: int, val: int) -> None:
        self.bus.write_byte_data(self.addr, reg, val)

    def _r(self, reg: int) -> int:
        return self.bus.read_byte_data(self.addr, reg)

    def init_chip(self) -> None:
        who = self._r(WHO_AM_I)
        if who not in (0x68, 0x98, 0x70, 0x72):
            log.warning("MPU6050 WHO_AM_I=0x%02x (clone?)", who)
        self._w(PWR_MGMT_1, 0x80)               # device reset
        time.sleep(0.1)
        self._w(PWR_MGMT_1, 0x01)               # wake, clock = PLL on gyro X
        time.sleep(0.05)
        self._w(CONFIG, self.dlpf & 0x07)       # DLPF 2 -> ~94 Hz, gyro rate 1 kHz
        self._w(SMPLRT_DIV, int(1000 / self.rate_hz) - 1)   # 4 -> 200 Hz
        self._w(GYRO_CONFIG, 0x08)              # +-500 dps
        self._w(ACCEL_CONFIG, 0x10)             # +-8 g (pothole hits clip at +-2 g)
        self.reset_fifo()

    def reset_fifo(self) -> None:
        self._w(FIFO_EN, 0x00)
        self._w(USER_CTRL, 0x04)                # FIFO_RESET
        time.sleep(0.005)
        self._w(USER_CTRL, 0x40)                # FIFO enable
        self._w(FIFO_EN, 0x78)                  # accel + gyro xyz
        self._r(INT_STATUS)                     # clear overflow flag
        self.segment += 1
        self.index = 0
        self._partial = b""
        self.clock.reset()

    # ------------------------------------------------------------------- read
    def _read_block(self, n: int) -> bytes:
        from smbus2 import i2c_msg

        out = bytearray()
        while n > 0:                            # i2c-dev handles large reads; chunk to be safe
            chunk = min(n, 252)
            w, r = i2c_msg.write(self.addr, [FIFO_R_W]), i2c_msg.read(self.addr, chunk)
            self.bus.i2c_rdwr(w, r)
            out += bytes(r)
            n -= chunk
        return bytes(out)

    def read(self) -> ImuBatch | None:
        """Drain the FIFO. Returns None if there was nothing (or an overflow forced a reset)."""
        try:
            if self._r(INT_STATUS) & OFLOW_BIT:
                self.overflows += 1
                log.warning("MPU6050 FIFO overflow #%d: resetting, gap marked", self.overflows)
                self.reset_fifo()
                return None
            hi, lo = self.bus.read_i2c_block_data(self.addr, FIFO_COUNT_H, 2)
            count = (hi << 8) | lo
            read_ns = time.monotonic_ns()
            if count >= FIFO_SIZE - SAMPLE_BYTES:   # about to overflow: data may be torn
                self.overflows += 1
                self.reset_fifo()
                return None
            count -= count % SAMPLE_BYTES
            if count == 0:
                return None
            raw = self._read_block(count)
        except OSError as exc:
            self.i2c_errors += 1
            log.warning("I2C error (%s); reinitialising", exc)
            time.sleep(0.05)
            try:
                self.init_chip()
            except OSError:
                time.sleep(0.5)
            return None
        return self.decode(raw, read_ns)

    def decode(self, raw: bytes, read_ns: int) -> ImuBatch:
        v = np.frombuffer(raw, dtype=">i2").reshape(-1, 6).astype(np.float64)
        n = len(v)
        first = self.index
        self.index += n
        self.clock.add(self.index - 1, read_ns)
        t = self.clock.times(first, n)
        acc = v[:, 0:3] * (G / ACCEL_LSB_PER_G)
        gyro = np.deg2rad(v[:, 3:6] / GYRO_LSB_PER_DPS)
        return ImuBatch(t, acc, gyro, self.segment)

    def close(self) -> None:
        try:
            self._w(FIFO_EN, 0)
            self.bus.close()
        except OSError:
            pass
