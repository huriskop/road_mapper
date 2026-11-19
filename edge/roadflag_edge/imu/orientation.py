"""E2 - mount calibration and attitude: rotate specific force into the vertical.

The MPU6050 measures specific force f_B. With the gravity direction u (unit "up" in the
sensor frame) the vertical acceleration is

    a_z = (f_B - b_a) . u - g

which is the z row of a_W = R(f_B - b_a) + g_W from the plan; yaw never matters for it.
u starts from calibrate_mount.py (10 s parked) and is then tracked by a complementary
filter: the gyro rotates it quickly, a slow pull toward the low-passed accelerometer
direction follows road grade and mount sag. |f| - g is kept only as a debug feature.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .mpu6050 import G


@dataclass
class MountCalibration:
    up_body: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])   # unit gravity-up in sensor frame
    accel_bias: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # along-up error removed
    gyro_bias: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])   # rad/s
    accel_noise_std: float | None = None
    samples: int = 0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "MountCalibration":
        if path.exists():
            return cls(**json.loads(path.read_text()))
        return cls()

    @classmethod
    def from_parked(cls, acc: np.ndarray, gyro: np.ndarray) -> "MountCalibration":
        """acc, gyro: (n, 3) arrays recorded while parked on level ground."""
        mean = acc.mean(axis=0)
        mag = float(np.linalg.norm(mean))
        if mag < 5 or mag > 15:
            raise ValueError(f"|a| = {mag:.2f} m/s^2 while parked; is the sensor connected and still?")
        up = mean / mag
        # Scale error along up: report |a| as exactly g when parked.
        bias = up * (mag - G)
        noise = float(np.std(acc @ up))
        return cls(up.tolist(), bias.tolist(), gyro.mean(axis=0).tolist(), noise, len(acc))


class GravityTracker:
    def __init__(self, cal: MountCalibration, fs: float = 200.0, tau_s: float = 5.0) -> None:
        self.u = np.asarray(cal.up_body, dtype=np.float64)
        self.u /= np.linalg.norm(self.u)
        self.bias = np.asarray(cal.accel_bias, dtype=np.float64)
        self.gyro_bias = np.asarray(cal.gyro_bias, dtype=np.float64)
        self.dt = 1.0 / fs
        self.alpha = self.dt / tau_s   # accel pull per sample
        self.calibrated = cal.samples > 0

    def update(self, acc: np.ndarray, gyro: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (az_w vertical accel without gravity, horizontal accel magnitude), each (n,)."""
        # Plain floats: per-sample numpy calls cost ~10x more on the Pi.
        f = (acc - self.bias).tolist()
        w = (gyro - self.gyro_bias).tolist()
        n = len(f)
        az = [0.0] * n
        horiz = [0.0] * n
        ux, uy, uz = self.u.tolist()
        dt, alpha, lo, hi = self.dt, self.alpha, 0.5 * G, 1.5 * G
        for k in range(n):
            fx, fy, fz = f[k]
            wx, wy, wz = w[k]
            # gravity is fixed in the world, so in the body frame du/dt = u x w
            ux, uy, uz = (ux + (uy * wz - uz * wy) * dt,
                          uy + (uz * wx - ux * wz) * dt,
                          uz + (ux * wy - uy * wx) * dt)
            norm = (fx * fx + fy * fy + fz * fz) ** 0.5
            if lo < norm < hi:               # only trust the accelerometer when it's not a hit
                ux += alpha * (fx / norm - ux)
                uy += alpha * (fy / norm - uy)
                uz += alpha * (fz / norm - uz)
            un = (ux * ux + uy * uy + uz * uz) ** 0.5
            ux, uy, uz = ux / un, uy / un, uz / un
            v = fx * ux + fy * uy + fz * uz
            az[k] = v - G
            hx, hy, hz = fx - v * ux, fy - v * uy, fz - v * uz
            horiz[k] = (hx * hx + hy * hy + hz * hz) ** 0.5
        self.u = np.array([ux, uy, uz])
        return np.asarray(az), np.asarray(horiz)


def magnitude_minus_g(acc: np.ndarray) -> np.ndarray:
    """Debug feature only: mixes braking, turning and bumps."""
    return np.linalg.norm(acc, axis=1) - G
