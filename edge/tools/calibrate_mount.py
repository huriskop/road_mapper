#!/usr/bin/env python3
"""Mount calibration: 10 s parked on level ground -> gravity direction, biases, noise.

    python tools/calibrate_mount.py [--seconds 10] [--config config.yaml]

Run once per car and mount, engine off (or idling), nobody moving in the car.
"""
import argparse
import time

import _common  # noqa: F401
import numpy as np

from roadflag_edge.config import load
from roadflag_edge.imu.mpu6050 import G, MPU6050
from roadflag_edge.imu.orientation import MountCalibration


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(_common.DEFAULT_CONFIG))
    p.add_argument("--seconds", type=float, default=10.0)
    a = p.parse_args()
    cfg = load(a.config)
    mpu = MPU6050(cfg.imu.bus, cfg.imu.addr, cfg.imu.rate_hz, cfg.imu.dlpf)
    print(f"Keep the car still for {a.seconds:.0f} s ...")
    acc, gyro = [], []
    end = time.monotonic() + a.seconds
    time.sleep(0.2)
    mpu.reset_fifo()
    while time.monotonic() < end:
        time.sleep(0.05)
        b = mpu.read()
        if b is not None:
            acc.append(b.acc)
            gyro.append(b.gyro)
    acc, gyro = np.concatenate(acc), np.concatenate(gyro)
    cal = MountCalibration.from_parked(acc, gyro)
    out = cfg.path(cfg.mount_file)
    cal.save(out)
    mag = np.linalg.norm(acc.mean(axis=0))
    tilt = np.degrees(np.arccos(np.clip(cal.up_body[2], -1, 1)))
    print(f"samples        {len(acc)} ({len(acc) / a.seconds:.1f} Hz), FIFO overflows {mpu.overflows}")
    print(f"|a| parked     {mag:.3f} m/s^2 (g = {G:.3f}; plan wants 9.81 +- 0.1)")
    print(f"up in sensor   [{cal.up_body[0]:+.3f} {cal.up_body[1]:+.3f} {cal.up_body[2]:+.3f}]  tilt {tilt:.1f} deg from z")
    print(f"accel noise    {cal.accel_noise_std:.4f} m/s^2 std along up")
    print(f"gyro bias      {np.degrees(cal.gyro_bias).round(3)} deg/s")
    print(f"saved          {out}")
    mpu.close()


if __name__ == "__main__":
    main()
