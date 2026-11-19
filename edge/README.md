# Edge device (Raspberry Pi 4 + MPU6050 + NEO-6M + camera)

Implements E1–E8 of *Road Defect Flagger — Implementation Plan*. One Python process, one
thread per sensor, every sample stamped with `time.monotonic_ns()` when it was sensed.

| Module | File |
|---|---|
| E1 clock | `roadflag_edge/clock.py` |
| E2 IMU (FIFO, 200 Hz, ±8 g) + attitude | `imu/mpu6050.py`, `imu/orientation.py` |
| E3 GPS (5 Hz, 38400 baud) + trajectory | `gps/neo6m.py`, `gps/trajectory.py` |
| E4 5 s camera ring (YUV420) | `camera/ring.py` |
| E5 trigger + zero-phase localisation | `detect/detector.py`, `detect/localize.py` |
| E6 4 m look-back frame selection | `evidence/select.py` |
| E7 crash-safe store (SQLite WAL + fsync'd files) | `store/observations.py` |
| E8 uploader (lease, backoff, idempotency) + supervisor | `uplink/uploader.py`, `app.py` |

The upload format is `shared/observation.schema.json`; the server side is
`POST /api/v1/observations` in `backend/`.

## Install on the Pi

```bash
# from the PC: copy the code
tar --exclude=__pycache__ -cf - edge shared | ssh pi 'mkdir -p ~/roadflag && cd ~/roadflag && tar xf -'
# on the Pi, once
cd ~/roadflag && python3 -m venv --system-site-packages .venv && .venv/bin/pip install scipy
echo '<device token>' > ~/roadflag/device_token        # same token as RDM_DEVICE_TOKENS on the server
```

Recommended (needs sudo, then reboot): I2C at 400 kHz instead of 100 kHz, so a FIFO drain takes
~4 ms instead of ~15 ms: add `dtparam=i2c_arm_baudrate=400000` to `/boot/firmware/config.txt`.

## Bench checks (M1–M4)

```bash
cd ~/roadflag/edge
PY=../.venv/bin/python
$PY tools/calibrate_mount.py                      # 10 s still -> data/mount.json
$PY tools/bench_check.py imu --seconds 3600       # M1: 200.0 ± 0.5 Hz, 0 FIFO overflows
$PY tools/bench_check.py tap                      # tap the sensor: one TAP line per tap
$PY tools/bench_check.py gps --configure          # M2: UBX setup (38400 baud, 5 Hz, saved)
$PY tools/bench_check.py camera --seconds 3600    # M3: 24 FPS, 41.7 ± 1 ms, flat memory
$PY tools/latency_led_test.py                     # M4: LED on GPIO17 in view, |error| < 10 ms
$PY tools/calibrate_lookahead.py                  # d_l: guide image, then --set <metres>
```

After `bench_check.py gps --configure` the module stays at 38400 baud, so the older
`sensor test/test_gps.py` needs `--baud 38400`.

## Drives (M5–M6)

```bash
$PY tools/record_drive.py                 # Enter = mark a pothole/bump (type a label first)
$PY tools/replay.py ~/roadflag/drives/<folder> [--set detector.tau_a0=5]
```

`replay.py` prints every detection with its evidence status, marks found / missed, unmarked
detections per km, the 99.9th-percentile A and J per speed band (the suggested thresholds), and
writes `replay/contact_sheet.jpg` to check that the defect sits in the lower-middle of each frame.

## Run

```bash
$PY -m roadflag_edge.app                  # or: --no-camera --no-gps --no-upload --duration 60
sudo cp systemd/roadflag-edge.service /etc/systemd/system/ && sudo systemctl enable --now roadflag-edge
```

A health line is logged every 10 s (also `data/health.json`): IMU rate, FIFO overflows, camera
FPS, dropped frames, GPS fix, queue depth, free disk, CPU temperature.

## Tests (no hardware)

```bash
python -m pytest edge/tests
```
