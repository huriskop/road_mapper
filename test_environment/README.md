# Equipment test bench (Raspberry Pi 4)

A web page for checking the camera, the MPU-6050 and the GPS before a drive.

| Page | URL | What it shows |
|---|---|---|
| Equipment check | `http://<pi-address>:8080/` | Camera on the left; a sphere that turns with the MPU-6050 (top right); GPS position, satellites and track (bottom right) |
| Auto capture | `http://<pi-address>:8080/capture` | Same panels plus a log: when the sensor jolts **down** (pothole) or **up** (bump) past the threshold, the camera takes a photo and it is added to the log |

Any device that cannot be opened is **simulated** and its badge says so, so the page also runs on a laptop:

```
python test_environment/server.py --sim      # Windows: py test_environment/server.py --sim
```

The server uses the Python standard library only. Hardware libraries are optional.

## Wiring

| MPU-6050 | Pi pin |
|---|---|
| VCC | 3.3 V (pin 1) |
| GND | GND (pin 6) |
| SDA | GPIO 2 / SDA (pin 3) |
| SCL | GPIO 3 / SCL (pin 5) |

| GPS (NEO-6M / 7M / 8M) | Pi pin |
|---|---|
| VCC | 5 V (pin 2) or 3.3 V, check your module |
| GND | GND (pin 9) |
| TX | GPIO 15 / RXD (pin 10) |
| RX | GPIO 14 / TXD (pin 8), optional |

Camera: Pi camera on the CSI ribbon, or a USB webcam.

Mount the MPU-6050 flat and rigidly, with its **X arrow pointing forward**.

## Set up the Pi (Raspberry Pi OS Bookworm, 64-bit)

```bash
sudo raspi-config          # Interface Options: I2C -> Yes; Serial Port: login shell -> No, hardware -> Yes
sudo reboot
sudo apt install -y python3-smbus2 python3-serial python3-picamera2 i2c-tools
i2cdetect -y 1             # should show 68 (or 69 when AD0 is high)
python3 test_environment/server.py
```

Use the system `python3`, not a venv, so the apt packages (especially `picamera2`) are found.
The console prints which devices are real and which are simulated, plus the page address.

**Stop the main edge app first** (`sudo systemctl stop roadflag-edge`). The camera and the GPS
port can only be used by one program: with both running, one gets "Device or resource busy"
and the other gets half the GPS data. The bench names the other program when this happens.

The GPS baud rate is detected automatically. The roadflag edge app switches the module to
38400 and saves it, so a reader fixed at the u-blox default of 9600 sees only garbage.

Options: `--gps-port /dev/ttyUSB0` (USB GPS), `--gps-baud 38400` (skip detection), `--mpu-address 0x69`,
`--camera opencv --camera-index 0` (USB webcam), `--threshold 0.4`, `--refractory 1.5`, `--port 8080`.

## How the auto capture works

1. The MPU-6050 is read at 100 Hz.
2. A complementary filter gives roll, pitch and yaw. Yaw drifts slowly, because there is no
   magnetometer; use **Reset yaw** to bring it back to 0.
3. **Vertical acceleration** = acceleration along gravity minus 1 g. Gravity's direction comes
   from the gyro-tracked orientation, so turning the sensor by hand is not a jolt.
4. When it goes past **± threshold** (default 0.4 g), the latest camera frame is saved right away.
   The direction of the first crossing gives the label:
   - **DOWN**, a pothole: the wheel drops first.
   - **UP**, a bump: the wheel is lifted first.
5. The peak is measured over the next 0.3 s. After that no new photo is taken for 1.5 s, so the
   front and rear wheels on one hole give one photo.

Photos are saved in `test_environment/captures/` (not committed) and listed at `/api/captures`.

## Camera detector on every photo

Each photo, from the trigger or from **Take test photo**, goes through the project's detector
(`vision_detector.py`). It runs in one background worker, so the sensor loop never waits.
The log entry starts as "analyzing…" and then shows:
- the labelled photo, with boxes, saved as `*_labelled.jpg`
- what the camera found: potholes with model confidence, speed bumps, cracks
- whether the camera **agrees with the sensor**: DOWN should be a pothole, UP should be a speed bump

The trained models live in `models/` (see `models/README.md`; they are not in git). Without them
the detector falls back to image rules for potholes and skips the speed-bump model.

Time per 640 px photo on a Pi 4:

| `--detect` | Time per photo |
|---|---|
| `pothole,crack` | about 1.6–2.5 s |
| `pothole,crack,speed_bump` (default) | about 5–7 s; the speed-bump model is a large YOLOv8m |

The speed-bump model misses most rubber and painted bumps, so the accelerometer is the better
bump signal. Use `--detect pothole,crack` for faster results, or `--detect ""` to turn
detection off. If more than 5 photos are waiting, new ones are logged but not analyzed.

To test at a desk, flick the sensor sharply down or up, or press the simulator buttons when the
MPU-6050 is simulated. In a car, start at 0.4 g; raise the threshold if rough asphalt triggers it.
