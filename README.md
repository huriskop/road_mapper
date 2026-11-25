# Road defect monitor

Finds potholes, speed bumps and cracks with a Raspberry Pi 4. Everything runs on the Pi 4: the sensors,
the detection and the web pages. `run.py` refuses to start on any other computer; a laptop only opens the pages.

1. The camera films all the time; the Pi keeps the last 6 seconds of video in memory.
2. The MPU-6050 feels a jolt: **down** means pothole, **up** means speed bump. The GPS gives the position.
3. When the wheel hits the hole the camera has already passed it, so the detector goes back through the
   6 seconds before the jolt (one frame every 0.5 s), keeps the frame that shows the defect best, and saves
   the 6 s as a video (`.avi`).
4. The result goes into the log and onto the map.

## Start

```bash
python3 run.py
```

| Page | What it shows |
|---|---|
| `http://<pi-address>:8080/` | Equipment check: live camera, MPU-6050, GPS, test photo |
| `http://<pi-address>:8080/capture` | Auto capture: a log of jolts with the best labelled frame and the video |
| `http://<pi-address>:8080/map` | Map, filters and incident log (map tiles need internet) |
| `http://<pi-address>:8080/test` | Try the detector on your own photos |

Useful options (`python3 run.py --help` lists them all):

| Option | Use |
|---|---|
| `--fallback-location 37.4563,126.7052` | Indoors: position to use while the GPS has no fix |
| `--roi 0.06,0.12,1,0.72` | The camera looks at a picture on a screen: analyze only that part |
| `--detect pothole,crack` | Report only these types |
| `--threshold 0.4` | How strong a jolt must be, in g |
| `--rotation 0` | The camera is mounted the right way up (default 180: ours is upside down) |

Photos and `incidents.json` (the map's data) are saved in `data/`. Jolts without a GPS fix are logged
but not put on the map, unless `--fallback-location` is given. Test photos never go on the map by themselves.

## Set up the Pi (once)

```bash
sudo raspi-config     # Interface Options: I2C on; Serial Port: login shell off, hardware on. Then reboot
sudo apt install -y python3-smbus2 python3-serial rpicam-apps python3-opencv python3-numpy
```

Use the system `python3`, not a venv. The trained model is in git, nothing to download. See
[models/README.md](models/README.md).

| MPU-6050 | Pi pin | | GPS | Pi pin |
|---|---|---|---|---|
| VCC | 3.3 V (1) | | VCC | 5 V (2) or 3.3 V, check your module |
| GND | GND (6) | | GND | GND (9) |
| SDA | GPIO 2 (3) | | TX | GPIO 15 / RXD (10) |
| SCL | GPIO 3 (5) | | RX | GPIO 14 / TXD (8), optional |

The Pi camera goes on the ribbon connector. Mount the MPU-6050 flat and rigid, X arrow forward.

## If something does not work

- **Camera "used by [...]"**: another program has the camera, for example `roadflag-edge`. Stop it with
  `sudo systemctl stop roadflag-edge`.
- **Camera "no new frame"**: it froze. The program restarts it by itself. On our Pi the cause is the Wi-Fi
  transmitter: with pages open, 4 freezes in 2 minutes at full power and none at 5 dBm. Before a test, run

  ```bash
  sudo iw dev wlan0 set txpower fixed 500      # 5 dBm; a reboot sets it back to full power
  ```

  Better: move the camera ribbon away from the Wi-Fi antenna, or use a 5 GHz hotspot or Ethernet. Do not copy
  large files from the Pi during a test. The live view runs at only 3 frames per second to send less;
  jolt photos use the full-rate camera.
- **GPS no fix**: needs open sky; the first fix can take a few minutes.

## Files

| File | What it does |
|---|---|
| `run.py` | Starts everything and serves the pages |
| `app/sensors.py` | Reads the MPU-6050, GPS and camera |
| `app/capture.py` | Jolt trigger and photo log |
| `app/detection.py` | Runs the detector on each photo |
| `app/incidents.py` | Defects on the map; reports within 5 m become one incident |
| `web/` | The pages |
| `vision_detector.py`, `pothole_model.py` | Camera detection algorithm |
| `sensor_detector.py` | Separate tool: finds potholes and bumps in a recorded accelerometer CSV (needs scipy) |
| `test_images/`, `tests/` | Road photos and tests |

## The detection

**Jolt:** vertical acceleration = acceleration along gravity minus 1 g, at 100 Hz. Gravity's direction comes
from the gyro, so turning the sensor is not a jolt. Past ±0.4 g a photo is taken; the direction of the first
crossing gives the type and the peak over the next 0.3 s gives the severity.

**Camera:** one trained YOLOv8n model finds potholes, speed bumps and cracks, about 0.5 s per photo on the Pi 4.
For a jolt it checks 13 frames from the 6 s before it (about 6 s of work): the type the jolt suggests wins, then
any pothole or bump, then cracks, and the most confident frame among those.
A camera pothole or bump sets the type over the jolt's direction; camera cracks count only when there was no jolt.

**Known limits:** this is a first, short training run. Close-up cracks are often missed or called potholes, and
some humps are missed, so the jolt is still the better bump signal. See [models/README.md](models/README.md).

```bash
python3 vision_detector.py road.jpg -o out.jpg     # one photo
python3 test_images/evaluate.py                    # score all test photos
python3 -m pytest                                  # tests (sudo apt install python3-pytest)
```
