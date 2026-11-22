# Road defect monitor

Finds potholes, speed bumps and cracks with a Raspberry Pi 4. Everything runs on the Pi: the sensors,
the detection and the web pages.

1. The MPU-6050 feels a jolt: **down** means pothole, **up** means speed bump.
2. The camera takes a photo at that moment, and the GPS gives the position.
3. The detector looks for potholes, speed bumps and cracks in the photo.
4. The result goes into the log and onto the map.

## Start

```bash
python3 run.py
```

| Page | What it shows |
|---|---|
| `http://<pi-address>:8080/` | Equipment check: live camera, MPU-6050, GPS, test photo |
| `http://<pi-address>:8080/capture` | Auto capture: a log of jolts with the labelled photos |
| `http://<pi-address>:8080/map` | Map, filters and incident log (map tiles need internet) |
| `http://<pi-address>:8080/test` | Try the detector on your own photos |

Useful options (`python3 run.py --help` lists them all):

| Option | Use |
|---|---|
| `--fallback-location 37.4563,126.7052` | Indoors: position to use while the GPS has no fix |
| `--roi 0.06,0.12,1,0.72` | The camera looks at a picture on a screen: analyze only that part |
| `--detect pothole,crack` | Skip the slow speed-bump model: about 2 s per photo instead of 5–7 s |
| `--threshold 0.4` | How strong a jolt must be, in g |

Photos and `incidents.json` (the map's data) are saved in `data/`. Jolts without a GPS fix are logged
but not put on the map, unless `--fallback-location` is given. Test photos never go on the map by themselves.

## Set up the Pi (once)

```bash
sudo raspi-config     # Interface Options: I2C on; Serial Port: login shell off, hardware on. Then reboot
sudo apt install -y python3-smbus2 python3-serial rpicam-apps python3-opencv python3-numpy
```

Use the system `python3`, not a venv. Download the two model files as described in
[models/README.md](models/README.md). Without them, potholes fall back to image rules and speed bumps
are found only by their painted markings.

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
- **Camera "no new frame"**: it froze. The program restarts it by itself. On our Pi this happens while the Pi
  *sends* a lot over Wi-Fi: at full transmit power the camera got 0 frames, at 5 dBm it ran normally. Move the
  camera ribbon away from the Wi-Fi antenna, use a 5 GHz hotspot or Ethernet, and do not copy files from the Pi
  during a test. This is why the live view runs at only 3 frames per second; photos are not affected by that.
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

**Camera:** potholes use a trained YOLOv8 model (8 of 12 test photos). Cracks use image rules, limited to the
road surface. Speed bumps use a trained model plus rules for painted bumps; it misses most bumps (3 of 15), so
the jolt is the better bump signal. A camera pothole or bump sets the type over the jolt's direction; camera
cracks count only when there was no jolt.

**Known limit:** the crack rules expect a road. Indoors they report desk edges and shadows as cracks.

```bash
python3 vision_detector.py road.jpg -o out.jpg     # one photo
python3 test_images/evaluate.py                    # score all test photos
python3 -m pytest                                  # tests (sudo apt install python3-pytest)
```
