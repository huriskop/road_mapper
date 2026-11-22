"""Hardware readers for the equipment test bench: MPU-6050, GPS (NMEA over serial), camera.

Each reader falls back to a simulator when its hardware or library is missing, so the
page can also be tried on a laptop. Only the standard library is required; the optional
libraries are smbus2 (MPU-6050), pyserial (GPS), picamera2 or opencv (camera).
"""
from __future__ import annotations

import collections
import io
import math
import os
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def other_users(dev: str, ignore: tuple[str, ...] = ()) -> str:
    """Other processes that have a device open (Linux only), e.g. "1318 python -m roadflag_edge.app".
    The camera and the GPS port cannot be shared: a second reader gets 'busy' or half the data."""
    target = os.path.realpath(dev)
    found = []
    for p in Path("/proc").glob("[0-9]*"):
        if int(p.name) == os.getpid():
            continue
        try:
            if any(os.path.realpath(f) == target for f in (p / "fd").iterdir()):
                cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
                if not any(i in cmd for i in ignore):
                    found.append(f"{p.name} {cmd[:90]}")
        except OSError:                                       # process ended, or not ours to inspect
            continue
    return "; ".join(found)


# =========================================================================== MPU-6050
class _Mpu6050:
    """Raw register access. Accelerometer +-4 g, gyro +-500 deg/s, 44 Hz low-pass filter."""

    ACC_LSB_PER_G = 8192.0
    GYRO_LSB_PER_DPS = 65.5

    def __init__(self, bus: int, address: int) -> None:
        try:
            from smbus2 import SMBus
        except ImportError:
            from smbus import SMBus  # Raspberry Pi OS package python3-smbus
        self.bus = SMBus(bus)
        self.addr = address
        who = self.bus.read_byte_data(address, 0x75)          # WHO_AM_I: 0x68 on genuine parts
        self.who_am_i = who
        self.bus.write_byte_data(address, 0x6B, 0x00)         # wake up
        time.sleep(0.05)
        self.bus.write_byte_data(address, 0x1A, 0x03)         # DLPF 44 Hz
        self.bus.write_byte_data(address, 0x19, 0x09)         # sample rate 1 kHz / (1 + 9) = 100 Hz
        self.bus.write_byte_data(address, 0x1C, 0x08)         # accel +-4 g
        self.bus.write_byte_data(address, 0x1B, 0x08)         # gyro +-500 deg/s

    def read(self) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
        b = self.bus.read_i2c_block_data(self.addr, 0x3B, 14)
        v = [int.from_bytes(bytes(b[i:i + 2]), "big", signed=True) for i in range(0, 14, 2)]
        acc = tuple(x / self.ACC_LSB_PER_G for x in v[0:3])
        gyro = tuple(x / self.GYRO_LSB_PER_DPS for x in v[4:7])
        return acc, gyro, v[3] / 340.0 + 36.53                 # type: ignore[return-value]


class Imu:
    """Orientation (complementary filter) and vertical acceleration at 100 Hz.

    dyn = acceleration along the gravity direction minus 1 g, in g.
    dyn > 0: the sensor is pushed UP (speed bump); dyn < 0: it DROPS (pothole).
    Gravity's direction comes from the gyro-tracked orientation, so turning the sensor
    by hand does not look like a jolt.
    """

    RATE_HZ = 100

    def __init__(self, bus: int = 1, address: int = 0x68, simulate: bool = False) -> None:
        self.lock = threading.Lock()
        self.source, self.error, self.ok = "sim", None, False
        self.roll = self.pitch = self.yaw = 0.0
        self.acc, self.gyro, self.temp, self.dyn = (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), None, 0.0
        self.series: collections.deque[tuple[float, float]] = collections.deque(maxlen=400)
        self.listeners: list = []                 # called as f(t, dyn) for every sample
        self._bias = [0.0, 0.0, 0.0]
        self._calib: list[tuple[float, float, float]] = []
        self._calib_left = self.RATE_HZ * 2       # 2 s of gyro samples for the bias
        self._g_mag: float | None = None
        self._inject: collections.deque[float] = collections.deque()
        self._dev = None
        if not simulate:
            try:
                self._dev = _Mpu6050(bus, address)
                self.source = "mpu6050"
                if self._dev.who_am_i != 0x68:
                    self.error = f"WHO_AM_I = 0x{self._dev.who_am_i:02x} (expected 0x68, maybe a clone)"
            except Exception as exc:  # no I2C bus, no library, nothing at the address
                self.error = f"MPU-6050 not found ({exc}); simulating"
        threading.Thread(target=self._run, daemon=True, name="imu").start()

    # ------------------------------------------------------------------ public
    @property
    def calibrating(self) -> bool:
        return self._calib_left > 0

    def zero(self) -> None:
        """Yaw back to 0 and re-measure the gyro bias (keep the sensor still for 1 s)."""
        with self.lock:
            self.yaw = 0.0
            self._calib, self._calib_left = [], self.RATE_HZ

    def simulate_event(self, kind: str) -> None:
        """Queue a synthetic jolt (simulator only): pothole = drop then impact, bump = lift then drop."""
        n = lambda s: int(s * self.RATE_HZ)  # noqa: E731
        if kind == "pothole":
            wave = [-0.9 * math.sin(math.pi * i / n(0.08)) for i in range(n(0.08))]
            wave += [1.3 * math.sin(math.pi * i / n(0.06)) for i in range(n(0.06))]
        else:
            wave = [0.9 * math.sin(math.pi * i / n(0.12)) for i in range(n(0.12))]
            wave += [-0.6 * math.sin(math.pi * i / n(0.1)) for i in range(n(0.1))]
        self._inject.extend(wave)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "ok": self.ok, "source": self.source, "error": self.error, "calibrating": self.calibrating,
                "roll": round(self.roll, 2), "pitch": round(self.pitch, 2), "yaw": round(self.yaw, 2),
                "acc": [round(x, 3) for x in self.acc], "gyro": [round(x, 2) for x in self.gyro],
                "temp": round(self.temp, 1) if self.temp is not None else None, "dyn": round(self.dyn, 3),
            }

    def series_since(self, t: float) -> tuple[list[list[float]], float]:
        with self.lock:
            pts = [[round(ts, 3), round(d, 3)] for ts, d in self.series if ts > t]
        return pts, (pts[-1][0] if pts else t)

    # ------------------------------------------------------------------ loop
    def _run(self) -> None:
        dt = 1.0 / self.RATE_HZ
        t0 = time.monotonic()
        nxt = time.perf_counter()
        while True:
            nxt += dt
            try:
                acc, gyro, temp = self._dev.read() if self._dev else self._sim(time.monotonic() - t0)
                self._update(acc, gyro, temp, dt)
                if self._dev and not self.ok and self.error and self.error.startswith("I2C"):
                    self.error = None                         # recovered after a read error
                self.ok = True
            except OSError as exc:                            # loose wire, bus error
                self.ok, self.error = False, f"I2C read failed: {exc}"
                time.sleep(0.5)
                nxt = time.perf_counter()
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.perf_counter()

    def _update(self, acc, gyro, temp, dt: float) -> None:
        if self._calib_left:
            self._calib.append(gyro)
            self._calib_left -= 1
            if not self._calib_left:
                n = len(self._calib)
                self._bias = [sum(g[i] for g in self._calib) / n for i in range(3)]
        gx, gy, gz = (gyro[i] - self._bias[i] for i in range(3))
        ax, ay, az = acc
        roll_acc = math.degrees(math.atan2(ay, az))
        pitch_acc = math.degrees(math.atan2(-ax, math.hypot(ay, az)))
        k = 0.98
        with self.lock:
            if self._g_mag is None:                           # first sample: trust the accelerometer
                self.roll, self.pitch = roll_acc, pitch_acc
            else:
                pred = self.roll + gx * dt
                diff = (roll_acc - pred + 180) % 360 - 180    # blend across the +-180 wrap
                self.roll = (pred + (1 - k) * diff + 180) % 360 - 180
                self.pitch = k * (self.pitch + gy * dt) + (1 - k) * pitch_acc
                if not self._calib_left:
                    self.yaw = (self.yaw + gz * dt + 180) % 360 - 180
            r, p = math.radians(self.roll), math.radians(self.pitch)
            down = (-math.sin(p), math.cos(p) * math.sin(r), math.cos(p) * math.cos(r))  # gravity, body frame
            along = ax * down[0] + ay * down[1] + az * down[2]
            # slow (0.1 Hz) estimate of 1 g in sensor units cancels the accelerometer's scale error
            self._g_mag = along if self._g_mag is None else self._g_mag + 0.0063 * (along - self._g_mag)
            self.dyn = along - self._g_mag
            self.acc, self.gyro, self.temp = acc, (gx, gy, gz), temp
            t = time.monotonic()
            self.series.append((t, self.dyn))
            dyn = self.dyn
        for f in self.listeners:
            f(t, dyn)

    def _sim(self, t: float):
        """Gentle sway plus noise, with any queued jolt added along the vertical."""
        r, p = 4 * math.sin(0.4 * t), 3 * math.sin(0.27 * t + 1)
        dr, dp, dy = 1.6 * math.cos(0.4 * t), 0.81 * math.cos(0.27 * t + 1), 2.0
        vert = (self._inject.popleft() if self._inject else 0.0) + random.gauss(0, 0.02)
        rr, pr = math.radians(r), math.radians(p)
        g = 1.0 + vert
        acc = (-math.sin(pr) * g, math.cos(pr) * math.sin(rr) * g, math.cos(pr) * math.cos(rr) * g)
        acc = tuple(a + random.gauss(0, 0.01) for a in acc)
        gyro = (dr + random.gauss(0, 0.3), dp + random.gauss(0, 0.3), dy + random.gauss(0, 0.3))
        return acc, gyro, 31.5


# =========================================================================== GPS
def _nmea_valid(line: str) -> bool:
    if not line.startswith("$") or "*" not in line:
        return False
    body, _, cs = line[1:].partition("*")
    calc = 0
    for ch in body:
        calc ^= ord(ch)
    try:
        return calc == int(cs[:2], 16)
    except ValueError:
        return False


def _nmea_deg(value: str, hemi: str) -> float | None:
    if not value:
        return None
    v = float(value)
    deg = int(v / 100)
    x = deg + (v - deg * 100) / 60
    return -x if hemi in ("S", "W") else x


class Gps:
    """NMEA reader (GGA + RMC) for u-blox NEO-6M/7M/8M and similar modules.

    baud=0 detects the rate: u-blox modules ship at 9600, but software (for example the
    roadflag edge app) can switch them to 38400 and save it, and the wrong rate gives only garbage."""

    BAUD_RATES = (38400, 9600, 115200, 57600, 4800)

    def __init__(self, port: str = "/dev/serial0", baud: int = 0, simulate: bool = False,
                 sim_center: tuple[float, float] = (37.4563, 126.7052)) -> None:
        self.lock = threading.Lock()
        self.source, self.error = "sim", None
        self.port, self.baud = port, baud
        self.d = {"fix": False, "quality": 0, "sats": None, "hdop": None, "lat": None, "lng": None,
                  "alt": None, "speed_kmh": None, "course": None, "utc": None}
        self.last_line, self.sentences, self.last_rx = "", 0, 0.0
        self._serial = None
        if not simulate:
            try:
                import serial  # pyserial
                self._serial = serial.Serial(port, baud or self.BAUD_RATES[0], timeout=1)
                self.source = "serial"
                busy = other_users(port)
                if busy:
                    self.error = f"{port} is also open in [{busy}]: stop that program, the data is split between you"
            except Exception as exc:
                self.error = f"GPS port {port} not available ({exc}); simulating"
        self._center = sim_center
        target = self._read_serial if self._serial else self._simulate
        threading.Thread(target=target, daemon=True, name="gps").start()

    def snapshot(self) -> dict:
        with self.lock:
            out = dict(self.d)
            out.update(source=self.source, error=self.error, last_line=self.last_line, sentences=self.sentences)
            out["baud"] = self.baud
        age = time.monotonic() - self.last_rx if self.last_rx else None
        if self.source == "serial":
            if not self.baud:
                out["status"] = "detecting baud rate…"
            elif age is None or age > 3:
                out["status"] = f"no data on {self.port}: check TX->RX wiring, baud rate and that serial is enabled"
            elif not out["fix"]:
                out["status"] = "receiving, no fix yet (needs open sky; first fix can take a few minutes)"
            else:
                out["status"] = "fix"
        else:
            out["status"] = "fix (simulated)"
        out["ok"] = bool(out["fix"])
        return out

    def _detect_baud(self) -> None:
        best_n, best = 0, None
        for b in self.BAUD_RATES:
            self._serial.baudrate = b
            self._serial.reset_input_buffer()
            t, n = time.monotonic(), 0
            while time.monotonic() - t < 1.5:
                try:
                    n += _nmea_valid(self._serial.readline().decode("ascii", "ignore").strip())
                except Exception as exc:          # port shared with another program, or unplugged
                    self.error = f"serial read failed: {exc}"
                    time.sleep(0.2)
            if n > best_n:
                best_n, best = n, b
            if n >= 3:
                break
        if best is None:
            self.error = (f"no valid NMEA on {self.port} at any of {self.BAUD_RATES} baud: check GPS TX -> Pi RX "
                          "(pin 10), power, and that no other program reads the port")
            best = self.BAUD_RATES[0]
        self._serial.baudrate = self.baud = best

    def _read_serial(self) -> None:
        if not self.baud:
            self._detect_baud()
        while True:
            try:
                raw = self._serial.readline()
            except Exception as exc:
                self.error = f"serial read failed: {exc}"
                time.sleep(1)
                continue
            line = raw.decode("ascii", "ignore").strip()
            if not line:
                continue
            with self.lock:
                self.last_line, self.last_rx = line[:120], time.monotonic()
            if _nmea_valid(line):
                self._parse(line)

    def _parse(self, line: str) -> None:
        f = line.split("*")[0].split(",")
        kind = f[0][3:]
        with self.lock:
            self.sentences += 1
            try:
                if kind == "GGA" and len(f) > 9:
                    q = int(f[6] or 0)
                    self.d.update(quality=q, fix=q > 0, sats=int(f[7] or 0), hdop=float(f[8]) if f[8] else None,
                                  alt=float(f[9]) if f[9] else None)
                    if q > 0:
                        self.d.update(lat=_nmea_deg(f[2], f[3]), lng=_nmea_deg(f[4], f[5]))
                elif kind == "RMC" and len(f) > 9:
                    if f[1]:
                        self.d["utc"] = f"{f[1][0:2]}:{f[1][2:4]}:{f[1][4:6]}"
                    if f[2] == "A":
                        self.d.update(lat=_nmea_deg(f[3], f[4]), lng=_nmea_deg(f[5], f[6]),
                                      speed_kmh=round(float(f[7] or 0) * 1.852, 1),
                                      course=float(f[8]) if f[8] else None)
                    else:
                        self.d["fix"] = False
            except ValueError:
                pass                                          # malformed field: skip the sentence

    def _simulate(self) -> None:
        """Drive a 150 m circle at 30 km/h, one update per second like a real module."""
        lat0, lng0 = self._center
        radius, speed = 150.0, 30 / 3.6
        t = 0.0
        while True:
            ang = speed * t / radius
            x, y = radius * math.sin(ang), radius * (1 - math.cos(ang))
            x, y = x + random.gauss(0, 0.8), y + random.gauss(0, 0.8)
            lat = lat0 + y / 111_320
            lng = lng0 + x / (111_320 * math.cos(math.radians(lat0)))
            with self.lock:
                self.d.update(fix=True, quality=1, sats=9, hdop=0.9, lat=round(lat, 7), lng=round(lng, 7),
                              alt=12.0, speed_kmh=round(30 + random.gauss(0, 0.5), 1),
                              course=round((90 - math.degrees(ang)) % 360, 1),
                              utc=datetime.now(timezone.utc).strftime("%H:%M:%S"))
                self.sentences += 2
                self.last_line, self.last_rx = "(simulated)", time.monotonic()
            time.sleep(1)
            t += 1


# =========================================================================== camera
def _jpeg_encoder():
    """BGR numpy image -> JPEG bytes, with OpenCV if present, else Pillow."""
    try:
        import cv2
        return lambda bgr: cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
    except ImportError:
        from PIL import Image

        def enc(bgr):
            buf = io.BytesIO()
            Image.fromarray(bgr[..., ::-1]).save(buf, "JPEG", quality=80)
            return buf.getvalue()
        return enc


class Camera:
    """Keeps the latest frame. kind: auto | picamera2 | opencv | sim.

    A camera that stops sending frames is reported as an error and reopened until it works again.
    Meanwhile latest() returns no frame, so an old picture is never saved or sent as a new photo."""

    STALL_S = 3.0        # no frame for this long: close and reopen the camera
    STALE_S = 1.0        # a frame older than this is not a photo of "now"
    RETRY_S = 2.0        # pause before each reopen attempt
    CLOSE_S = 5.0        # a close that takes longer is reported as a stuck driver

    def __init__(self, kind: str = "auto", size: tuple[int, int] = (640, 480), index: int = 0, fps: float = 15) -> None:
        self.source, self.error, self.mime = "sim", None, "image/svg+xml"
        self.size, self.fps_limit, self.fps = size, fps, 0.0
        self._frame: bytes | None = None
        self._frame_at = 0.0                                  # time.monotonic() of the latest frame
        self._seq = 0
        self._index = index
        self.restarts = 0
        self._cond = threading.Condition()
        grab = close = None
        errors = []
        for k in (["picamera2", "opencv"] if kind == "auto" else [kind]):
            if k == "sim":
                break
            try:
                grab, close = self._open(k)
                self.source, self.mime = k, "image/jpeg"
                break
            except Exception as exc:
                errors.append(f"{k}: {exc}")
        if grab is None:
            grab = self._sim_frame
            if errors:
                self.error = "no camera (" + "; ".join(errors) + "); simulating"
        if self.source != "sim" or errors:
            # pipewire/wireplumber always hold the device; anything else steals frames ("resource busy")
            busy = other_users("/dev/video0", ignore=("pipewire", "wireplumber")) if os.path.exists("/dev/video0") else ""
            if busy:
                self.error = f"camera is also used by [{busy}]: stop that program. " + (self.error or "")
        threading.Thread(target=self._run, args=(grab, close), daemon=True, name="camera").start()

    def _open(self, kind: str):
        """(grab, close) functions for a real camera."""
        return self._open_picamera2() if kind == "picamera2" else self._open_opencv(self._index)

    def _open_picamera2(self):
        from picamera2 import Picamera2
        enc = _jpeg_encoder()
        cam = Picamera2()
        try:
            # "RGB888" is stored as B, G, R in memory, the same order OpenCV uses
            cam.configure(cam.create_video_configuration(main={"size": self.size, "format": "RGB888"}))
            cam.start()
        except Exception:
            cam.close()                                       # or the next attempt finds it busy
            raise

        def grab():
            # capture_array() alone waits forever when the sensor stops sending frames
            job = cam.capture_array(wait=False)
            try:
                return enc(cam.wait(job, timeout=self.STALL_S))
            except TimeoutError:
                raise RuntimeError(f"no new frame for {self.STALL_S:g} s") from None
        return grab, cam.close

    def _open_opencv(self, index: int):
        import cv2
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        ok, _ = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"no frames from camera index {index}")
        enc = _jpeg_encoder()

        def grab():
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("camera stopped sending frames")
            return enc(frame)
        return grab, cap.release

    def _sim_frame(self) -> bytes:
        w, h = self.size
        t = time.time()
        x = int((t * 120) % (w + 80)) - 40
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
                f'<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#3a4250"/>'
                f'<stop offset="1" stop-color="#1b1f26"/></linearGradient></defs>'
                f'<rect width="{w}" height="{h}" fill="url(#g)"/>'
                f'<polygon points="{w*0.42},{h*0.45} {w*0.58},{h*0.45} {w*0.95},{h} {w*0.05},{h}" fill="#2b2f36"/>'
                f'<rect x="{w/2-4}" y="{h*0.6}" width="8" height="{h*0.12}" fill="#d8d27a"/>'
                f'<rect x="{x}" y="{h-18}" width="40" height="8" fill="#5aa9ff"/>'
                f'<text x="20" y="40" fill="#e8ecf2" font-family="monospace" font-size="22">CAMERA SIMULATION</text>'
                f'<text x="20" y="70" fill="#9aa4b2" font-family="monospace" font-size="18">{stamp}  frame {self._seq}</text>'
                f'</svg>').encode()

    def _run(self, grab, close) -> None:
        period = 1.0 / self.fps_limit
        last = time.perf_counter()
        while True:
            start = time.perf_counter()
            try:
                if grab is None:
                    grab, close = self._open(self.source)
                frame = grab()
            except Exception as exc:
                if self.source == "sim":
                    self.error = f"capture failed: {exc}"
                    time.sleep(1)
                    continue
                self.restarts += 1
                self.error = (f"camera stopped: {exc}. Reopening (attempt {self.restarts}); "
                              "if it keeps stopping, check the ribbon cable")
                self._close(close)
                grab = close = None
                time.sleep(self.RETRY_S)
                continue
            if self.restarts and (self.error or "").startswith("camera st"):
                self.error = None                             # it works again
            with self._cond:
                self._frame = frame
                self._frame_at = time.monotonic()
                self._seq += 1
                self._cond.notify_all()
            now = time.perf_counter()
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - last, 1e-3))
            last = now
            time.sleep(max(0.0, period - (now - start)))

    def _close(self, close) -> None:
        """Close the camera. Returns only once it is closed: a second open would find it busy."""
        if close is None:
            return

        def run():
            try:
                close()
            except Exception:                                 # already broken; the reopen will tell
                pass
        t = threading.Thread(target=run, daemon=True, name="camera-close")
        t.start()
        t.join(self.CLOSE_S)
        while t.is_alive():
            self.error = "camera stuck: the driver does not close it. Stop this program and check the ribbon cable"
            t.join(self.CLOSE_S)

    def _fresh(self) -> bool:
        return self._frame is not None and time.monotonic() - self._frame_at <= self.STALE_S

    def latest(self) -> tuple[bytes | None, str]:
        """The current frame, or None when the camera has stopped sending them."""
        with self._cond:
            return (self._frame if self._fresh() else None), self.mime

    def wait(self, seq: int, timeout: float = 2.0) -> tuple[bytes | None, int]:
        """Block until a frame newer than seq exists."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != seq, timeout)
            return self._frame, self._seq

    def info(self) -> dict:
        fresh = self._fresh()
        return {"ok": fresh, "source": self.source, "error": self.error, "mime": self.mime,
                "fps": round(self.fps, 1) if fresh else 0.0, "size": list(self.size), "restarts": self.restarts}


# =========================================================================== captures
class CaptureStore:
    """Photos taken by the trigger (or by hand), kept on disk plus a short in-memory log.

    Every add or update bumps `seq`, so the live stream can send changed entries
    (for example when the detector finishes) as well as new ones."""

    def __init__(self, directory: Path, keep: int = 200) -> None:
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.items: collections.deque[dict] = collections.deque(maxlen=keep)
        self.last_id = 0
        self.seq = 0
        self.on_add: list = []          # called as f(item, frame, mime, file_name) after a photo is saved

    def add(self, frame: bytes | None, mime: str, reason: str, peak_g: float | None, gps: dict, when: str) -> dict:
        ext = "jpg" if mime == "image/jpeg" else "svg"
        name = None
        if frame:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
            name = f"{stamp}_{reason}.{ext}"
            (self.dir / name).write_bytes(frame)
        with self.lock:
            self.last_id += 1
            item = {
                "id": self.last_id, "time": when, "reason": reason,
                "guess": {"down": "pothole?", "up": "bump?"}.get(reason),
                "peak_g": round(peak_g, 2) if peak_g is not None else None,
                "image": f"/captures/{name}" if name else None,
                "lat": gps.get("lat"), "lng": gps.get("lng"), "gps_fix": gps.get("fix"),
                "speed_kmh": gps.get("speed_kmh"),
                "annotated": None, "analysis": None,
            }
            self.seq += 1
            item["seq"] = self.seq
            self.items.append(item)
        for f in self.on_add:
            f(item, frame, mime, name)
        return dict(item)

    def update(self, item_id: int, **fields) -> None:
        with self.lock:
            for item in self.items:
                if item["id"] == item_id:
                    item.update(fields)
                    self.seq += 1
                    item["seq"] = self.seq
                    return

    def since(self, last_seq: int) -> list[dict]:
        """New or changed entries, oldest first."""
        with self.lock:
            return sorted((dict(i) for i in self.items if i["seq"] > last_seq), key=lambda i: i["seq"])

    def all(self) -> list[dict]:
        with self.lock:
            return [dict(i) for i in self.items]


class Trigger:
    """Takes a photo when the vertical acceleration passes the threshold.

    The photo is grabbed at the first sample over the threshold (the moment of the jolt);
    the peak is then measured over the next 0.3 s, and nothing fires during the refractory
    time, so the front and rear wheels on the same hole give one photo."""

    WINDOW_S = 0.3

    def __init__(self, imu: Imu, camera: Camera, gps: Gps, store: CaptureStore,
                 threshold_g: float = 0.4, refractory_s: float = 1.5) -> None:
        self.imu, self.camera, self.gps, self.store = imu, camera, gps, store
        self.threshold, self.refractory, self.enabled = threshold_g, refractory_s, True
        self._active = False
        self._quiet_until = 0.0
        self._pending: dict = {}
        imu.listeners.append(self.feed)

    def info(self) -> dict:
        return {"enabled": self.enabled, "threshold_g": self.threshold, "refractory_s": self.refractory}

    def feed(self, t: float, dyn: float) -> None:
        if self._active:
            p = self._pending
            if abs(dyn) > abs(p["peak"]):
                p["peak"] = dyn
            if t - p["t0"] >= self.WINDOW_S:
                self._active = False
                self._quiet_until = t + self.refractory
                threading.Thread(target=self.store.add, daemon=True, args=(
                    p["frame"], p["mime"], p["reason"], p["peak"], p["gps"], p["when"])).start()
            return
        if not self.enabled or self.imu.calibrating or t < self._quiet_until or abs(dyn) < self.threshold:
            return
        frame, mime = self.camera.latest()
        self._pending = {"t0": t, "peak": dyn, "reason": "up" if dyn > 0 else "down", "frame": frame,
                         "mime": mime, "gps": self.gps.snapshot(), "when": now_iso()}
        self._active = True

    def manual(self) -> dict:
        frame, mime = self.camera.latest()
        return self.store.add(frame, mime, "manual", None, self.gps.snapshot(), now_iso())
