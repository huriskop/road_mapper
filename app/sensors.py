"""Hardware readers for the Raspberry Pi: MPU-6050 (I2C), GPS (NMEA over serial), Pi camera.

A device that is missing or fails shows an error on the page and is retried until it works.
Needs smbus2, pyserial, picamera2 and cv2 (apt packages, see README.md)."""
from __future__ import annotations

import collections
import math
import os
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
        from smbus2 import SMBus
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
    RETRY_S = 2.0

    def __init__(self, bus: int = 1, address: int = 0x68) -> None:
        self.lock = threading.Lock()
        self.error, self.ok = None, False
        self.bus, self.address = bus, address
        self.roll = self.pitch = self.yaw = 0.0
        self.acc, self.gyro, self.temp, self.dyn = (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), None, 0.0
        self.series: collections.deque[tuple[float, float]] = collections.deque(maxlen=400)
        self.listeners: list = []                 # called as f(t, dyn) for every sample
        self._bias = [0.0, 0.0, 0.0]
        self._calib: list[tuple[float, float, float]] = []
        self._calib_left = self.RATE_HZ * 2       # 2 s of gyro samples for the bias
        self._g_mag: float | None = None
        self._dev = None
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

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "ok": self.ok, "error": self.error, "calibrating": self.calibrating,
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
        nxt = time.perf_counter()
        while True:
            nxt += dt
            try:
                if self._dev is None:
                    self._dev = _Mpu6050(self.bus, self.address)
                    who = self._dev.who_am_i
                    self.error = None if who == 0x68 else f"WHO_AM_I = 0x{who:02x} (expected 0x68, maybe a clone)"
                self._update(*self._dev.read(), dt)
                self.ok = True
            except (OSError, ImportError) as exc:             # not wired, loose wire, no smbus2: try again
                self.ok, self._dev = False, None
                self.error = f"MPU-6050 not working: {exc}"
                time.sleep(self.RETRY_S)
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

    def __init__(self, port: str = "/dev/serial0", baud: int = 0) -> None:
        self.lock = threading.Lock()
        self.error = None
        self.port, self.baud = port, baud
        self.d = {"fix": False, "quality": 0, "sats": None, "hdop": None, "lat": None, "lng": None,
                  "alt": None, "speed_kmh": None, "course": None, "utc": None}
        self.last_line, self.sentences, self.last_rx = "", 0, 0.0
        self._serial = None
        threading.Thread(target=self._read_serial, daemon=True, name="gps").start()

    def snapshot(self) -> dict:
        with self.lock:
            out = dict(self.d)
            out.update(error=self.error, last_line=self.last_line, sentences=self.sentences)
            out["baud"] = self.baud
        age = time.monotonic() - self.last_rx if self.last_rx else None
        if not self._serial:
            out["status"] = "serial port not open"
        elif not self.baud:
            out["status"] = "detecting baud rate…"
        elif age is None or age > 3:
            out["status"] = f"no data on {self.port}: check TX->RX wiring, baud rate and that serial is enabled"
        elif not out["fix"]:
            out["status"] = "receiving, no fix yet (needs open sky; first fix can take a few minutes)"
        else:
            out["status"] = "fix"
        out["ok"] = bool(out["fix"])
        return out

    def _open(self) -> None:
        """Open the port, retrying until it works."""
        while True:
            try:
                import serial  # pyserial
                self._serial = serial.Serial(self.port, self.baud or self.BAUD_RATES[0], timeout=1)
                break
            except (OSError, ImportError) as exc:             # SerialException is an OSError
                self.error = f"GPS port {self.port} not available ({exc})"
                time.sleep(2)
        busy = other_users(self.port)
        self.error = f"{self.port} is also open in [{busy}]: stop that program, the data is split between you" if busy else None

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
        self._open()
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


# =========================================================================== camera
class Camera:
    """Keeps the latest frame of the Pi camera as a JPEG.

    A camera that cannot be opened or stops sending frames is reported as an error and reopened
    until it works. Meanwhile latest() returns no frame, so an old picture is never saved as a new photo."""

    STALL_S = 3.0        # no frame for this long: close and reopen the camera
    STALE_S = 1.0        # a frame older than this is not a photo of "now"
    RETRY_S = 2.0        # pause before each reopen attempt
    CLOSE_S = 5.0        # a close that takes longer is reported as a stuck driver

    def __init__(self, size: tuple[int, int] = (640, 480), fps: float = 15) -> None:
        self.error: str | None = None
        self.size, self.fps_limit, self.fps = size, fps, 0.0
        self.restarts = 0
        self._stop = threading.Event()
        self._frame: bytes | None = None
        self._frame_at = 0.0                                  # time.monotonic() of the latest frame
        self._seq = 0
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._run, daemon=True, name="camera")
        self._thread.start()

    def _open(self):
        """(grab, close) functions for the Pi camera."""
        import cv2
        from picamera2 import Picamera2
        try:
            cam = Picamera2()
        except Exception as exc:
            # pipewire/wireplumber always hold the device; any other program keeps the camera busy
            busy = other_users("/dev/video0", ignore=("pipewire", "wireplumber"))
            raise RuntimeError(f"used by [{busy}], stop that program" if busy else f"cannot open it ({exc})") from exc
        try:
            # "RGB888" is stored as B, G, R in memory, the same order OpenCV uses
            cam.configure(cam.create_video_configuration(main={"size": self.size, "format": "RGB888"}))
            cam.start()
        except Exception:
            cam.close()                                       # or the next attempt finds it busy
            raise

        def grab() -> bytes:
            # capture_array() alone waits forever when the sensor stops sending frames
            job = cam.capture_array(wait=False)
            try:
                bgr = cam.wait(job, timeout=self.STALL_S)
            except TimeoutError:
                raise RuntimeError(f"no new frame for {self.STALL_S:g} s") from None
            return cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        return grab, cam.close

    def _run(self) -> None:
        period = 1.0 / self.fps_limit
        last = time.perf_counter()
        grab = close = None
        while not self._stop.is_set():
            start = time.perf_counter()
            try:
                if grab is None:
                    grab, close = self._open()
                frame = grab()
            except Exception as exc:
                self.restarts += 1
                self.error = (f"camera not working: {exc}. Reopening (attempt {self.restarts}); "
                              "if this repeats, check the ribbon cable")
                self._close(close)
                grab = close = None
                time.sleep(self.RETRY_S)
                continue
            self.error = None
            with self._cond:
                self._frame = frame
                self._frame_at = time.monotonic()
                self._seq += 1
                self._cond.notify_all()
            now = time.perf_counter()
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - last, 1e-3))
            last = now
            time.sleep(max(0.0, period - (now - start)))
        self._close(close)

    def shutdown(self) -> None:
        """Release the camera before the program ends."""
        self._stop.set()
        self._thread.join(8)

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

    def latest(self) -> bytes | None:
        """The current frame, or None when the camera has stopped sending them."""
        with self._cond:
            return self._frame if self._fresh() else None

    def wait(self, seq: int, timeout: float = 2.0) -> tuple[bytes | None, int]:
        """Block until a frame newer than seq exists."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != seq, timeout)
            return self._frame, self._seq

    def info(self) -> dict:
        fresh = self._fresh()
        return {"ok": fresh, "error": self.error,
                "fps": round(self.fps, 1) if fresh else 0.0, "size": list(self.size), "restarts": self.restarts}
