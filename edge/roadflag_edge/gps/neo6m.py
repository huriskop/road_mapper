"""E3 - u-blox NEO-6M reader: NMEA RMC + GGA at 5 Hz, 38400 baud.

The module ships at 1 Hz / 9600 baud with six sentences per epoch. `configure()` sends UBX
commands once (baud 38400, 200 ms rate, only RMC + GGA) and saves them to the module's
flash/BBR, so later boots come up configured. `open_port()` finds whichever baud it is on.
"""
from __future__ import annotations

import logging
import struct
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timezone

log = logging.getLogger(__name__)

KNOT = 0.514444


@dataclass
class Fix:
    t_ns: int                 # monotonic time of the epoch (arrival - NMEA latency)
    lat: float | None
    lon: float | None
    speed_mps: float | None
    heading: float | None
    hdop: float | None
    quality: int              # GGA fix quality, 0 = none
    utc: datetime | None
    sats: int | None = None

    @property
    def valid(self) -> bool:
        return self.quality > 0 and self.lat is not None and self.lon is not None


# ---------------------------------------------------------------------- NMEA
def nmea_checksum_ok(line: str) -> bool:
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


def _deg(value: str, hemi: str) -> float | None:
    if not value:
        return None
    dot = value.index(".")
    deg = float(value[: dot - 2]) + float(value[dot - 2:]) / 60.0
    return -deg if hemi in ("S", "W") else deg


def _num(value: str) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None


def _utc(hms: str, dmy: str | None) -> datetime | None:
    if not hms or len(hms) < 6:
        return None
    t = dtime(int(hms[0:2]), int(hms[2:4]), int(hms[4:6]),
              int(round(float("0" + hms[6:]) * 1e6)) if len(hms) > 6 else 0, tzinfo=timezone.utc)
    if dmy and len(dmy) == 6:
        d = date(2000 + int(dmy[4:6]), int(dmy[2:4]), int(dmy[0:2]))
    else:
        d = datetime.now(timezone.utc).date()
    return datetime.combine(d, t)


class NmeaAssembler:
    """Combines the RMC and GGA of one epoch (same hhmmss.ss) into a Fix."""

    def __init__(self, latency_ns: int) -> None:
        self.latency_ns = latency_ns
        self._rmc: dict | None = None
        self._gga: dict | None = None

    def feed(self, line: str, arrival_ns: int) -> Fix | None:
        line = line.strip()
        if not nmea_checksum_ok(line):
            return None
        f = line[1:].split("*")[0].split(",")
        kind = f[0][2:]
        if kind == "RMC" and len(f) >= 10:
            self._rmc = {"time": f[1], "ok": f[2] == "A", "lat": _deg(f[3], f[4]), "lon": _deg(f[5], f[6]),
                         "speed": _num(f[7]), "course": _num(f[8]), "date": f[9], "t": arrival_ns}
        elif kind == "GGA" and len(f) >= 9:
            self._gga = {"time": f[1], "lat": _deg(f[2], f[3]), "lon": _deg(f[4], f[5]),
                         "q": int(f[6] or 0), "sats": int(f[7] or 0), "hdop": _num(f[8]), "t": arrival_ns}
        else:
            return None
        r, g = self._rmc, self._gga
        if r and g and r["time"] == g["time"]:
            self._rmc = self._gga = None
            quality = g["q"] if r["ok"] else 0
            return Fix(
                t_ns=min(r["t"], g["t"]) - self.latency_ns,
                lat=g["lat"] if g["lat"] is not None else r["lat"],
                lon=g["lon"] if g["lon"] is not None else r["lon"],
                speed_mps=r["speed"] * KNOT if r["speed"] is not None else None,
                heading=r["course"],
                hdop=g["hdop"],
                quality=quality,
                utc=_utc(r["time"], r["date"]),
                sats=g["sats"],
            )
        return None


# ----------------------------------------------------------------------- UBX
def ubx(cls: int, mid: int, payload: bytes) -> bytes:
    body = bytes([cls, mid]) + struct.pack("<H", len(payload)) + payload
    a = b = 0
    for x in body:
        a = (a + x) & 0xFF
        b = (b + a) & 0xFF
    return b"\xb5\x62" + body + bytes([a, b])


def cfg_prt(baud: int) -> bytes:
    # UART1, 8N1, UBX+NMEA in and out
    return ubx(0x06, 0x00, struct.pack("<BBHIIHHHH", 1, 0, 0, 0x08D0, baud, 0x03, 0x03, 0, 0))


def cfg_rate(period_ms: int) -> bytes:
    return ubx(0x06, 0x08, struct.pack("<HHH", period_ms, 1, 1))


def cfg_msg_nmea(msg_id: int, rate: int) -> bytes:
    return ubx(0x06, 0x01, bytes([0xF0, msg_id, rate]))


def cfg_save() -> bytes:
    return ubx(0x06, 0x09, struct.pack("<IIIB", 0, 0xFFFF, 0, 0x17))


NMEA_IDS = {"GGA": 0x00, "GLL": 0x01, "GSA": 0x02, "GSV": 0x03, "RMC": 0x04, "VTG": 0x05}


class Neo6m:
    def __init__(self, port: str = "/dev/serial0", baud: int = 38400, rate_hz: int = 5,
                 nmea_latency_ms: float = 80.0) -> None:
        self.port, self.baud, self.rate_hz = port, baud, rate_hz
        self.asm = NmeaAssembler(int(nmea_latency_ms * 1e6))
        self.ser = None
        self.bad_lines = 0

    def _try(self, baud: int, seconds: float = 2.5) -> bool:
        import serial

        if self.ser:
            self.ser.close()
        self.ser = serial.Serial(self.port, baud, timeout=0.5)
        self.ser.reset_input_buffer()
        end = time.monotonic() + seconds
        good = 0
        while time.monotonic() < end:
            line = self.ser.readline().decode("ascii", "replace").strip()
            if nmea_checksum_ok(line):
                good += 1
                if good >= 3:
                    return True
        return False

    def open_port(self) -> int:
        for baud in (self.baud, 9600, 38400, 115200):
            if self._try(baud):
                log.info("GPS answering at %d baud", baud)
                return baud
        raise RuntimeError(f"no NMEA on {self.port} at any baud (wiring? enable_uart=1?)")

    def configure(self) -> None:
        """Switch to the target baud and rate, keep only RMC + GGA, save to the module."""
        current = self.open_port()
        if current != self.baud:
            self.ser.write(cfg_prt(self.baud))
            self.ser.flush()
            time.sleep(0.3)
            if not self._try(self.baud):
                raise RuntimeError(f"GPS did not come back at {self.baud} baud")
        for name, mid in NMEA_IDS.items():
            self.ser.write(cfg_msg_nmea(mid, 1 if name in ("RMC", "GGA") else 0))
            time.sleep(0.05)
        self.ser.write(cfg_rate(int(1000 / self.rate_hz)))
        time.sleep(0.1)
        self.ser.write(cfg_save())
        self.ser.flush()
        time.sleep(0.3)
        log.info("GPS configured: %d baud, %d Hz, RMC+GGA, saved", self.baud, self.rate_hz)

    def lines(self):
        """Yields (line, arrival monotonic ns) forever."""
        while True:
            raw = self.ser.readline()
            if raw:
                yield raw.decode("ascii", "replace"), time.monotonic_ns()

    def fixes(self):
        for line, t in self.lines():
            if not nmea_checksum_ok(line.strip()):
                self.bad_lines += 1
                continue
            fix = self.asm.feed(line, t)
            if fix is not None:
                yield fix
