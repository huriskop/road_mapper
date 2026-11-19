"""Edge-device telemetry simulator.

Mimics several vehicles driving virtual routes over a synthetic road network that
contains ground-truth defects. Each vehicle samples a synthetic vertical
accelerometer signal; when the edge-side detector fires it captures a (generated)
road photo and POSTs a packet to /api/v1/telemetry. Several devices share routes,
so the same physical defect is reported by multiple device_ids with GPS noise,
which exercises the server's spatial clustering.

Examples
--------
    python simulator/simulate.py                       # fast replay, defaults
    python simulator/simulate.py --devices 10 --defects 40 --passes 3
    python simulator/simulate.py --time-scale 20       # real-time-ish streaming (20x)
    python simulator/simulate.py --images-dir ./samples  # use your own road photos
    python simulator/simulate.py --base64              # send JSON + base64 instead of multipart
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont

M_PER_DEG = 111_320.0
SAMPLE_HZ = 100
DETECT_THRESHOLD_G = 0.28
REFRACTORY_S = 1.5

DEFECT_PROFILES = {
    # type: (amplitude range g, waveform)
    "pothole": ((0.45, 2.2), "drop"),
    "bump": ((0.35, 1.6), "rise"),
    "crack": ((0.30, 0.9), "chatter"),
}


# --------------------------------------------------------------------------- geometry
@dataclass
class Frame:
    lat0: float
    lng0: float

    def to_latlng(self, x: float, y: float) -> tuple[float, float]:
        return (self.lat0 + y / M_PER_DEG, self.lng0 + x / (M_PER_DEG * math.cos(math.radians(self.lat0))))


@dataclass
class Route:
    name: str
    points: list[tuple[float, float]]  # local metres
    cum: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cum = [0.0]
        for (x1, y1), (x2, y2) in zip(self.points, self.points[1:]):
            self.cum.append(self.cum[-1] + math.hypot(x2 - x1, y2 - y1))

    @property
    def length(self) -> float:
        return self.cum[-1]

    def at(self, s: float) -> tuple[float, float]:
        s = max(0.0, min(s, self.length))
        for i in range(1, len(self.cum)):
            if s <= self.cum[i]:
                seg = self.cum[i] - self.cum[i - 1] or 1.0
                t = (s - self.cum[i - 1]) / seg
                (x1, y1), (x2, y2) = self.points[i - 1], self.points[i]
                return x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
        return self.points[-1]


def make_route(rng: random.Random, name: str, extent: float, segments: int) -> Route:
    """Manhattan-style route snapped to a 100 m street grid."""
    grid = 100.0
    x = rng.randint(-int(extent / grid), int(extent / grid)) * grid
    y = rng.randint(-int(extent / grid), int(extent / grid)) * grid
    pts = [(x, y)]
    heading = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
    for _ in range(segments):
        if rng.random() < 0.6:
            heading = rng.choice([(heading[1], heading[0]), (-heading[1], -heading[0])])
        length = rng.randint(2, 5) * grid
        nx, ny = x + heading[0] * length, y + heading[1] * length
        if abs(nx) > extent or abs(ny) > extent:
            heading = (-heading[0], -heading[1])
            nx, ny = x + heading[0] * length, y + heading[1] * length
        x, y = nx, ny
        pts.append((x, y))
    return Route(name, pts)


@dataclass
class Defect:
    idx: int
    route: Route
    s: float           # arc position along route (m)
    kind: str
    amplitude: float   # g
    x: float
    y: float


@dataclass
class Event:
    t: datetime
    device_id: str
    lat: float
    lng: float
    peak_g: float
    rms_g: float
    speed_kmh: float
    hint: str
    truth: int | None  # ground-truth defect index, None = false trigger
    window: list[float]


# ---------------------------------------------------------------- signal + detection
def impulse(kind: str, amp: float, n: int, rng: random.Random) -> list[float]:
    out = []
    for i in range(n):
        u = i / max(n - 1, 1)
        if kind == "drop":      # wheel drops into hole then slams the far edge
            v = -amp * 0.6 * math.sin(math.pi * min(u * 2, 1)) if u < 0.5 else amp * math.sin(math.pi * (u - 0.5) * 2)
        elif kind == "rise":    # compression over bump then rebound
            v = amp * math.sin(math.pi * u) - (0.35 * amp * math.sin(math.pi * (u - 0.6) / 0.4) if u > 0.6 else 0)
        else:                   # crack: short high-frequency chatter
            v = amp * math.sin(u * math.pi * 9) * math.exp(-3 * u)
        out.append(v + rng.gauss(0, 0.03))
    return out


def edge_classify(window: list[float]) -> str:
    """Very rough on-device waveform heuristic (preliminary only)."""
    peak_i = max(range(len(window)), key=lambda i: abs(window[i]))
    pos, neg = max(window), -min(window)
    # Count sign flips between significant excursions only (hysteresis ignores sensor noise).
    signs = [v > 0 for v in window if abs(v) > 0.15]
    crossings = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
    if crossings >= 4 and max(pos, neg) < 1.0:
        return "crack"
    first_neg = next((i for i, v in enumerate(window) if v < -0.15), None)
    if first_neg is not None and first_neg < peak_i and neg > 0.2:
        return "pothole"
    return "bump" if pos >= neg else "unclassified"


def drive(
    rng: random.Random, frame: Frame, device_id: str, route: Route, defects: list[Defect],
    start: datetime, gps_sigma: float, hit_prob: float, false_rate: float,
) -> list[Event]:
    speed = rng.uniform(7, 14)                     # m/s
    dt = 1 / SAMPLE_HZ
    n_samples = int(route.length / speed / dt)
    on_route = sorted((d for d in defects if d.route is route), key=lambda d: d.s)
    signal = [rng.gauss(0, 0.06) for _ in range(n_samples)]
    truth_at: dict[int, int] = {}

    # Inject defect impulses (vehicle may straddle the defect and miss it).
    for d in on_route:
        if rng.random() > hit_prob:
            continue
        i0 = int(d.s / speed / dt)
        n = rng.randint(8, 18)
        amp = d.amplitude * rng.uniform(0.8, 1.15) * (speed / 10) ** 0.5
        for k, v in enumerate(impulse(DEFECT_PROFILES[d.kind][1], amp, n, rng)):
            if 0 <= i0 + k < n_samples:
                signal[i0 + k] += v
                truth_at[i0 + k] = d.idx
    # Random false triggers (manhole covers, expansion joints, door slams...).
    for _ in range(int(n_samples * dt * false_rate)):
        i0 = rng.randrange(n_samples)
        for k, v in enumerate(impulse("rise", rng.uniform(0.3, 0.55), 6, rng)):
            if i0 + k < n_samples:
                signal[i0 + k] += v

    events: list[Event] = []
    last_trigger = -1e9
    i = 0
    while i < n_samples:
        if abs(signal[i]) >= DETECT_THRESHOLD_G and (i - last_trigger) * dt > REFRACTORY_S:
            lo, hi = max(0, i - 10), min(n_samples, i + 25)
            window = signal[lo:hi]
            peak = max(window, key=abs)
            truth = next((truth_at[j] for j in range(lo, hi) if j in truth_at), None)
            if truth is not None:  # GPS fix of the actual defect + receiver noise
                d = defects[truth]
                x, y = d.x, d.y
            else:
                x, y = route.at(i * dt * speed)
            x += rng.gauss(0, gps_sigma)
            y += rng.gauss(0, gps_sigma)
            lat, lng = frame.to_latlng(x, y)
            events.append(Event(
                t=start + timedelta(seconds=i * dt), device_id=device_id, lat=lat, lng=lng,
                peak_g=round(abs(peak), 3), rms_g=round(math.sqrt(sum(v * v for v in window) / len(window)), 3),
                speed_kmh=round(speed * 3.6, 1), hint=edge_classify(window), truth=truth,
                window=[round(v, 3) for v in window],
            ))
            last_trigger = i
            i = hi
        else:
            i += 1
    return events


# --------------------------------------------------------------------------- images
def synth_road_image(rng: random.Random, kind: str | None, caption: str) -> bytes:
    w, h = 640, 480
    base = Image.effect_noise((w, h), 38).convert("RGB")
    tint = Image.new("RGB", (w, h), (rng.randint(70, 95),) * 3)
    img = Image.blend(base, tint, 0.65).filter(ImageFilter.GaussianBlur(0.6))
    d = ImageDraw.Draw(img)
    # lane marking in perspective
    for k in range(5):
        y0 = 60 + k * 90
        d.polygon([(300 + k * 4, y0), (340 + k * 4, y0), (346 + k * 6, y0 + 50), (294 + k * 2, y0 + 50)], fill=(225, 220, 190))
    cx, cy = rng.randint(180, 460), rng.randint(200, 330)
    if kind == "pothole":
        rx, ry = rng.randint(60, 130), rng.randint(35, 70)
        d.ellipse([cx - rx - 8, cy - ry - 6, cx + rx + 8, cy + ry + 6], fill=(58, 55, 52))
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(28, 26, 25))
        d.ellipse([cx - rx // 2, cy - ry // 3, cx + rx // 3, cy + ry // 2], fill=(45, 50, 58))  # puddle
    elif kind == "crack":
        x, y = cx - 200, cy
        for _ in range(28):
            nx, ny = x + rng.randint(8, 20), y + rng.randint(-14, 14)
            d.line([(x, y), (nx, ny)], fill=(20, 20, 20), width=rng.randint(2, 5))
            if rng.random() < 0.25:
                d.line([(nx, ny), (nx + rng.randint(-25, 25), ny + rng.randint(-30, 30))], fill=(30, 30, 30), width=2)
            x, y = nx, ny
    elif kind == "bump":
        for k in range(8):
            shade = 120 + k * 10
            d.ellipse([cx - 220 + k * 6, cy - 30 + k * 3, cx + 220 - k * 6, cy + 30 - k * 3], fill=(shade, shade, shade - 5))
    else:  # false trigger: manhole cover
        d.ellipse([cx - 55, cy - 30, cx + 55, cy + 30], fill=(60, 60, 64), outline=(30, 30, 30), width=4)
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()
    d.rectangle([0, h - 28, w, h], fill=(0, 0, 0))
    d.text((8, h - 23), caption, fill=(255, 255, 255), font=font)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=78)
    return buf.getvalue()


def load_sample_images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


# --------------------------------------------------------------------------- main
def build_world(args, rng: random.Random):
    routes = [make_route(rng, f"route-{i + 1}", args.extent, rng.randint(5, 9)) for i in range(args.routes)]
    defects: list[Defect] = []
    for i in range(args.defects):
        r = rng.choice(routes)
        s = rng.uniform(20, r.length - 20)
        kind = rng.choices(list(DEFECT_PROFILES), weights=[5, 2, 3])[0]
        lo, hi = DEFECT_PROFILES[kind][0]
        x, y = r.at(s)
        defects.append(Defect(i, r, s, kind, rng.uniform(lo, hi), x, y))
    return routes, defects


def post_event(session: requests.Session, url: str, ev: Event, image: bytes, mime: str, use_b64: bool) -> dict:
    meta = {
        "device_id": ev.device_id,
        "timestamp": ev.t.isoformat().replace("+00:00", "Z"),
        "location": {"lat": ev.lat, "lng": ev.lng},
        "sensor_metadata": {
            "peak_g": ev.peak_g, "rms_g": ev.rms_g, "speed_kmh": ev.speed_kmh,
            "sample_rate_hz": SAMPLE_HZ, "defect_type_hint": ev.hint, "window_g": ev.window,
        },
    }
    if use_b64:
        meta["image"] = f"data:{mime};base64," + base64.b64encode(image).decode()
        resp = session.post(url, json=meta, timeout=30)
    else:
        resp = session.post(url, data={"metadata": json.dumps(meta)},
                            files={"image": (f"{ev.device_id}.jpg", image, mime)}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api", default="http://127.0.0.1:8000", help="backend base URL")
    p.add_argument("--center", default="37.4563,126.7052", help="lat,lng of the simulated area (default: Incheon City Hall)")
    p.add_argument("--extent", type=float, default=900, help="half-size of the area in metres")
    p.add_argument("--routes", type=int, default=4)
    p.add_argument("--devices", type=int, default=8)
    p.add_argument("--defects", type=int, default=30)
    p.add_argument("--passes", type=int, default=2, help="how many times each device drives its route")
    p.add_argument("--gps-sigma", type=float, default=1.2, help="GPS noise std-dev in metres")
    p.add_argument("--hit-prob", type=float, default=0.8, help="chance a wheel hits a given defect")
    p.add_argument("--false-rate", type=float, default=0.004, help="false triggers per second of driving")
    p.add_argument("--time-scale", type=float, default=0, help="0 = send as fast as possible; N = replay at N x real time")
    p.add_argument("--images-dir", type=Path, help="folder of sample road photos to use instead of synthetic ones")
    p.add_argument("--base64", action="store_true", help="send JSON with base64 image instead of multipart")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="simulate only; do not send")
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    lat0, lng0 = (float(v) for v in args.center.split(","))
    frame = Frame(lat0, lng0)
    routes, defects = build_world(args, rng)
    samples = load_sample_images(args.images_dir) if args.images_dir else []
    if args.images_dir and not samples:
        print(f"no images found in {args.images_dir}", file=sys.stderr)
        return 2

    # Devices are assigned round-robin so every route is driven by several vehicles.
    start = datetime.now(timezone.utc) - timedelta(hours=args.passes * 2)
    events: list[Event] = []
    for dev in range(args.devices):
        device_id = f"SIM-{dev + 1:03d}"
        route = routes[dev % len(routes)]
        for pss in range(args.passes):
            t0 = start + timedelta(minutes=pss * 90 + rng.uniform(0, 60))
            events += drive(rng, frame, device_id, route, defects, t0, args.gps_sigma, args.hit_prob, args.false_rate)
    events.sort(key=lambda e: e.t)

    true_hits = {e.truth for e in events if e.truth is not None}
    print(f"world: {len(routes)} routes, {len(defects)} defects ({len(true_hits)} detected at least once)")
    print(f"generated {len(events)} packets from {args.devices} devices "
          f"({sum(e.truth is None for e in events)} false triggers)")
    if args.dry_run:
        return 0

    url = args.api.rstrip("/") + "/api/v1/telemetry"
    session = requests.Session()
    incidents: set[str] = set()
    new_incidents = 0
    prev_t = events[0].t if events else None
    for n, ev in enumerate(events, 1):
        if args.time_scale > 0 and prev_t is not None:
            time.sleep(max(0.0, (ev.t - prev_t).total_seconds() / args.time_scale))
        prev_t = ev.t
        if samples:
            path = rng.choice(samples)
            image = path.read_bytes()
            mime = {".png": "image/png", ".webp": "image/webp"}.get(path.suffix.lower(), "image/jpeg")
        else:
            kind = defects[ev.truth].kind if ev.truth is not None else None
            image = synth_road_image(rng, kind, f"{ev.device_id}  {ev.t:%Y-%m-%d %H:%M:%S}Z  {ev.peak_g:.2f}g")
            mime = "image/jpeg"
        try:
            res = post_event(session, url, ev, image, mime, args.base64)
        except requests.RequestException as exc:
            print(f"[{n}/{len(events)}] send failed: {exc}", file=sys.stderr)
            if isinstance(exc, requests.ConnectionError):
                print("Is the backend running? (see README)", file=sys.stderr)
                return 1
            continue
        incidents.add(res["incident_id"])
        new_incidents += res["incident_created"]
        tag = "NEW   " if res["incident_created"] else "MERGED"
        print(f"[{n:4}/{len(events)}] {tag} {ev.device_id} -> {res['incident_id'][:8]} "
              f"(devices={res['report_count']}, peak={ev.peak_g:.2f}g, hint={ev.hint})")

    print("\n--- clustering summary ---")
    print(f"packets sent:            {len(events)}")
    print(f"distinct incidents:      {len(incidents)}")
    print(f"ground-truth defects hit:{len(true_hits):>4}  (+ false triggers create their own incidents)")
    print(f"dedup ratio:             {len(events) / max(len(incidents), 1):.2f} packets per incident")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
