#!/usr/bin/env python3
"""Road defect monitor for the Raspberry Pi: sensors, detection and map in one program.

    python3 run.py
    python3 run.py --fallback-location 37.4563,126.7052     # indoors: position to use without a GPS fix
    python3 run.py --roi 0.06,0.12,1,0.72                   # the camera looks at a picture on a screen

Pages:  http://<pi-address>:8080/          equipment check (camera, MPU-6050, GPS)
        http://<pi-address>:8080/capture   a jolt takes a photo, the detector labels it
        http://<pi-address>:8080/map       detected defects on a map
        http://<pi-address>:8080/test      try the detector on your own photos

Stop any other program that uses the camera or the GPS first.
"""
from __future__ import annotations

import argparse
import json
import signal
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from app.capture import CaptureStore, Trigger
from app.detection import TYPES, Analyzer
from app.incidents import Incidents
from app.sensors import Camera, Gps, Imu, now_iso

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
PAGES = {"/": "index.html", "/capture": "capture.html", "/map": "map.html", "/test": "test.html"}
MIME = {".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "application/javascript", ".jpg": "image/jpeg"}
VIDEO_FPS = 3        # live view only. Sending much data over Wi-Fi stalls the camera (see README.md)
CLIENT_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


def parse_location(text: str) -> tuple[float, float]:
    lat, lng = (float(v) for v in text.split(","))
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        raise ValueError("not a position on Earth")
    return lat, lng


def parse_roi(text: str) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(v) for v in text.split(","))
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("fractions of the frame, e.g. 0.05,0.12,1,0.72")
    return x0, y0, x1, y1


def parse_detect(text: str) -> tuple[str, ...]:
    kinds = tuple(k.strip() for k in text.split(",") if k.strip())
    if not kinds or set(kinds) - set(TYPES):
        raise ValueError(f"choose from {', '.join(TYPES)}")
    return kinds


class Monitor:
    """Connects the parts: a jolt takes a photo, the detector checks it, the result goes on the map."""

    def __init__(self, args: argparse.Namespace) -> None:
        data = Path(args.data)
        self.fallback = args.fallback_location
        self.imu, self.gps, self.camera = Imu(), Gps(args.gps_port, args.gps_baud), Camera()
        self.store = CaptureStore(data / "captures")
        self.incidents = Incidents(data / "incidents.json")
        self.analyzer = Analyzer(self.store, args.detect, args.pothole_conf, roi=args.roi, on_result=self.to_map)
        self.trigger = Trigger(self.imu, self.camera, self.gps, self.store, args.threshold)

    def to_map(self, item: dict) -> None:
        if item["reason"] == "manual":                        # a test photo, not a road event
            return
        if item["gps_fix"] and item["lat"] is not None:
            (lat, lng), source = (item["lat"], item["lng"]), "gps"
        elif self.fallback:
            (lat, lng), source = self.fallback, "fallback"
        else:
            return self.store.update(item["id"], map={"status": "no GPS fix, not on the map"})
        inc = self.incidents.add(lat=lat, lng=lng, time=item["time"], image=item["annotated"] or item["image"],
                                 reason=item["reason"], peak_g=item["peak_g"], analysis=item["analysis"], location=source)
        self.store.update(item["id"], map={"id": inc["id"], "type": inc["defect_type"], "severity": inc["severity"],
                                           "location": source})

    def check_photo(self, photo: bytes, detect: tuple[str, ...], conf: float | None, where: tuple[float, float] | None) -> dict:
        """The photo test page: label one photo and, when a position is given, put it on the map."""
        if self.analyzer.vd is None:
            raise ValueError(self.analyzer.error)
        labelled, result = self.analyzer.analyze(photo, detect, conf)
        name = f"upload_{time.strftime('%Y%m%d-%H%M%S')}_{time.time_ns() % 1_000_000:06d}.jpg"
        out = {**result, "detect": detect, "models": self.analyzer.models}
        if where:
            (self.store.dir / name).write_bytes(labelled)
            out["incident"] = self.incidents.add(lat=where[0], lng=where[1], time=now_iso(), image=f"/captures/{name}",
                                                 reason="upload", peak_g=None, analysis=result, location="typed in")
        else:
            import base64
            out["image"] = "data:image/jpeg;base64," + base64.b64encode(labelled).decode()
        return out

    def state(self) -> dict:
        return {"t": time.time(), "imu": self.imu.snapshot(), "gps": self.gps.snapshot(), "camera": self.camera.info(),
                "trigger": self.trigger.info(), "detector": self.analyzer.info()}


MON: Monitor


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):                        # keep the console for errors only
        pass

    # ------------------------------------------------------------------ helpers
    def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status: int = 200) -> None:
        self._send(json.dumps(data).encode(), "application/json", status)

    def _file(self, path: Path) -> None:
        if not path.is_file():
            return self._json({"error": "not found"}, 404)
        self._send(path.read_bytes(), MIME.get(path.suffix, "application/octet-stream"))

    def _raw(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def _body(self) -> dict:
        data = json.loads(self._raw() or b"{}")
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data

    def _handle(self, route) -> None:
        url = urlsplit(self.path)
        try:
            route(url.path, {k: v[0] for k, v in parse_qs(url.query).items()})
        except CLIENT_GONE:
            pass
        except (TypeError, ValueError) as exc:
            self._json({"error": str(exc)}, 422)

    def do_GET(self) -> None:
        self._handle(self._get)

    def do_POST(self) -> None:
        self._handle(self._post)

    def do_PATCH(self) -> None:
        self._handle(self._patch)

    def do_DELETE(self) -> None:
        self._handle(self._delete)

    # ------------------------------------------------------------------ routes
    def _get(self, path: str, q: dict) -> None:
        if path in PAGES:
            self._file(WEB / PAGES[path])
        elif path.startswith("/web/"):
            self._file(WEB / Path(path).name)
        elif path.startswith("/captures/"):
            self._file(MON.store.dir / Path(path).name)
        elif path == "/api/state":
            self._json(MON.state())
        elif path == "/api/captures":
            self._json({"items": MON.store.all()})
        elif path == "/api/incidents":
            unchanged = q.get("rev") == str(MON.incidents.rev)
            self._json({"rev": MON.incidents.rev} if unchanged else MON.incidents.all())
        elif path == "/api/camera.mjpg":
            self._mjpeg()
        elif path == "/api/stream":
            self._stream()
        else:
            self._json({"error": "not found"}, 404)

    def _post(self, path: str, q: dict) -> None:
        if path == "/api/photo":                              # body: the image file itself
            photo = self._raw()
            where = parse_location(f"{q['lat']},{q['lng']}") if "lat" in q and "lng" in q else None
            return self._json(MON.check_photo(photo, parse_detect(q.get("detect", ",".join(TYPES))),
                                              float(q["conf"]) if "conf" in q else None, where))
        body = self._body()
        if path == "/api/trigger":
            t = MON.trigger
            if "threshold_g" in body:
                t.threshold = min(4.0, max(0.05, float(body["threshold_g"])))
            if "enabled" in body:
                t.enabled = bool(body["enabled"])
            self._json(t.info())
        elif path == "/api/snapshot":
            self._json(MON.trigger.manual())
        elif path == "/api/imu/zero":
            MON.imu.zero()
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)

    def _patch(self, path: str, q: dict) -> None:
        body = self._body()
        inc = MON.incidents.edit(path.rsplit("/", 1)[-1], body) if path.startswith("/api/incidents/") else None
        self._json(inc) if inc else self._json({"error": "not found"}, 404)

    def _delete(self, path: str, q: dict) -> None:
        ok = path.startswith("/api/incidents/") and MON.incidents.delete(path.rsplit("/", 1)[-1])
        self._json({"ok": True}) if ok else self._json({"error": "not found"}, 404)

    # ------------------------------------------------------------------ live streams
    def _stream(self) -> None:
        """Server-sent events: state 20x per second (with every 100 Hz IMU sample), plus new or
        changed capture log entries (a photo, then its detector result)."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        last_seq = MON.store.seq
        last_t = time.monotonic()
        self.wfile.write(b"retry: 2000\n\n")
        while True:
            state = MON.state()
            state["imu"]["series"], last_t = MON.imu.series_since(last_t)
            out = b"data: " + json.dumps(state).encode() + b"\n\n"
            for item in MON.store.since(last_seq):
                out += b"event: capture\ndata: " + json.dumps(item).encode() + b"\n\n"
                last_seq = item["seq"]
            self.wfile.write(out)
            self.wfile.flush()
            time.sleep(0.05)

    def _mjpeg(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        seq = -1
        while True:
            frame, seq = MON.camera.wait(seq)
            if frame:
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
            time.sleep(1 / VIDEO_FPS)


def lan_address() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))                  # no packet is sent; picks the outgoing interface
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def main() -> None:
    global MON
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--threshold", type=float, default=0.4, help="jolt that takes a photo, in g (gravity removed)")
    ap.add_argument("--detect", type=parse_detect, default=TYPES,
                    help="what to look for (default pothole,crack,speed_bump)")
    ap.add_argument("--pothole-conf", type=float, help="pothole model confidence cut-off (0-1, default 0.25)")
    ap.add_argument("--fallback-location", type=parse_location, metavar="LAT,LNG",
                    help="position for jolts while the GPS has no fix")
    ap.add_argument("--roi", type=parse_roi, metavar="X0,Y0,X1,Y1", help="analyze only this part of each photo (fractions)")
    ap.add_argument("--gps-port", default="/dev/serial0")
    ap.add_argument("--gps-baud", type=int, default=0, help="0 = detect")
    ap.add_argument("--data", default=str(ROOT / "data"), help="folder for photos and incidents.json")
    args = ap.parse_args()
    try:
        server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    except OSError as exc:
        raise SystemExit(f"cannot listen on port {args.port} ({exc.strerror}): is run.py already running?")
    server.daemon_threads = True
    MON = Monitor(args)
    print(f"Open http://{lan_address()}:{args.port}/   (Ctrl+C stops)", flush=True)

    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)                        # also when started in the background
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        MON.camera.shutdown()


if __name__ == "__main__":
    main()
