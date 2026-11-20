#!/usr/bin/env python3
"""Equipment test bench for the Raspberry Pi: camera, MPU-6050 and GPS on one web page.

    python3 test_environment/server.py            # real hardware where found, simulators otherwise
    python3 test_environment/server.py --sim      # everything simulated (try it on a laptop)

Pages:  http://<pi-address>:8080/          equipment check (camera, IMU sphere, GPS)
        http://<pi-address>:8080/capture   auto photo when the sensor jolts up or down

Standard library only. Optional: smbus2 (MPU-6050), pyserial (GPS), picamera2 or opencv (camera).
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from detection import Analyzer
from sensors import Camera, CaptureStore, Gps, Imu, Trigger

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
PAGES = {"/": "index.html", "/capture": "capture.html"}
MIME = {".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "application/javascript",
        ".jpg": "image/jpeg", ".svg": "image/svg+xml"}
CLIENT_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


class Bench:
    def __init__(self, args: argparse.Namespace) -> None:
        self.imu = Imu(args.i2c_bus, int(args.mpu_address, 0), simulate=args.sim)
        self.gps = Gps(args.gps_port, args.gps_baud, simulate=args.sim)
        self.camera = Camera("sim" if args.sim else args.camera, (args.width, args.height), args.camera_index)
        self.store = CaptureStore(Path(args.captures))
        self.trigger = Trigger(self.imu, self.camera, self.gps, self.store, args.threshold, args.refractory)
        detect = tuple(k.strip() for k in args.detect.split(",") if k.strip())
        roi = tuple(float(v) for v in args.roi.split(",")) if args.roi else None
        if roi and not (len(roi) == 4 and 0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
            raise SystemExit("--roi needs x0,y0,x1,y1 as fractions of the frame, e.g. 0.05,0.12,1,0.72")
        self.analyzer = Analyzer(self.store, detect, args.pothole_conf, roi=roi) if detect else None

    def state(self) -> dict:
        detector = self.analyzer.info() if self.analyzer else {"available": False, "error": "turned off (--detect '')"}
        return {"t": time.time(), "imu": self.imu.snapshot(), "gps": self.gps.snapshot(),
                "camera": self.camera.info(), "trigger": self.trigger.info(), "detector": detector}


BENCH: Bench


class Handler(BaseHTTPRequestHandler):
    server_version = "RDMTestBench/1.0"

    def log_message(self, fmt, *args):  # keep the console for errors only
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

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    # ------------------------------------------------------------------ GET
    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        try:
            if path in PAGES:
                self._file(STATIC / PAGES[path])
            elif path.startswith("/static/"):
                self._file(STATIC / Path(path).name)          # basename only: no path traversal
            elif path.startswith("/captures/"):
                self._file(BENCH.store.dir / Path(path).name)
            elif path == "/api/state":
                self._json(BENCH.state())
            elif path == "/api/captures":
                self._json({"items": BENCH.store.all()})
            elif path == "/api/frame":
                frame, mime = BENCH.camera.latest()
                self._send(frame, mime) if frame else self._json({"error": "no frame yet"}, 503)
            elif path == "/api/camera.mjpg":
                self._mjpeg()
            elif path == "/api/stream":
                self._stream()
            else:
                self._json({"error": "not found"}, 404)
        except CLIENT_GONE:
            pass

    def _stream(self) -> None:
        """Server-sent events: state 20x per second (with every 100 Hz IMU sample), plus new or
        changed capture log entries (a photo, then its detector result)."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        last_seq = BENCH.store.seq
        last_t = time.monotonic()
        self.wfile.write(b"retry: 2000\n\n")
        while True:
            state = BENCH.state()
            state["imu"]["series"], last_t = BENCH.imu.series_since(last_t)
            out = b"data: " + json.dumps(state).encode() + b"\n\n"
            for item in BENCH.store.since(last_seq):
                out += b"event: capture\ndata: " + json.dumps(item).encode() + b"\n\n"
                last_seq = item["seq"]
            self.wfile.write(out)
            self.wfile.flush()
            time.sleep(0.05)

    def _mjpeg(self) -> None:
        if BENCH.camera.mime != "image/jpeg":
            return self._json({"error": "camera is simulated; poll /api/frame"}, 409)
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        seq = -1
        while True:
            frame, seq = BENCH.camera.wait(seq)
            if frame:
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")

    # ------------------------------------------------------------------ POST
    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        body = self._body()
        try:
            if path == "/api/trigger":
                t = BENCH.trigger
                if "threshold_g" in body:
                    t.threshold = min(4.0, max(0.05, float(body["threshold_g"])))
                if "refractory_s" in body:
                    t.refractory = min(10.0, max(0.2, float(body["refractory_s"])))
                if "enabled" in body:
                    t.enabled = bool(body["enabled"])
                self._json(t.info())
            elif path == "/api/snapshot":
                self._json(BENCH.trigger.manual())
            elif path == "/api/imu/zero":
                BENCH.imu.zero()
                self._json({"ok": True})
            elif path == "/api/simulate":
                if BENCH.imu.source != "sim":
                    return self._json({"error": "only available while the MPU-6050 is simulated"}, 409)
                kind = body.get("kind")
                if kind not in ("pothole", "bump"):
                    return self._json({"error": "kind must be pothole or bump"}, 422)
                BENCH.imu.simulate_event(kind)
                self._json({"ok": True})
            else:
                self._json({"error": "not found"}, 404)
        except (TypeError, ValueError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.UNPROCESSABLE_ENTITY)
        except CLIENT_GONE:
            pass


def lan_address() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))                  # no packet is sent; picks the outgoing interface
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def parser(description: str = __doc__) -> argparse.ArgumentParser:
    """Command-line options (also used by system_test/run.py)."""
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--sim", action="store_true", help="simulate all three devices")
    ap.add_argument("--i2c-bus", type=int, default=1)
    ap.add_argument("--mpu-address", default="0x68", help="0x68, or 0x69 when AD0 is high")
    ap.add_argument("--gps-port", default="/dev/serial0", help="e.g. /dev/serial0, /dev/ttyUSB0, COM3")
    ap.add_argument("--gps-baud", type=int, default=0, help="0 = detect (tries 38400, 9600, 115200, 57600, 4800)")
    ap.add_argument("--camera", default="auto", choices=["auto", "picamera2", "opencv", "sim"])
    ap.add_argument("--camera-index", type=int, default=0, help="USB camera index for opencv")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--threshold", type=float, default=0.4, help="trigger level in g (vertical, gravity removed)")
    ap.add_argument("--refractory", type=float, default=1.5, help="seconds after a photo before the next one")
    ap.add_argument("--captures", default=str(HERE / "captures"), help="folder for trigger photos")
    ap.add_argument("--detect", default="pothole,crack,speed_bump",
                    help="what the camera detector looks for in each photo; '' turns it off")
    ap.add_argument("--pothole-conf", type=float, default=0.5, help="pothole model confidence cut-off (0-1)")
    ap.add_argument("--roi", default="",
                    help="analyse only this part of each photo: x0,y0,x1,y1 as fractions (0-1). Use it when "
                         "the camera looks at a picture on a screen, so the bezel and desk are left out")
    return ap


def serve(bench: Bench, args: argparse.Namespace, handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    global BENCH
    BENCH = bench
    for name, dev in (("MPU-6050", bench.imu), ("GPS", bench.gps), ("Camera", bench.camera)):
        print(f"{name:9} {dev.source:10} {dev.error or ''}")
    try:
        server = ThreadingHTTPServer((args.host, args.port), handler)
    except OSError as exc:
        raise SystemExit(f"\ncannot listen on port {args.port} ({exc.strerror}): is the test bench already "
                         f"running? Stop it, or use --port {args.port + 1}")
    server.daemon_threads = True
    return server


def main() -> None:
    args = parser().parse_args()
    server = serve(Bench(args), args, Handler)
    print(f"\nEquipment check: http://{lan_address()}:{args.port}/")
    print(f"Auto capture:    http://{lan_address()}:{args.port}/capture")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
