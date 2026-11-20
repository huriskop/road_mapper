#!/usr/bin/env python3
"""System test: the Pi's sensors report road defects to the web map.

Starts two pages:
    http://<pi-address>:8080/   Sensors: live camera, MPU-6050, GPS, and every photo sent to the map
    http://<pi-address>:8001/   Map: the project's web console (backend/ + frontend/)

When the vertical jolt passes the threshold (or you press "Photo now"), the photo is sent with
the GPS position and the jolt to POST /api/v1/telemetry. The map server runs the camera
detector on it, merges reports less than 5 m apart into one incident, and the map updates live.

    sudo apt install python3-fastapi python3-uvicorn python3-multipart     # once, on the Pi
    python3 system_test/run.py
    python3 system_test/run.py --fallback-location 37.4563,126.7052       # indoors, no GPS fix
    python3 system_test/run.py --roi 0.06,0.12,1,0.72                     # camera looks at a screen
    python3 system_test/run.py --map-url http://192.168.0.5:8001          # use a map server elsewhere

Stop roadflag-edge first (it holds the camera and the GPS port).
"""
from __future__ import annotations

import importlib.util
import json
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "test_environment"))
import server as bench  # noqa: E402  (test_environment/server.py: sensors, camera, trigger, web handler)

MAX_QUEUE = 5          # photos waiting to be sent; more than this are skipped
HINT = {"down": "pothole", "up": "bump"}      # what the jolt suggests, before the camera's verdict


def multipart(fields: dict[str, str], name: str, data: bytes, mime: str) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    out = b""
    for k, v in fields.items():
        out += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    out += (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
            f"Content-Type: {mime}\r\n\r\n").encode() + data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return out, f"multipart/form-data; boundary={boundary}"


class Uploader:
    """Sends every capture to the map server, one at a time, and writes the result into the
    capture log entry (field "upload"), so the sensors page shows it."""

    def __init__(self, store, map_url: str, device_id: str, fallback: tuple[float, float] | None,
                 roi: tuple[float, float, float, float] | None) -> None:
        self.store, self.map_url, self.device_id = store, map_url.rstrip("/"), device_id
        self.fallback, self.roi = fallback, roi
        self.online, self.error = False, None
        self.sent = self.failed = 0
        self.queue: queue.Queue = queue.Queue()
        store.on_add.append(self.submit)
        threading.Thread(target=self._run, daemon=True, name="uploader").start()
        threading.Thread(target=self._watch, daemon=True, name="map-health").start()

    def info(self) -> dict:
        return {"url": self.map_url, "online": self.online, "error": self.error, "sent": self.sent,
                "failed": self.failed, "queued": self.queue.qsize(), "device_id": self.device_id,
                "fallback": self.fallback, "roi": self.roi}

    def submit(self, item: dict, frame: bytes | None, mime: str, name: str | None) -> None:
        if not frame or mime != "image/jpeg":
            return self.store.update(item["id"], upload={"status": "skipped", "message": "no camera photo (camera is simulated)"})
        if item.get("gps_fix") and item.get("lat") is not None:
            where = (item["lat"], item["lng"], "gps")
        elif self.fallback:
            where = (*self.fallback, "fallback")
        else:
            return self.store.update(item["id"], upload={
                "status": "skipped", "message": "no GPS fix; go outside, or start with --fallback-location LAT,LNG"})
        if self.queue.qsize() >= MAX_QUEUE:
            return self.store.update(item["id"], upload={"status": "skipped", "message": "map server busy, photo skipped"})
        self.queue.put((item, frame, name, where))
        self.store.update(item["id"], upload={"status": "queued", "location": where[2]})

    def _crop(self, frame: bytes) -> bytes:
        """With --roi, send only that part of the photo (e.g. the picture on a screen)."""
        if not self.roi:
            return frame
        import cv2
        import numpy as np
        bgr = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        x0, y0, x1, y1 = self.roi
        ok, buf = cv2.imencode(".jpg", bgr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)],
                               [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes() if ok else frame

    def _request(self, path: str, body: bytes | None = None, ctype: str = "", timeout: float = 5) -> dict:
        req = urllib.request.Request(self.map_url + path, data=body, headers={"Content-Type": ctype} if ctype else {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")

    def _send(self, item: dict, frame: bytes, name: str, where: tuple) -> dict:
        lat, lng, source = where
        meta = {
            "device_id": self.device_id,
            "timestamp": item["time"],
            "location": {"lat": lat, "lng": lng},
            "sensor_metadata": {
                "source": "system_test", "trigger": item["reason"], "location_source": source,
                "peak_g": abs(item["peak_g"]) if item.get("peak_g") is not None else None,
                "direction": item["reason"] if item["reason"] in HINT else None,
                "defect_type_hint": HINT.get(item["reason"]),
                "speed_kmh": item.get("speed_kmh"),
            },
        }
        body, ctype = multipart({"metadata": json.dumps(meta)}, name, self._crop(frame), "image/jpeg")
        # the map server runs the detector before answering: several seconds on a Pi
        result = self._request("/api/v1/telemetry", body, ctype, timeout=90)
        inc = self._request(f"/api/v1/incidents/{result['incident_id']}")
        mine = next((i for i in inc.get("images", []) if i.get("packet_id") == result.get("packet_id")), {})
        vision = (mine.get("sensor_metadata") or {}).get("vision") or {}
        return {"status": "sent", "location": source, "incident_id": result["incident_id"],
                "created": result["incident_created"], "reports": result.get("report_count"),
                "defect_type": inc.get("defect_type"), "severity": inc.get("severity"),
                "image": mine.get("url"), "camera": {k: vision.get(k) for k in ("potholes", "speed_bumps", "cracks")}}

    def _run(self) -> None:
        while True:
            item, frame, name, where = self.queue.get()
            self.store.update(item["id"], upload={"status": "sending", "location": where[2]})
            try:
                upload = self._send(item, frame, name, where)
                self.sent += 1
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:300]
                try:
                    detail = json.loads(detail).get("detail", detail)
                except ValueError:
                    pass
                upload = {"status": "error", "message": f"map server said {exc.code}: {detail}"}
                self.failed += 1
            except Exception as exc:                    # server down, timeout, bad photo: keep going
                upload = {"status": "error", "message": f"could not reach the map server ({exc})"}
                self.failed += 1
            self.store.update(item["id"], upload=upload)

    def _watch(self) -> None:
        while True:
            try:
                self._request("/api/v1/stats", timeout=3)
                self.online, self.error = True, None
            except Exception as exc:
                self.online, self.error = False, f"map server not answering at {self.map_url} ({exc})"
            time.sleep(3)


class System(bench.Bench):
    def __init__(self, args, uploader_args: dict) -> None:
        super().__init__(args)
        self.uploader = Uploader(self.store, **uploader_args)

    def state(self) -> dict:
        s = super().state()
        s["map"] = self.uploader.info()
        return s


class Handler(bench.Handler):
    server_version = "RDMSystemTest/1.0"

    def do_GET(self) -> None:
        if urlsplit(self.path).path == "/":
            return self._file(HERE / "static" / "sensors.html")
        super().do_GET()


def start_map_server(port: int) -> subprocess.Popen:
    missing = [m for m in ("fastapi", "uvicorn", "python_multipart") if importlib.util.find_spec(m) is None]
    if missing:
        raise SystemExit(f"the map server needs {', '.join(missing)}. On the Pi:\n"
                         "    sudo apt install python3-fastapi python3-uvicorn python3-multipart\n"
                         "or use a map server running elsewhere with --map-url http://<host>:<port>")
    env = {**os.environ, "RDM_DATA_DIR": os.environ.get("RDM_DATA_DIR", str(HERE / "data"))}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0",
                             "--port", str(port), "--log-level", "warning"], cwd=ROOT, env=env)
    time.sleep(1.0)
    if proc.poll() is not None:
        raise SystemExit(f"the map server stopped right away (exit code {proc.returncode}); "
                         f"is port {port} already in use? Try --map-port {port + 1}")
    return proc


def main() -> None:
    ap = bench.parser(__doc__)
    ap.add_argument("--map-port", type=int, default=8001,
                    help="port for the map server this script starts (8000 is roadflag-server)")
    ap.add_argument("--map-url", default="",
                    help="send to a map server that is already running (http://host:port); none is started")
    ap.add_argument("--device-id", default="pi-system-test", help="vehicle name shown on the map")
    ap.add_argument("--fallback-location", default="",
                    help="LAT,LNG to use while the GPS has no fix (indoor tests); must be inside Incheon")
    ap.set_defaults(detect="",                           # the map server runs the detector
                    captures=str(HERE / "captures"))
    args = ap.parse_args()

    fallback = None
    if args.fallback_location:
        try:
            lat, lng = (float(v) for v in args.fallback_location.split(","))
            fallback = (lat, lng)
        except ValueError:
            raise SystemExit("--fallback-location needs LAT,LNG, e.g. 37.4563,126.7052")
    roi = tuple(float(v) for v in args.roi.split(",")) if args.roi else None
    if roi and not (len(roi) == 4 and 0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        raise SystemExit("--roi needs x0,y0,x1,y1 as fractions of the frame, e.g. 0.06,0.12,1,0.72")
    args.roi = ""                                        # the crop happens before sending, not in the bench

    proc = None if args.map_url else start_map_server(args.map_port)
    map_url = args.map_url or f"http://127.0.0.1:{args.map_port}"
    system = System(args, dict(map_url=map_url, device_id=args.device_id, fallback=fallback, roi=roi))
    server = bench.serve(system, args, Handler)
    host = bench.lan_address()
    shown_map = map_url.replace("127.0.0.1", host)
    print(f"\nSensors: http://{host}:{args.port}/")
    print(f"Map:     {shown_map}/")
    if fallback:
        print(f"Without a GPS fix, photos are placed at {fallback[0]}, {fallback[1]}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=10)


if __name__ == "__main__":
    main()
