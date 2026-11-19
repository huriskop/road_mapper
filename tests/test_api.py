import base64
import io
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402

BASE = (37.4563, 126.7052)  # Incheon City Hall
M = 1 / 111_320  # ~1 metre of latitude in degrees


def jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (90, 90, 90)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, cluster_radius_m=5))
    with TestClient(app) as c:
        yield c


def send(client, device="D1", lat=BASE[0], lng=BASE[1], ts="2026-11-01T10:00:00Z", peak=0.5, hint="pothole"):
    meta = {"device_id": device, "timestamp": ts, "location": {"lat": lat, "lng": lng},
            "sensor_metadata": {"peak_g": peak, "defect_type_hint": hint}}
    r = client.post("/api/v1/telemetry", data={"metadata": json.dumps(meta)},
                    files={"image": ("a.jpg", jpeg(), "image/jpeg")})
    assert r.status_code == 201, r.text
    return r.json()


def test_clustering_merges_nearby_packets_from_different_devices(client):
    a = send(client, "D1")
    b = send(client, "D2", lat=BASE[0] + 3 * M, ts="2026-11-01T11:00:00Z", peak=1.2)
    c = send(client, "D2", lat=BASE[0] + 1 * M, ts="2026-11-01T12:00:00Z")  # same device again
    far = send(client, "D3", lat=BASE[0] + 20 * M)

    assert a["incident_created"] and not b["incident_created"] and not c["incident_created"]
    assert b["incident_id"] == a["incident_id"] == c["incident_id"]
    assert far["incident_created"] and far["incident_id"] != a["incident_id"]

    inc = client.get(f"/api/v1/incidents/{a['incident_id']}").json()
    assert inc["report_count"] == 2          # unique devices
    assert inc["packet_count"] == 3
    assert len(inc["images"]) == 3
    assert inc["created_at"] == "2026-11-01T10:00:00.000Z"
    assert inc["last_report_at"] == "2026-11-01T12:00:00.000Z"
    assert inc["severity"] == "high"         # auto-classified from max peak 1.2 g
    assert inc["defect_type"] == "pothole"
    assert client.get(inc["images"][0]["url"]).status_code == 200


def test_completed_incidents_are_not_merged(client):
    a = send(client, "D1")
    client.patch(f"/api/v1/incidents/{a['incident_id']}", json={"repair_status": "completed"})
    b = send(client, "D2")
    assert b["incident_created"] and b["incident_id"] != a["incident_id"]


def test_base64_json_ingest(client):
    body = {"device_id": "D9", "timestamp": "2026-11-01T10:00:00Z", "location": {"lat": 37.45, "lng": 126.70},
            "image": "data:image/jpeg;base64," + base64.b64encode(jpeg()).decode()}
    r = client.post("/api/v1/telemetry", json=body)
    assert r.status_code == 201, r.text


def test_rejects_bad_payloads(client):
    meta = {"device_id": "D1", "timestamp": "2026-11-01T10:00:00Z", "location": {"lat": 999, "lng": 0}}
    r = client.post("/api/v1/telemetry", data={"metadata": json.dumps(meta)}, files={"image": ("a.jpg", jpeg(), "image/jpeg")})
    assert r.status_code == 422
    meta["location"]["lat"] = 1
    r = client.post("/api/v1/telemetry", data={"metadata": json.dumps(meta)}, files={"image": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 422
    r = client.post("/api/v1/telemetry", data={"metadata": json.dumps(meta)})
    assert r.status_code == 422


def test_patch_locks_manual_severity(client):
    a = send(client, "D1", peak=0.4)
    r = client.patch(f"/api/v1/incidents/{a['incident_id']}", json={"severity": "false_positive", "defect_type": "crack"})
    assert r.status_code == 200 and r.json()["severity"] == "false_positive"
    send(client, "D2", peak=2.5, hint="bump")
    inc = client.get(f"/api/v1/incidents/{a['incident_id']}").json()
    assert inc["severity"] == "false_positive" and inc["defect_type"] == "crack"
    assert client.patch(f"/api/v1/incidents/{a['incident_id']}", json={"severity": "nope"}).status_code == 422


def test_filter_sort_paginate_delete(client):
    ids = []
    for i, (peak, hint) in enumerate([(0.2, "bump"), (0.7, "pothole"), (2.0, "crack"), (1.2, "pothole")]):
        ids.append(send(client, f"D{i}", lat=BASE[0] + i * 50 * M, ts=f"2026-11-0{i + 1}T10:00:00Z", peak=peak, hint=hint)["incident_id"])

    r = client.get("/api/v1/incidents", params={"sort": "severity", "order": "desc"}).json()
    assert [x["severity"] for x in r["items"]] == ["critical", "high", "medium", "no_repair_needed"]
    assert r["items"][0]["thumbnail_url"]

    r = client.get("/api/v1/incidents", params={"defect_type": "pothole"}).json()
    assert r["total"] == 2
    r = client.get("/api/v1/incidents", params=[("severity", "critical"), ("severity", "high")]).json()
    assert r["total"] == 2
    r = client.get("/api/v1/incidents", params={"severity": "critical,high", "defect_type": "crack"}).json()
    assert r["total"] == 1
    r = client.get("/api/v1/incidents", params={"start": "2026-11-02T00:00:00Z", "end": "2026-11-03T23:59:59Z"}).json()
    assert r["total"] == 2
    r = client.get("/api/v1/incidents", params={"page": 2, "page_size": 3}).json()
    assert len(r["items"]) == 1 and r["pages"] == 2
    assert client.get("/api/v1/incidents", params={"severity": "bogus"}).status_code == 422

    assert len(client.get("/api/v1/incidents/map").json()["items"]) == 4
    assert client.delete(f"/api/v1/incidents/{ids[0]}").status_code == 204
    assert client.get(f"/api/v1/incidents/{ids[0]}").status_code == 404
    assert client.delete(f"/api/v1/incidents/{ids[0]}").status_code == 404
    assert client.get("/api/v1/stats").json()["incidents"] == 3


def test_uploaded_photo_is_annotated_by_vision_detector(client):
    pytest.importorskip("cv2")
    import cv2
    sys.path.insert(0, str(ROOT))
    import vision_detector

    ok, buf = cv2.imencode(".jpg", vision_detector.synthetic_road())
    meta = {"device_id": "CAM1", "timestamp": "2026-11-01T10:00:00Z", "location": {"lat": BASE[0], "lng": BASE[1]},
            "sensor_metadata": {"peak_g": 0.9, "defect_type_hint": "pothole"}}
    r = client.post("/api/v1/telemetry", data={"metadata": json.dumps(meta)},
                    files={"image": ("road.jpg", buf.tobytes(), "image/jpeg")})
    assert r.status_code == 201, r.text
    inc = client.get(f"/api/v1/incidents/{r.json()['incident_id']}").json()
    stored = client.get(inc["images"][0]["url"]).content
    assert stored != buf.tobytes()                       # the labelled version was stored
    img = cv2.imdecode(__import__("numpy").frombuffer(stored, "uint8"), cv2.IMREAD_COLOR)
    red = (img[..., 2] > 200) & (img[..., 1] < 60) & (img[..., 0] < 60)
    assert red.sum() > 500                               # red POTHOLE box drawn on the photo


def test_analyze_endpoint_returns_detections_and_saves_nothing(client):
    pytest.importorskip("cv2")
    import cv2
    sys.path.insert(0, str(ROOT))
    import vision_detector

    ok, buf = cv2.imencode(".jpg", vision_detector.synthetic_road())
    r = client.post("/api/v1/analyze", data={"detect": "pothole,crack", "conf": "0.5"},
                    files={"image": ("road.jpg", buf.tobytes(), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image"].startswith("data:image/jpeg;base64,")
    assert {d["kind"] for d in body["defects"]} & {"longitudinal_crack", "transverse_crack", "alligator_crack"}
    assert client.get("/api/v1/stats").json()["incidents"] == 0          # test photos never reach the map

    r = client.post("/api/v1/analyze", files={"image": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 422
    assert client.get("/test").status_code == 200


def test_camera_classification_beats_vibration_guess_but_not_operator(client):
    photo = ROOT / "test_images" / "user" / "bump" / "rubber_bump_yellow_black.jpg"
    if not photo.exists():
        pytest.skip("user test photo not present")
    pytest.importorskip("cv2")

    def post(device, ts, hint):
        meta = {"device_id": device, "timestamp": ts, "location": {"lat": BASE[0], "lng": BASE[1]},
                "sensor_metadata": {"peak_g": 0.8, "defect_type_hint": hint}}
        r = client.post("/api/v1/telemetry", data={"metadata": json.dumps(meta)},
                        files={"image": ("bump.jpg", photo.read_bytes(), "image/jpeg")})
        assert r.status_code == 201, r.text
        return r.json()["incident_id"]

    iid = post("D1", "2026-11-01T10:00:00Z", "pothole")          # vibration says pothole
    inc = client.get(f"/api/v1/incidents/{iid}").json()
    assert inc["defect_type"] == "bump" and inc["type_source"] == "camera"
    assert inc["images"][0]["sensor_metadata"]["vision"]["camera_type"] == "bump"

    client.patch(f"/api/v1/incidents/{iid}", json={"defect_type": "pothole"})   # operator overrides
    post("D2", "2026-11-01T11:00:00Z", "bump")
    inc = client.get(f"/api/v1/incidents/{iid}").json()
    assert inc["defect_type"] == "pothole" and inc["type_source"] == "operator"


def test_old_database_is_upgraded(tmp_path):
    import sqlite3
    from app.db import SCHEMA, Database
    old_schema = SCHEMA.replace(
        "defect_type_manual INTEGER NOT NULL DEFAULT 0,\n    type_source        TEXT NOT NULL DEFAULT 'sensor'   -- sensor | camera | operator\n",
        "defect_type_manual INTEGER NOT NULL DEFAULT 0\n")
    assert "type_source" not in old_schema
    db_path = tmp_path / "road_defects.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(old_schema)
    conn.execute("INSERT INTO incidents (id, lat, lng, created_at, updated_at, last_report_at, defect_type_manual) "
                 "VALUES ('a', 37.4, 126.7, 'x', 'x', 'x', 1)")
    conn.commit(); conn.close()
    Database(db_path)
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT type_source FROM incidents").fetchone()[0] == "operator"


def test_annotation_can_be_switched_off(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, annotate_images=False))
    with TestClient(app) as c:
        raw = jpeg()
        meta = {"device_id": "D1", "timestamp": "2026-11-01T10:00:00Z", "location": {"lat": BASE[0], "lng": BASE[1]}}
        r = c.post("/api/v1/telemetry", data={"metadata": json.dumps(meta)}, files={"image": ("a.jpg", raw, "image/jpeg")})
        inc = c.get(f"/api/v1/incidents/{r.json()['incident_id']}").json()
        assert c.get(inc["images"][0]["url"]).content == raw


def test_rejects_packets_outside_incheon(client):
    meta = {"device_id": "D1", "timestamp": "2026-11-01T10:00:00Z", "location": {"lat": 37.5665, "lng": 126.9780}}  # Seoul
    r = client.post("/api/v1/telemetry", data={"metadata": json.dumps(meta)}, files={"image": ("a.jpg", jpeg(), "image/jpeg")})
    assert r.status_code == 422 and "Incheon" in r.text
    cfg = client.get("/api/v1/config").json()
    assert cfg["service_area"]["name"] == "Incheon"
