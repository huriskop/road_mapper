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


def test_rejects_packets_outside_incheon(client):
    meta = {"device_id": "D1", "timestamp": "2026-11-01T10:00:00Z", "location": {"lat": 37.5665, "lng": 126.9780}}  # Seoul
    r = client.post("/api/v1/telemetry", data={"metadata": json.dumps(meta)}, files={"image": ("a.jpg", jpeg(), "image/jpeg")})
    assert r.status_code == 422 and "Incheon" in r.text
    cfg = client.get("/api/v1/config").json()
    assert cfg["service_area"]["name"] == "Incheon"
