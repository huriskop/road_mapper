"""POST /api/v1/observations (Pi upload), the classification worker, de-dup and review (M8)."""
import hashlib
import io
import json
import math
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import worker as worker_mod  # noqa: E402
from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402

BASE = (37.4563, 126.7052)
TOKEN = "secret-token"


def road_jpeg(seed=0, flat=False) -> bytes:
    rng = np.random.default_rng(seed)
    arr = np.full((240, 320, 3), 120, np.uint8) if flat else rng.integers(60, 200, (240, 320, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def offset(lat, lon, heading, metres):
    h = math.radians(heading)
    return (lat + math.degrees(metres * math.cos(h) / 6_371_008.8),
            lon + math.degrees(metres * math.sin(h) / (6_371_008.8 * math.cos(math.radians(lat)))))


def make(image=b"", status="AVAILABLE", lat=BASE[0], lon=BASE[1], heading=90.0, score=2.5, device="rpi4-01"):
    meta = {
        "schema_version": 1, "observation_id": str(uuid.uuid4()), "device_id": device,
        "evidence_status": status, "t_impact_mono_ns": 123, "t_impact_utc": "2026-11-19T10:41:07.412Z",
        "alignment_error_ms": 11.8, "speed_before_mps": 8.4, "lookahead_m": 4.0,
        "defect": {"lat": lat, "lon": lon} if lat is not None else None, "heading_deg": heading, "hdop": 1.1,
        "imu": {"peak_az_mps2": 14.2, "peak_jz_mps3": 610, "score": score, "axle2_seen": True},
        "image_sha256": hashlib.sha256(image).hexdigest() if image else None, "firmware": "0.3.0",
    }
    return meta, image


def post(client, meta, image, token=TOKEN, key=None):
    files = {"metadata": (None, json.dumps(meta), "application/json")}
    if image:
        files["image"] = ("o.jpg", image, "image/jpeg")
    headers = {"Authorization": f"Bearer {token}", "Idempotency-Key": key or meta["observation_id"]}
    return client.post("/api/v1/observations", files=files, headers=headers)


@pytest.fixture()
def app(tmp_path):
    a = create_app(Settings(data_dir=tmp_path, worker_enabled=False, device_tokens={"rpi4-01": TOKEN}))
    with TestClient(a) as c:
        a.state.client = c
        yield a


@pytest.fixture()
def client(app):
    return app.state.client


@pytest.fixture()
def verdict(monkeypatch):
    """Set what the classifier says: verdict('pothole', 0.9)."""
    state = {"v": ("pothole", 0.9)}

    def fake(image, settings):
        label, conf = state["v"]
        return label, conf, image, {"camera_type": label if label in ("pothole", "bump") else None}

    monkeypatch.setattr(worker_mod, "classify", fake)
    return lambda label, conf: state.update(v=(label, conf))


def work(app):
    out = []
    while (r := app.state.worker.run_once()) is not None:
        out.append(r)
    return out


# --------------------------------------------------------------- ingest
def test_upload_is_idempotent(client):
    meta, img = make(road_jpeg())
    r1 = post(client, meta, img)
    assert r1.status_code == 201, r1.text
    r2 = post(client, meta, img)
    assert r2.status_code == 200 and r2.json() == r1.json()
    assert r1.json()["observation_id"] == meta["observation_id"]
    assert client.get("/api/v1/observations").json()["total"] == 1


def test_same_id_different_content_conflicts(client):
    meta, img = make(road_jpeg(1))
    assert post(client, meta, img).status_code == 201
    img2 = road_jpeg(2)
    meta2 = {**meta, "image_sha256": hashlib.sha256(img2).hexdigest()}
    assert post(client, meta2, img2).status_code == 409


@pytest.mark.parametrize("change,code", [
    ({"token": "nope"}, 401),
    ({"device": "someone-else"}, 403),
    ({"key": "00000000-0000-0000-0000-000000000000"}, 400),
    ({"sha": "0" * 64}, 422),
    ({"drop_image": True}, 422),
    ({"lat": 10.0}, 422),            # outside the service area
])
def test_rejections(client, change, code):
    meta, img = make(road_jpeg(), lat=change.get("lat", BASE[0]), device=change.get("device", "rpi4-01"))
    if "sha" in change:
        meta["image_sha256"] = change["sha"]
    if change.get("drop_image"):
        img = b""
    r = post(client, meta, img, token=change.get("token", TOKEN), key=change.get("key"))
    assert r.status_code == code, r.text


def test_imageless_observation_accepted(client):
    meta, _ = make(status="NO_GPS_FIX", lat=None)
    r = post(client, meta, b"")
    assert r.status_code == 201, r.text


# --------------------------------------------------------------- worker
def test_confirmed_creates_defect_and_retry_never_double_counts(app, client, verdict):
    meta, img = make(road_jpeg())
    post(client, meta, img)
    [res] = work(app)
    assert res["status"] == "CONFIRMED" and res["incident"]["incident_created"]
    post(client, meta, img)                       # Pi retry after a lost response
    assert work(app) == []
    geo = client.get("/api/v1/defects").json()
    [f] = geo["features"]
    assert f["properties"]["label"] == "pothole" and f["properties"]["observation_count"] == 1
    assert f["properties"]["confidence"] == 0.9
    assert f["geometry"]["coordinates"] == pytest.approx([BASE[1], BASE[0]])


def test_dedupe_by_distance_heading_and_label(app, client, verdict):
    seed = iter(range(100))
    def send(lat, lon, heading, label="pothole"):
        verdict(label, 0.9)
        m, i = make(road_jpeg(next(seed)), lat=lat, lon=lon, heading=heading)
        assert post(client, m, i).status_code == 201
        [r] = work(app)
        return r["incident"]

    first = send(*BASE, 90)
    near = send(*offset(*BASE, 0, 5), 100)                 # 5 m away, same direction: same defect
    assert near["incident_id"] == first["incident_id"] and not near["incident_created"]
    opposite = send(*offset(*BASE, 0, 4), 270)             # other lane
    assert opposite["incident_created"]
    far = send(*offset(*BASE, 0, 20), 90)                  # 20 m: new defect
    assert far["incident_created"]
    crack = send(*offset(*BASE, 0, 2), 90, "crack")        # same spot, different label
    assert crack["incident_created"]
    inc = client.get(f"/api/v1/incidents/{first['incident_id']}").json()
    assert inc["packet_count"] == 2


def test_decisions(app, client, verdict):
    verdict("pothole", 0.55)
    m, i = make(road_jpeg(1))
    post(client, m, i)
    assert work(app)[0]["status"] == "NEEDS_REVIEW"
    verdict("nothing", 0.75)
    m, i = make(road_jpeg(2))
    post(client, m, i)
    assert work(app)[0]["status"] == "REJECTED"
    verdict("pothole", 0.95)
    m, i = make(road_jpeg(3, flat=True))                  # featureless = blurred: quality gate
    post(client, m, i)
    r = work(app)[0]
    assert r["status"] == "NEEDS_REVIEW"
    assert client.get(f"/api/v1/observations/{m['observation_id']}").json()["quality"]["reason"] == "blurred"


def test_imageless_goes_by_imu_score(app, client, verdict):
    hi, _ = make(status="BUFFER_MISS", score=3.0)
    lo, _ = make(status="LOW_SPEED", score=1.2)
    post(client, hi, b"")
    post(client, lo, b"")
    got = {r["observation_id"]: r["status"] for r in work(app)}
    assert got == {hi["observation_id"]: "NEEDS_REVIEW", lo["observation_id"]: "REJECTED"}
    r = client.patch(f"/api/v1/observations/{hi['observation_id']}/review", json={"decision": "confirm",
                                                                                   "label": "pothole"})
    assert r.status_code == 422                            # no photo: cannot go on the map


def test_review_confirm_then_conflict(app, client, verdict):
    verdict("crack", 0.6)
    m, i = make(road_jpeg())
    post(client, m, i)
    work(app)
    queue = client.get("/api/v1/observations", params={"status": "NEEDS_REVIEW"}).json()
    assert queue["total"] == 1 and queue["items"][0]["annotated_url"]
    oid = m["observation_id"]
    r = client.patch(f"/api/v1/observations/{oid}/review", json={"decision": "confirm", "label": "pothole"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "CONFIRMED" and body["label"] == "pothole" and body["decided_by"] == "operator"
    inc = client.get(f"/api/v1/incidents/{body['incident_id']}").json()
    assert inc["defect_type"] == "pothole" and inc["type_source"] == "operator"
    again = client.patch(f"/api/v1/observations/{oid}/review", json={"decision": "reject"})
    assert again.status_code == 409
    stats = client.get("/api/v1/stats").json()
    assert stats["observations"] == {"CONFIRMED": 1} and stats["review_backlog"] == 0


def test_worker_crash_retries_then_fails(app, client, monkeypatch):
    def boom(image, settings):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(worker_mod, "classify", boom)
    m, i = make(road_jpeg())
    post(client, m, i)
    results = work(app)
    assert [r["status"] for r in results] == ["RETRY"] * 3
    assert client.get(f"/api/v1/observations/{m['observation_id']}").json()["status"] == "FAILED"


def test_expired_lease_is_retried(app, client, verdict):
    m, i = make(road_jpeg())
    post(client, m, i)
    db = app.state.db
    assert db.claim_job(lease_s=-1, max_attempts=3) is not None   # a worker took it and died
    [r] = work(app)
    assert r["status"] == "CONFIRMED"


def test_defects_geojson_filters(app, client, verdict):
    m, i = make(road_jpeg())
    post(client, m, i)
    work(app)
    w, s, e, n = BASE[1] - 0.01, BASE[0] - 0.01, BASE[1] + 0.01, BASE[0] + 0.01
    assert len(client.get("/api/v1/defects", params={"bbox": f"{w},{s},{e},{n}"}).json()["features"]) == 1
    assert len(client.get("/api/v1/defects", params={"bbox": f"{e},{n},{e + 1},{n + 1}"}).json()["features"]) == 0
    assert len(client.get("/api/v1/defects", params={"label": "crack"}).json()["features"]) == 0
    assert client.get("/api/v1/defects", params={"bbox": "1,2"}).status_code == 422
