"""The edge's metadata (edge/roadflag_edge/pipeline.py) is accepted by the server, and the
shared JSON schema, the edge writer and the server model agree on field names."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "edge"))

pytest.importorskip("scipy")

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models import ObservationMetadata  # noqa: E402
from roadflag_edge import pipeline  # noqa: E402
from roadflag_edge.camera.ring import Frame, FrameRing  # noqa: E402
from roadflag_edge.clock import Clock  # noqa: E402
from roadflag_edge.config import Config  # noqa: E402
from roadflag_edge.detect.detector import Candidate  # noqa: E402
from roadflag_edge.detect.localize import ImuRing  # noqa: E402
from roadflag_edge.gps.neo6m import Fix  # noqa: E402
from roadflag_edge.gps.trajectory import Trajectory, offset_latlon  # noqa: E402
from roadflag_edge.store.observations import ObservationStore, new_observation_id  # noqa: E402

SEC = 1_000_000_000
SCHEMA = json.loads((ROOT / "shared" / "observation.schema.json").read_text())


def test_schema_and_server_model_agree():
    assert set(SCHEMA["required"]) <= set(ObservationMetadata.model_fields)
    assert set(ObservationMetadata.model_fields) <= set(SCHEMA["properties"])
    assert set(SCHEMA["properties"]["evidence_status"]["enum"]) == \
        {e.value for e in ObservationMetadata.model_fields["evidence_status"].annotation}


def build_real_observation(tmp_path):
    """Run the edge pipeline on a synthetic hit with GPS and camera, persist it like app.py does."""
    from datetime import datetime, timedelta, timezone

    cfg = Config()
    clock = Clock()
    traj = Trajectory(keep_s=60)
    t0 = datetime(2026, 11, 19, 10, 41, tzinfo=timezone.utc)
    for k in range(60):
        la, lo = offset_latlon(37.4563, 126.7052, 90.0, 8.0 * k * 0.2)
        f = Fix(k * SEC // 5, la, lo, 8.0, 90.0, 1.1, 1, t0 + timedelta(seconds=k * 0.2))
        traj.add(f)
        clock.add_gps_fix(f.t_ns, f.utc)
    ring = FrameRing(120, 64, 48)
    for k in range(int(10 * 24)):
        y = np.random.default_rng(k).integers(0, 255, (72, 64), dtype=np.uint8)
        ring.push(int(k * SEC / 24), k, y)
    imu = ImuRing(12, 200)
    t = np.arange(0, 10 * 200, dtype=np.int64) * (SEC // 200)
    az = np.zeros(len(t))
    az[1800:1830] = 15 * np.sin(np.linspace(0, 3 * np.pi, 30))
    imu.push(t, az, np.zeros(len(t)), np.zeros(len(t)), 1)
    cand = Candidate(t_d=int(9.05 * SEC), peak_a=14, peak_j=900, score=2.3, speed=8.0,
                     t_start=int(9.0 * SEC), t_end=int(9.1 * SEC))
    b = pipeline.build(cand, imu, traj, ring, cfg)
    oid = new_observation_id()
    meta = pipeline.metadata(oid, b, clock, cfg)
    image = b.evidence.frame.to_jpeg()
    store = ObservationStore(tmp_path / "edge")
    store.persist(oid, meta, image, score=cand.score)
    return store, meta, image


def test_edge_observation_accepted_by_server(tmp_path):
    store, meta, image = build_real_observation(tmp_path)
    assert meta["evidence_status"] == "AVAILABLE"
    app = create_app(Settings(data_dir=tmp_path / "srv", worker_enabled=False,
                              device_tokens={meta["device_id"]: "tok"}))
    row = store.claim()
    with TestClient(app) as c:
        r = c.post("/api/v1/observations",
                   files={"metadata": (None, row["metadata"], "application/json"),
                          "image": ("o.jpg", store.read_image(row), "image/jpeg")},
                   headers={"Authorization": "Bearer tok", "Idempotency-Key": row["id"]})
        assert r.status_code == 201, r.text
        obs = c.get(f"/api/v1/observations/{row['id']}").json()
    assert obs["location"]["lat"] == pytest.approx(meta["defect"]["lat"])
    assert obs["t_impact"].startswith("2026-11-19T10:41:")
