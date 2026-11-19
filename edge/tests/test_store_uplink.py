"""E7-E8: crash safety of the store, exactly-once delivery against a misbehaving server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from roadflag_edge.store.observations import ObservationStore, new_observation_id, sha256
from roadflag_edge.uplink.uploader import Uploader


def meta(obs_id, status="AVAILABLE"):
    return {"schema_version": 1, "observation_id": obs_id, "device_id": "t", "evidence_status": status,
            "t_impact_mono_ns": 1}


def test_persist_sets_hash_and_queues(tmp_path):
    s = ObservationStore(tmp_path)
    oid = new_observation_id()
    m = meta(oid)
    s.persist(oid, m, b"\xff\xd8\xffjpeg", {"imu.npz": b"npz"})
    row = s.claim()
    assert row["id"] == oid and row["image_sha256"] == sha256(b"\xff\xd8\xffjpeg")
    assert json.loads(row["metadata"])["image_sha256"] == row["image_sha256"]
    assert s.read_image(row) == b"\xff\xd8\xffjpeg"
    assert s.claim() is None                               # leased: not handed out twice


def test_recover_after_crash_between_steps(tmp_path):
    s = ObservationStore(tmp_path)
    # crash after the file rename, before the insert: orphan file
    s._atomic_write("evidence/20261119/orphan.jpg", b"x")
    # crash during a write: temp file
    (tmp_path / "evidence/20261119/half.jpg.tmp").write_bytes(b"x")
    # a good row, and a row whose file vanished (SD card trouble)
    good, lost = new_observation_id(), new_observation_id()
    s.persist(good, meta(good), b"a")
    s.persist(lost, meta(lost), b"b")
    (tmp_path / f"evidence/{__import__('time').strftime('%Y%m%d')}/{lost}.jpg").unlink()
    s.claim()                                              # leased by a process that then died
    rec = ObservationStore(tmp_path).recover()
    assert rec == {"leases_released": 1, "orphan_files": 1, "missing_files": 1, "tmp_files": 1}
    s2 = ObservationStore(tmp_path)
    assert s2.counts() == {"QUEUED": 1, "BLOCKED": 1}
    # invariant: every remaining file has a row and every queued row has its file
    for p in (tmp_path / "evidence").rglob("*"):
        if p.is_file():
            assert good in p.name
    assert s2.claim()["id"] == good


def test_observation_without_image(tmp_path):
    s = ObservationStore(tmp_path)
    oid = new_observation_id()
    s.persist(oid, meta(oid, "BUFFER_MISS"), None)
    row = s.claim()
    assert row["image_path"] is None and json.loads(row["metadata"])["image_sha256"] is None


def test_disk_budget_prunes_accepted_then_low_score(tmp_path):
    s = ObservationStore(tmp_path, budget_bytes=2500)
    ids = [new_observation_id() for _ in range(3)]
    s.persist(ids[0], meta(ids[0]), b"x" * 1000, score=5)
    s.accept(ids[0], "{}")
    s.persist(ids[1], meta(ids[1]), b"x" * 1000, score=1.1)
    s.persist(ids[2], meta(ids[2]), b"x" * 1000, score=9)  # 3000 > 2500: accepted files go first
    assert s.counts() == {"ACCEPTED": 1, "QUEUED": 2}
    assert s.used_bytes() == 2000
    big = new_observation_id()
    s.persist(big, meta(big), b"x" * 1000, score=3)         # 3000 again: lowest-score queued dropped
    assert s.counts()["DROPPED"] == 1
    with s.connect() as c:
        assert c.execute("SELECT id FROM observations WHERE state='DROPPED'").fetchone()[0] == ids[1]


# ---------------------------------------------------------------- uploader
class FakeServer:
    """Idempotent like the real one; modes simulate a bad network/server."""

    def __init__(self):
        self.stored: dict[str, dict] = {}
        self.requests = 0
        self.mode = "ok"
        self.fail_next = 0
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                srv.requests += 1
                body = self.rfile.read(int(self.headers["Content-Length"]))
                key = self.headers["Idempotency-Key"]
                if self.headers.get("Authorization") != "Bearer tok":
                    return self._send(401, {"detail": "bad token"})
                if srv.fail_next > 0:
                    srv.fail_next -= 1
                    return self._send(500, {"detail": "boom"})
                assert key.encode() in body
                new = key not in srv.stored
                srv.stored.setdefault(key, {"observation_id": key, "receipt_id": f"r-{len(srv.stored)}"})
                if srv.mode == "drop_after_commit":
                    srv.mode = "ok"
                    self.close_connection = True
                    self.connection.close()          # committed, but the Pi never hears back
                    return
                self._send(201 if new else 200, srv.stored[key])

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.close()


def drain(up, store, rounds=20):
    for _ in range(rounds):
        with store.connect() as c:   # ignore backoff in tests
            c.execute("UPDATE observations SET next_attempt = 0")
        if not up.step():
            break


def test_exactly_once_with_lost_responses_and_500s(tmp_path, server):
    store = ObservationStore(tmp_path)
    ids = [new_observation_id() for _ in range(4)]
    for i in ids:
        store.persist(i, meta(i), b"\xff\xd8\xffimg")
    up = Uploader(store, server.url, "tok", timeout_s=2)
    server.mode = "drop_after_commit"
    server.fail_next = 0
    up.step()                                   # first upload committed, response lost
    server.fail_next = 2                        # then two 500s
    drain(up, store)
    assert store.counts() == {"ACCEPTED": 4}
    assert sorted(server.stored) == sorted(ids)  # each ID exactly once on the server
    assert server.requests >= 7


def test_offline_server_backs_off_then_delivers(tmp_path):
    store = ObservationStore(tmp_path)
    oid = new_observation_id()
    store.persist(oid, meta(oid), b"i")
    up = Uploader(store, "http://127.0.0.1:9", "tok", timeout_s=0.5)
    assert up.step()
    with store.connect() as c:
        row = c.execute("SELECT * FROM observations").fetchone()
    assert row["state"] == "QUEUED" and row["attempts"] == 1 and row["last_error"].startswith("network")
    assert store.claim() is None                # backing off
    srv = FakeServer()
    try:
        up.url = srv.url + "/api/v1/observations"
        drain(up, store)
        assert store.counts() == {"ACCEPTED": 1} and list(srv.stored) == [oid]
    finally:
        srv.close()


def test_bad_token_blocks_and_alarms(tmp_path, server):
    store = ObservationStore(tmp_path)
    oid = new_observation_id()
    store.persist(oid, meta(oid), b"i")
    up = Uploader(store, server.url, "wrong")
    up.step()
    assert store.counts() == {"BLOCKED": 1} and up.auth_alarm
