"""E7 - nothing is lost if power or network drops.

Order of a save (the review's persist_observation):
  1. write each evidence file to a temp name, fsync, rename (atomic on the same filesystem)
  2. one SQLite transaction inserts the observation row in state QUEUED
The observation ID (UUID) and the image SHA-256 are created once; every retry reuses them.
On startup, `recover()` returns expired leases to the queue, deletes files no row owns, and
blocks rows whose files went missing.

States: QUEUED -> (leased while uploading) -> ACCEPTED, or BLOCKED / DROPPED.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id            TEXT PRIMARY KEY,
    created_at    REAL NOT NULL,           -- wall clock (s), for ordering on this device only
    state         TEXT NOT NULL CHECK (state IN ('QUEUED','ACCEPTED','BLOCKED','DROPPED')),
    evidence      TEXT NOT NULL,
    score         REAL NOT NULL DEFAULT 0,
    metadata      TEXT NOT NULL,
    image_path    TEXT,                    -- relative to the store root
    image_sha256  TEXT,
    extra_paths   TEXT NOT NULL DEFAULT '[]',
    bytes         INTEGER NOT NULL DEFAULT 0,
    attempts      INTEGER NOT NULL DEFAULT 0,
    next_attempt  REAL NOT NULL DEFAULT 0,
    lease_until   REAL,
    last_error    TEXT,
    receipt       TEXT,
    accepted_at   REAL,
    files_pruned  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_obs_queue ON observations(state, next_attempt);
"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def new_observation_id() -> str:
    return str(uuid.uuid4())


class ObservationStore:
    def __init__(self, root: Path, budget_bytes: int = 2 * 1024**3) -> None:
        self.root = Path(root)
        self.files = self.root / "evidence"
        self.files.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "observations.db"
        self.budget = budget_bytes
        self._lock = threading.Lock()
        with self.connect() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode = WAL")
        c.execute("PRAGMA synchronous = FULL")
        try:
            yield c
        finally:
            c.close()

    # ---------------------------------------------------------------- write
    def _atomic_write(self, rel: str, data: bytes) -> None:
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
        try:  # make the rename itself durable
            dfd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except (OSError, AttributeError):
            pass  # not supported on Windows

    def persist(self, obs_id: str, metadata: dict[str, Any], image: bytes | None,
                extras: dict[str, bytes] | None = None, score: float = 0.0) -> None:
        """Files first, then one transaction. metadata['image_sha256'] is set here."""
        day = time.strftime("%Y%m%d")
        image_rel = None
        written: list[str] = []
        total = 0
        if image is not None:
            image_rel = f"evidence/{day}/{obs_id}.jpg"
            self._atomic_write(image_rel, image)
            written.append(image_rel)
            total += len(image)
            metadata["image_sha256"] = sha256(image)
        else:
            metadata["image_sha256"] = None
        extra_rels = []
        for name, data in (extras or {}).items():
            rel = f"evidence/{day}/{obs_id}.{name}"
            self._atomic_write(rel, data)
            extra_rels.append(rel)
            total += len(data)
        with self._lock, self.connect() as c:
            c.execute(
                "INSERT INTO observations (id, created_at, state, evidence, score, metadata, image_path, "
                "image_sha256, extra_paths, bytes) VALUES (?, ?, 'QUEUED', ?, ?, ?, ?, ?, ?, ?)",
                (obs_id, time.time(), metadata["evidence_status"], score, json.dumps(metadata), image_rel,
                 metadata["image_sha256"], json.dumps(extra_rels), total),
            )
        self.enforce_budget()

    # ---------------------------------------------------------------- queue
    def claim(self, lease_s: float = 60.0) -> sqlite3.Row | None:
        now = time.time()
        with self._lock, self.connect() as c:
            return c.execute(
                """UPDATE observations SET lease_until = ?
                   WHERE id = (SELECT id FROM observations
                               WHERE state = 'QUEUED' AND next_attempt <= ?
                                 AND (lease_until IS NULL OR lease_until < ?)
                               ORDER BY created_at LIMIT 1)
                   RETURNING *""",
                (now + lease_s, now, now),
            ).fetchone()

    def accept(self, obs_id: str, receipt: str) -> None:
        with self._lock, self.connect() as c:
            c.execute("UPDATE observations SET state='ACCEPTED', receipt=?, accepted_at=?, lease_until=NULL, "
                      "last_error=NULL WHERE id=?", (receipt, time.time(), obs_id))

    def block(self, obs_id: str, error: str) -> None:
        with self._lock, self.connect() as c:
            c.execute("UPDATE observations SET state='BLOCKED', last_error=?, lease_until=NULL WHERE id=?",
                      (error, obs_id))

    def retry_later(self, obs_id: str, error: str, base_s: float = 5.0, cap_s: float = 600.0) -> float:
        with self._lock, self.connect() as c:
            attempts = c.execute("SELECT attempts FROM observations WHERE id=?", (obs_id,)).fetchone()[0] + 1
            delay = min(cap_s, base_s * 2 ** (attempts - 1)) * random.uniform(0.5, 1.0)
            c.execute("UPDATE observations SET attempts=?, next_attempt=?, last_error=?, lease_until=NULL "
                      "WHERE id=?", (attempts, time.time() + delay, error, obs_id))
        return delay

    def read_image(self, row: sqlite3.Row) -> bytes | None:
        if not row["image_path"]:
            return None
        return (self.root / row["image_path"]).read_bytes()

    # ------------------------------------------------------------ recovery
    def recover(self) -> dict[str, int]:
        """Run once at startup, before the uploader."""
        out = {"leases_released": 0, "orphan_files": 0, "missing_files": 0, "tmp_files": 0}
        with self._lock, self.connect() as c:
            out["leases_released"] = c.execute(
                "UPDATE observations SET lease_until = NULL WHERE lease_until IS NOT NULL").rowcount
            owned: set[str] = set()
            for r in c.execute("SELECT id, image_path, extra_paths, state, files_pruned FROM observations"):
                paths = ([r["image_path"]] if r["image_path"] else []) + json.loads(r["extra_paths"])
                owned.update(paths)
                if r["state"] == "QUEUED" and not r["files_pruned"] and r["image_path"] \
                        and not (self.root / r["image_path"]).exists():
                    c.execute("UPDATE observations SET state='BLOCKED', last_error='image file missing' "
                              "WHERE id=?", (r["id"],))
                    out["missing_files"] += 1
        for p in self.files.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(self.root).as_posix()
            if p.name.endswith(".tmp"):
                p.unlink(missing_ok=True)
                out["tmp_files"] += 1
            elif rel not in owned:            # crashed between rename and insert
                p.unlink(missing_ok=True)
                out["orphan_files"] += 1
        return out

    # --------------------------------------------------------------- budget
    def used_bytes(self) -> int:
        with self.connect() as c:
            return c.execute("SELECT COALESCE(SUM(bytes),0) FROM observations WHERE files_pruned=0").fetchone()[0]

    def _delete_files(self, c, row) -> None:
        for rel in ([row["image_path"]] if row["image_path"] else []) + json.loads(row["extra_paths"]):
            (self.root / rel).unlink(missing_ok=True)

    def enforce_budget(self) -> list[str]:
        """Free space: accepted files first, then the oldest low-score queued ones. Returns dropped IDs."""
        dropped: list[str] = []
        used = self.used_bytes()
        if used <= self.budget:
            return dropped
        with self._lock, self.connect() as c:
            for r in c.execute("SELECT * FROM observations WHERE state IN ('ACCEPTED','BLOCKED') AND files_pruned=0 "
                               "ORDER BY created_at").fetchall():
                if used <= self.budget:
                    break
                self._delete_files(c, r)
                c.execute("UPDATE observations SET files_pruned=1 WHERE id=?", (r["id"],))
                used -= r["bytes"]
            for r in c.execute("SELECT * FROM observations WHERE state='QUEUED' AND files_pruned=0 "
                               "ORDER BY score ASC, created_at ASC").fetchall():
                if used <= self.budget:
                    break
                self._delete_files(c, r)
                c.execute("UPDATE observations SET state='DROPPED', files_pruned=1, "
                          "last_error='disk budget' WHERE id=?", (r["id"],))
                used -= r["bytes"]
                dropped.append(r["id"])
                log.warning("disk budget: dropped queued observation %s (score %.2f)", r["id"], r["score"])
        return dropped

    # ---------------------------------------------------------------- stats
    def counts(self) -> dict[str, int]:
        with self.connect() as c:
            return {r[0]: r[1] for r in c.execute("SELECT state, COUNT(*) FROM observations GROUP BY state")}

    def free_disk_mb(self) -> float:
        return shutil.disk_usage(self.root).free / 1e6
