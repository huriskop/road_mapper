"""SQLite persistence with an R*Tree spatial index.

Tables
------
incidents       one row per Incident Node (aggregated defect)
incident_rtree  R*Tree spatial index over incident coordinates (kept in sync by triggers)
report_packets  one row per edge-device packet, FK -> incidents (cascade delete)
observations    one row per Pi upload (one pass over one spot); id = the device's observation_id
jobs            classification queue (leased rows, claimed under BEGIN IMMEDIATE)
devices         edge units: SHA-256 of the token, last_seen, firmware
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from .geo import bounding_box, haversine_m
from .models import (
    SEVERITY_RANK,
    STATUS_RANK,
    UNRESOLVED_STATUSES,
    DefectType,
    RepairStatus,
    Severity,
    classify_severity,
    utc_now_iso,
)

_enum_check = lambda col, enum: f"CHECK ({col} IN ({', '.join(repr(e.value) for e in enum)}))"  # noqa: E731

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS incidents (
    pk                 INTEGER PRIMARY KEY AUTOINCREMENT,
    id                 TEXT NOT NULL UNIQUE,
    lat                REAL NOT NULL,
    lng                REAL NOT NULL,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    last_report_at     TEXT NOT NULL,
    defect_type        TEXT NOT NULL DEFAULT 'unclassified' {_enum_check('defect_type', DefectType)},
    severity           TEXT NOT NULL DEFAULT 'low' {_enum_check('severity', Severity)},
    repair_status      TEXT NOT NULL DEFAULT 'awaiting_repair' {_enum_check('repair_status', RepairStatus)},
    report_count       INTEGER NOT NULL DEFAULT 0,
    packet_count       INTEGER NOT NULL DEFAULT 0,
    peak_intensity     REAL,
    severity_manual    INTEGER NOT NULL DEFAULT 0,
    defect_type_manual INTEGER NOT NULL DEFAULT 0,
    type_source        TEXT NOT NULL DEFAULT 'sensor'   -- sensor | camera | operator
);
CREATE INDEX IF NOT EXISTS ix_incidents_created  ON incidents(created_at);
CREATE INDEX IF NOT EXISTS ix_incidents_status   ON incidents(repair_status);
CREATE INDEX IF NOT EXISTS ix_incidents_severity ON incidents(severity);
CREATE INDEX IF NOT EXISTS ix_incidents_type     ON incidents(defect_type);

CREATE VIRTUAL TABLE IF NOT EXISTS incident_rtree USING rtree(pk, min_lat, max_lat, min_lng, max_lng);

CREATE TRIGGER IF NOT EXISTS trg_incidents_ai AFTER INSERT ON incidents BEGIN
    INSERT INTO incident_rtree VALUES (new.pk, new.lat, new.lat, new.lng, new.lng);
END;
CREATE TRIGGER IF NOT EXISTS trg_incidents_au AFTER UPDATE OF lat, lng ON incidents BEGIN
    UPDATE incident_rtree SET min_lat = new.lat, max_lat = new.lat, min_lng = new.lng, max_lng = new.lng
    WHERE pk = new.pk;
END;
CREATE TRIGGER IF NOT EXISTS trg_incidents_ad AFTER DELETE ON incidents BEGIN
    DELETE FROM incident_rtree WHERE pk = old.pk;
END;

CREATE TABLE IF NOT EXISTS report_packets (
    id              TEXT PRIMARY KEY,
    incident_id     TEXT NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    device_id       TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    lat             REAL NOT NULL,
    lng             REAL NOT NULL,
    image_path      TEXT NOT NULL,
    sensor_metadata TEXT,
    received_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_packets_incident ON report_packets(incident_id, timestamp);

CREATE TABLE IF NOT EXISTS devices (
    id          TEXT PRIMARY KEY,
    token_hash  TEXT NOT NULL UNIQUE,
    last_seen   TEXT,
    firmware    TEXT
);

CREATE TABLE IF NOT EXISTS observations (
    id               TEXT PRIMARY KEY,           -- observation_id from the device (idempotency key)
    receipt_id       TEXT NOT NULL UNIQUE,
    device_id        TEXT NOT NULL,
    received_at      TEXT NOT NULL,
    t_impact         TEXT,
    evidence_status  TEXT NOT NULL,
    lat              REAL,
    lng              REAL,
    heading_deg      REAL,
    speed_mps        REAL,
    imu_score        REAL,
    peak_az_mps2     REAL,
    image_path       TEXT,                       -- photo as uploaded
    annotated_path   TEXT,                       -- with the detector's boxes
    sha256           TEXT,
    metadata         TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'QUEUED',
    label            TEXT,
    confidence       REAL,
    quality          TEXT,                       -- JSON: blur, brightness, ok, reason
    vision           TEXT,                       -- JSON: detector summary
    incident_id      TEXT,
    reviewed_at      TEXT,
    decided_by       TEXT                        -- worker | operator
);
CREATE INDEX IF NOT EXISTS ix_obs_status ON observations(status, received_at);
CREATE INDEX IF NOT EXISTS ix_obs_incident ON observations(incident_id);

CREATE TABLE IF NOT EXISTS jobs (
    observation_id  TEXT PRIMARY KEY REFERENCES observations(id) ON DELETE CASCADE,
    state           TEXT NOT NULL DEFAULT 'QUEUED',   -- QUEUED | LEASED | DONE | FAILED
    attempts        INTEGER NOT NULL DEFAULT 0,
    lease_until     REAL,
    last_error      TEXT,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_jobs_state ON jobs(state, created_at);
"""


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def heading_diff(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)

SORT_COLUMNS = {
    "created_at": "i.created_at",
    "updated_at": "i.updated_at",
    "last_report_at": "i.last_report_at",
    "severity": "CASE i.severity " + " ".join(f"WHEN '{k}' THEN {v}" for k, v in SEVERITY_RANK.items()) + " END",
    "repair_status": "CASE i.repair_status " + " ".join(f"WHEN '{k}' THEN {v}" for k, v in STATUS_RANK.items()) + " END",
    "report_count": "i.report_count",
    "defect_type": "i.defect_type",
}


@dataclass
class IncidentFilters:
    start: str | None = None
    end: str | None = None
    defect_types: Sequence[str] = ()
    severities: Sequence[str] = ()
    statuses: Sequence[str] = ()

    def where(self) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if self.start:
            clauses.append("i.created_at >= ?")
            params.append(self.start)
        if self.end:
            clauses.append("i.created_at <= ?")
            params.append(self.end)
        for col, values in (
            ("i.defect_type", self.defect_types),
            ("i.severity", self.severities),
            ("i.repair_status", self.statuses),
        ):
            if values:
                clauses.append(f"{col} IN ({', '.join('?' * len(values))})")
                params.extend(values)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


class Database:
    def __init__(self, path: Path, media_url_prefix: str = "/media") -> None:
        self.path = path
        self.media_url_prefix = media_url_prefix.rstrip("/")
        # Serialises "find nearby incident -> insert/merge" so concurrent packets
        # for the same defect never create duplicate incidents.
        self._ingest_lock = threading.Lock()
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            # upgrade databases created before the camera classified incidents
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(incidents)")}
            if "type_source" not in cols:
                conn.execute("ALTER TABLE incidents ADD COLUMN type_source TEXT NOT NULL DEFAULT 'sensor'")
                conn.execute("UPDATE incidents SET type_source = 'operator' WHERE defect_type_manual = 1")
            if "heading_deg" not in cols:   # direction of travel; keeps two lanes apart in de-dup
                conn.execute("ALTER TABLE incidents ADD COLUMN heading_deg REAL")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------ ingest
    def find_cluster_target(self, conn: sqlite3.Connection, lat: float, lng: float, radius_m: float,
                            defect_type: str | None = None, heading: float | None = None,
                            heading_tol: float = 45.0) -> sqlite3.Row | None:
        """Nearest unresolved incident within radius_m, using the R*Tree as a pre-filter.
        With defect_type, only incidents of that type match; with heading, only incidents seen
        travelling the same way (within heading_tol degrees) or with no heading recorded."""
        min_lat, max_lat, min_lng, max_lng = bounding_box(lat, lng, radius_m)
        rows = conn.execute(
            f"""
            SELECT i.* FROM incident_rtree r JOIN incidents i ON i.pk = r.pk
            WHERE r.max_lat >= ? AND r.min_lat <= ? AND r.max_lng >= ? AND r.min_lng <= ?
              AND i.repair_status IN ({', '.join('?' * len(UNRESOLVED_STATUSES))})
            """,
            (min_lat, max_lat, min_lng, max_lng, *UNRESOLVED_STATUSES),
        ).fetchall()
        best, best_d = None, float("inf")
        for row in rows:
            if defect_type is not None and row["defect_type"] != defect_type:
                continue
            if heading is not None and row["heading_deg"] is not None \
                    and heading_diff(heading, row["heading_deg"]) > heading_tol:
                continue
            d = haversine_m(lat, lng, row["lat"], row["lng"])
            if d <= radius_m and d < best_d:
                best, best_d = row, d
        return best

    def ingest_packet(
        self,
        *,
        device_id: str,
        timestamp: str,
        lat: float,
        lng: float,
        image_path: str,
        sensor_metadata: dict[str, Any] | None,
        peak: float | None,
        defect_hint: DefectType | None,
        radius_m: float,
        hint_source: str = "sensor",
        packet_id: str | None = None,
        heading: float | None = None,
        match_type: bool = False,
        heading_tol: float = 45.0,
        conn: sqlite3.Connection | None = None,
    ) -> dict[str, Any]:
        """hint_source: "camera" when defect_hint comes from the photo detector, else "sensor".
        Camera findings outrank the vibration guess; an operator's edit outranks both.
        match_type / heading: de-duplicate only against incidents with the same type and travel
        direction (observations path). conn: run inside the caller's open transaction."""
        if conn is None:
            with self._ingest_lock, self.connect() as own:
                own.execute("BEGIN IMMEDIATE")
                try:
                    out = self.ingest_packet(
                        device_id=device_id, timestamp=timestamp, lat=lat, lng=lng, image_path=image_path,
                        sensor_metadata=sensor_metadata, peak=peak, defect_hint=defect_hint, radius_m=radius_m,
                        hint_source=hint_source, packet_id=packet_id, heading=heading, match_type=match_type,
                        heading_tol=heading_tol, conn=own)
                    own.execute("COMMIT")
                except BaseException:
                    own.execute("ROLLBACK")
                    raise
                return out
        return self._ingest_in(conn, device_id, timestamp, lat, lng, image_path, sensor_metadata, peak,
                               defect_hint, radius_m, hint_source, packet_id or str(uuid.uuid4()), heading,
                               match_type, heading_tol)

    def _ingest_in(self, conn, device_id, timestamp, lat, lng, image_path, sensor_metadata, peak, defect_hint,
                   radius_m, hint_source, packet_id, heading, match_type, heading_tol) -> dict[str, Any]:
        now = utc_now_iso()
        meta_json = json.dumps(sensor_metadata) if sensor_metadata is not None else None
        target = self.find_cluster_target(
            conn, lat, lng, radius_m,
            defect_type=(defect_hint.value if (match_type and defect_hint) else None),
            heading=heading, heading_tol=heading_tol)
        if target is None:
            incident_id = str(uuid.uuid4())
            severity = classify_severity(peak) or Severity.low
            conn.execute(
                """INSERT INTO incidents (id, lat, lng, created_at, updated_at, last_report_at,
                       defect_type, severity, repair_status, report_count, packet_count, peak_intensity,
                       type_source, heading_deg, defect_type_manual)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'awaiting_repair', 1, 1, ?, ?, ?, ?)""",
                (incident_id, lat, lng, timestamp, timestamp, timestamp,
                 (defect_hint or DefectType.unclassified).value, severity.value, peak,
                 hint_source if defect_hint else "sensor", heading,
                 1 if hint_source == "operator" else 0),
            )
            created = True
        else:
            incident_id = target["id"]
            created = False
        conn.execute(
            """INSERT INTO report_packets (id, incident_id, device_id, timestamp, lat, lng,
                   image_path, sensor_metadata, received_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (packet_id, incident_id, device_id, timestamp, lat, lng, image_path, meta_json, now),
        )
        if not created:
            self._merge_into(conn, target, timestamp, peak, defect_hint, hint_source)
        row = conn.execute("SELECT report_count, packet_count FROM incidents WHERE id = ?",
                           (incident_id,)).fetchone()

        return {
            "packet_id": packet_id,
            "incident_id": incident_id,
            "incident_created": created,
            "report_count": row["report_count"],
            "packet_count": row["packet_count"],
        }

    def _merge_into(self, conn, target: sqlite3.Row, timestamp: str, peak: float | None, hint: DefectType | None,
                    hint_source: str = "sensor") -> None:
        incident_id = target["id"]
        agg = conn.execute(
            """SELECT AVG(lat) AS lat, AVG(lng) AS lng, COUNT(DISTINCT device_id) AS devices, COUNT(*) AS packets,
                      MIN(timestamp) AS first_ts, MAX(timestamp) AS last_ts
               FROM report_packets WHERE incident_id = ?""",
            (incident_id,),
        ).fetchone()
        peaks = [p for p in (target["peak_intensity"], peak) if p is not None]
        new_peak = max(peaks) if peaks else None

        severity = target["severity"]
        if not target["severity_manual"]:
            auto = classify_severity(new_peak)
            if auto is not None:
                severity = auto.value
        defect_type, source = target["defect_type"], target["type_source"]
        if not target["defect_type_manual"]:
            camera = self._camera_vote(conn, incident_id)
            if camera:                                   # majority of the camera's verdicts
                defect_type, source = camera, "camera"
            elif defect_type == DefectType.unclassified.value and hint:
                defect_type, source = hint.value, hint_source

        conn.execute(
            """UPDATE incidents SET lat = ?, lng = ?, report_count = ?, packet_count = ?, peak_intensity = ?,
                   severity = ?, defect_type = ?, type_source = ?, created_at = MIN(created_at, ?),
                   last_report_at = ?, updated_at = MAX(updated_at, ?)
               WHERE id = ?""",
            (agg["lat"], agg["lng"], agg["devices"], agg["packets"], new_peak, severity, defect_type, source,
             agg["first_ts"], agg["last_ts"], timestamp, incident_id),
        )

    @staticmethod
    def _camera_vote(conn, incident_id: str) -> str | None:
        """Most common camera classification among the incident's photos (ties: pothole first)."""
        votes: dict[str, int] = {}
        for (meta,) in conn.execute("SELECT sensor_metadata FROM report_packets WHERE incident_id = ?", (incident_id,)):
            try:
                kind = (json.loads(meta or "{}").get("vision") or {}).get("camera_type")
            except (ValueError, AttributeError):
                kind = None
            if kind in (DefectType.pothole.value, DefectType.bump.value):
                votes[kind] = votes.get(kind, 0) + 1
        if not votes:
            return None
        return max(votes, key=lambda k: (votes[k], k == DefectType.pothole.value))

    # ------------------------------------------------------------------- reads
    def _image_url(self, path: str) -> str:
        return f"{self.media_url_prefix}/{path}"

    def _serialize(self, row: sqlite3.Row, images: list[dict[str, Any]] | None = None, thumb: str | None = None) -> dict[str, Any]:
        data = {
            "id": row["id"],
            "coordinates": {"lat": row["lat"], "lng": row["lng"]},
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_report_at": row["last_report_at"],
            "defect_type": row["defect_type"],
            "type_source": row["type_source"] if "type_source" in row.keys() else None,
            "severity": row["severity"],
            "repair_status": row["repair_status"],
            "report_count": row["report_count"],
            "packet_count": row["packet_count"],
            "peak_intensity": row["peak_intensity"],
        }
        if images is not None:
            data["images"] = images
        if thumb is not None or "thumb_path" in row.keys():
            tp = thumb or row["thumb_path"]
            data["thumbnail_url"] = self._image_url(tp) if tp else None
        return data

    def _images_for(self, conn, incident_id: str) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT id, device_id, timestamp, image_path, lat, lng, sensor_metadata FROM report_packets "
            "WHERE incident_id = ? ORDER BY timestamp",
            (incident_id,),
        ).fetchall()
        return [
            {
                "packet_id": r["id"],
                "url": self._image_url(r["image_path"]),
                "timestamp": r["timestamp"],
                "device_id": r["device_id"],
                "location": {"lat": r["lat"], "lng": r["lng"]},
                "sensor_metadata": json.loads(r["sensor_metadata"]) if r["sensor_metadata"] else None,
            }
            for r in rows
        ]

    def get_incident(self, incident_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
            if row is None:
                return None
            return self._serialize(row, images=self._images_for(conn, incident_id))

    _THUMB_SQL = (
        "(SELECT p.image_path FROM report_packets p WHERE p.incident_id = i.id ORDER BY p.timestamp LIMIT 1) AS thumb_path"
    )

    def list_incidents(
        self, filters: IncidentFilters, sort: str, order: str, page: int, page_size: int
    ) -> tuple[list[dict[str, Any]], int]:
        where, params = filters.where()
        sort_sql = SORT_COLUMNS.get(sort, SORT_COLUMNS["created_at"])
        direction = "ASC" if order.lower() == "asc" else "DESC"
        with self.connect() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM incidents i{where}", params).fetchone()[0]
            rows = conn.execute(
                f"SELECT i.*, {self._THUMB_SQL} FROM incidents i{where} "
                f"ORDER BY {sort_sql} {direction}, i.created_at DESC, i.pk DESC LIMIT ? OFFSET ?",
                [*params, page_size, (page - 1) * page_size],
            ).fetchall()
        return [self._serialize(r) for r in rows], total

    def map_incidents(self, filters: IncidentFilters) -> list[dict[str, Any]]:
        where, params = filters.where()
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT i.id, i.lat, i.lng, i.severity, i.repair_status, i.defect_type, i.report_count, "
                f"i.created_at, i.last_report_at FROM incidents i{where}",
                params,
            ).fetchall()
        return [
            {
                "id": r["id"],
                "lat": r["lat"],
                "lng": r["lng"],
                "severity": r["severity"],
                "repair_status": r["repair_status"],
                "defect_type": r["defect_type"],
                "report_count": r["report_count"],
                "created_at": r["created_at"],
                "last_report_at": r["last_report_at"],
            }
            for r in rows
        ]

    def stats(self) -> dict[str, Any]:
        with self.connect() as conn:
            out: dict[str, Any] = {
                "incidents": conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0],
                "packets": conn.execute("SELECT COUNT(*) FROM report_packets").fetchone()[0],
                "devices": conn.execute("SELECT COUNT(DISTINCT device_id) FROM report_packets").fetchone()[0],
            }
            for col in ("severity", "repair_status", "defect_type"):
                out[f"by_{col}"] = {
                    r[0]: r[1] for r in conn.execute(f"SELECT {col}, COUNT(*) FROM incidents GROUP BY {col}")
                }
            out["observations"] = {
                r[0]: r[1] for r in conn.execute("SELECT status, COUNT(*) FROM observations GROUP BY status")
            }
            out["review_backlog"] = out["observations"].get("NEEDS_REVIEW", 0)
            week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
            out["new_this_week"] = conn.execute("SELECT COUNT(*) FROM incidents WHERE created_at >= ?",
                                                (week_ago,)).fetchone()[0]
        return out

    # ------------------------------------------------------------------ writes
    def update_incident(self, incident_id: str, changes: dict[str, str]) -> dict[str, Any] | None:
        if not changes:
            return self.get_incident(incident_id)
        sets = [f"{k} = ?" for k in changes]
        params: list[Any] = list(changes.values())
        if "severity" in changes:
            sets.append("severity_manual = 1")
        if "defect_type" in changes:
            sets.append("defect_type_manual = 1")
            sets.append("type_source = 'operator'")
        sets.append("updated_at = ?")
        params.append(utc_now_iso())
        with self.connect() as conn:
            cur = conn.execute(f"UPDATE incidents SET {', '.join(sets)} WHERE id = ?", [*params, incident_id])
            if cur.rowcount == 0:
                return None
        return self.get_incident(incident_id)

    def delete_incident(self, incident_id: str) -> list[str] | None:
        """Delete an incident and its packets. Returns image paths to remove, or None if missing."""
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            paths = [r[0] for r in conn.execute("SELECT image_path FROM report_packets WHERE incident_id = ?", (incident_id,))]
            # observation photos are owned by the observation rows, not the incident
            owned = {r[0] for r in conn.execute("SELECT annotated_path FROM observations WHERE incident_id = ?",
                                                (incident_id,)) if r[0]}
            paths = [p for p in paths if p not in owned]
            conn.execute("UPDATE observations SET incident_id = NULL WHERE incident_id = ?", (incident_id,))
            cur = conn.execute("DELETE FROM incidents WHERE id = ?", (incident_id,))
            conn.execute("COMMIT")
        return paths if cur.rowcount else None

    # ------------------------------------------------------------- devices
    def sync_devices(self, tokens: dict[str, str]) -> None:
        """Make the devices table match the configured tokens (hash only)."""
        with self.connect() as conn:
            for dev, tok in tokens.items():
                conn.execute(
                    "INSERT INTO devices (id, token_hash) VALUES (?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET token_hash = excluded.token_hash", (dev, token_hash(tok)))
            if tokens:
                conn.execute(f"DELETE FROM devices WHERE id NOT IN ({', '.join('?' * len(tokens))})", list(tokens))
            else:
                conn.execute("DELETE FROM devices")

    def device_for_token(self, token: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT id FROM devices WHERE token_hash = ?", (token_hash(token),)).fetchone()
        return row["id"] if row else None

    def touch_device(self, device_id: str, firmware: str | None) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE devices SET last_seen = ?, firmware = COALESCE(?, firmware) WHERE id = ?",
                         (utc_now_iso(), firmware, device_id))

    # -------------------------------------------------------- observations
    def insert_observation(self, meta: dict[str, Any], image_path: str | None) -> tuple[dict[str, Any], bool]:
        """Insert once, keyed by observation_id, and queue its classification job in the same
        transaction. Returns (receipt, created). A repeat returns the stored receipt untouched;
        a repeat with a different image hash raises ValueError (conflict)."""
        obs_id = meta["observation_id"]
        imu = meta.get("imu") or {}
        defect = meta.get("defect") or {}
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                old = conn.execute("SELECT * FROM observations WHERE id = ?", (obs_id,)).fetchone()
                if old is not None:
                    conn.execute("ROLLBACK")
                    if old["sha256"] != meta.get("image_sha256") or old["device_id"] != meta["device_id"]:
                        raise ValueError("observation_id already used for different content")
                    return self._receipt(old), False
                receipt_id = str(uuid.uuid4())
                conn.execute(
                    """INSERT INTO observations (id, receipt_id, device_id, received_at, t_impact, evidence_status,
                           lat, lng, heading_deg, speed_mps, imu_score, peak_az_mps2, image_path, sha256, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (obs_id, receipt_id, meta["device_id"], utc_now_iso(), meta.get("t_impact_utc"),
                     meta["evidence_status"], defect.get("lat"), defect.get("lon"), meta.get("heading_deg"),
                     meta.get("speed_before_mps"), imu.get("score"), imu.get("peak_az_mps2"), image_path,
                     meta.get("image_sha256"), json.dumps(meta)),
                )
                conn.execute("INSERT INTO jobs (observation_id, created_at) VALUES (?, ?)", (obs_id, time.time()))
                row = conn.execute("SELECT * FROM observations WHERE id = ?", (obs_id,)).fetchone()
                conn.execute("COMMIT")
            except ValueError:
                raise
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return self._receipt(row), True

    @staticmethod
    def _receipt(row: sqlite3.Row) -> dict[str, Any]:
        return {"observation_id": row["id"], "receipt_id": row["receipt_id"], "received_at": row["received_at"],
                "status": row["status"]}

    def claim_job(self, lease_s: float, max_attempts: int) -> sqlite3.Row | None:
        """Lease the oldest queued job (or one whose lease expired: its worker crashed)."""
        now = time.time()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                while True:
                    job = conn.execute(
                        "SELECT * FROM jobs WHERE state = 'QUEUED' OR (state = 'LEASED' AND lease_until < ?) "
                        "ORDER BY created_at LIMIT 1", (now,)).fetchone()
                    if job is None:
                        conn.execute("COMMIT")
                        return None
                    if job["attempts"] >= max_attempts:
                        conn.execute("UPDATE jobs SET state = 'FAILED', lease_until = NULL WHERE observation_id = ?",
                                     (job["observation_id"],))
                        conn.execute("UPDATE observations SET status = 'FAILED' WHERE id = ?", (job["observation_id"],))
                        continue
                    conn.execute("UPDATE jobs SET state = 'LEASED', lease_until = ?, attempts = attempts + 1 "
                                 "WHERE observation_id = ?", (now + lease_s, job["observation_id"]))
                    obs = conn.execute("SELECT * FROM observations WHERE id = ?", (job["observation_id"],)).fetchone()
                    conn.execute("COMMIT")
                    return obs
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def fail_job(self, obs_id: str, error: str, max_attempts: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET state = CASE WHEN attempts >= ? THEN 'FAILED' ELSE 'QUEUED' END, "
                "lease_until = NULL, last_error = ? WHERE observation_id = ?", (max_attempts, error, obs_id))
            conn.execute("UPDATE observations SET status = 'FAILED' WHERE id = ? AND "
                         "(SELECT state FROM jobs WHERE observation_id = ?) = 'FAILED'", (obs_id, obs_id))

    def finish_observation(self, obs_id: str, *, status: str, label: str | None, confidence: float | None,
                           quality: dict | None, vision: dict | None, annotated_path: str | None,
                           decided_by: str, attach: dict[str, Any] | None = None,
                           radius_m: float = 8.0, heading_tol: float = 45.0) -> dict[str, Any] | None:
        """Record the verdict; with `attach`, also join/create the incident in the same transaction.
        attach = {"defect_type": DefectType, "image_path": ..., "sensor_metadata": ..., "peak": g}.
        Returns the ingest result (incident id, created) when attached."""
        result = None
        with self._ingest_lock, self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                obs = conn.execute("SELECT * FROM observations WHERE id = ?", (obs_id,)).fetchone()
                if attach is not None and obs["incident_id"] is None and obs["lat"] is not None:
                    result = self._ingest_in(
                        conn, obs["device_id"], obs["t_impact"] or obs["received_at"], obs["lat"], obs["lng"],
                        attach["image_path"], attach.get("sensor_metadata"), attach.get("peak"),
                        attach["defect_type"], radius_m, attach.get("source", "camera"), obs_id,
                        obs["heading_deg"], True, heading_tol)
                conn.execute(
                    """UPDATE observations SET status = ?, label = ?, confidence = ?, quality = ?, vision = ?,
                           annotated_path = COALESCE(?, annotated_path), decided_by = ?,
                           incident_id = COALESCE(?, incident_id),
                           reviewed_at = CASE WHEN ? = 'operator' THEN ? ELSE reviewed_at END
                       WHERE id = ?""",
                    (status, label, confidence, json.dumps(quality) if quality is not None else obs["quality"],
                     json.dumps(vision) if vision is not None else obs["vision"], annotated_path, decided_by,
                     result["incident_id"] if result else None, decided_by, utc_now_iso(), obs_id))
                conn.execute("UPDATE jobs SET state = 'DONE', lease_until = NULL WHERE observation_id = ?", (obs_id,))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return result

    def _obs_dict(self, r: sqlite3.Row) -> dict[str, Any]:
        url = lambda p: self._image_url(p) if p else None  # noqa: E731
        return {
            "observation_id": r["id"], "receipt_id": r["receipt_id"], "device_id": r["device_id"],
            "received_at": r["received_at"], "t_impact": r["t_impact"], "evidence_status": r["evidence_status"],
            "location": {"lat": r["lat"], "lng": r["lng"]} if r["lat"] is not None else None,
            "heading_deg": r["heading_deg"], "speed_mps": r["speed_mps"], "imu_score": r["imu_score"],
            "peak_az_mps2": r["peak_az_mps2"], "status": r["status"], "label": r["label"],
            "confidence": r["confidence"], "quality": json.loads(r["quality"]) if r["quality"] else None,
            "vision": json.loads(r["vision"]) if r["vision"] else None, "incident_id": r["incident_id"],
            "decided_by": r["decided_by"], "reviewed_at": r["reviewed_at"],
            "image_url": url(r["image_path"]), "annotated_url": url(r["annotated_path"]),
            "metadata": json.loads(r["metadata"]),
        }

    def get_observation(self, obs_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            r = conn.execute("SELECT * FROM observations WHERE id = ?", (obs_id,)).fetchone()
        return self._obs_dict(r) if r else None

    def list_observations(self, statuses: Sequence[str] = (), limit: int = 50, offset: int = 0
                          ) -> tuple[list[dict[str, Any]], int]:
        where, params = "", []
        if statuses:
            where = f" WHERE status IN ({', '.join('?' * len(statuses))})"
            params = list(statuses)
        with self.connect() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM observations{where}", params).fetchone()[0]
            rows = conn.execute(f"SELECT * FROM observations{where} ORDER BY received_at DESC LIMIT ? OFFSET ?",
                                [*params, limit, offset]).fetchall()
        return [self._obs_dict(r) for r in rows], total

    def defects_geojson(self, bbox: tuple[float, float, float, float] | None, types: Sequence[str],
                        statuses: Sequence[str], since: str | None) -> dict[str, Any]:
        clauses, params = [], []
        if bbox:   # west, south, east, north (GeoJSON order)
            clauses.append("i.lng >= ? AND i.lat >= ? AND i.lng <= ? AND i.lat <= ?")
            params += list(bbox)
        if types:
            clauses.append(f"i.defect_type IN ({', '.join('?' * len(types))})")
            params += list(types)
        if statuses:
            clauses.append(f"i.repair_status IN ({', '.join('?' * len(statuses))})")
            params += list(statuses)
        if since:
            clauses.append("i.last_report_at >= ?")
            params.append(since)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self.connect() as conn:
            rows = conn.execute(
                f"""SELECT i.*, {self._THUMB_SQL},
                       (SELECT MAX(o.confidence) FROM observations o WHERE o.incident_id = i.id) AS best_conf,
                       (SELECT AVG(o.speed_mps) FROM observations o WHERE o.incident_id = i.id) AS avg_speed
                    FROM incidents i{where}""", params).fetchall()
        feats = [{
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["lng"], r["lat"]]},
            "properties": {
                "id": r["id"], "label": r["defect_type"], "severity": r["severity"],
                "status": r["repair_status"], "confidence": r["best_conf"],
                "speed_kmh": round(r["avg_speed"] * 3.6, 1) if r["avg_speed"] is not None else None,
                "first_seen": r["created_at"], "last_seen": r["last_report_at"],
                "observation_count": r["packet_count"], "heading_deg": r["heading_deg"],
                "image_url": self._image_url(r["thumb_path"]) if r["thumb_path"] else None,
            },
        } for r in rows]
        return {"type": "FeatureCollection", "features": feats}
