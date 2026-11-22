"""SQLite persistence with an R*Tree spatial index.

Tables
------
incidents       one row per Incident Node (aggregated defect)
incident_rtree  R*Tree spatial index over incident coordinates (kept in sync by triggers)
report_packets  one row per edge-device packet, FK -> incidents (cascade delete)
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
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
"""

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
    def find_cluster_target(self, conn: sqlite3.Connection, lat: float, lng: float, radius_m: float) -> sqlite3.Row | None:
        """Nearest unresolved incident within radius_m, using the R*Tree as a pre-filter."""
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
    ) -> dict[str, Any]:
        """hint_source: "camera" when defect_hint comes from the photo detector, else "sensor".
        Camera findings outrank the vibration guess; an operator's edit outranks both."""
        packet_id = str(uuid.uuid4())
        now = utc_now_iso()
        meta_json = json.dumps(sensor_metadata) if sensor_metadata is not None else None

        with self._ingest_lock, self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                target = self.find_cluster_target(conn, lat, lng, radius_m)
                if target is None:
                    incident_id = str(uuid.uuid4())
                    severity = classify_severity(peak) or Severity.low
                    conn.execute(
                        """INSERT INTO incidents (id, lat, lng, created_at, updated_at, last_report_at,
                               defect_type, severity, repair_status, report_count, packet_count, peak_intensity,
                               type_source)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'awaiting_repair', 1, 1, ?, ?)""",
                        (incident_id, lat, lng, timestamp, timestamp, timestamp,
                         (defect_hint or DefectType.unclassified).value, severity.value, peak,
                         hint_source if defect_hint else "sensor"),
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
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            row = conn.execute("SELECT report_count, packet_count FROM incidents WHERE id = ?", (incident_id,)).fetchone()

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
            cur = conn.execute("DELETE FROM incidents WHERE id = ?", (incident_id,))
            conn.execute("COMMIT")
        return paths if cur.rowcount else None
