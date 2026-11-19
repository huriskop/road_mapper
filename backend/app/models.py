"""Enums, request/response schemas and small domain helpers."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DefectType(str, Enum):
    pothole = "pothole"
    bump = "bump"
    crack = "crack"
    unclassified = "unclassified"


class Severity(str, Enum):
    false_positive = "false_positive"
    no_repair_needed = "no_repair_needed"
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class RepairStatus(str, Enum):
    awaiting_repair = "awaiting_repair"
    in_progress = "in_progress"
    completed = "completed"


# Ordinal ranks used for sorting ("highest/lowest").
SEVERITY_RANK = {s.value: i for i, s in enumerate(Severity)}
STATUS_RANK = {s.value: i for i, s in enumerate(RepairStatus)}
UNRESOLVED_STATUSES = (RepairStatus.awaiting_repair.value, RepairStatus.in_progress.value)


def to_utc_iso(dt: datetime) -> str:
    """Normalise to a fixed-width UTC ISO-8601 string so lexical order == time order."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def utc_now_iso() -> str:
    return to_utc_iso(datetime.now(timezone.utc))


class Location(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)


class TelemetryMetadata(BaseModel):
    """Everything in the edge packet except the image itself."""

    device_id: str = Field(min_length=1, max_length=64)
    timestamp: datetime
    location: Location
    sensor_metadata: dict[str, Any] | None = None

    @field_validator("device_id")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("device_id must not be blank")
        return v


class EvidenceStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    LOW_SPEED = "LOW_SPEED"
    INSUFFICIENT_TRAJECTORY = "INSUFFICIENT_TRAJECTORY"
    BUFFER_MISS = "BUFFER_MISS"
    ALIGNMENT_REJECTED = "ALIGNMENT_REJECTED"
    NO_GPS_FIX = "NO_GPS_FIX"
    NO_CAMERA_DATA = "NO_CAMERA_DATA"


class ObservationStatus(str, Enum):
    QUEUED = "QUEUED"              # stored, waiting for the classification worker
    CONFIRMED = "CONFIRMED"        # attached to a defect on the map
    NEEDS_REVIEW = "NEEDS_REVIEW"
    REJECTED = "REJECTED"          # kept to measure false triggers
    FAILED = "FAILED"              # worker gave up after max attempts


class LatLon(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class ObservationMetadata(BaseModel):
    """Metadata part of POST /api/v1/observations. Mirrors shared/observation.schema.json."""

    model_config = {"extra": "allow"}

    schema_version: int = Field(ge=1, le=1)
    observation_id: str = Field(pattern=r"^[0-9a-fA-F-]{36}$")
    device_id: str = Field(min_length=1, max_length=64)
    evidence_status: EvidenceStatus
    t_impact_mono_ns: int
    t_impact_utc: datetime | None = None
    t_frame_utc: datetime | None = None
    utc_estimated: bool | None = None
    alignment_error_ms: float | None = None
    speed_before_mps: float | None = Field(default=None, ge=0)
    lookahead_m: float | None = Field(default=None, ge=0)
    defect: LatLon | None = None
    vehicle_at_frame: LatLon | None = None
    heading_deg: float | None = Field(default=None, ge=0, le=360)
    hdop: float | None = Field(default=None, ge=0)
    imu: dict[str, Any] | None = None
    image_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    firmware: str | None = None


class ReviewDecision(str, Enum):
    confirm = "confirm"
    reject = "reject"


class ObservationReview(BaseModel):
    decision: ReviewDecision
    label: DefectType | None = None   # required to confirm unless the worker already labelled it


class IncidentPatch(BaseModel):
    defect_type: DefectType | None = None
    severity: Severity | None = None
    repair_status: RepairStatus | None = None


# ---------------------------------------------------------------------------
# Preliminary automated classification from sensor metadata
# ---------------------------------------------------------------------------
_PEAK_KEYS = ("peak_g", "peak_acceleration_g", "vibration_peak", "peak", "intensity")

# (upper bound of peak vertical acceleration in g, severity)
_SEVERITY_THRESHOLDS = (
    (0.30, Severity.no_repair_needed),
    (0.60, Severity.low),
    (1.00, Severity.medium),
    (1.60, Severity.high),
)


def extract_peak(sensor_metadata: dict[str, Any] | None) -> float | None:
    if not sensor_metadata:
        return None
    for key in _PEAK_KEYS:
        value = sensor_metadata.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return abs(float(value))
    return None


def classify_severity(peak: float | None) -> Severity | None:
    if peak is None:
        return None
    for bound, sev in _SEVERITY_THRESHOLDS:
        if peak < bound:
            return sev
    return Severity.critical


def extract_defect_hint(sensor_metadata: dict[str, Any] | None) -> DefectType | None:
    if not sensor_metadata:
        return None
    hint = sensor_metadata.get("defect_type_hint") or sensor_metadata.get("defect_type")
    try:
        return DefectType(str(hint).lower()) if hint else None
    except ValueError:
        return None
