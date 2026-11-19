"""Runtime configuration (overridable through environment variables)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


def _parse_tokens(value: str) -> dict[str, str]:
    out = {}
    for part in value.split(","):
        if ":" in part:
            dev, tok = part.split(":", 1)
            if dev.strip() and tok.strip():
                out[dev.strip()] = tok.strip()
    return out


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: _env_path("RDM_DATA_DIR", PROJECT_ROOT / "data"))
    db_path: Path | None = None
    media_dir: Path | None = None
    frontend_dir: Path = PROJECT_ROOT / "frontend"
    # Packets closer than this (metres) to an unresolved incident are merged into it.
    cluster_radius_m: float = float(os.getenv("RDM_CLUSTER_RADIUS_M", "5"))
    max_image_bytes: int = int(float(os.getenv("RDM_MAX_IMAGE_MB", "8")) * 1024 * 1024)
    media_url_prefix: str = "/media"
    # Browser key for the Google Maps JavaScript API (restrict it by HTTP referrer in Google Cloud).
    google_maps_api_key: str = os.getenv("RDM_GOOGLE_MAPS_API_KEY", "")
    # Service area: Incheon Metropolitan City (mainland, Yeongjong, Ganghwa and nearby islands).
    # (min_lat, min_lng, max_lat, max_lng); telemetry outside it is rejected and the map is locked to it.
    service_area: tuple[float, float, float, float] = tuple(  # type: ignore[assignment]
        float(v) for v in os.getenv("RDM_SERVICE_AREA", "37.30,126.10,37.82,126.80").split(",")
    )
    service_area_name: str = os.getenv("RDM_SERVICE_AREA_NAME", "Incheon")
    # Run vision_detector.py on uploaded photos and store the labelled version (boxes + POTHOLE/CRACK).
    annotate_images: bool = os.getenv("RDM_ANNOTATE_IMAGES", "1").lower() not in ("0", "false", "no")
    # Which types the photo detector marks (comma-separated): pothole, crack, speed_bump.
    detect: tuple[str, ...] = tuple(
        k.strip() for k in os.getenv("RDM_DETECT", "pothole,crack,speed_bump").split(",") if k.strip()
    )

    # ---- edge devices (POST /api/v1/observations) -------------------------------------
    # "device_id:token,device_id:token". Only SHA-256 hashes of the tokens are stored.
    device_tokens: dict[str, str] = field(default_factory=lambda: _parse_tokens(os.getenv("RDM_DEVICE_TOKENS", "")))
    # De-duplication: a confirmed observation joins a defect with the same label within this
    # radius (~3x the NEO-6M's 2.5 m error) whose heading is within heading_tolerance_deg.
    dedupe_radius_m: float = float(os.getenv("RDM_DEDUPE_RADIUS_M", "8"))
    heading_tolerance_deg: float = float(os.getenv("RDM_HEADING_TOLERANCE_DEG", "45"))
    # Classification decision (starting values to tune)
    confirm_confidence: float = float(os.getenv("RDM_CONFIRM_CONF", "0.70"))
    review_confidence: float = float(os.getenv("RDM_REVIEW_CONF", "0.40"))
    review_imu_score: float = float(os.getenv("RDM_REVIEW_IMU_SCORE", "2.0"))  # image-less observations
    crack_confidence: float = 0.60     # crack rules have no score; 0.6 sends cracks to review
    blur_min_laplacian_var: float = float(os.getenv("RDM_BLUR_MIN", "40"))
    brightness_range: tuple[float, float] = (35.0, 225.0)
    worker_enabled: bool = os.getenv("RDM_WORKER", "1").lower() not in ("0", "false", "no")
    job_lease_s: float = 120.0
    job_max_attempts: int = 3

    def __post_init__(self) -> None:
        if self.db_path is None:
            self.db_path = _env_path("RDM_DB_PATH", self.data_dir / "road_defects.db")
        if self.media_dir is None:
            self.media_dir = _env_path("RDM_MEDIA_DIR", self.data_dir / "media")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)
