"""Runtime configuration (overridable through environment variables)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


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

    def __post_init__(self) -> None:
        if self.db_path is None:
            self.db_path = _env_path("RDM_DB_PATH", self.data_dir / "road_defects.db")
        if self.media_dir is None:
            self.media_dir = _env_path("RDM_MEDIA_DIR", self.data_dir / "media")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)
