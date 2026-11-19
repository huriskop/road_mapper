"""config.yaml -> typed settings. Every key is optional; defaults are the plan's starting values."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from .detect.detector import DetectorConfig
from .evidence.select import EvidenceConfig


@dataclass
class ImuConfig:
    bus: int = 1
    addr: int = 0x68
    rate_hz: int = 200
    dlpf: int = 2
    read_interval_ms: float = 50.0
    ring_seconds: float = 10.0
    attitude_tau_s: float = 5.0


@dataclass
class GpsConfig:
    enabled: bool = True
    port: str = "/dev/serial0"
    baud: int = 38400
    rate_hz: int = 5
    nmea_latency_ms: float = 80.0
    configure_on_start: bool = True
    keep_s: float = 30.0
    max_gap_s: float = 1.0
    max_hdop: float = 5.0


@dataclass
class CameraConfig:
    enabled: bool = True
    width: int = 1280
    height: int = 720
    fps: float = 24.0
    exposure_us: int | None = 1500
    ring_seconds: float = 5.0
    frame_time_offset_ms: float = 0.0
    jpeg_quality: int = 90


@dataclass
class UplinkConfig:
    enabled: bool = True
    server_url: str = "http://127.0.0.1:8000"
    token: str = ""
    token_file: str = ""
    timeout_s: float = 20.0
    lease_s: float = 60.0


@dataclass
class Config:
    device_id: str = "rpi4-01"
    data_dir: str = "~/roadflag/data"
    mount_file: str = "~/roadflag/data/mount.json"
    budget_mb: float = 2048
    health_interval_s: float = 10.0
    snippet_s: float = 1.0
    imu: ImuConfig = field(default_factory=ImuConfig)
    gps: GpsConfig = field(default_factory=GpsConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    uplink: UplinkConfig = field(default_factory=UplinkConfig)

    def path(self, value: str) -> Path:
        return Path(value).expanduser()

    def device_token(self) -> str:
        if self.uplink.token:
            return self.uplink.token
        if self.uplink.token_file:
            return self.path(self.uplink.token_file).read_text().strip()
        return ""


def _fill(obj: Any, data: dict[str, Any], where: str = "") -> Any:
    known = {f.name: f for f in fields(obj)}
    for key, value in (data or {}).items():
        if key not in known:
            raise ValueError(f"unknown config key '{where}{key}'")
        current = getattr(obj, key)
        if is_dataclass(current):
            _fill(current, value, f"{where}{key}.")
        else:
            setattr(obj, key, value)
    return obj


def load(path: str | Path | None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    p = Path(path).expanduser()
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    return _fill(cfg, data)
