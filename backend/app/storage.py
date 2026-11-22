"""Local file storage for uploaded road-surface images.

Swap `LocalImageStorage` for an S3/MinIO implementation with the same interface
to move to object storage; the DB only keeps the relative key.
"""
from __future__ import annotations

import base64
import binascii
import uuid
from datetime import datetime, timezone
from pathlib import Path


class ImageError(ValueError):
    pass


def sniff_image_type(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    raise ImageError("unsupported image format (expected JPEG, PNG or WebP)")


def decode_base64_image(value: str) -> bytes:
    value = value.strip()
    if value.startswith("data:") and "," in value:
        value = value.split(",", 1)[1]
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageError("image is not valid base64") from exc


class LocalImageStorage:
    def __init__(self, root: Path, max_bytes: int) -> None:
        self.root = root
        self.max_bytes = max_bytes

    def save(self, data: bytes) -> str:
        if not data:
            raise ImageError("image is empty")
        if len(data) > self.max_bytes:
            raise ImageError(f"image exceeds {self.max_bytes // (1024 * 1024)} MB limit")
        ext = sniff_image_type(data)
        now = datetime.now(timezone.utc)
        key = f"{now:%Y/%m/%d}/{uuid.uuid4().hex}.{ext}"
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return key

    def delete(self, key: str) -> None:
        path = (self.root / key).resolve()
        if self.root.resolve() in path.parents:
            path.unlink(missing_ok=True)
