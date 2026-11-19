"""FastAPI application: telemetry ingestion, incident CRUD, live events, static frontend."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from enum import Enum
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from .config import Settings
from .db import SORT_COLUMNS, Database, IncidentFilters
from .events import EventBus
from .models import (
    DefectType,
    IncidentPatch,
    RepairStatus,
    Severity,
    TelemetryMetadata,
    extract_defect_hint,
    extract_peak,
    to_utc_iso,
)
from .storage import ImageError, LocalImageStorage, decode_base64_image


def _parse_multi(values: list[str] | None, enum: type[Enum], name: str) -> list[str]:
    """Accept both ?x=a&x=b and ?x=a,b."""
    out: list[str] = []
    allowed = {e.value for e in enum}
    for raw in values or []:
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if part not in allowed:
                raise HTTPException(422, f"invalid {name} '{part}'; allowed: {sorted(allowed)}")
            out.append(part)
    return out


def _filters(start, end, defect_type, severity, repair_status) -> IncidentFilters:
    return IncidentFilters(
        start=to_utc_iso(start) if start else None,
        end=to_utc_iso(end) if end else None,
        defect_types=_parse_multi(defect_type, DefectType, "defect_type"),
        severities=_parse_multi(severity, Severity, "severity"),
        statuses=_parse_multi(repair_status, RepairStatus, "repair_status"),
    )


async def _read_telemetry(request: Request) -> tuple[TelemetryMetadata, bytes]:
    """Parse multipart (metadata JSON + image file) or a JSON body with a base64 image."""
    ctype = request.headers.get("content-type", "")
    image_bytes: bytes | None = None
    try:
        if ctype.startswith(("multipart/form-data", "application/x-www-form-urlencoded")):
            form = await request.form()
            if "metadata" in form:
                raw = json.loads(str(form["metadata"]))
            else:  # flat form fields fallback
                raw = {
                    "device_id": form.get("device_id"),
                    "timestamp": form.get("timestamp"),
                    "location": {"lat": form.get("lat"), "lng": form.get("lng")},
                    "sensor_metadata": json.loads(str(form["sensor_metadata"])) if form.get("sensor_metadata") else None,
                }
            image = form.get("image")
            if isinstance(image, UploadFile):
                image_bytes = await image.read()
            elif isinstance(image, str) and image:
                image_bytes = decode_base64_image(image)
            elif isinstance(raw.get("image"), str):
                image_bytes = decode_base64_image(raw["image"])
        elif ctype.startswith("application/json"):
            raw = await request.json()
            if isinstance(raw, dict) and isinstance(raw.get("image"), str):
                image_bytes = decode_base64_image(raw["image"])
        else:
            raise HTTPException(415, "use multipart/form-data or application/json")
        if not isinstance(raw, dict):
            raise HTTPException(422, "metadata must be a JSON object")
        meta = TelemetryMetadata.model_validate({k: v for k, v in raw.items() if k != "image"})
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"invalid JSON: {exc.msg}") from exc
    except ValidationError as exc:
        raise HTTPException(422, json.loads(exc.json(include_url=False))) from exc
    except ImageError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not image_bytes:
        raise HTTPException(422, "image is required (multipart file field 'image' or base64 'image')")
    return meta, image_bytes


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    db = Database(settings.db_path, settings.media_url_prefix)
    storage = LocalImageStorage(settings.media_dir, settings.max_image_bytes)
    bus = EventBus()

    app = FastAPI(title="Road Defect Monitoring & Mapping API", version="1.0.0")
    app.state.settings, app.state.db, app.state.bus = settings, db, bus
    api = APIRouter(prefix="/api/v1")

    # -------------------------------------------------------------- ingestion
    @api.post("/telemetry", status_code=201)
    async def ingest_telemetry(request: Request) -> dict[str, Any]:
        meta, image_bytes = await _read_telemetry(request)
        min_lat, min_lng, max_lat, max_lng = settings.service_area
        if not (min_lat <= meta.location.lat <= max_lat and min_lng <= meta.location.lng <= max_lng):
            raise HTTPException(422, f"location is outside the {settings.service_area_name} service area")
        try:
            image_key = storage.save(image_bytes)
        except ImageError as exc:
            raise HTTPException(422, str(exc)) from exc
        try:
            result = db.ingest_packet(
                device_id=meta.device_id,
                timestamp=to_utc_iso(meta.timestamp),
                lat=meta.location.lat,
                lng=meta.location.lng,
                image_path=image_key,
                sensor_metadata=meta.sensor_metadata,
                peak=extract_peak(meta.sensor_metadata),
                defect_hint=extract_defect_hint(meta.sensor_metadata),
                radius_m=settings.cluster_radius_m,
            )
        except Exception:
            storage.delete(image_key)
            raise
        bus.publish("incident.created" if result["incident_created"] else "incident.updated",
                    {"id": result["incident_id"]})
        return result

    # ------------------------------------------------------------------ reads
    @api.get("/incidents")
    async def list_incidents(
        start: datetime | None = None,
        end: datetime | None = None,
        defect_type: list[str] | None = Query(None),
        severity: list[str] | None = Query(None),
        repair_status: list[str] | None = Query(None),
        sort: str = Query("created_at", description=f"one of {sorted(SORT_COLUMNS)}"),
        order: str = Query("desc", pattern="^(asc|desc)$"),
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=200),
    ) -> dict[str, Any]:
        if sort not in SORT_COLUMNS:
            raise HTTPException(422, f"invalid sort; allowed: {sorted(SORT_COLUMNS)}")
        filters = _filters(start, end, defect_type, severity, repair_status)
        items, total = db.list_incidents(filters, sort, order, page, page_size)
        return {"items": items, "total": total, "page": page, "page_size": page_size,
                "pages": max(1, -(-total // page_size))}

    @api.get("/incidents/map")
    async def map_incidents(
        start: datetime | None = None,
        end: datetime | None = None,
        defect_type: list[str] | None = Query(None),
        severity: list[str] | None = Query(None),
        repair_status: list[str] | None = Query(None),
    ) -> dict[str, Any]:
        """Lightweight, unpaginated feed for the map's marker + heatmap layers."""
        filters = _filters(start, end, defect_type, severity, repair_status)
        return {"items": db.map_incidents(filters)}

    @api.get("/incidents/{incident_id}")
    async def get_incident(incident_id: str) -> dict[str, Any]:
        incident = db.get_incident(incident_id)
        if incident is None:
            raise HTTPException(404, "incident not found")
        return incident

    @api.get("/config")
    async def client_config() -> dict[str, Any]:
        min_lat, min_lng, max_lat, max_lng = settings.service_area
        return {
            "google_maps_api_key": settings.google_maps_api_key,
            "service_area": {"name": settings.service_area_name,
                             "south": min_lat, "west": min_lng, "north": max_lat, "east": max_lng},
        }

    @api.get("/stats")
    async def stats() -> dict[str, Any]:
        return db.stats()

    # ----------------------------------------------------------------- writes
    @api.patch("/incidents/{incident_id}")
    async def patch_incident(incident_id: str, patch: IncidentPatch) -> dict[str, Any]:
        changes = {k: v.value for k, v in patch.model_dump(exclude_none=True).items()}
        incident = db.update_incident(incident_id, changes)
        if incident is None:
            raise HTTPException(404, "incident not found")
        bus.publish("incident.updated", {"id": incident_id})
        return incident

    @api.delete("/incidents/{incident_id}", status_code=204)
    async def delete_incident(incident_id: str) -> None:
        paths = db.delete_incident(incident_id)
        if paths is None:
            raise HTTPException(404, "incident not found")
        for p in paths:
            storage.delete(p)
        bus.publish("incident.deleted", {"id": incident_id})

    # ------------------------------------------------------------ live events
    @api.get("/events")
    async def events(request: Request) -> StreamingResponse:
        queue = bus.subscribe()

        async def stream():
            try:
                yield "retry: 3000\n\n"
                while not await request.is_disconnected():
                    try:
                        msg = await asyncio.wait_for(queue.get(), timeout=15)
                        yield f"data: {msg}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
            finally:
                bus.unsubscribe(queue)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    app.include_router(api)

    # ------------------------------------------------------- static / frontend
    app.mount(settings.media_url_prefix, StaticFiles(directory=settings.media_dir), name="media")
    if settings.frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=settings.frontend_dir), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(settings.frontend_dir / "index.html")

    return app


app = create_app()
