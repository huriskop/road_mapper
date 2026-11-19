"""FastAPI application: telemetry ingestion, incident CRUD, live events, static frontend."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from . import vision
from .config import Settings
from .db import SORT_COLUMNS, Database, IncidentFilters
from .events import EventBus
from .models import (
    DefectType,
    EvidenceStatus,
    IncidentPatch,
    ObservationMetadata,
    ObservationReview,
    ObservationStatus,
    RepairStatus,
    Severity,
    TelemetryMetadata,
    extract_defect_hint,
    extract_peak,
    to_utc_iso,
)
from .storage import ImageError, LocalImageStorage, decode_base64_image, sniff_image_type
from .worker import ReviewConflict, Worker

log = logging.getLogger(__name__)


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
    db.sync_devices(settings.device_tokens)
    storage = LocalImageStorage(settings.media_dir, settings.max_image_bytes)
    bus = EventBus()
    worker = Worker(db, storage, settings)
    wake = asyncio.Event()

    def publish_result(res: dict[str, Any]) -> None:
        bus.publish("observation.updated", {"id": res["observation_id"], "status": res["status"]})
        inc = res.get("incident")
        if inc:
            bus.publish("incident.created" if inc["incident_created"] else "incident.updated",
                        {"id": inc["incident_id"]})

    async def worker_loop() -> None:
        while True:
            try:
                res = await run_in_threadpool(worker.run_once)
            except Exception:  # never let the loop die
                log.exception("worker loop error")
                res = None
            if res is None:
                wake.clear()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(wake.wait(), 2.0)
                continue
            publish_result(res)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        task = asyncio.create_task(worker_loop()) if settings.worker_enabled else None
        yield
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="Road Defect Monitoring & Mapping API", version="1.1.0", lifespan=lifespan)
    app.state.settings, app.state.db, app.state.bus, app.state.worker = settings, db, bus, worker
    api = APIRouter(prefix="/api/v1")

    # ------------------------------------------------- edge observations (Pi)
    @api.post("/observations", status_code=201)
    async def ingest_observation(request: Request, response: Response) -> dict[str, Any]:
        """Pi upload: multipart `metadata` (shared/observation.schema.json) + optional `image` JPEG.
        Headers: Authorization: Bearer <device token>, Idempotency-Key: <observation_id>.
        201 new, 200 + the same receipt for a repeat; 409 if the ID was used for other content."""
        auth = request.headers.get("authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        device = db.device_for_token(token) if token else None
        if device is None:
            raise HTTPException(401, "unknown or missing device token")
        ctype = request.headers.get("content-type", "")
        if not ctype.startswith("multipart/form-data"):
            raise HTTPException(415, "use multipart/form-data with parts 'metadata' and 'image'")
        form = await request.form()
        raw_meta = form.get("metadata")
        if isinstance(raw_meta, UploadFile):
            raw_meta = (await raw_meta.read()).decode("utf-8", "replace")
        if not raw_meta:
            raise HTTPException(422, "metadata part is required")
        try:
            meta = ObservationMetadata.model_validate(json.loads(raw_meta))
        except json.JSONDecodeError as exc:
            raise HTTPException(422, f"metadata is not JSON: {exc.msg}") from exc
        except ValidationError as exc:
            raise HTTPException(422, json.loads(exc.json(include_url=False))) from exc
        if meta.device_id != device:
            raise HTTPException(403, f"token belongs to device '{device}', not '{meta.device_id}'")
        if request.headers.get("idempotency-key") != meta.observation_id:
            raise HTTPException(400, "Idempotency-Key header must equal metadata.observation_id")
        upload = form.get("image")
        image = await upload.read() if isinstance(upload, UploadFile) else None
        if meta.evidence_status == EvidenceStatus.AVAILABLE and not image:
            raise HTTPException(422, "evidence_status AVAILABLE needs the image part")
        if image and meta.evidence_status != EvidenceStatus.AVAILABLE:
            raise HTTPException(422, f"evidence_status {meta.evidence_status.value} must not carry an image")
        if image:
            if len(image) > settings.max_image_bytes:
                raise HTTPException(413, f"image exceeds {settings.max_image_bytes // (1024 * 1024)} MB limit")
            try:
                sniff_image_type(image)
            except ImageError as exc:
                raise HTTPException(422, str(exc)) from exc
            if hashlib.sha256(image).hexdigest() != meta.image_sha256:
                raise HTTPException(422, "image_sha256 does not match the image")
        elif meta.image_sha256:
            raise HTTPException(422, "image_sha256 given but no image part")
        if meta.defect is not None:
            min_lat, min_lng, max_lat, max_lng = settings.service_area
            if not (min_lat <= meta.defect.lat <= max_lat and min_lng <= meta.defect.lon <= max_lng):
                raise HTTPException(422, f"defect is outside the {settings.service_area_name} service area")
        key = storage.save(image) if image else None
        try:
            receipt, created = db.insert_observation(meta.model_dump(mode="json"), key)
        except ValueError as exc:
            if key:
                storage.delete(key)
            raise HTTPException(409, str(exc)) from exc
        except Exception:
            if key:
                storage.delete(key)
            raise
        if not created and key:
            storage.delete(key)     # a retry: keep the first copy
        db.touch_device(device, meta.firmware)
        if created:
            wake.set()
            bus.publish("observation.created", {"id": meta.observation_id})
        response.status_code = 201 if created else 200
        return receipt

    @api.get("/observations")
    async def list_observations(
        status: list[str] | None = Query(None),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        statuses = _parse_multi(status, ObservationStatus, "status")
        items, total = db.list_observations(statuses, limit, offset)
        return {"items": items, "total": total}

    @api.get("/observations/{obs_id}")
    async def get_observation(obs_id: str) -> dict[str, Any]:
        obs = db.get_observation(obs_id)
        if obs is None:
            raise HTTPException(404, "observation not found")
        return obs

    @api.patch("/observations/{obs_id}/review")
    async def review_observation(obs_id: str, body: ObservationReview) -> dict[str, Any]:
        """A person confirms (with a label) or rejects a NEEDS_REVIEW observation."""
        try:
            res = await run_in_threadpool(worker.review, obs_id, body.decision.value, body.label)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ReviewConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        publish_result(res)
        return db.get_observation(obs_id)

    @api.get("/defects")
    async def defects_geojson(
        bbox: str | None = Query(None, description="west,south,east,north"),
        label: list[str] | None = Query(None),
        status: list[str] | None = Query(None, description="repair status"),
        since: datetime | None = None,
    ) -> dict[str, Any]:
        """GeoJSON FeatureCollection, one feature per defect (incident)."""
        box = None
        if bbox:
            try:
                box = tuple(float(v) for v in bbox.split(","))
                assert len(box) == 4
            except (ValueError, AssertionError):
                raise HTTPException(422, "bbox must be west,south,east,north")
        return db.defects_geojson(box, _parse_multi(label, DefectType, "label"),
                                  _parse_multi(status, RepairStatus, "status"),
                                  to_utc_iso(since) if since else None)

    # -------------------------------------------------------------- ingestion
    @api.post("/telemetry", status_code=201)
    async def ingest_telemetry(request: Request) -> dict[str, Any]:
        meta, image_bytes = await _read_telemetry(request)
        min_lat, min_lng, max_lat, max_lng = settings.service_area
        if not (min_lat <= meta.location.lat <= max_lat and min_lng <= meta.location.lng <= max_lng):
            raise HTTPException(422, f"location is outside the {settings.service_area_name} service area")
        found = None
        try:
            sniff_image_type(image_bytes)  # reject non-images before running the detector
            if settings.annotate_images:
                image_bytes, found = await run_in_threadpool(vision.annotate, image_bytes, settings.detect)
                if found is not None:
                    meta.sensor_metadata = {**(meta.sensor_metadata or {}), "vision": found}
            image_key = storage.save(image_bytes)
        except ImageError as exc:
            raise HTTPException(422, str(exc)) from exc
        # Type: the camera's pothole / speed-bump finding beats the vibration guess; camera cracks
        # (image rules, less reliable) are only used when the sensor gave no type at all.
        hint, hint_source = extract_defect_hint(meta.sensor_metadata), "sensor"
        if found and found.get("camera_type"):
            hint, hint_source = DefectType(found["camera_type"]), "camera"
        elif found and found.get("cracks") and hint in (None, DefectType.unclassified):
            hint, hint_source = DefectType.crack, "camera"
        try:
            result = db.ingest_packet(
                device_id=meta.device_id,
                timestamp=to_utc_iso(meta.timestamp),
                lat=meta.location.lat,
                lng=meta.location.lng,
                image_path=image_key,
                sensor_metadata=meta.sensor_metadata,
                peak=extract_peak(meta.sensor_metadata),
                defect_hint=hint,
                hint_source=hint_source,
                radius_m=settings.cluster_radius_m,
            )
        except Exception:
            storage.delete(image_key)
            raise
        bus.publish("incident.created" if result["incident_created"] else "incident.updated",
                    {"id": result["incident_id"]})
        return result

    # ------------------------------------------------------ test a single photo
    @api.post("/analyze")
    async def analyze_photo(request: Request) -> dict[str, Any]:
        """Run the detector on one uploaded photo and return the labelled image + detections.
        Multipart form: `image` (file), optional `detect` (comma-separated: pothole, crack, speed_bump)
        and `conf` (pothole confidence cut-off, 0-1). Nothing is saved."""
        form = await request.form()
        upload = form.get("image")
        if not isinstance(upload, UploadFile):
            raise HTTPException(422, "send the photo as a multipart file field named 'image'")
        data = await upload.read()
        if len(data) > settings.max_image_bytes:
            raise HTTPException(422, f"image exceeds {settings.max_image_bytes // (1024 * 1024)} MB limit")
        detect = tuple(k.strip() for k in str(form.get("detect") or ",".join(settings.detect)).split(",")
                       if k.strip() in ("pothole", "crack", "speed_bump")) or settings.detect
        try:
            conf = min(0.95, max(0.05, float(form.get("conf") or 0.5)))
        except ValueError:
            raise HTTPException(422, "conf must be a number between 0 and 1")
        try:
            return await run_in_threadpool(vision.analyze_photo, data, detect, conf)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

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
    @api.get("/stream")
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

        @app.get("/test", include_in_schema=False)
        async def test_page() -> FileResponse:
            return FileResponse(settings.frontend_dir / "test.html")

        @app.get("/review", include_in_schema=False)
        async def review_page() -> FileResponse:
            return FileResponse(settings.frontend_dir / "review.html")

    return app


app = create_app()
