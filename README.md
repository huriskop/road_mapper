# Road Defect Monitoring & Mapping System

Implementation of `road_defect_monitoring_technical_specification.txt`.

| Part | Tech | Location |
|---|---|---|
| Backend API + spatial DB | FastAPI, SQLite with an **R\*Tree** spatial index | `backend/app/` |
| Image storage | Local disk, served at `/media/...` (swap `LocalImageStorage` for S3/MinIO) | `data/media/` |
| Web map + dashboard | Vanilla JS, Leaflet, markercluster, leaflet.heat | `frontend/` |
| Telemetry simulator | Python (requests + Pillow) | `simulator/simulate.py` |
| Tests | pytest | `tests/` |

## Quick start (Windows / PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

# 1. start the server  ->  http://127.0.0.1:8000  (API docs at /docs)
.venv\Scripts\python -m uvicorn app.main:app --app-dir backend --port 8000

# 2. in a second terminal, stream simulated edge devices
.venv\Scripts\python simulator\simulate.py --seed 7
.venv\Scripts\python simulator\simulate.py --time-scale 30   # slower "live" stream; watch the map update
```

Run the tests with `.venv\Scripts\python -m pytest`.

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `RDM_CLUSTER_RADIUS_M` | `5` | merge radius for spatial clustering (metres) |
| `RDM_DATA_DIR` | `./data` | DB + media root |
| `RDM_DB_PATH` / `RDM_MEDIA_DIR` | under data dir | override individually |
| `RDM_MAX_IMAGE_MB` | `8` | upload limit |
| `RDM_GOOGLE_MAPS_API_KEY` | *(empty)* | Google Maps JavaScript API browser key. Without it the map falls back to OpenStreetMap and shows a notice. |
| `RDM_SERVICE_AREA` | `37.30,126.10,37.82,126.80` | `min_lat,min_lng,max_lat,max_lng` (Incheon). The map is locked to this box and telemetry outside it is rejected with 422. |
| `RDM_SERVICE_AREA_NAME` | `Incheon` | label used in messages |

### Google Maps setup

1. In Google Cloud Console, enable the **Maps JavaScript API** and create an API key.
2. Restrict the key to HTTP referrers such as `http://127.0.0.1:8000/*` and `http://localhost:8000/*`.
3. Start the server with the key:
   ```powershell
   $env:RDM_GOOGLE_MAPS_API_KEY = "YOUR_KEY"
   .venv\Scripts\python -m uvicorn app.main:app --app-dir backend --port 8000
   ```
The map uses Google Roadmap, Satellite and Hybrid layers (switcher in the top-right corner) through the Leaflet GoogleMutant plugin. Marker clustering and the heatmap work the same on every layer.

## API (`/api/v1`)

| Method | Path | Notes |
|---|---|---|
| POST | `/telemetry` | multipart: `metadata` (JSON) + `image` (file). Also accepts `application/json` with a base64 `image` (data-URI ok). |
| GET | `/incidents` | `start`, `end` (ISO), `defect_type`, `severity`, `repair_status` (repeat or comma-separate), `sort` (`created_at`, `last_report_at`, `severity`, `repair_status`, `report_count`, `defect_type`), `order` (`asc`/`desc`), `page`, `page_size` |
| GET | `/incidents/map` | same filters, unpaginated lightweight feed for map layers |
| GET | `/incidents/{id}` | full incident incl. `images[]` (url, timestamp, device_id) |
| PATCH | `/incidents/{id}` | `{defect_type?, severity?, repair_status?}` |
| DELETE | `/incidents/{id}` | deletes incident, its packets and image files |
| GET | `/stats` | totals and breakdowns |
| GET | `/events` | Server-Sent Events stream (`incident.created/updated/deleted`) |

Telemetry metadata example:

```json
{
  "device_id": "SIM-001",
  "timestamp": "2026-11-18T07:06:18Z",
  "location": {"lat": 51.5091, "lng": -0.1350},
  "sensor_metadata": {"peak_g": 1.12, "rms_g": 0.31, "defect_type_hint": "pothole"}
}
```

## How aggregation works

1. **Spatial lookup** – the R\*Tree finds candidate incidents in a bounding box around the packet, then an exact haversine check keeps those within `RDM_CLUSTER_RADIUS_M`. Only *unresolved* incidents (`awaiting_repair`, `in_progress`) are merge targets; a defect reported again after `completed` opens a new incident.
2. **Merge** – the packet is attached (its photo joins the gallery), the incident's coordinates become the centroid of all its packets, `report_count` = number of **unique devices**, `packet_count` = total packets, `created_at` = earliest report, `last_report_at` / `updated_at` = latest.
3. **Preliminary classification** – severity comes from the max vibration peak (`peak_g`: <0.3 no repair needed, <0.6 low, <1.0 medium, <1.6 high, otherwise critical). Defect type comes from the device's `defect_type_hint`. Once an operator edits severity or type, it is locked and new packets no longer overwrite it.
4. Ingestion is serialised (lock + `BEGIN IMMEDIATE`), so two devices reporting the same defect at the same moment cannot create duplicate incidents.

The date range filter applies to `created_at` (first detection).

## Frontend

- **Map**: severity- or status-coloured pins (the number on a pin is its device count), clustering until zoom 18 (a cluster takes the colour of its worst member), a toggleable heatmap weighted by severity × log(report frequency), and a legend.
- **Detail drawer** (click a marker or a row), in the order the spec requires: photo carousel (arrows, thumbnails, swipe) → timestamp (first and latest report, devices, coordinates) → defect type → severity → repair status → **Edit / Save Changes / Delete Incident**.
- **Hide log**: the "▼ Hide log" button collapses the incident log so the map gets the full height. "▲ Show incident log" brings it back, and the browser remembers the choice.
- **Log table**: thumbnail, coordinates, date/time, type, severity, status and report count; sortable headers; pagination; multi-criteria filters shared with the map. Clicking a row flies the map to that marker and highlights it.
- **Updates**: edits and deletes update the UI right away and roll back if the request fails. SSE pushes new packets and changes from other operators to every open browser.

## Simulator

`simulator/simulate.py` builds a street-grid road network with ground-truth defects (pothole, bump, crack). It then drives N virtual vehicles along shared routes, sampling a 100 Hz vertical accelerometer signal. Defect impulses each have their own shape, and the signal includes noise and random false triggers. An edge-side threshold detector picks the events, and a rough waveform classifier produces the type hint. Each packet gets GPS noise and a generated road photo, or one of your own photos via `--images-dir`. When the run ends it prints a clustering summary: packets sent vs. distinct incidents. See `--help` for all options (`--devices`, `--defects`, `--passes`, `--gps-sigma`, `--time-scale`, `--base64`, `--dry-run`, `--center`). The default centre is Incheon City Hall (37.4563, 126.7052).
