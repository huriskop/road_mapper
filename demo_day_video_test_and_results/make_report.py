"""Build README.md for a saved test run.

Reads every captures_log_*.json (the capture page's list, saved before each restart) and incidents.json
(the map). Run from this folder: python make_report.py
"""
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).parent
LOCAL = timezone(timedelta(hours=9))      # the Pi's clock (KST); capture times are stored in UTC
FROM = "16:00"                            # "findings from" section, local time

caps = {}
for f in sorted(HERE.glob("captures_log_*.json")):
    for c in json.loads(f.read_text())["items"]:
        caps[c["image"]] = c                  # ids restart with the program; the photo name does not
caps = sorted(caps.values(), key=lambda c: c["time"])
incs = json.loads((HERE / "incidents.json").read_text())


def local_time(iso):
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(LOCAL)


def local(path):
    """'/captures/x.jpg' -> 'captures/x.jpg' if the file was saved, else None."""
    if path and (HERE / path.lstrip("/")).exists():
        return path.lstrip("/")
    return None


def video_for(c):
    v = local(c.get("video"))
    if v:
        return v
    stamp = Path(c["image"]).name.split("_")[0]
    hits = sorted((HERE / "captures").glob(f"{stamp}_*.avi"))
    return f"captures/{hits[0].name}" if hits else None


def maplink(lat, lng):
    return f"[{lat:.6f}, {lng:.6f}](https://www.google.com/maps?q={lat:.6f},{lng:.6f})"


def verdict_of(c):
    a = c.get("analysis") or {}
    return a.get("verdict") or a.get("status") or "?"


def summary(items):
    return ", ".join(f"{k}: {v}" for k, v in Counter(verdict_of(c) for c in items).most_common())


def jolt_table(items):
    out = ["| # | Time (KST) | Jolt | Peak g | Speed km/h | Camera verdict | Map | Location | Photo | Video |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for n, c in enumerate(items, 1):
        a = c.get("analysis") or {}
        conf = max((d["confidence"] or 0 for d in a.get("defects", [])), default=None)
        img = local(c.get("annotated")) or local(c.get("image"))
        vid = video_for(c)
        m = c.get("map") or {}
        on_map = f"{m['type']} ({m['severity']})" if m.get("type") else m.get("status", "")
        where = maplink(c["lat"], c["lng"]) if c.get("lat") is not None else "no GPS fix"
        photo = '<img src="%s" width="180">' % img if img else ""
        video = "[%s s](%s)" % (c.get("clip_s") or 6, vid) if vid else ""
        peak = c["peak_g"] if c.get("peak_g") is not None else ""
        conf_s = " (%.2f)" % conf if conf else ""
        out.append(f"| {n} | {local_time(c['time']):%H:%M:%S} | {c['reason']} | {peak} | {c.get('speed_kmh', '')} | "
                   f"{verdict_of(c)}{conf_s} | {on_map} | {where} | {photo} | {video} |")
    return out


day = local_time(caps[0]["time"])
late = [c for c in caps if f"{local_time(c['time']):%H:%M}" >= FROM]
lines = [
    "# Demo day: video test and results",
    "",
    f"Road mapper on the Raspberry Pi 4, {day:%Y-%m-%d}, {local_time(caps[0]['time']):%H:%M} to "
    f"{local_time(caps[-1]['time']):%H:%M} KST. Each jolt saved the 6 s of video before it; the detector picked "
    "the best frame and labelled the defect. Locations are from the GPS.",
    "",
    f"- **Jolts captured:** {len(caps)}",
    f"- **Camera verdicts:** {summary(caps)}",
    f"- **Incidents on the map:** {len(incs)} (reports within 5 m are merged): "
    + ", ".join(f"{k}: {v}" for k, v in Counter(i.get("defect_type") for i in incs).most_common()),
    f"- **Findings from {FROM}:** {len(late)} jolts ({summary(late)})",
    "",
    "Files:",
    "- `captures/`: photos (`_best` = frame the detector chose, `_labelled` = with boxes) and `.avi` videos",
    "- `captures_log_*.json`: the sensor log of every jolt (time, jolt direction, peak g, speed, GPS, detector result)",
    "- `incidents.json`: the map data (location, type, severity, every report); `incidents_api_*.json`: the same as the map page served it",
    "- `run.log`: the program's messages, including each change of the Detect choice",
    "",
    "## Map incidents",
    "",
    "| Type | Severity | Reports | First seen (KST) | Location | Photo |",
    "|---|---|---|---|---|---|",
]
for i in sorted(incs, key=lambda i: i["created_at"]):
    img = local(i["reports"][-1].get("image"))
    photo = '<img src="%s" width="180">' % img if img else ""
    lines.append(f"| {i.get('defect_type', '?')} | {i.get('severity', '?')} | {i.get('report_count', len(i['reports']))} | "
                 f"{local_time(i['created_at']):%H:%M:%S} | {maplink(i['lat'], i['lng'])} | {photo} |")

lines += ["", f"## Findings from {FROM} KST", ""] + (jolt_table(late) if late else ["None."])
lines += ["", "## Every jolt", ""] + jolt_table(caps)

(HERE / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"README.md: {len(caps)} jolts ({len(late)} from {FROM}), {len(incs)} incidents")
