"""Build README.md for a saved test run from captures.json and incidents_api.json.

Run from this folder: python make_report.py
"""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
caps = json.loads((HERE / "captures.json").read_text())["items"]
incs = json.loads((HERE / "incidents_api.json").read_text())["items"]


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


verdicts = Counter((c["analysis"].get("verdict") or c["analysis"].get("status")) for c in caps)
lines = [
    "# Demo day: video test and results",
    "",
    f"Road mapper on the Raspberry Pi 4, {caps[0]['time'][:10]}. Each jolt saved the 6 s of video before it; "
    "the detector picked the best frame and labelled the defect. Locations are from the GPS.",
    "",
    f"- **Jolts captured:** {len(caps)}",
    "- **Camera verdicts:** " + ", ".join(f"{k}: {v}" for k, v in verdicts.most_common()),
    f"- **Incidents on the map:** {len(incs)} (reports within 5 m are merged)",
    "",
    "Files: `captures/` (photos and `.avi` videos), `captures.json` (every jolt), "
    "`incidents_api.json` and `incidents.json` (the map data).",
    "",
    "## Map incidents",
    "",
    "| Type | Severity | Reports | Location | Photo |",
    "|---|---|---|---|---|",
]
for i in incs:
    img = local(i["reports"][-1].get("image"))
    photo = '<img src="%s" width="200">' % img if img else ""
    lines.append(f"| {i.get('defect_type', '?')} | {i.get('severity', '?')} | {i.get('report_count', len(i['reports']))} | "
                 f"{maplink(i['lat'], i['lng'])} | {photo} |")

lines += ["", "## Every jolt", "",
          "| # | Time (UTC) | Jolt | Peak g | Speed km/h | Camera verdict | Location | Photo | Video |",
          "|---|---|---|---|---|---|---|---|---|"]
for c in caps:
    a = c["analysis"]
    verdict = a.get("verdict") or a.get("status")
    conf = max((d["confidence"] for d in a.get("defects", [])), default=None)
    img = local(c.get("annotated")) or local(c.get("image"))
    vid = video_for(c)
    where = maplink(c["lat"], c["lng"]) if c.get("lat") is not None else "no fix"
    photo = '<img src="%s" width="200">' % img if img else ""
    video = "[%s s video](%s)" % (c.get("clip_s") or 6, vid) if vid else ""
    peak = c["peak_g"] if c["peak_g"] is not None else ""
    conf_s = " (%.2f)" % conf if conf else ""
    lines.append(f"| {c['id']} | {c['time'][11:19]} | {c['reason']} | {peak} | {c.get('speed_kmh', '')} | "
                 f"{verdict}{conf_s} | {where} | {photo} | {video} |")

(HERE / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"README.md: {len(caps)} jolts, {len(incs)} incidents")
