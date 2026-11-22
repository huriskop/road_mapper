"""Map data: merging, type and severity rules. No hardware needed."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.incidents import Incidents, distance_m  # noqa: E402


def result(verdict="none", cracks=0):
    return {"status": "done", "verdict": verdict, "cracks": cracks, "potholes": 0, "speed_bumps": 0}


def add(inc, lat=37.45, lng=126.7, reason="down", peak=0.5, analysis=None, t="2026-11-22T09:00:00.000Z"):
    return inc.add(lat=lat, lng=lng, time=t, image="/captures/a.jpg", reason=reason, peak_g=peak,
                   analysis=analysis, location="gps")


def test_close_reports_merge_and_far_ones_do_not(tmp_path):
    inc = Incidents(tmp_path / "i.json")
    a = add(inc)
    assert add(inc, lat=37.45 + 3 / 111_320)["id"] == a["id"]
    assert add(inc, lat=37.45 + 30 / 111_320)["id"] != a["id"]
    assert [i["report_count"] for i in inc.items] == [2, 1]
    assert 2.9 < distance_m(37.45, 126.7, 37.45 + 3 / 111_320, 126.7) < 3.1


def test_repaired_incident_gets_no_new_reports(tmp_path):
    inc = Incidents(tmp_path / "i.json")
    a = add(inc)
    inc.edit(a["id"], {"repair_status": "completed"})
    assert add(inc)["id"] != a["id"]


@pytest.mark.parametrize("reason, analysis, expected", [
    ("down", None, "pothole"), ("up", None, "bump"),
    ("up", result("pothole"), "pothole"), ("down", result("speed_bump"), "bump"),
    ("down", result("crack", 3), "pothole"),              # a jolt beats camera cracks
    ("upload", result("crack", 3), "crack"), ("upload", result(), "unclassified"),
])
def test_type(tmp_path, reason, analysis, expected):
    assert add(Incidents(tmp_path / "i.json"), reason=reason, analysis=analysis)["defect_type"] == expected


@pytest.mark.parametrize("peak, expected", [(None, "low"), (0.5, "low"), (-0.8, "medium"), (1.2, "high"), (3.1, "critical")])
def test_severity(tmp_path, peak, expected):
    assert add(Incidents(tmp_path / "i.json"), peak=peak)["severity"] == expected


def test_hand_edits_stay_and_survive_a_restart(tmp_path):
    inc = Incidents(tmp_path / "i.json")
    a = add(inc)
    inc.edit(a["id"], {"defect_type": "crack", "severity": "false_positive"})
    again = Incidents(tmp_path / "i.json")
    b = add(again, peak=3.0, analysis=result("pothole"))
    assert (b["defect_type"], b["severity"], b["type_source"], b["report_count"]) == ("crack", "false_positive", "operator", 2)


def test_bad_edit_and_delete(tmp_path):
    inc = Incidents(tmp_path / "i.json")
    a = add(inc)
    for bad in ({"severity": "huge"}, {"lat": 1}):
        with pytest.raises(ValueError):
            inc.edit(a["id"], bad)
    assert inc.edit("nope", {"severity": "low"}) is None
    assert inc.delete(a["id"]) and not inc.delete(a["id"])
