"""Look-back: the detector picks the frame from the 6 s before a jolt that shows the defect best."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.detection import Analyzer  # noqa: E402


def fake(verdicts):
    """An Analyzer whose detector answers from a table: frame bytes -> (verdict, confidence)."""
    a = Analyzer.__new__(Analyzer)
    a.detect, a.conf, a.roi = ("pothole", "crack", "speed_bump"), None, None

    def analyze(jpg, *_):
        v, c = verdicts.get(jpg, ("none", None))
        return b"L" + jpg, {"verdict": v, f"{v}_confidence": c, "time_ms": 500,
                            "defects": [{"confidence": c}] if c else []}
    a.analyze = analyze
    return a


# 6 s at 10 fps, oldest first; frame names say how far before the jolt they are
CLIP = [(round(6 - i / 10, 1), f"f{i}".encode()) for i in range(61)]


def test_finds_the_pothole_seen_before_the_jolt():
    jpg, labelled, r = fake({b"f20": ("pothole", 0.4), b"f30": ("pothole", 0.8), b"f60": ("none", None)}).scan(CLIP, "pothole")
    assert jpg == b"f30" and labelled == b"Lf30"
    assert r["seconds_before"] == 3.0 and r["frames_checked"] == 13 and r["time_ms"] == 13 * 500


def test_the_type_of_the_jolt_wins_over_a_more_confident_other_type():
    jpg, _, _ = fake({b"f10": ("speed_bump", 0.9), b"f40": ("pothole", 0.3)}).scan(CLIP, "pothole")
    assert jpg == b"f40"


def test_nothing_found_keeps_the_frame_of_the_jolt():
    jpg, _, r = fake({}).scan(CLIP, "pothole")
    assert jpg == b"f60" and r["verdict"] == "none" and r["seconds_before"] == 0.0
