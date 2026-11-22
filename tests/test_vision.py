"""Regression tests on real photos the detector has got wrong before.

Photos live in test_images/user/<expected type>/. Tests are skipped when the photo or the
trained model files (see models/README.md) are not present.
"""
import sys
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import vision_detector as vd  # noqa: E402

USER = ROOT / "test_images" / "user"
needs_models = pytest.mark.skipif(
    not (ROOT / "models" / "road_damage_5class_v8n.onnx").exists(), reason="trained model file not present")


def kinds(path: Path) -> set[str]:
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    defects, _ = vd.analyze(cv2.imread(str(path)))
    return {d.kind for d in defects}


@pytest.mark.parametrize("name", ["painted_hump_white.jpg", "rubber_bump_yellow_black.jpg"])
def test_speed_bumps_are_found_and_not_called_potholes(name):
    found = kinds(USER / "bump" / name)
    assert "speed_bump" in found
    assert "pothole" not in found


@needs_models
@pytest.mark.parametrize("name", ["hole_behind_barrier.jpg", "many_potholes_dirt_road.jpg"])
def test_potholes_are_found_and_not_called_speed_bumps(name):
    found = kinds(USER / "pothole" / name)
    assert "pothole" in found
    assert "speed_bump" not in found


def test_painted_marking_detector_ignores_lane_dashes():
    import numpy as np
    img = np.full((480, 640, 3), 90, np.uint8)
    for i in range(5):                      # dashes running up the image, like a lane line
        y = 440 - i * 80
        cv2.rectangle(img, (310, y - 40), (330, y), (240, 240, 240), -1)
    assert vd.detect_bump_markings(img) == []


@needs_models
def test_detect_limits_the_types_and_nothing_is_found_on_a_blank_picture():
    import numpy as np
    assert vd.road_model() is not None
    assert vd.analyze(np.full((480, 640, 3), 90, np.uint8))[0] == []
    img = cv2.imread(str(ROOT / "test_images" / "real" / "pothole" / "Gurt_Pothole_3bf7d8.jpg"))
    assert {d.kind for d in vd.analyze(img)[0]} == {"pothole"}
    assert vd.analyze(img, detect=("crack",))[0] == []
