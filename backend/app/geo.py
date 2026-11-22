"""Small geodesy helpers for spatial clustering."""
from __future__ import annotations

import math

EARTH_RADIUS_M = 6_371_008.8
METRES_PER_DEG_LAT = 111_320.0


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def bounding_box(lat: float, lng: float, radius_m: float) -> tuple[float, float, float, float]:
    """(min_lat, max_lat, min_lng, max_lng) of a box enclosing a circle of radius_m.

    Padded slightly because the R*Tree stores 32-bit floats.
    """
    pad = radius_m * 1.05 + 1.0
    dlat = pad / METRES_PER_DEG_LAT
    cos_lat = max(math.cos(math.radians(lat)), 1e-6)
    dlng = pad / (METRES_PER_DEG_LAT * cos_lat)
    return lat - dlat, lat + dlat, lng - dlng, lng + dlng
