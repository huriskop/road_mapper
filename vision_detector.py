"""
Crack and pothole detection in road images / video with classical computer vision
(OpenCV only, no training data needed).

Cracks
------
1. Grayscale -> CLAHE (fixes uneven lighting / shadows).
2. Black-hat morphology (closing - image): keeps thin DARK structures narrower
   than the kernel and removes the smooth asphalt background.
   Two kernel sizes (15 and 31 px) catch both hairline and wider cracks.
3. Hysteresis threshold on the black-hat response: the strongest pixels are seeds,
   grown along connected moderately-dark pixels. Thresholds adapt to the image
   (noise estimate, capped by percentiles so coarse real asphalt still works).
4. Connected components -> skeleton. A component is a crack when
   skeleton length >= min_len and mean width (area / length) <= max_width,
   i.e. long and thin; asphalt grain and stains are rejected.
5. Type: PCA orientation of the skeleton
      ~ along the driving direction -> longitudinal
      ~ across                       -> transverse
      many branch points / mesh      -> alligator (fatigue cracking)

Potholes
--------
1. Local background = heavy median blur; "darkness" = background - image
   (potholes are darker: shadowed walls, wet / exposed base layer).
2. Threshold, open/close, keep blobs that are big, compact (solidity),
   not line-shaped (aspect ratio), and more textured than the surroundings.

Usage:
    python vision_detector.py road.jpg -o result.jpg
    python vision_detector.py dashcam.mp4 -o result.mp4 --roi-top 0.55
    python vision_detector.py --demo
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class Defect:
    kind: str                 # longitudinal_crack | transverse_crack | alligator_crack | pothole | speed_bump
    bbox: tuple[int, int, int, int]   # x, y, w, h
    length_px: float = 0.0
    width_px: float = 0.0
    area_px: int = 0
    severity: str = "low"
    confidence: float | None = None   # set when a trained model made the detection


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Morphological skeleton (works with plain opencv, no contrib)."""
    if hasattr(cv2, "ximgproc"):
        return cv2.ximgproc.thinning(mask)
    # Pad with background: cv2.erode treats out-of-image pixels as foreground, so a
    # component that fills its own bounding box would otherwise never erode away.
    img = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    skel = np.zeros_like(img)
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(img):
        eroded = cv2.erode(img, k)
        skel |= cv2.subtract(img, cv2.dilate(eroded, k))
        img = eroded
    return skel[1:-1, 1:-1]


def branch_points(skel: np.ndarray) -> int:
    s = (skel > 0).astype(np.uint8)
    nb = cv2.filter2D(s, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - s
    return int(np.sum((s == 1) & (nb >= 3)))


def preprocess(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.GaussianBlur(gray, (3, 3), 0)


# --------------------------------------------------------------------------- #
# cracks
# --------------------------------------------------------------------------- #
def hysteresis(resp: np.ndarray, low: float, high: float) -> np.ndarray:
    """Keep connected regions of resp >= low that contain at least one pixel >= high."""
    n, lab = cv2.connectedComponents((resp >= low).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[resp >= high])] = True
    keep[0] = False
    return keep[lab].astype(np.uint8) * 255


def detect_cracks(gray: np.ndarray, *, kernels: tuple[int, ...] = (15, 31), min_contrast: int = 25,
                  k_sigma: float = 5.0, p_low: float = 92.0, p_high: float = 99.0,
                  min_len: int = 60, max_width: float = 8.0,
                  exclude: np.ndarray | None = None) -> tuple[list[Defect], np.ndarray]:
    # Black-hat at two scales: thin hairline cracks and wider (open) cracks.
    bh = np.zeros_like(gray)
    for k in kernels:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bh = np.maximum(bh, cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se))
    # Hysteresis threshold. Seeds = strongest line pixels; regions then grow along
    # moderately dark connected pixels. On coarse real asphalt the gaps between stones
    # make the noise estimate (med + k*sigma) huge, so it is capped by a high percentile.
    med = float(np.median(bh))
    sigma = 1.4826 * float(np.median(np.abs(bh.astype(np.float32) - med)))
    high = max(min_contrast, min(med + k_sigma * sigma, float(np.percentile(bh, p_high))))
    low = min(high, max(min_contrast / 2, float(np.percentile(bh, p_low))))
    mask = hysteresis(bh, low, high)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    max_width = max_width * max(kernels) / 15     # wider kernel admits wider cracks
    cell_area = min(kernels) ** 2
    if exclude is not None:
        mask[exclude > 0] = 0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out_mask = np.zeros_like(mask)
    defects = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if max(w, h) < min_len:
            continue
        comp = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        skel = skeletonize(comp)
        length = float(cv2.countNonZero(skel))
        if length < min_len:
            continue
        width = float(area) / length
        if width > max_width:
            continue

        # alligator cracking = a mesh -> the crack pattern encloses several "cells"
        cnts, hier = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cells = 0 if hier is None else sum(
            1 for j, hh in enumerate(hier[0]) if hh[3] != -1 and cv2.contourArea(cnts[j]) > cell_area)
        ys, xs = np.nonzero(skel)
        if cells >= 3 or (cells >= 1 and branch_points(skel) >= 8):
            kind = "alligator_crack"
        else:
            cov = np.cov(np.vstack([xs, ys]))
            evals, evecs = np.linalg.eigh(cov)
            vx, vy = evecs[:, -1]
            angle = abs(np.degrees(np.arctan2(vy, vx)))      # 0 = horizontal in image
            # camera looks down the road -> driving direction is vertical in the image
            kind = "longitudinal_crack" if 45 < angle < 135 else "transverse_crack"

        sev = "high" if width > 5 or kind == "alligator_crack" else "medium" if width > 3 else "low"
        defects.append(Defect(kind, (int(x), int(y), int(w), int(h)),
                              round(length, 1), round(width, 2), int(area), sev))
        out_mask[labels == i] = 255
    return defects, out_mask


# --------------------------------------------------------------------------- #
# potholes
# --------------------------------------------------------------------------- #
def detect_potholes(gray: np.ndarray, *, bg_kernel: int = 0, min_dark: int = 25,
                    min_area_frac: float = 0.002, min_solidity: float = 0.6,
                    max_aspect: float = 4.0, min_fill: float = 0.5) -> tuple[list[Defect], np.ndarray]:
    h, w = gray.shape
    if bg_kernel <= 0:
        bg_kernel = (max(h, w) // 3) | 1   # must be ~2x larger than the biggest pothole
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bg = cv2.medianBlur(small, max(3, (bg_kernel // 4) | 1))
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    dark = cv2.subtract(bg, cv2.medianBlur(gray, 9))

    _, mask = cv2.threshold(dark, min_dark, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    c = max(15, (max(h, w) // 25) | 1)    # join fragments of one hole (rough, wet interior)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c, c)))

    pixel_dark = cv2.subtract(bg, cv2.GaussianBlur(gray, (5, 5), 0)) > min_dark
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    global_tex = float(np.median(lap)) + 1e-6

    defects, out_mask = [], np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_frac * h * w:
            continue
        hull = cv2.convexHull(c)
        solidity = area / max(cv2.contourArea(hull), 1)
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        aspect = max(rw, rh) / max(min(rw, rh), 1)
        if solidity < min_solidity or aspect > max_aspect:
            continue
        blob = np.zeros_like(mask)
        cv2.drawContours(blob, [c], -1, 255, -1)
        texture = float(np.mean(lap[blob > 0])) / global_tex
        if texture < 0.8:               # flat dark area = shadow / fresh patch, not a hole
            continue
        if np.mean(pixel_dark[blob > 0]) < min_fill:   # closed-up crack mesh, not a solid hole
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        frac = area / (h * w)
        sev = "high" if frac > 0.02 else "medium" if frac > 0.008 else "low"
        defects.append(Defect("pothole", (x, y, bw, bh), area_px=int(area), severity=sev))
        out_mask |= blob
    return defects, out_mask


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
COLORS = {"pothole": (0, 0, 255), "longitudinal_crack": (0, 200, 255),
          "transverse_crack": (255, 160, 0), "alligator_crack": (255, 0, 255),
          "speed_bump": (60, 170, 40)}


LABELS = {"pothole": "POTHOLE", "longitudinal_crack": "CRACK (longitudinal)",
          "transverse_crack": "CRACK (transverse)", "alligator_crack": "CRACK (alligator)",
          "speed_bump": "SPEED BUMP"}
SEV_RANK = {"info": -1, "low": 0, "medium": 1, "high": 2}


def group_boxes(defects: list[Defect], gap: int) -> list[tuple[tuple[int, int, int, int], list[Defect]]]:
    """Merge boxes that overlap or lie within `gap` px of each other (for drawing only)."""
    groups = [((d.bbox[0], d.bbox[1], d.bbox[0] + d.bbox[2], d.bbox[1] + d.bbox[3]), [d]) for d in defects]
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                (a, da), (b, db) = groups[i], groups[j]
                if a[0] - gap <= b[2] and b[0] - gap <= a[2] and a[1] - gap <= b[3] and b[1] - gap <= a[3]:
                    box = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    groups[i] = (box, da + db)
                    del groups[j]
                    merged = True
                    break
            if merged:
                break
    return groups


def draw_label(img: np.ndarray, text: str, x: int, y: int, color, scale: float, thick: int) -> None:
    """Text on a solid colour band, placed above the box (or inside it at the top edge)."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = max(3, int(th * 0.3))
    x = max(0, min(x, img.shape[1] - tw - 2 * pad))
    top = y - th - base - 2 * pad
    if top < 0:
        top = y
    cv2.rectangle(img, (x, top), (x + tw + 2 * pad, top + th + base + 2 * pad), color, -1)
    lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    cv2.putText(img, text, (x + pad, top + pad + th), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0) if lum > 140 else (255, 255, 255), thick, cv2.LINE_AA)


def annotate(bgr: np.ndarray, defects: list[Defect], crack_mask: np.ndarray | None = None,
             y0: int = 0) -> np.ndarray:
    """Draw a labelled box per pothole and per group of nearby crack pieces."""
    vis = bgr.copy()
    h, w = vis.shape[:2]
    if crack_mask is not None:
        roi = vis[y0:]
        roi[crack_mask > 0] = (0, 255, 255)
    unit = max(h, w) / 1000
    thick = max(2, round(3 * unit))
    scale = max(0.5, 0.8 * unit)
    potholes = [d for d in defects if d.kind in ("pothole", "speed_bump")]   # one box each
    cracks = [d for d in defects if d.kind.endswith("_crack")]
    items = [((d.bbox[0], d.bbox[1], d.bbox[0] + d.bbox[2], d.bbox[1] + d.bbox[3]), [d]) for d in potholes]
    items += group_boxes(cracks, gap=int(0.03 * max(h, w)))
    for (x1, y1, x2, y2), members in items:
        kinds = {m.kind for m in members}
        if kinds in ({"pothole"}, {"speed_bump"}):
            kind = next(iter(kinds))
            conf = members[0].confidence
            text = LABELS[kind] if conf is None else f"{LABELS[kind]} {conf:.0%}"
            color = COLORS[kind]
            if kind == "speed_bump":     # a road feature, not damage: no severity
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
                draw_label(vis, text, x1, y1, color, scale, max(1, thick - 1))
                continue
        elif len(kinds) == 1:
            kind = next(iter(kinds))
            text = LABELS[kind] if len(members) == 1 else f"CRACKS ({kind.split('_')[0]})"
            color = COLORS[kind]
        else:   # a group of mixed crack types; alligator dominates if present
            text = "CRACKS (alligator)" if "alligator_crack" in kinds else "CRACKS"
            color = COLORS["alligator_crack" if "alligator_crack" in kinds else "transverse_crack"]
        sev = max((m.severity for m in members), key=SEV_RANK.get)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
        draw_label(vis, f"{text} - {sev}", x1, y1, color, scale, max(1, thick - 1))
    if y0:
        cv2.line(vis, (0, y0), (w, y0), (128, 128, 128), 1)
    return vis


def road_mask(bgr: np.ndarray, min_frac: float = 0.15, max_hole_frac: float = 0.06,
              min_row_cover: float = 0.3, top_trim: float = 0.04) -> np.ndarray:
    """Estimate the road surface so trees, grass, buildings, cars and sky are ignored.

    The road colour is sampled just in front of the camera (bottom-centre of the frame).
    The road is the connected area, reaching the bottom edge, whose colour and brightness
    stay close to that sample. Small enclosed holes (potholes, puddles, patches) are filled
    back in so the defects themselves are not masked out. Returns all-255 when no clear
    road is found (e.g. a close-up that is all defect), so the detector still runs."""
    h, w = bgr.shape[:2]
    s = 320 / max(h, w)
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (7, 7), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_patch = lab[int(sh * 0.65):int(sh * 0.9), int(sw * 0.25):int(sw * 0.75)].reshape(-1, 3)
    ref = np.median(ref_patch, axis=0)
    # brightness counts half: shadows and wet patches change lightness more than colour
    diff = lab - ref
    dist = np.sqrt(0.25 * diff[..., 0] ** 2 + diff[..., 1] ** 2 + diff[..., 2] ** 2)
    ref_dist = np.sqrt(0.25 * (ref_patch[:, 0] - ref[0]) ** 2 + (ref_patch[:, 1] - ref[1]) ** 2
                       + (ref_patch[:, 2] - ref[2]) ** 2)
    # tolerance from the sample's own spread, capped so a dark patch or puddle in the
    # sample area cannot make every colour in the photo count as road
    thr = float(np.clip(2.0 * np.percentile(ref_dist, 75), 10.0, 22.0))
    similar = (dist < thr).astype(np.uint8) * 255
    similar = cv2.morphologyEx(similar, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # bridge thin dark lines (the cracks themselves) so they do not cut the road into pieces
    similar = cv2.morphologyEx(similar, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    # keep only regions that overlap the sampled patch (the road just in front of the camera;
    # works even when the photo has a black frame along the bottom edge)
    n, labels = cv2.connectedComponents(similar, connectivity=4)
    bottom_ids = set(np.unique(labels[int(sh * 0.65):int(sh * 0.9), int(sw * 0.25):int(sw * 0.75)])) - {0}
    road = np.isin(labels, list(bottom_ids)).astype(np.uint8) * 255
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    if road.mean() / 255 < min_frac:
        return np.full((h, w), 255, np.uint8)
    # Horizon cut for street-level photos: scanning up from the bottom, the road fills less
    # and less of each row. Above the first row where it drops under min_row_cover, the
    # "road-coloured" pixels are walls, cars and sky, so drop them.
    cover = np.convolve(road.mean(axis=1) / 255, np.ones(5) / 5, mode="same")[::-1]   # bottom-up
    start = np.nonzero(cover >= min_row_cover)[0]
    if start.size:
        below = np.nonzero(cover[start[0]:] < min_row_cover)[0]
        if below.size:
            road[:sh - (start[0] + below[0])] = 0
    # fill small enclosed holes (potholes, puddles); big ones (cars, islands) stay out
    cnts, hier = cv2.findContours(road, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is not None:
        for c, hh in zip(cnts, hier[0]):
            if hh[3] != -1 and cv2.contourArea(c) < max_hole_frac * sh * sw:
                cv2.drawContours(road, [c], -1, 255, -1)
    # potholes cut into the road edge: close with a large kernel, but only up to the road's top
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    # trim a band along the road's upper edges, where it meets tyres, kerbs and walls
    k = max(2, int(sh * top_trim))
    above = np.zeros_like(road)
    above[k:] = road[:-k]
    road &= above
    return cv2.resize(road, (w, h), interpolation=cv2.INTER_NEAREST)


_MODEL_CACHE: dict[str, object] = {}


# cut-offs of the road damage model, chosen on validation photos
ROAD_CONF = {"pothole": 0.25, "crack": 0.15, "speed_bump": 0.15}


def road_model():
    """Our trained model for all types (models/road_damage_5class_v8n.onnx), or None if it is not there."""
    if "road" not in _MODEL_CACHE:
        try:
            import pothole_model as pm
            path = pm.ROOT / "models" / "road_damage_5class_v8n.onnx"
            _MODEL_CACHE["road"] = pm.YoloOnnx(path) if path.exists() else None
        except Exception as exc:
            print(f"road damage model not loaded ({exc}); using the older models", file=sys.stderr)
            _MODEL_CACHE["road"] = None
    return _MODEL_CACHE["road"]


def detect_road_damage(bgr: np.ndarray, model, detect: tuple[str, ...], pothole_conf: float | None) -> list[Defect]:
    cut = dict(ROAD_CONF) if pothole_conf is None else {**ROAD_CONF, "pothole": pothole_conf}
    h, w = bgr.shape[:2]
    defects = []
    for d in model(bgr, conf=min(cut.values()), classes=None):
        group = "crack" if d.kind.endswith("_crack") else d.kind
        if group not in detect or d.confidence < cut[group]:
            continue
        area = d.bbox[2] * d.bbox[3]
        frac = area / (h * w)
        sev = "info" if group == "speed_bump" else "high" if frac > 0.04 else "medium" if frac > 0.01 else "low"
        defects.append(Defect(d.kind, d.bbox, area_px=area, severity=sev, confidence=round(d.confidence, 3)))
    return defects


def pothole_model():
    """The trained YOLOv8 pothole model (models/pothole_1class.onnx), or None if it is not there."""
    if "model" not in _MODEL_CACHE:
        try:
            here = str(__import__("pathlib").Path(__file__).resolve().parent)
            if here not in sys.path:
                sys.path.insert(0, here)
            import pothole_model as pm
            _MODEL_CACHE["model"] = pm.YoloOnnx(pm.DEFAULT_MODEL) if pm.DEFAULT_MODEL.exists() else None
        except Exception as exc:  # broken file or old OpenCV: fall back to the rule-based detector
            print(f"pothole model not loaded ({exc}); using rule-based pothole detection", file=sys.stderr)
            _MODEL_CACHE["model"] = None
    return _MODEL_CACHE["model"]


def speed_bump_model():
    """Trained 7-class road-anomaly model (models/road_anomaly_7class.onnx), used for its
    Speed-Bump class only. None if the file is missing."""
    if "bump" not in _MODEL_CACHE:
        try:
            pothole_model()   # makes sure pothole_model is importable
            import pothole_model as pm
            path = pm.ROOT / "models" / "road_anomaly_7class.onnx"
            _MODEL_CACHE["bump"] = pm.YoloOnnx(path) if path.exists() else None
        except Exception as exc:
            print(f"speed bump model not loaded ({exc})", file=sys.stderr)
            _MODEL_CACHE["bump"] = None
    return _MODEL_CACHE["bump"]


def detect_speed_bumps(bgr: np.ndarray, model, conf: float = 0.5) -> list[Defect]:
    """Speed bumps are road features, not damage: severity is 'info'."""
    return [Defect("speed_bump", d.bbox, area_px=d.bbox[2] * d.bbox[3], severity="info",
                   confidence=round(d.confidence, 3))
            for d in model(bgr, conf=conf, classes=("speed_bump",))]


def detect_bump_markings(bgr: np.ndarray, road: np.ndarray | None = None, min_blocks: int = 3,
                         debug: dict | None = None) -> list[Defect]:
    """Painted speed humps: a row of >= 3 similar white blocks / triangles, evenly spaced,
    running ACROSS the road.

    Rejected on purpose:
      - lane dashes: they line up along the road (row runs towards the horizon) and
        shrink with distance
      - zebra crossings: long stripes (high aspect ratio) rather than short blocks
    """
    h, w = bgr.shape[:2]
    s = 640 / max(h, w)
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    sh, sw = small.shape[:2]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sat, val = hsv[..., 1].astype(np.int32), hsv[..., 2].astype(np.int32)
    if road is not None and road.mean() < 250:
        road_s = cv2.resize(road, (sw, sh), interpolation=cv2.INTER_NEAREST)
        # Paint (and a rubber bump) differs in colour from asphalt, so it is NOT inside the road
        # mask. Accept it anywhere below the top of the road; each blob must also have road
        # somewhere around it (checked per blob below).
        rows_with_road = np.nonzero((road_s > 0).mean(axis=1) > 0.05)[0]
        top = int(rows_with_road[0]) if rows_with_road.size else 0
        road_area = np.zeros((sh, sw), bool)
        # A bump across the road cuts the road mask in two (its black rubber is not road
        # coloured), so the mask stops AT the bump: allow paint a bit above the road's top.
        road_area[max(0, top - sh // 4):] = True
        road_val = float(np.median(val[road_s > 0])) if (road_s > 0).any() else float(np.median(val))
    else:
        road_s = None
        road_area = np.ones((sh, sw), bool)
        road_val = float(np.median(val))
    hue = hsv[..., 0].astype(np.int32)
    white = (sat < 60) & (val > max(150.0, road_val + 45)) & road_area
    # yellow: painted humps and the yellow/black stripes of rubber speed bumps
    yellow = (hue >= 12) & (hue <= 38) & (sat > 90) & (val > 110) & road_area
    # asphalt-like surface (grey or dark, not colourful): fresh hump asphalt and black rubber count
    asphalt = ((sat < 70) & (val > 15) & (val < road_val + 40)).astype(np.uint8)
    found: list[Defect] = []
    paints = []
    for colour in (white, yellow):    # separately, so a white line never pairs with a yellow block
        paint = cv2.morphologyEx(colour.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        paints.append(paint)
        found += _bump_rows(paint, s, sh, sw, min_blocks, debug, asphalt, val, road_val)
    if debug is not None:
        debug["paint"] = cv2.bitwise_or(*paints)
    return _merge_boxes(found)


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Share of box a that lies inside box b."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / max(1, a[2] * a[3])


def _merge_boxes(defects: list[Defect]) -> list[Defect]:
    """Merge overlapping speed bump boxes (e.g. white and yellow rows of the same hump)."""
    out: list[Defect] = []
    for d in sorted(defects, key=lambda d: -(d.confidence or 0)):
        x, y, w, h = d.bbox
        for o in out:
            ox, oy, ow, oh = o.bbox
            if x < ox + ow and ox < x + w and y < oy + oh and oy < y + h:
                nx, ny = min(x, ox), min(y, oy)
                o.bbox = (nx, ny, max(x + w, ox + ow) - nx, max(y + h, oy + oh) - ny)
                break
        else:
            out.append(d)
    return out


def _bump_rows(paint: np.ndarray, s: float, sh: int, sw: int, min_blocks: int,
               debug: dict | None, asphalt: np.ndarray, val: np.ndarray, road_val: float) -> list[Defect]:
    n, lab, stats, cents = cv2.connectedComponentsWithStats(paint, connectivity=8)
    blobs = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 0.0004 * sh * sw or area > 0.06 * sh * sw:
            continue
        piece = (lab[y:y + bh, x:x + bw] == i).astype(np.uint8)
        cnt, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnt, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = area / max(hull_area, 1)
        (_, _), (rw, rh), rect_angle = cv2.minAreaRect(c)
        aspect = max(rw, rh) / max(1.0, min(rw, rh))
        if solidity < 0.75 or aspect > 4.5:          # blocks and triangles, not long lines
            continue
        # the paint must sit on an asphalt-like surface (not grass, a wall or a car body)
        r = max(4, int(0.5 * max(bw, bh)))
        y0r, y1r, x0r, x1r = max(0, y - r), min(sh, y + bh + r), max(0, x - r), min(sw, x + bw + r)
        ring = asphalt[y0r:y1r, x0r:x1r].astype(bool) & (lab[y0r:y1r, x0r:x1r] != i)
        if ring.sum() < 0.4 * ((y1r - y0r) * (x1r - x0r) - area):
            continue
        fill = area / max(1.0, rw * rh)              # ~0.5 for a triangle, ~1 for a block
        corners = len(cv2.approxPolyDP(c, 0.06 * cv2.arcLength(c, True), True))
        long_axis = rect_angle if rw >= rh else rect_angle + 90   # degrees, image coordinates
        blobs.append(dict(i=i, x=x, y=y, w=bw, h=bh, area=area, cx=cents[i][0], cy=cents[i][1], aspect=aspect,
                          triangle=corners == 3 and 0.35 < fill < 0.68 and solidity > 0.9,
                          axis=long_axis % 180))
    if debug is not None:
        debug.setdefault("blobs", []).extend(blobs)
    if len(blobs) < 2:
        return []

    # Grow rows: chain blobs left to right whose size is similar and whose next neighbour
    # sits about one block-width to the side and at a similar height.
    blobs.sort(key=lambda b: b["cx"])
    found: list[Defect] = []
    used, rows = set(), []
    for a in range(len(blobs)):
        if a in used:
            continue
        row = [a]
        while True:
            last = blobs[row[-1]]
            best, best_d = None, None
            for b in range(len(blobs)):
                if b in used or b in row or blobs[b]["cx"] <= last["cx"]:
                    continue
                o = blobs[b]
                if not (0.4 < o["area"] / last["area"] < 2.5):
                    continue
                dx, dy = o["cx"] - last["cx"], o["cy"] - last["cy"]
                size = max(last["w"], last["h"], o["w"], o["h"])
                if dx > 4 * size or abs(dy) > max(0.8 * dx, 0.6 * size):   # next to it, across the road
                    continue
                d = dx * dx + dy * dy
                if best is None or d < best_d:
                    best, best_d = b, d
            if best is None:
                break
            row.append(best)
        # two matching triangles side by side are the classic hump marking; otherwise need min_blocks
        pair_of_triangles = len(row) == 2 and all(blobs[k]["triangle"] for k in row)
        if len(row) >= min_blocks or pair_of_triangles:
            rows.append(row)
            used.update(row)

    for row in rows:
        bs = [blobs[k] for k in row]
        xs = np.array([b["cx"] for b in bs]); ys = np.array([b["cy"] for b in bs])
        angle = abs(np.degrees(np.arctan2(ys[-1] - ys[0], xs[-1] - xs[0])))
        if angle > 40:                                    # runs along the road: lane dashes
            continue
        areas = np.array([b["area"] for b in bs])
        steady = np.all(np.diff(areas) > 0) or np.all(np.diff(areas) < 0)   # perspective: shrinks with distance
        if areas.max() / areas.min() > (5.0 if steady else 3.5):           # sizes too uneven
            continue
        gaps = np.diff(xs)
        if len(gaps) >= 2 and gaps.std() / gaps.mean() > 0.6:   # irregular spacing
            continue
        if np.median([b["aspect"] for b in bs]) > 3.5:    # long stripes: zebra crossing
            continue
        # pieces of one broken line point ALONG the row; hump blocks point across it
        row_dir = np.degrees(np.arctan2(ys[-1] - ys[0], xs[-1] - xs[0])) % 180
        elongated = [b for b in bs if b["aspect"] > 1.8 and not b["triangle"]]
        if elongated:
            off = [min(abs(b["axis"] - row_dir), 180 - abs(b["axis"] - row_dir)) for b in elongated]
            # ...unless the gaps between the blocks are much darker than the paint: that is the
            # paint / black rubber striping of a rubber speed bump seen from the side
            if np.median(off) < 30 and not _dark_gaps(bs, lab, val, road_val, debug):
                continue
        x0 = min(b["x"] for b in bs); y0 = min(b["y"] for b in bs)
        x1 = max(b["x"] + b["w"] for b in bs); y1 = max(b["y"] + b["h"] for b in bs)
        pad = int(0.25 * (y1 - y0)) + 2
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(sw, x1 + pad), min(sh, y1 + pad)
        box = (int(x0 / s), int(y0 / s), int((x1 - x0) / s), int((y1 - y0) / s))
        # a simple 0-1 score: more blocks and more regular spacing -> higher
        regular = 1.0 - min(1.0, gaps.std() / gaps.mean()) if len(gaps) >= 2 else 0.5
        score = round(min(0.95, 0.45 + 0.08 * len(bs) + 0.25 * regular), 3)
        found.append(Defect("speed_bump", box, area_px=box[2] * box[3], severity="info", confidence=score))
    return found


def _dark_gaps(bs: list[dict], lab: np.ndarray, val: np.ndarray, road_val: float, debug: dict | None) -> bool:
    """True when the stretch between each pair of neighbouring blocks is much darker than both
    the blocks and the asphalt (yellow/black rubber bump), not plain asphalt (a broken line)."""
    paint_v = np.median([np.median(val[lab == b["i"]]) for b in bs])
    gap_vs = []
    for a, b in zip(bs, bs[1:]):
        n = max(2, int(np.hypot(b["cx"] - a["cx"], b["cy"] - a["cy"])))
        xs = np.linspace(a["cx"], b["cx"], n).astype(int)
        ys = np.linspace(a["cy"], b["cy"], n).astype(int)
        between = lab[ys, xs] == 0                    # off both blocks
        if between.sum() < 3:
            return False
        gap_vs.append(float(np.median(val[ys, xs][between])))
    if debug is not None:
        debug.setdefault("gaps", []).append((float(paint_v), gap_vs))
    return max(gap_vs) < min(0.6 * paint_v, road_val - 30)


def detect_potholes_model(bgr: np.ndarray, model, conf: float = 0.5) -> tuple[list[Defect], np.ndarray]:
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    defects = []
    for d in model(bgr, conf=conf):
        x, y, bw, bh = d.bbox
        frac = bw * bh / (h * w)
        sev = "high" if frac > 0.04 else "medium" if frac > 0.01 else "low"
        defects.append(Defect("pothole", d.bbox, area_px=bw * bh, severity=sev, confidence=round(d.confidence, 3)))
        mask[y:y + bh, x:x + bw] = 255
    return defects, mask


def analyze(bgr: np.ndarray, roi_top: float = 0.0, road_only: bool = True, use_model: bool = True,
            pothole_conf: float | None = None, detect: tuple[str, ...] = ("pothole", "crack", "speed_bump"),
            bump_conf: float = 0.5, **kw) -> tuple[list[Defect], np.ndarray]:
    """Run the detectors. roi_top: ignore the top fraction of the frame (sky, cars).
    road_only: only report cracks on the estimated road surface (not trees, buildings, cars).
    use_model: use the trained models. models/road_damage_5class_v8n.onnx finds all types in one pass;
    without it, the older pothole and speed bump models and the image rules below are used.
    pothole_conf: pothole cut-off (default 0.25 for the road damage model, 0.5 for the older one).
    detect: which types to look for ("pothole", "crack", "speed_bump"). Speed bumps need
    models/road_anomaly_7class.onnx and are skipped without it."""
    y0 = int(roi_top * bgr.shape[0])
    if use_model and road_model() is not None:
        defects = detect_road_damage(bgr[y0:], road_model(), detect, pothole_conf)
        for d in defects:
            x, y, w, h = d.bbox
            d.bbox = (x, y + y0, w, h)
        return defects, annotate(bgr, defects, None, y0)
    pothole_conf = 0.5 if pothole_conf is None else pothole_conf
    gray = preprocess(bgr[y0:])
    road = road_mask(bgr[y0:]) if road_only else np.full(gray.shape, 255, np.uint8)
    off_road = cv2.bitwise_not(road)
    model = pothole_model() if use_model else None
    if model is not None:
        potholes, pmask = detect_potholes_model(bgr[y0:], model, pothole_conf)
    else:
        potholes, pmask = detect_potholes(gray)
        keep = [i for i, d in enumerate(potholes)
                if road[d.bbox[1]:d.bbox[1] + d.bbox[3], d.bbox[0]:d.bbox[0] + d.bbox[2]].mean() > 0.7 * 255]
        if len(keep) != len(potholes):
            potholes = [potholes[i] for i in keep]
            kept = np.zeros_like(pmask)
            for d in potholes:
                x, y, w, h = d.bbox
                kept[y:y + h, x:x + w] = pmask[y:y + h, x:x + w]
            pmask = kept
    # pothole rims look like thick dark lines -> don't report them as cracks too
    if "crack" in detect:
        excl = cv2.bitwise_or(cv2.dilate(pmask, np.ones((15, 15), np.uint8)), off_road)
        cracks, cmask = detect_cracks(gray, exclude=excl, **kw)
    else:
        cracks, cmask = [], None
    if "pothole" not in detect:
        potholes = []
    bumps = []
    if "speed_bump" in detect:
        # painted / striped speed bumps (white or yellow blocks in a row across the road)
        bumps = detect_bump_markings(bgr[y0:], road)
        if use_model and speed_bump_model() is not None:
            bumps = _merge_boxes(bumps + detect_speed_bumps(bgr[y0:], speed_bump_model(), bump_conf))
    if bumps:
        # a rubber or painted speed bump can look like a pothole to the pothole model, and its
        # painted edges like cracks: drop those
        potholes = [p for p in potholes if not any(_overlap(p.bbox, b.bbox) > 0.3 for b in bumps)]
        cracks = [c for c in cracks if not any(_overlap(c.bbox, b.bbox) > 0.5 for b in bumps)]

    defects = potholes + bumps + cracks
    for d in defects:
        x, y, w, h = d.bbox
        d.bbox = (x, y + y0, w, h)
    return defects, annotate(bgr, defects, cmask, y0)


def synthetic_road(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = 600, 800
    img = rng.normal(125, 12, (h, w)).astype(np.float32)
    img = cv2.GaussianBlur(img, (3, 3), 0) + np.linspace(-20, 20, w)[None, :]   # lighting gradient

    def walk(x, y, dx, dy, steps, thick):
        pts = []
        for _ in range(steps):
            pts.append((int(x), int(y)))
            x += dx + rng.normal(0, 1.5)
            y += dy + rng.normal(0, 1.5)
        cv2.polylines(img, [np.array(pts)], False, 60, thick)

    walk(120, 40, 0.3, 3, 120, 2)      # longitudinal
    walk(300, 120, 4, 0.2, 90, 3)      # transverse
    for i in range(6):                 # alligator mesh
        walk(470 + rng.integers(0, 40), 380 + i * 25, 3, 0.3, 50, 2)
        walk(480 + i * 28, 370, 0.3, 3, 50, 2)
    # pothole: dark, rough, irregular ellipse
    hole = np.zeros((h, w), np.uint8)
    cv2.ellipse(hole, (220, 450), (80, 45), 10, 0, 360, 255, -1)
    img[hole > 0] = rng.normal(70, 25, int(np.sum(hole > 0)))
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="image or video file")
    ap.add_argument("-o", "--output", help="annotated image / video output")
    ap.add_argument("--json", help="write detections to JSON")
    ap.add_argument("--roi-top", type=float, default=0.0, help="ignore top fraction of frame (dashcam: ~0.5)")
    ap.add_argument("--every", type=int, default=5, help="video: analyse every N-th frame")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--only", help="comma-separated types to look for: pothole, crack, speed_bump "
                                   "(default: all three)")
    a = ap.parse_args()
    only = tuple(k.strip() for k in a.only.split(",")) if a.only else ("pothole", "crack", "speed_bump")
    bad = [k for k in only if k not in ("pothole", "crack", "speed_bump")]
    if bad:
        ap.error(f"--only: unknown type(s) {bad}; use pothole, crack, speed_bump")

    if a.demo:
        img = synthetic_road()
        cv2.imwrite("demo_input.jpg", img)
        defects, vis = analyze(img, detect=only)
        cv2.imwrite(a.output or "demo_result.jpg", vis)
        results = [asdict(d) for d in defects]
    elif a.input and a.input.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        cap = cv2.VideoCapture(a.input)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer, results, i, vis = None, [], 0, None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % a.every == 0:
                defects, vis = analyze(frame, a.roi_top, detect=only)
                results += [{"frame": i, "time_s": round(i / fps, 2), **asdict(d)} for d in defects]
            if a.output:
                if writer is None:
                    writer = cv2.VideoWriter(a.output, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                             (frame.shape[1], frame.shape[0]))
                writer.write(vis if vis is not None else frame)
            i += 1
        if writer:
            writer.release()
    elif a.input:
        img = cv2.imread(a.input)
        if img is None:
            sys.exit(f"cannot read {a.input}")
        defects, vis = analyze(img, a.roi_top, detect=only)
        if a.output:
            cv2.imwrite(a.output, vis)
        results = [asdict(d) for d in defects]
    else:
        ap.error("give an image/video or --demo")

    for r in results:
        print(r)
    print(f"{len(results)} defects")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
