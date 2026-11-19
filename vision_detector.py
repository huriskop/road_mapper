"""
Crack and pothole detection in road images / video with classical computer vision
(OpenCV only, no training data needed).

Cracks
------
1. Grayscale -> CLAHE (fixes uneven lighting / shadows).
2. Black-hat morphology (closing - image): keeps thin DARK structures narrower
   than the kernel and removes the smooth asphalt background.
3. Otsu threshold on the black-hat response (+ minimum contrast floor).
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
    kind: str                 # longitudinal_crack | transverse_crack | alligator_crack | pothole
    bbox: tuple[int, int, int, int]   # x, y, w, h
    length_px: float = 0.0
    width_px: float = 0.0
    area_px: int = 0
    severity: str = "low"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Morphological skeleton (works with plain opencv, no contrib)."""
    if hasattr(cv2, "ximgproc"):
        return cv2.ximgproc.thinning(mask)
    skel = np.zeros_like(mask)
    img = mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(img):
        eroded = cv2.erode(img, k)
        skel |= cv2.subtract(img, cv2.dilate(eroded, k))
        img = eroded
    return skel


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
def detect_cracks(gray: np.ndarray, *, kernel: int = 15, min_contrast: int = 18, k_sigma: float = 5.0,
                  min_len: int = 40, max_width: float = 8.0,
                  exclude: np.ndarray | None = None) -> tuple[list[Defect], np.ndarray]:
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
    # noise-adaptive threshold: asphalt grain gives a broad low response, cracks are outliers
    med = float(np.median(bh))
    sigma = 1.4826 * float(np.median(np.abs(bh.astype(np.float32) - med)))
    _, mask = cv2.threshold(bh, max(min_contrast, med + k_sigma * sigma), 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
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
            1 for j, hh in enumerate(hier[0]) if hh[3] != -1 and cv2.contourArea(cnts[j]) > kernel * kernel)
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
          "transverse_crack": (255, 160, 0), "alligator_crack": (255, 0, 255)}


def analyze(bgr: np.ndarray, roi_top: float = 0.0, **kw) -> tuple[list[Defect], np.ndarray]:
    """Run both detectors. roi_top: ignore the top fraction of the frame (sky, cars)."""
    y0 = int(roi_top * bgr.shape[0])
    gray = preprocess(bgr[y0:])
    potholes, pmask = detect_potholes(gray)
    # pothole rims look like thick dark lines -> don't report them as cracks too
    excl = cv2.dilate(pmask, np.ones((15, 15), np.uint8))
    cracks, cmask = detect_cracks(gray, exclude=excl, **kw)

    defects = potholes + cracks
    vis = bgr.copy()
    roi = vis[y0:]
    roi[cmask > 0] = (0, 255, 255)
    for d in defects:
        x, y, w, h = d.bbox
        y += y0
        d.bbox = (x, y, w, h)
        cv2.rectangle(vis, (x, y), (x + w, y + h), COLORS[d.kind], 2)
        cv2.putText(vis, f"{d.kind.replace('_crack', '')} {d.severity}", (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[d.kind], 1, cv2.LINE_AA)
    if y0:
        cv2.line(vis, (0, y0), (vis.shape[1], y0), (128, 128, 128), 1)
    return defects, vis


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
    a = ap.parse_args()

    if a.demo:
        img = synthetic_road()
        cv2.imwrite("demo_input.jpg", img)
        defects, vis = analyze(img)
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
                defects, vis = analyze(frame, a.roi_top)
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
        defects, vis = analyze(img, a.roi_top)
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
