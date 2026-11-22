"""Pothole detection with a trained YOLOv8 model (ONNX), run through OpenCV's DNN module.

No PyTorch needed: only opencv-python. Models live in models/ (see models/README.md).

    python pothole_model.py road.jpg -o out.jpg
    python pothole_model.py road.jpg --model models/road_damage_5class.onnx --conf 0.3
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = {
    # file name -> class names, in the order the model was trained with
    "pothole_1class.onnx": ["pothole"],
    "road_damage_5class.onnx": ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole", "other"],
    "road_anomaly_7class.onnx": ["heavy_vehicle", "light_vehicle", "pedestrian", "crack", "severe_crack",
                                 "pothole", "speed_bump"],
}
DEFAULT_MODEL = ROOT / "models" / "pothole_1class.onnx"


@dataclass
class Detection:
    kind: str
    confidence: float
    bbox: tuple[int, int, int, int]   # x, y, w, h in original image pixels


class YoloOnnx:
    def __init__(self, path: Path | str = DEFAULT_MODEL, size: int = 640) -> None:
        self.path = Path(path)
        self.names = MODELS.get(self.path.name)
        if self.names is None:
            raise ValueError(f"unknown model {self.path.name}; add its class names to MODELS")
        self.net = cv2.dnn.readNetFromONNX(str(self.path))
        self.size = size

    def __call__(self, bgr: np.ndarray, conf: float = 0.35, iou: float = 0.45,
                 classes: tuple[str, ...] | None = ("pothole",)) -> list[Detection]:
        h, w = bgr.shape[:2]
        # letterbox: keep aspect ratio, pad to a square
        s = self.size / max(h, w)
        nh, nw = round(h * s), round(w * s)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255, (self.size, self.size), swapRB=True)
        self.net.setInput(blob)
        out = self.net.forward()[0].T                     # (8400, 4 + n_classes)
        scores = out[:, 4:]
        cls = scores.argmax(1)
        score = scores[np.arange(len(cls)), cls]
        keep = score >= conf
        if classes is not None:
            wanted = [i for i, n in enumerate(self.names) if n in classes]
            keep &= np.isin(cls, wanted)
        boxes, score, cls = out[keep, :4], score[keep], cls[keep]
        # cx, cy, w, h (letterboxed) -> x, y, w, h (original)
        xywh = np.column_stack([(boxes[:, 0] - boxes[:, 2] / 2 - left) / s,
                                (boxes[:, 1] - boxes[:, 3] / 2 - top) / s,
                                boxes[:, 2] / s, boxes[:, 3] / s])
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), score.tolist(), conf, iou)
        dets = []
        for i in np.array(idx).flatten():
            x, y, bw, bh = xywh[i]
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(w, int(x + bw)), min(h, int(y + bh))
            dets.append(Detection(self.names[cls[i]], float(score[i]), (x0, y0, x1 - x0, y1 - y0)))
        return sorted(dets, key=lambda d: -d.confidence)


def draw(bgr: np.ndarray, dets: list[Detection]) -> np.ndarray:
    from vision_detector import draw_label
    vis = bgr.copy()
    unit = max(vis.shape[:2]) / 1000
    thick, scale = max(2, round(3 * unit)), max(0.5, 0.8 * unit)
    for d in dets:
        x, y, w, h = d.bbox
        color = (0, 0, 255) if d.kind == "pothole" else (255, 160, 0)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thick)
        draw_label(vis, f"{d.kind.upper().replace('_', ' ')} {d.confidence:.0%}", x, y, color, scale, max(1, thick - 1))
    return vis


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("-o", "--output")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--conf", type=float, default=0.35)
    a = ap.parse_args()
    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit(f"cannot read {a.image}")
    dets = YoloOnnx(a.model)(img, conf=a.conf)
    for d in dets:
        print(d)
    print(f"{len(dets)} potholes")
    if a.output:
        cv2.imwrite(a.output, draw(img, dets))


if __name__ == "__main__":
    main()
