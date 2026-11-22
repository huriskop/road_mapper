"""Run vision_detector on every real photo and show what was found per photo folder.

    python test_images/evaluate.py                          # all types; images go to test_images/results/
    python test_images/evaluate.py --only pothole           # one type
    python test_images/evaluate.py --only pothole,speed_bump

Photos live in test_images/real/<folder>/ where folder is pothole, crack or bump. The table
counts, for each folder, how many photos got at least one box of each type: the diagonal
is what should be found, everything else is a false alarm. This is an image-level check,
not a box-accurate benchmark (look at the saved images for that).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import vision_detector as vd  # noqa: E402

FOLDERS = {"pothole": "pothole", "crack": "crack", "bump": "speed_bump"}   # folder -> expected type
TYPES = ("pothole", "crack", "speed_bump")


def group(kind: str) -> str:
    return "crack" if kind.endswith("_crack") else kind


def main() -> int:
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    detect = tuple(only.split(",")) if only else TYPES
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    table: dict[str, Counter] = {}
    totals = Counter()
    for folder, expected in FOLDERS.items():
        table[folder] = Counter()
        for path in sorted((ROOT / "real" / folder).glob("*.jpg")):
            img = cv2.imread(str(path))
            defects, vis = vd.analyze(img, detect=detect)
            kinds = Counter(group(d.kind) for d in defects)
            for t in kinds:
                table[folder][t] += 1
            totals[folder] += 1
            cv2.imwrite(str(out_dir / f"{folder}__{path.name}"), vis)
            hit = "HIT " if kinds[expected] else ("MISS" if expected in detect else "    ")
            found = ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())) or "nothing"
            print(f"{hit} {folder:7} {path.name[:48]:48} -> {found}")

    cols = [t for t in TYPES if t in detect]
    print("\nphotos with at least one box of each type (rows: what the photos show)")
    print(f"{'':12}" + "".join(f"{c:>14}" for c in cols))
    for folder, expected in FOLDERS.items():
        cells = []
        for c in cols:
            cell = f"{table[folder][c]}/{totals[folder]}"
            cells.append(f"{cell + (' *' if c == expected else '  '):>14}")
        print(f"{folder:12}" + "".join(cells))
    print("* = should be found; other cells are false alarms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
