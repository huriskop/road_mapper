#!/usr/bin/env python3
"""Look-ahead d_l: how far ahead of the front wheel is the road patch at the chosen image row?

1. Park on a flat road. Run:   python tools/calibrate_lookahead.py --row 0.70
   It saves data/lookahead_guide.jpg with a yellow line across the chosen row (70 % down).
2. Put a marker (tape, a coin) on the road so it sits on that line; re-run to check.
3. Tape-measure along the ground from the marker to where the front tyre touches the road.
4. Save it:                    python tools/calibrate_lookahead.py --set 4.25

The row should be the lower-middle of the image, where the classifier looks for defects.
"""
import argparse
import re
import threading
import time

import _common  # noqa: F401

from roadflag_edge.camera.ring import FrameRing, PiCameraSource
from roadflag_edge.config import load


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(_common.DEFAULT_CONFIG))
    p.add_argument("--row", type=float, default=0.70, help="image row as a fraction of the height")
    p.add_argument("--set", type=float, metavar="METRES", help="write evidence.lookahead_m to the config")
    a = p.parse_args()
    if a.set is not None:
        text = open(a.config).read()
        new, n = re.subn(r"(?m)^(\s*lookahead_m:\s*)[0-9.]+", rf"\g<1>{a.set:.2f}", text)
        if n != 1:
            raise SystemExit("could not find 'lookahead_m:' under evidence: in the config")
        open(a.config, "w").write(new)
        print(f"evidence.lookahead_m = {a.set:.2f} m saved to {a.config}")
        return

    import cv2

    cfg = load(a.config)
    c = cfg.camera
    ring = FrameRing(8, c.width, c.height)
    # auto exposure for the still: the guide must be visible, not motion-blur-proof
    src = PiCameraSource(ring, c.width, c.height, c.fps, None)
    threading.Thread(target=src.run, daemon=True).start()
    time.sleep(3)
    src.stop()
    if not len(ring):
        raise SystemExit("no frames from the camera")
    f, *_ = ring.nearest(ring.span()[1])
    bgr = cv2.cvtColor(f.yuv, cv2.COLOR_YUV2BGR_I420)
    h, w = bgr.shape[:2]
    y = int(h * a.row)
    cv2.line(bgr, (0, y), (w - 1, y), (0, 255, 255), 2)
    cv2.line(bgr, (w // 2, y - 15), (w // 2, y + 15), (0, 255, 255), 2)
    cv2.putText(bgr, f"row {a.row:.2f}: put the marker on this line", (10, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    out = cfg.path(cfg.data_dir) / "lookahead_guide.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), bgr)
    print(f"saved {out}\nNow measure marker -> front tyre contact, then: --set <metres>")


if __name__ == "__main__":
    main()
