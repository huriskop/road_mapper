#!/usr/bin/env python3
"""M4 - is "frame time = SensorTimestamp + exposure/2" right on this device?

Wire an LED + 330 ohm resistor from GPIO17 (pin 11) to GND (pin 9) and put it in view of the
camera, ideally at the image row the evidence uses (lower-middle). The script blinks it N
times at random phases. For each blink, the first lit frame's time minus the moment the LED
switched on is >= 0 and spread over one frame period, so the *minimum* over many blinks
converges on the camera's timing error. Same for the switch-off edge.

    python tools/latency_led_test.py [--pin 17] [--blinks 40]

Done when |error| < 10 ms. If it is larger, put  camera.frame_time_offset_ms: <-error>  in
config.yaml and run again. (IMU timing is checked by bench_check.py: 200.0 Hz, sample clock.)
"""
import argparse
import random
import threading
import time

import _common  # noqa: F401
import numpy as np

from roadflag_edge.camera.ring import FrameRing, PiCameraSource
from roadflag_edge.config import load


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(_common.DEFAULT_CONFIG))
    p.add_argument("--pin", type=int, default=17)
    p.add_argument("--blinks", type=int, default=40)
    a = p.parse_args()
    from gpiozero import LED

    cfg = load(a.config)
    c = cfg.camera
    led = LED(a.pin)
    ring = FrameRing(int(c.ring_seconds * c.fps), c.width, c.height)
    src = PiCameraSource(ring, c.width, c.height, c.fps, c.exposure_us, c.frame_time_offset_ms)
    threading.Thread(target=src.run, daemon=True).start()
    time.sleep(3)
    h = c.height

    # find the LED: the pixels that change most between off and on
    led.off()
    time.sleep(0.3)
    off = ring.nearest(ring.span()[1])[0].yuv[:h].astype(np.int16)
    led.on()
    time.sleep(0.3)
    on = ring.nearest(ring.span()[1])[0].yuv[:h].astype(np.int16)
    led.off()
    diff = on - off
    y, x = np.unravel_index(np.argmax(diff), diff.shape)
    if diff[y, x] < 40:
        raise SystemExit("cannot see the LED change (max brightness change %d); aim the camera at it" % diff[y, x])
    r = 6
    roi = (slice(max(0, y - r), y + r), slice(max(0, x - r), x + r))
    thr = (off[roi].mean() + on[roi].mean()) / 2
    print(f"LED at x={x} y={y} ({y / h:.2f} of height), off {off[roi].mean():.0f} on {on[roi].mean():.0f}")

    rise, fall = [], []
    for k in range(a.blinks):
        time.sleep(0.3 + random.random() * 0.05)       # random phase vs the frame clock
        t_on = time.monotonic_ns()
        led.on()
        time.sleep(0.25)
        t_off = time.monotonic_ns()
        led.off()
        time.sleep(0.25)
        frames = ring.frames_between(t_on - 150_000_000, t_off + 200_000_000)
        lit = [(f.t_ns, f.yuv[roi].mean() > thr) for f in frames]
        first_on = next((t for t, l in lit if l and t > t_on - 100_000_000), None)
        last_on = max((t for t, l in lit if l), default=None)
        if first_on is None or last_on is None:
            continue
        rise.append((first_on - t_on) / 1e6)
        fall.append((last_on - t_off) / 1e6)
    src.stop()
    led.off()
    if not rise:
        raise SystemExit("no blinks detected")
    period = 1000 / c.fps
    rise_err = min(rise)                        # ~0 if timing is right (first lit frame can be early by exp/2)
    fall_err = max(fall)                        # ~0 if timing is right
    err = (rise_err + fall_err) / 2
    print(f"blinks seen {len(rise)}/{a.blinks}")
    print(f"rise: first lit frame - LED on   min {rise_err:+.1f} ms (spread {min(rise):.0f}..{max(rise):.0f}, "
          f"expect 0..{period:.0f})")
    print(f"fall: last lit frame - LED off   max {fall_err:+.1f} ms (expect -{period:.0f}..0)")
    print(f"camera timing error ~ {err:+.1f} ms -> {'PASS' if abs(err) < 10 else 'set camera.frame_time_offset_ms: %.1f' % (-err)}")


if __name__ == "__main__":
    main()
