"""E1 - one time base for everything.

Every sample and frame is stamped with time.monotonic_ns() at the moment it was sensed.
UTC is attached later, only for display, from a running offset UTC - monotonic learned from
GPS fixes (the Pi has no RTC, and NTP may jump the wall clock after boot).
"""
from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from datetime import datetime, timezone

now_ns = time.monotonic_ns


class Clock:
    def __init__(self, window: int = 20) -> None:
        self._offsets: deque[int] = deque(maxlen=window)
        self._lock = threading.Lock()

    def add_gps_fix(self, epoch_mono_ns: int, fix_utc: datetime) -> None:
        """epoch_mono_ns: monotonic time of the fix epoch (sentence arrival minus the calibrated
        NMEA latency, see gps.neo6m.Fix.t_ns); fix_utc: the UTC time inside the sentence."""
        utc_ns = int(fix_utc.timestamp() * 1e9)
        with self._lock:
            self._offsets.append(utc_ns - epoch_mono_ns)

    @property
    def synced(self) -> bool:
        return len(self._offsets) >= 3

    def offset_ns(self) -> int | None:
        with self._lock:
            if not self._offsets:
                return None
            return int(statistics.median(self._offsets))

    def spread_ms(self) -> float | None:
        """Max - min of the offsets in the window: a stability indicator for the health line."""
        with self._lock:
            if len(self._offsets) < 2:
                return None
            return (max(self._offsets) - min(self._offsets)) / 1e6

    def mono_to_utc(self, t_ns: int | None) -> datetime | None:
        off = self.offset_ns()
        if off is None or t_ns is None or not self.synced:
            return None
        return datetime.fromtimestamp((t_ns + off) / 1e9, tz=timezone.utc)


def iso_utc(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
