"""E8 - deliver every observation exactly once.

Claim one job with a lease, POST it with Idempotency-Key = observation_id, check the
receipt names the same ID. The server enforces uniqueness, so a lost response just means
the retry gets 200 + the same receipt. Timeouts / 5xx / no network back off (5 s doubling to
10 min, with jitter); 400/401/409/413/422 mean retrying cannot help, so the job is BLOCKED.
"""
from __future__ import annotations

import json
import logging
import threading

from ..store.observations import ObservationStore

log = logging.getLogger(__name__)

PERMANENT = {400, 401, 403, 409, 413, 422}


class Uploader:
    def __init__(self, store: ObservationStore, server_url: str, token: str, timeout_s: float = 20.0,
                 lease_s: float = 60.0, idle_s: float = 1.0, backoff_base_s: float = 5.0,
                 backoff_cap_s: float = 600.0) -> None:
        import requests

        self.store = store
        self.url = server_url.rstrip("/") + "/api/v1/observations"
        self.token = token
        self.timeout = timeout_s
        self.lease_s = lease_s
        self.idle_s = idle_s
        self.backoff = (backoff_base_s, backoff_cap_s)
        self.session = requests.Session()
        self.wake = threading.Event()
        self._stop = threading.Event()
        self.sent = 0
        self.failed = 0
        self.auth_alarm = False
        self.last_error: str | None = None

    def stop(self) -> None:
        self._stop.set()
        self.wake.set()

    def run(self) -> None:
        while not self._stop.is_set():
            if not self.step():
                self.wake.wait(self.idle_s)
                self.wake.clear()

    def step(self) -> bool:
        """Try one job. Returns False when there was nothing to do."""
        import requests

        row = self.store.claim(self.lease_s)
        if row is None:
            return False
        obs_id = row["id"]
        files = {"metadata": (None, row["metadata"], "application/json")}
        image = self.store.read_image(row) if row["image_path"] and not row["files_pruned"] else None
        if image is not None:
            files["image"] = (f"{obs_id}.jpg", image, "image/jpeg")
        headers = {"Authorization": f"Bearer {self.token}", "Idempotency-Key": obs_id}
        try:
            resp = self.session.post(self.url, files=files, headers=headers, timeout=self.timeout)
        except requests.RequestException as exc:
            self._retry(obs_id, f"network: {type(exc).__name__}")
            return True
        if resp.status_code in (200, 201):
            try:
                receipt = resp.json()
            except ValueError:
                self._retry(obs_id, "receipt not JSON")
                return True
            if receipt.get("observation_id") != obs_id:
                self._retry(obs_id, f"receipt for wrong id {receipt.get('observation_id')!r}")
                return True
            self.store.accept(obs_id, json.dumps(receipt))
            self.sent += 1
            self.last_error = None
            return True
        if resp.status_code in PERMANENT:
            detail = resp.text[:300]
            self.store.block(obs_id, f"HTTP {resp.status_code}: {detail}")
            self.failed += 1
            self.last_error = f"HTTP {resp.status_code}"
            if resp.status_code in (401, 403):
                self.auth_alarm = True
                log.error("server rejected the device token (HTTP %d)", resp.status_code)
            else:
                log.error("observation %s blocked: HTTP %d %s", obs_id, resp.status_code, detail)
            return True
        self._retry(obs_id, f"HTTP {resp.status_code}")
        return True

    def _retry(self, obs_id: str, why: str) -> None:
        delay = self.store.retry_later(obs_id, why, *self.backoff)
        self.last_error = why
        log.info("upload %s failed (%s); retry in %.0f s", obs_id, why, delay)
