"""In-process pub/sub used to push live updates to browsers over Server-Sent Events."""
from __future__ import annotations

import asyncio
import json
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[str]] = set()

    def subscribe(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=256)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        self._subscribers.discard(q)

    def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        message = json.dumps({"type": event_type, **payload})
        for q in list(self._subscribers):
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:  # slow client: drop the event, it will resync on next one
                pass
