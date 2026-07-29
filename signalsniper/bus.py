"""In-process async pub/sub with backpressure that drops rather than blocks.

A trading pipeline must never let a slow consumer stall the feed reader. Queues
are bounded; on overflow we drop the *oldest* item and count it. A dropped stale
quote is free; a stalled EDGAR poller costs the whole edge.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, AsyncIterator, Callable, Awaitable

log = logging.getLogger("signalsniper.bus")

Handler = Callable[[Any], Awaitable[None]]


class Subscription:
    __slots__ = ("topic", "queue", "dropped", "_bus", "_name")

    def __init__(self, bus: "EventBus", topic: str, maxsize: int, name: str) -> None:
        self._bus = bus
        self._name = name
        self.topic = topic
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    def offer(self, item: Any) -> None:
        """Non-blocking put. Evicts oldest on overflow."""
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except asyncio.QueueEmpty:  # pragma: no cover - race, harmless
                pass
            self.dropped += 1
            if self.dropped % 100 == 1:
                log.warning("subscriber %s dropped %d msgs on %s", self._name, self.dropped, self.topic)
            try:
                self.queue.put_nowait(item)
            except asyncio.QueueFull:  # pragma: no cover
                pass

    async def get(self) -> Any:
        return await self.queue.get()

    async def stream(self) -> AsyncIterator[Any]:
        while True:
            yield await self.queue.get()

    def close(self) -> None:
        self._bus.unsubscribe(self)


class EventBus:
    """Topic-routed fanout. Publish is synchronous and non-blocking by design."""

    def __init__(self) -> None:
        self._subs: dict[str, list[Subscription]] = defaultdict(list)
        self.published: dict[str, int] = defaultdict(int)

    def subscribe(self, topic: str, maxsize: int = 1024, name: str = "anon") -> Subscription:
        sub = Subscription(self, topic, maxsize, name)
        self._subs[topic].append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        subs = self._subs.get(sub.topic)
        if subs and sub in subs:
            subs.remove(sub)

    def publish(self, topic: str, item: Any) -> int:
        """Returns how many subscribers received it. Never awaits, never raises."""
        self.published[topic] += 1
        subs = self._subs.get(topic)
        if not subs:
            return 0
        for sub in subs:
            sub.offer(item)
        return len(subs)

    def stats(self) -> dict[str, Any]:
        return {
            "published": dict(self.published),
            "subscribers": {t: len(s) for t, s in self._subs.items() if s},
            "dropped": {
                t: sum(x.dropped for x in s) for t, s in self._subs.items() if any(x.dropped for x in s)
            },
        }


# Canonical topic names -- keep them here so typos fail loudly at import.
TOPIC_RAW = "raw"          # RawDoc off a feed
TOPIC_EVENT = "event"      # classified Event
TOPIC_SIGNAL = "signal"    # actionable Signal
TOPIC_QUOTE = "quote"      # market data tick
TOPIC_HEALTH = "health"    # feed heartbeats / errors
