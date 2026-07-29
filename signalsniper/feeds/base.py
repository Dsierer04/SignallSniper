"""Polling feed machinery.

The whole game here is bytes-on-the-wire latency. Three things buy that:

1. **Conditional GET.** ETag / If-Modified-Since means an unchanged endpoint
   costs a 304 with no body. That lets us poll fast without moving megabytes.
2. **A persistent connection pool.** TLS handshakes cost 2 RTTs. httpx keeps
   connections warm so a poll is one round trip.
3. **A token bucket, not a sleep.** Publishers rate-limit (SEC is 10 req/s
   across your whole IP). A shared bucket lets us spend that budget where it
   matters instead of hard-coding a conservative interval per feed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import httpx

from ..bus import TOPIC_HEALTH, TOPIC_RAW, EventBus
from ..models import RawDoc, mono_ns

log = logging.getLogger("signalsniper.feeds")


class TokenBucket:
    """Shared rate budget. `take()` waits only as long as strictly necessary."""

    __slots__ = ("rate", "capacity", "_tokens", "_last", "_lock")

    def __init__(self, rate: float, capacity: float | None = None) -> None:
        self.rate = rate
        self.capacity = capacity if capacity is not None else rate
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def take(self, n: float = 1.0) -> float:
        """Consume n tokens, waiting if needed. Returns seconds spent waiting."""
        waited = 0.0
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= n:
                    self._tokens -= n
                    return waited
                deficit = (n - self._tokens) / self.rate
                waited += deficit
                await asyncio.sleep(deficit)


@dataclass
class FeedStats:
    polls: int = 0
    not_modified: int = 0
    changed: int = 0
    errors: int = 0
    docs: int = 0
    last_ok_ns: int = 0
    latency_ms_ewma: float = 0.0

    def record_latency(self, ms: float, alpha: float = 0.2) -> None:
        self.latency_ms_ewma = ms if self.latency_ms_ewma == 0 else (
            alpha * ms + (1 - alpha) * self.latency_ms_ewma
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "polls": self.polls,
            "304s": self.not_modified,
            "changed": self.changed,
            "errors": self.errors,
            "docs": self.docs,
            "latency_ms": round(self.latency_ms_ewma, 1),
        }


class PollingFeed:
    """Base class: subclasses implement `parse()` and get dedupe + pacing free."""

    #: Override in subclasses.
    name: str = "feed"
    #: Tokens charged per poll against the shared bucket.
    cost: float = 1.0

    def __init__(
        self,
        url: str,
        bus: EventBus,
        client: httpx.AsyncClient,
        bucket: TokenBucket,
        interval: float = 1.0,
        seen_max: int = 20_000,
    ) -> None:
        self.url = url
        self.bus = bus
        self.client = client
        self.bucket = bucket
        self.interval = interval
        self.stats = FeedStats()
        self._etag: str | None = None
        self._last_modified: str | None = None
        self._seen: dict[str, None] = {}
        self._seen_max = seen_max
        self._backoff = 0.0
        # First poll primes the dedupe set without emitting: on startup the feed
        # is full of old news and we do not want to trade the backlog.
        self._primed = False

    # ---- subclass hook -------------------------------------------------

    def parse(self, body: bytes, headers: httpx.Headers) -> Iterable[RawDoc]:
        raise NotImplementedError

    # ---- machinery -----------------------------------------------------

    def _remember(self, doc_id: str) -> bool:
        """True if this is the first time we've seen doc_id."""
        if doc_id in self._seen:
            return False
        self._seen[doc_id] = None
        if len(self._seen) > self._seen_max:
            # dicts preserve insertion order -- drop the oldest quarter
            for k in list(self._seen)[: self._seen_max // 4]:
                del self._seen[k]
        return True

    async def poll_once(self) -> list[RawDoc]:
        headers: dict[str, str] = {}
        if self._etag:
            headers["If-None-Match"] = self._etag
        if self._last_modified:
            headers["If-Modified-Since"] = self._last_modified

        await self.bucket.take(self.cost)
        t0 = mono_ns()
        try:
            resp = await self.client.get(self.url, headers=headers)
        except Exception as exc:  # network flap -- back off, never die
            self.stats.errors += 1
            self._backoff = min(30.0, max(0.5, self._backoff * 2 or 0.5))
            self.bus.publish(TOPIC_HEALTH, {"feed": self.name, "error": repr(exc)})
            log.warning("%s poll failed: %r (backoff %.1fs)", self.name, exc, self._backoff)
            return []

        self.stats.polls += 1
        self.stats.record_latency((mono_ns() - t0) / 1e6)
        self._backoff = 0.0

        if resp.status_code == 304:
            self.stats.not_modified += 1
            return []
        if resp.status_code == 429:
            self.stats.errors += 1
            self._backoff = min(60.0, max(2.0, self._backoff * 2 or 2.0))
            log.warning("%s rate limited; backing off %.1fs", self.name, self._backoff)
            return []
        if resp.status_code >= 400:
            self.stats.errors += 1
            log.warning("%s HTTP %d", self.name, resp.status_code)
            return []

        self._etag = resp.headers.get("etag") or self._etag
        self._last_modified = resp.headers.get("last-modified") or self._last_modified
        self.stats.changed += 1
        self.stats.last_ok_ns = mono_ns()

        fresh: list[RawDoc] = []
        try:
            for doc in self.parse(resp.content, resp.headers):
                if self._remember(doc.doc_id):
                    fresh.append(doc)
        except Exception as exc:
            self.stats.errors += 1
            log.exception("%s parse failed: %r", self.name, exc)
            return []

        if not self._primed:
            self._primed = True
            log.info("%s primed with %d existing docs (not emitted)", self.name, len(fresh))
            return []

        self.stats.docs += len(fresh)
        return fresh

    async def run(self, stop: asyncio.Event | None = None) -> None:
        log.info("%s starting: %s (every %.2fs)", self.name, self.url, self.interval)
        while stop is None or not stop.is_set():
            docs = await self.poll_once()
            for doc in docs:
                self.bus.publish(TOPIC_RAW, doc)
            delay = self._backoff or self.interval
            try:
                if stop is not None:
                    await asyncio.wait_for(stop.wait(), timeout=delay)
                    return
                await asyncio.sleep(delay)
            except asyncio.TimeoutError:
                continue


def build_client(user_agent: str, timeout: float = 4.0) -> httpx.AsyncClient:
    """One warm, HTTP/2-capable pool for every feed.

    `timeout` is deliberately short: a poll that takes 4s is already useless,
    and failing fast lets the next poll try a healthy path.
    """
    return httpx.AsyncClient(
        headers={
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        },
        timeout=httpx.Timeout(timeout, connect=2.0),
        limits=httpx.Limits(max_keepalive_connections=32, max_connections=64,
                            keepalive_expiry=120.0),
        follow_redirects=True,
        http2=False,
    )
