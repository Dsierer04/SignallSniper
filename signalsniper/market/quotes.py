"""Market data sources.

Two implementations behind one interface:

* `AlpacaQuoteStream` -- live websocket. Note the feed tiers: `iex` is free but
  covers only IEX's ~2% of consolidated volume, which makes its prints thin and
  its spreads unrepresentative after hours. `sip` is the full consolidated tape
  and is what you actually need for the 16:05 window. Running the second-order
  logic on IEX data during a post-close earnings move will produce garbage
  reference prices; budget for the SIP subscription or use Polygon.

* `ReplaySource` -- deterministic playback for tests and dry runs, so the whole
  pipeline can be exercised without a market open or a network.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import AsyncIterator, Iterable, Protocol

from ..models import Quote, now_ns

log = logging.getLogger("signalsniper.quotes")


class QuoteSource(Protocol):
    def stream(self) -> AsyncIterator[Quote]: ...


def _parse_rfc3339_ns(raw: str) -> int:
    """Alpaca stamps RFC3339 with nanoseconds. We only need it for ordering."""
    try:
        return int(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp() * 1e9)
    except (ValueError, AttributeError):
        return now_ns()


class AlpacaQuoteStream:
    """Websocket quote/trade stream.

    Reconnects with capped exponential backoff. A dropped socket during the event
    window is the worst possible failure, so reconnection is aggressive at first
    (0.5s) and only backs off if the venue is genuinely down.
    """

    def __init__(
        self,
        key: str,
        secret: str,
        symbols: Iterable[str],
        feed: str = "iex",
        url: str | None = None,
    ) -> None:
        self.key = key
        self.secret = secret
        self.symbols = [s.upper() for s in symbols]
        self.feed = feed
        self.url = url or f"wss://stream.data.alpaca.markets/v2/{feed}"
        self.connected = False
        self.messages = 0

    async def stream(self) -> AsyncIterator[Quote]:
        try:
            import websockets
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("pip install websockets to use AlpacaQuoteStream") from exc

        backoff = 0.5
        while True:
            try:
                async with websockets.connect(self.url, ping_interval=10, max_queue=4096) as ws:
                    await ws.send(json.dumps(
                        {"action": "auth", "key": self.key, "secret": self.secret}
                    ))
                    await ws.recv()  # auth ack
                    await ws.send(json.dumps({
                        "action": "subscribe",
                        "quotes": self.symbols,
                        "trades": self.symbols,
                    }))
                    self.connected = True
                    backoff = 0.5
                    log.info("alpaca stream up: %d symbols on %s", len(self.symbols), self.feed)

                    async for raw in ws:
                        for q in self._decode(raw):
                            yield q
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.connected = False
                log.warning("alpaca stream dropped: %r; reconnecting in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(15.0, backoff * 2)

    def _decode(self, raw: str | bytes) -> list[Quote]:
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            return []
        if isinstance(payload, dict):
            payload = [payload]

        out: list[Quote] = []
        for msg in payload:
            if not isinstance(msg, dict):
                continue
            kind = msg.get("T")
            sym = msg.get("S")
            if not sym:
                continue
            self.messages += 1
            if kind == "q":
                bid = float(msg.get("bp") or 0.0)
                ask = float(msg.get("ap") or 0.0)
                if bid <= 0 and ask <= 0:
                    continue
                out.append(Quote(
                    ticker=sym,
                    bid=bid,
                    ask=ask,
                    last=(bid + ask) / 2 if bid > 0 and ask > 0 else (bid or ask),
                    volume=0,
                    t_source=_parse_rfc3339_ns(msg.get("t", "")),
                ))
            elif kind == "t":
                px = float(msg.get("p") or 0.0)
                if px <= 0:
                    continue
                out.append(Quote(
                    ticker=sym,
                    bid=0.0,
                    ask=0.0,
                    last=px,
                    volume=int(msg.get("s") or 0),
                    t_source=_parse_rfc3339_ns(msg.get("t", "")),
                ))
        return out


class ReplaySource:
    """Deterministic playback. `speed=0` replays instantly for tests."""

    def __init__(self, quotes: list[Quote], speed: float = 0.0) -> None:
        self.quotes = sorted(quotes, key=lambda q: q.t_source)
        self.speed = speed

    async def stream(self) -> AsyncIterator[Quote]:
        prev: int | None = None
        for q in self.quotes:
            if self.speed > 0 and prev is not None:
                gap = (q.t_source - prev) / 1e9 / self.speed
                if gap > 0:
                    await asyncio.sleep(min(gap, 5.0))
            prev = q.t_source
            yield q


def synth_walk(
    ticker: str,
    start_price: float,
    n: int,
    t0_ns: int,
    step_ns: int = 100_000_000,
    drift_bps_total: float = 0.0,
    spread_bps: float = 8.0,
) -> list[Quote]:
    """Build a deterministic price path. Used by tests and the dry-run demo.

    Deliberately not random: a reproducible path means a failing assertion is a
    real regression rather than an unlucky seed.
    """
    out: list[Quote] = []
    for i in range(n):
        frac = (i / max(1, n - 1))
        px = start_price * (1.0 + (drift_bps_total / 10_000.0) * frac)
        half = px * spread_bps / 2 / 10_000.0
        out.append(Quote(
            ticker=ticker.upper(),
            bid=round(px - half, 4),
            ask=round(px + half, 4),
            last=round(px, 4),
            volume=100,
            t_source=t0_ns + i * step_ns,
        ))
    return out
