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

from ..models import Quote, epoch_ns

log = logging.getLogger("signalsniper.quotes")


class QuoteSource(Protocol):
    def stream(self) -> AsyncIterator[Quote]: ...


def _parse_rfc3339_ns(raw: str) -> int:
    """Alpaca stamps RFC3339 with nanoseconds. We only need it for ordering."""
    try:
        return int(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp() * 1e9)
    except (ValueError, AttributeError):
        return epoch_ns()


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
        max_symbols: int = 200,
    ) -> None:
        self.key = key
        self.secret = secret
        self.symbols = [s.upper() for s in symbols]
        self.feed = feed
        self.url = url or f"wss://stream.data.alpaca.markets/v2/{feed}"
        self.connected = False
        self.messages = 0
        self.max_symbols = max_symbols
        self._ws = None
        self._pending: list[str] = []

    async def add_symbols(self, symbols: Iterable[str]) -> list[str]:
        """Subscribe to more symbols on a live socket.

        This is what makes the EDGAR long-tail path actually tradeable. That path
        ingests filings from *every* US issuer, but the engine needs a tape on a
        name to check liquidity and get a reference price -- so without dynamic
        subscription it can only ever signal on the pre-configured watchlist,
        which defeats the point of watching the whole market.

        Returns the symbols newly added.
        """
        fresh = [s.upper() for s in symbols
                 if s and s.upper() not in self.symbols]
        if not fresh:
            return []
        room = self.max_symbols - len(self.symbols)
        if room <= 0:
            log.warning("symbol cap %d reached; not subscribing %s",
                        self.max_symbols, ",".join(fresh))
            return []
        fresh = fresh[:room]
        self.symbols.extend(fresh)

        if self._ws is None:
            # Not connected yet -- they are in self.symbols and will go out with
            # the initial subscribe.
            return fresh
        try:
            await self._ws.send(json.dumps({
                "action": "subscribe", "quotes": fresh, "trades": fresh,
            }))
            log.info("subscribed mid-stream to %s", ",".join(fresh))
        except Exception as exc:
            log.warning("mid-stream subscribe failed for %s: %r", fresh, exc)
            self._pending.extend(fresh)
        return fresh

    async def stream(self) -> AsyncIterator[Quote]:
        try:
            import websockets
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("pip install websockets to use AlpacaQuoteStream") from exc

        backoff = 0.5
        while True:
            try:
                async with websockets.connect(self.url, ping_interval=10, max_queue=4096) as ws:
                    self._ws = ws
                    await ws.send(json.dumps(
                        {"action": "auth", "key": self.key, "secret": self.secret}
                    ))
                    # Alpaca sends [{"T":"success","msg":"connected"}] FIRST and
                    # the authenticated ack second. A single recv() consumes the
                    # connect frame and never checks auth at all -- so a wrong
                    # key produces a socket that connects, yields no quotes, and
                    # looks exactly like a quiet market. Read until we see the
                    # authenticated ack or an explicit error.
                    if not await self._await_auth(ws):
                        raise RuntimeError(
                            "alpaca auth failed -- check ALPACA_API_KEY / "
                            "ALPACA_SECRET_KEY and whether they are paper or live keys"
                        )
                    # Re-subscribe the FULL current set on every (re)connect,
                    # including anything added mid-stream. A reconnect that
                    # restored only the original watchlist would silently drop
                    # the dynamically-added names -- and those are exactly the
                    # ones with a live position or a pending signal.
                    await ws.send(json.dumps({
                        "action": "subscribe",
                        "quotes": self.symbols,
                        "trades": self.symbols,
                    }))
                    self._pending.clear()
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
                self._ws = None
                log.warning("alpaca stream dropped: %r; reconnecting in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(15.0, backoff * 2)

    async def _await_auth(self, ws, max_frames: int = 5) -> bool:
        """True once the venue confirms authentication."""
        for _ in range(max_frames):
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            except (asyncio.TimeoutError, Exception):
                return False
            try:
                payload = json.loads(raw)
            except (ValueError, TypeError):
                continue
            if isinstance(payload, dict):
                payload = [payload]
            for msg in payload:
                if not isinstance(msg, dict):
                    continue
                if msg.get("T") == "error":
                    log.error("alpaca auth error %s: %s",
                              msg.get("code"), msg.get("msg"))
                    return False
                if msg.get("T") == "success" and msg.get("msg") == "authenticated":
                    return True
        return False

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
