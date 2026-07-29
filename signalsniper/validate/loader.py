"""Historical bar loading with an on-disk cache.

The cache is not an optimisation, it is a correctness feature: a validation run
must be reproducible, and re-pulling from a provider whose data can be revised
means the same command gives different answers on different days. Pull once,
cache, and every subsequent run measures the same reality.

Alpaca's historical bars endpoint is the default because the credentials are
already required elsewhere. `feed="sip"` is essential here -- validating an
after-hours thesis against IEX bars measures IEX's ~2% of volume, not the market,
and will make almost every linked name look like it "didn't move" when in fact
you simply could not see it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import httpx

from .study import Bar, Series

log = logging.getLogger("signalsniper.validate.loader")

ALPACA_DATA = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"


class BarCache:
    def __init__(self, root: Path | str = ".cache/bars") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, start: datetime, end: datetime, tf: str) -> Path:
        key = f"{symbol.upper()}_{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}_{tf}.json"
        return self.root / key

    def get(self, symbol: str, start: datetime, end: datetime, tf: str) -> Series | None:
        p = self._path(symbol, start, end, tf)
        if not p.exists():
            return None
        try:
            raw = json.loads(p.read_text())
        except (ValueError, OSError):
            return None
        return Series(symbol.upper(), [_bar_from_json(b) for b in raw])

    def put(self, symbol: str, start: datetime, end: datetime, tf: str,
            series: Series) -> None:
        p = self._path(symbol, start, end, tf)
        p.write_text(json.dumps([{
            "t": b.t.isoformat(), "o": b.open, "h": b.high,
            "l": b.low, "c": b.close, "v": b.volume,
        } for b in series.bars]))


def _bar_from_json(d: dict) -> Bar:
    return Bar(
        t=datetime.fromisoformat(d["t"].replace("Z", "+00:00")),
        open=float(d.get("o", 0.0)), high=float(d.get("h", 0.0)),
        low=float(d.get("l", 0.0)), close=float(d.get("c", 0.0)),
        volume=int(d.get("v", 0)),
    )


class AlpacaBars:
    def __init__(self, key: str, secret: str, feed: str = "sip",
                 cache: BarCache | None = None) -> None:
        if not key or not secret:
            raise ValueError("Alpaca credentials required for historical bars")
        self.feed = feed
        self.cache = cache or BarCache()
        self._headers = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
        }
        if feed != "sip":
            log.warning(
                "loading historical bars from '%s' -- after-hours coverage will "
                "be thin and linked names will spuriously look 'unmoved'", feed,
            )

    async def load(self, client: httpx.AsyncClient, symbol: str,
                   start: datetime, end: datetime,
                   timeframe: str = "1Min") -> Series:
        cached = self.cache.get(symbol, start, end, timeframe)
        if cached is not None:
            return cached

        bars: list[Bar] = []
        page_token: str | None = None
        while True:
            params = {
                "start": start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                "end": end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                "timeframe": timeframe,
                "limit": "10000",
                "feed": self.feed,
                "adjustment": "raw",
            }
            if page_token:
                params["page_token"] = page_token

            r = await client.get(ALPACA_DATA.format(symbol=symbol.upper()),
                                 headers=self._headers, params=params)
            if r.status_code == 403:
                raise RuntimeError(
                    f"403 loading {symbol} on feed '{self.feed}' -- your data "
                    "subscription does not cover this feed"
                )
            r.raise_for_status()
            payload = r.json()
            for b in (payload.get("bars") or []):
                bars.append(_bar_from_json(b))
            page_token = payload.get("next_page_token")
            if not page_token:
                break

        series = Series(symbol.upper(), bars)
        self.cache.put(symbol, start, end, timeframe, series)
        return series


async def load_window(
    provider: AlpacaBars,
    client: httpx.AsyncClient,
    symbols: Iterable[str],
    t_event: datetime,
    before_min: float = 60.0,
    after_min: float = 300.0,
) -> dict[str, Series]:
    """Load every symbol around one event. Sequential on purpose.

    Alpaca rate-limits historical requests (200/min on the free tier). A
    validation run over 12 events x 12 symbols is 144 requests; firing those
    concurrently gets you 429s and a half-populated study that silently looks
    like 'no edge'.
    """
    start = t_event - timedelta(minutes=before_min)
    end = t_event + timedelta(minutes=after_min)
    out: dict[str, Series] = {}
    for sym in symbols:
        try:
            out[sym.upper()] = await provider.load(client, sym, start, end)
        except Exception as exc:
            log.error("failed loading %s around %s: %r", sym, t_event, exc)
            out[sym.upper()] = Series(sym.upper(), [])
    return out
