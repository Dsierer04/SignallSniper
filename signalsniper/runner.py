"""The wiring. Feeds -> classify -> engine -> risk -> (alert | order).

Everything is one asyncio loop in one process. That is not a limitation, it is
the design: the moment you put a queue between the classifier and the engine on
another host you have added a network hop to a path measured in microseconds.

Default posture is ALERT ONLY. Orders require an explicit opt-in in config, and
the risk manager gets the final veto even then.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from .bus import TOPIC_EVENT, TOPIC_QUOTE, TOPIC_RAW, TOPIC_SIGNAL, EventBus
from .config import Config
from .feeds.base import TokenBucket, build_client
from .feeds.edgar import EdgarCurrentFeed, TickerResolver
from .feeds.newswire import PUBLIC_WIRES, RssFeed
from .market.linkage import LinkageGraph
from .market.quotes import QuoteSource
from .market.tape import MarketState
from .models import Event, Quote, RawDoc, Signal
from .parse.classify import build_event
from .signal.engine import DEFAULT_MOVE_SCALE, EngineConfig, SignalEngine
from .signal.risk import Order, RiskConfig, RiskManager

log = logging.getLogger("signalsniper.runner")

SignalSink = Callable[[Signal], None]
OrderSink = Callable[[Order], None]


class Runner:
    def __init__(
        self,
        cfg: Config,
        market: MarketState | None = None,
        resolver: TickerResolver | None = None,
        on_signal: SignalSink | None = None,
        on_order: OrderSink | None = None,
    ) -> None:
        self.cfg = cfg
        self.bus = EventBus()
        self.market = market or MarketState()
        self.resolver = resolver or TickerResolver()
        self.graph = LinkageGraph()
        self.engine = SignalEngine(
            self.market,
            self.graph,
            EngineConfig(move_scale=dict(DEFAULT_MOVE_SCALE)),
        )
        self.risk = RiskManager(RiskConfig(
            equity=cfg.equity,
            risk_per_trade=cfg.risk_per_trade,
            max_daily_loss_frac=cfg.max_daily_loss,
            max_concurrent=cfg.max_concurrent,
        ))
        self.on_signal = on_signal or self._default_signal_sink
        self.on_order = on_order or self._default_order_sink
        self.stop = asyncio.Event()
        self.feeds: list = []
        self._watch = {t.upper() for t in cfg.watchlist}
        self.counts = {"docs": 0, "events": 0, "signals": 0, "orders": 0}

        # Subscribe eagerly at construction, not inside the loop coroutines. A
        # task does not run until the scheduler gets to it, so subscribing there
        # opens a window where a doc published by an already-running feed lands
        # on a topic with no subscribers and is silently dropped. On a normal day
        # that window is microseconds; on the one day a filing lands during
        # startup it is the whole trade.
        self._raw_sub = self.bus.subscribe(TOPIC_RAW, maxsize=4096, name="classifier")
        self._event_sub = self.bus.subscribe(TOPIC_EVENT, maxsize=2048, name="engine")

    # -----------------------------------------------------------------
    # sinks
    # -----------------------------------------------------------------

    def _default_signal_sink(self, sig: Signal) -> None:
        log.warning("SIGNAL %s", sig.describe())
        for note in sig.notes:
            if note:
                log.info("       | %s", note)
        log.info("       | %s", sig.event.doc.url or sig.event.doc.title)

    def _default_order_sink(self, order: Order) -> None:
        mode = "LIVE" if self.cfg.live else "DRY"
        log.warning("[%s] ORDER %s  (%s)", mode, order.describe(), order.reason)

    # -----------------------------------------------------------------
    # ticker attribution
    # -----------------------------------------------------------------

    def tickers_for(self, doc: RawDoc) -> tuple[str, ...]:
        # An IR feed pre-attributes its own issuer -- no ambiguity, no lookup.
        pre = doc.meta.get("tickers")
        if pre:
            return tuple(str(t).upper() for t in pre)

        if doc.source == "edgar":
            t = self.resolver.resolve(
                cik=str(doc.meta.get("cik", "")),
                company=str(doc.meta.get("company", "")),
            )
            return (t,) if t else ()

        # Newswire headlines: match only explicit $TICKER or an exchange-prefixed
        # form. Bare substring matching is what makes naive scrapers useless --
        # "AMC theaters", "the CEO said ALL of it", "GAAP" all false-positive.
        return _extract_cashtags(doc.title + " " + doc.body, self._watch)

    # -----------------------------------------------------------------
    # pipeline stages
    # -----------------------------------------------------------------

    async def _classify_loop(self) -> None:
        sub = self._raw_sub
        while not self.stop.is_set():
            doc: RawDoc = await sub.get()
            self.counts["docs"] += 1
            tickers = self.tickers_for(doc)
            event = build_event(doc, tickers)
            if event.materiality <= 0.0 and not tickers:
                continue
            self.counts["events"] += 1
            self.bus.publish(TOPIC_EVENT, event)

    async def _signal_loop(self) -> None:
        sub = self._event_sub
        while not self.stop.is_set():
            event: Event = await sub.get()
            for sig in self.engine.on_event(event):
                self.counts["signals"] += 1
                self.bus.publish(TOPIC_SIGNAL, sig)
                self.on_signal(sig)
                order = self.risk.size_order(sig)
                if order is not None:
                    self.counts["orders"] += 1
                    self.risk.open(order)
                    self.on_order(order)

    async def _quote_loop(self, source: QuoteSource) -> None:
        async for q in source.stream():
            if self.stop.is_set():
                return
            self.market.on_quote(q)
            self.bus.publish(TOPIC_QUOTE, q)
            # Exit checks ride the quote path so a stop is evaluated the instant
            # the price that would trigger it arrives, not on the next timer tick.
            pos = self.risk.positions.get(q.ticker)
            if pos is not None:
                self.risk.check_exits({q.ticker: q.mid or q.last})

    # -----------------------------------------------------------------
    # feed construction
    # -----------------------------------------------------------------

    def build_feeds(self, client, ir_feeds: dict[str, tuple[str, tuple[str, ...]]] | None = None):
        sec_bucket = TokenBucket(self.cfg.sec_rate)
        wire_bucket = TokenBucket(self.cfg.wire_rate)
        feeds: list = []

        for form in self.cfg.edgar_forms:
            feeds.append(EdgarCurrentFeed(
                self.bus, client, sec_bucket,
                interval=self.cfg.edgar_interval,
                form=form,
            ))

        if self.cfg.enable_wires:
            for label, url in PUBLIC_WIRES.items():
                feeds.append(RssFeed(
                    url, self.bus, client, wire_bucket,
                    interval=self.cfg.wire_interval,
                    label=label,
                ))

        # IR feeds are the sharpest wire source: pre-attributed and unaggregated.
        for label, (url, tickers) in (ir_feeds or {}).items():
            feeds.append(RssFeed(
                url, self.bus, client, wire_bucket,
                interval=max(0.5, self.cfg.wire_interval / 2),
                label=label, tickers=tickers,
            ))

        self.feeds = feeds
        return feeds

    # -----------------------------------------------------------------
    # lifecycle
    # -----------------------------------------------------------------

    async def run(self, quote_source: QuoteSource | None = None,
                  ir_feeds: dict[str, tuple[str, tuple[str, ...]]] | None = None) -> None:
        problems = self.cfg.validate()
        for p in problems:
            log.error("config: %s", p)
        if any("SEC_USER_AGENT" in p for p in problems):
            raise SystemExit("refusing to poll SEC without a declaring User-Agent")

        client = build_client(self.cfg.sec_user_agent)
        try:
            n = await self.resolver.load(client)
            log.info("loaded %d CIK->ticker mappings", n)
        except Exception as exc:
            log.error("ticker map load failed (%r) -- EDGAR attribution degraded", exc)

        self.build_feeds(client, ir_feeds)

        tasks = [asyncio.create_task(self._classify_loop(), name="classify"),
                 asyncio.create_task(self._signal_loop(), name="signal")]
        for feed in self.feeds:
            tasks.append(asyncio.create_task(feed.run(self.stop), name=feed.name))
        if quote_source is not None:
            tasks.append(asyncio.create_task(self._quote_loop(quote_source), name="quotes"))
        tasks.append(asyncio.create_task(self._heartbeat(), name="heartbeat"))

        try:
            await self.stop.wait()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await client.aclose()

    async def _heartbeat(self, every: float = 30.0) -> None:
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=every)
                return
            except asyncio.TimeoutError:
                pass
            log.info(
                "heartbeat counts=%s feeds=%s risk=%s",
                self.counts,
                {f.name: f.stats.as_dict() for f in self.feeds},
                self.risk.summary(),
            )
            rej = self.engine.stats()
            if rej:
                log.info("engine rejects: %s", rej)

    def shutdown(self) -> None:
        self.stop.set()


# ---------------------------------------------------------------------------

import re  # noqa: E402

_CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")
_EXCHANGE_RE = re.compile(
    r"\((?:NASDAQ|NYSE|NYSE\s+American|AMEX|OTC(?:QB|QX)?|CBOE)\s*:\s*([A-Z.\-]{1,6})\)",
    re.IGNORECASE,
)


def _extract_cashtags(text: str, universe: set[str]) -> tuple[str, ...]:
    """Explicit tickers only: `$AAPL` or `(NASDAQ: AAPL)`.

    Restricting to the configured universe when one is set kills the remaining
    false positives -- `$IT`, `$A`, `$ALL` are all real tickers and all appear
    constantly as ordinary words in press releases.
    """
    found: list[str] = []
    for m in _EXCHANGE_RE.finditer(text):
        found.append(m.group(1).upper())
    for m in _CASHTAG_RE.finditer(text):
        found.append(m.group(1).upper())
    seen: dict[str, None] = {}
    for t in found:
        if universe and t not in universe:
            continue
        seen.setdefault(t, None)
    return tuple(seen)
