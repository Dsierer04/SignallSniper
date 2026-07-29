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
from .execution.broker import Broker, Fill, OrderStatus
from .execution.session import current_session
from .feeds.base import TokenBucket, build_client
from .feeds.edgar import EdgarCurrentFeed, EdgarEnricher, TickerResolver
from .feeds.newswire import PUBLIC_WIRES, RssFeed
from .market.linkage import LinkageGraph
from .market.quotes import QuoteSource
from .market.tape import MarketState
from .models import Event, Quote, RawDoc, Signal, mono_ns
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
        broker: Broker | None = None,
    ) -> None:
        self.cfg = cfg
        self.broker = broker
        self.bus = EventBus()
        self.market = market or MarketState()
        self.resolver = resolver or TickerResolver()
        self.graph = LinkageGraph()
        self.engine = SignalEngine(
            self.market,
            self.graph,
            EngineConfig(move_scale=dict(DEFAULT_MOVE_SCALE),
                         blackout_tickers=frozenset(cfg.blackout)),
        )
        # for_equity keeps the proportions sane on a small account instead of
        # leaving a config that silently rejects every order. It does NOT widen
        # risk_per_trade or the daily loss cap -- those stay as configured.
        self.risk = RiskManager(RiskConfig.for_equity(
            cfg.equity,
            risk_per_trade=cfg.risk_per_trade,
            max_daily_loss_frac=cfg.max_daily_loss,
            max_concurrent=cfg.max_concurrent,
        ))
        for problem in self.risk.cfg.viability():
            log.warning("account: %s", problem)
        self.enricher: EdgarEnricher | None = None
        #: Set when a live quote source supports mid-stream subscription. Lets
        #: the broad EDGAR path signal on names outside the initial watchlist.
        self.quote_source = None
        self.on_signal = on_signal or self._default_signal_sink
        self.on_order = on_order or self._default_order_sink
        self.stop = asyncio.Event()
        self.feeds: list = []
        self._watch = {t.upper() for t in cfg.watchlist}
        self.counts = {"docs": 0, "events": 0, "signals": 0, "orders": 0,
                       "accepted": 0, "rejected": 0, "unfilled": 0,
                       "subscribed": 0, "adopted": 0}

        # Subscribe eagerly at construction, not inside the loop coroutines. A
        # task does not run until the scheduler gets to it, so subscribing there
        # opens a window where a doc published by an already-running feed lands
        # on a topic with no subscribers and is silently dropped. On a normal day
        # that window is microseconds; on the one day a filing lands during
        # startup it is the whole trade.
        self._raw_sub = self.bus.subscribe(TOPIC_RAW, maxsize=4096, name="classifier")
        self._event_sub = self.bus.subscribe(TOPIC_EVENT, maxsize=2048, name="engine")

        #: Tickers whose stop is enforced by this process rather than the broker.
        #: Extended-hours orders cannot carry a bracket leg, so if we go down
        #: with one of these open the position is unprotected. Tracked so
        #: shutdown can flatten them.
        self.client_side_stops: set[str] = set()

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

            # Item codes are not in the current-filings feed -- they live on the
            # per-filing index page. Fetch them only once the doc has resolved to
            # a ticker we care about, so the 16:05 flood of unrelated filings
            # does not spend the SEC rate budget.
            if (self.enricher is not None and doc.source == "edgar"
                    and tickers and not doc.meta.get("items")
                    and str(doc.meta.get("form", "")).startswith("8-K")):
                doc = await self.enricher.enrich(doc)

            event = build_event(doc, tickers)
            if event.materiality <= 0.0 and not tickers:
                continue

            # The EDGAR path watches every US filer, but the engine needs a tape
            # on a name to size or even gate a trade. Subscribe on demand so a
            # material filing on an unwatched small cap is actionable rather
            # than discarded as "illiquid" purely for want of data.
            if (self.quote_source is not None and tickers
                    and event.materiality >= self.engine.cfg.min_materiality):
                missing = [t for t in tickers if t not in self.market.tapes]
                if missing:
                    try:
                        added = await self.quote_source.add_symbols(missing)
                        if added:
                            self.counts["subscribed"] += len(added)
                            # The tape needs ticks before the engine can act, so
                            # this filing may not fire -- the next event on the
                            # name will. That is the cost of not knowing the
                            # universe in advance.
                            log.info("subscribed on demand: %s", ",".join(added))
                    except Exception as exc:
                        log.warning("on-demand subscribe failed: %r", exc)
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
                if order is None:
                    continue
                self.counts["orders"] += 1
                self.on_order(order)

                if not (self.cfg.live and self.broker is not None):
                    # Alert-only: record the intent so the day's log is complete,
                    # but nothing is at the broker.
                    self.risk.open(order)
                    continue

                fill = await self.broker.submit(order)
                if not fill.accepted:
                    self.counts["rejected"] += 1
                    log.error("broker rejected %s", fill.describe())
                    continue

                self.counts["accepted"] += 1
                log.warning("broker %s", fill.describe())
                self.risk.open(order)
                if fill.stop_is_client_side:
                    self.client_side_stops.add(order.ticker)
                # Do not block the signal loop waiting on a fill -- the next
                # event may already be in the queue.
                asyncio.create_task(self._reconcile(order, fill.broker_order_id))

    async def _quote_loop(self, source: QuoteSource) -> None:
        async for q in source.stream():
            if self.stop.is_set():
                return
            self.market.on_quote(q)
            self.bus.publish(TOPIC_QUOTE, q)
            # Exit checks ride the quote path so a stop is evaluated the instant
            # the price that would trigger it arrives, not on the next timer tick.
            pos = self.risk.positions.get(q.ticker)
            if pos is None:
                continue

            # Evaluate exits against the TAPE, not the raw quote. The tape holds
            # a two-tick outlier confirmation gate; reading q.mid directly walks
            # straight past it, so one bad print fires a real stop and sends a
            # real order. The gate is worthless if the exit path does not use it.
            tape = self.market.tapes.get(q.ticker)
            px = tape.last if tape is not None and tape.last > 0 else 0.0
            if px <= 0:
                continue

            # Unrealized P&L must reach the kill switch: several positions opened
            # off one event are one bet, and while they are all open the realized
            # figure is still zero.
            marks = {t: (self.market.tapes[t].last
                         if t in self.market.tapes and self.market.tapes[t].last > 0
                         else p.entry)
                     for t, p in self.risk.positions.items()}
            if self.risk.check_kill_switch(marks):
                await self._panic_flatten("kill switch tripped on total P&L")
                continue

            triggered = self.risk.pending_exits({q.ticker: px})
            for ticker, trigger_px, why in triggered:
                if ticker not in self.client_side_stops:
                    # A broker-side bracket already closed this; just book it.
                    self.risk.close(ticker, trigger_px, why)
                    continue

                # Our stop, our responsibility: send the order FIRST and only
                # book the exit once the broker accepts it. Booking first would
                # delete a still-live position from our memory.
                p2 = self.risk.positions.get(ticker)
                if p2 is None:
                    continue
                sent = True
                if self.cfg.live and self.broker is not None:
                    sent = await self._close_at_broker(
                        ticker, p2.direction, p2.shares, trigger_px, why, p2.signal)
                if sent:
                    self.client_side_stops.discard(ticker)
                    self.risk.close(ticker, trigger_px, why)
                else:
                    log.critical(
                        "%s hit its %s at %.2f but the closing order was REJECTED "
                        "-- position is STILL OPEN and unprotected. Close it "
                        "manually now.", ticker, why, trigger_px)

    async def adopt_broker_positions(self) -> int:
        """Pull existing broker positions into the risk manager on startup.

        Adopted positions get NO stop and NO target -- we do not know what thesis
        opened them or where its stop was, and inventing one would be worse than
        admitting we cannot manage them. They are logged loudly and counted
        against max_concurrent so the system does not pile new risk on top of
        risk it cannot see. Close them manually or with `flatten`.
        """
        if self.broker is None or not hasattr(self.broker, "positions"):
            return 0
        try:
            open_positions = await self.broker.positions()
        except Exception as exc:
            log.error("could not read existing broker positions (%r) -- if you "
                      "restarted with positions open, they are UNMANAGED", exc)
            return 0
        if not open_positions:
            return 0

        log.critical(
            "STARTUP: %d position(s) already open at the broker. These were not "
            "opened by this process, so their stops are unknown and this system "
            "CANNOT manage them: %s",
            len(open_positions),
            ", ".join(str(p.get("symbol")) for p in open_positions),
        )
        log.critical("Close them manually, or run: python3 -m signalsniper flatten")
        self.counts["adopted"] = len(open_positions)
        return len(open_positions)

    async def _panic_flatten(self, reason: str) -> None:
        """Flatten everything at the broker. Used when the kill switch trips."""
        log.critical("PANIC FLATTEN: %s", reason)
        if not (self.cfg.live and self.broker is not None):
            return
        try:
            n = await self.broker.flatten_all()
            log.critical("flattened %d position(s)", n)
            self.client_side_stops.clear()
        except Exception as exc:
            log.critical("FLATTEN FAILED (%r) -- close positions manually NOW", exc)

    async def _reconcile(self, order: Order, broker_order_id: str,
                         timeout_s: float = 45.0, poll_s: float = 1.5) -> None:
        """Make the risk manager's position match what the broker actually did.

        Without this the system trades on a fiction. A limit that never fills
        leaves a phantom position occupying a `max_concurrent` slot, and when
        the price crosses its stop the exit logic sends a closing order for
        shares that were never bought -- which does not flatten anything, it
        opens a real position in the opposite direction.
        """
        if not broker_order_id or self.broker is None:
            return

        deadline = mono_ns() + int(timeout_s * 1e9)
        last: Fill | None = None
        while mono_ns() < deadline:
            await asyncio.sleep(poll_s)
            status = await self.broker.order_status(broker_order_id)

            if status.status == "unknown":
                continue

            if status.filled_qty > 0 and status.filled_avg_price > 0:
                pos = self.risk.positions.get(order.ticker)
                if pos is not None and (
                    pos.shares != status.filled_qty
                    or abs(pos.entry - status.filled_avg_price) > 0.005
                ):
                    log.warning(
                        "reconcile %s: assumed %d @ %.2f, actual %d @ %.2f",
                        order.ticker, pos.shares, pos.entry,
                        status.filled_qty, status.filled_avg_price,
                    )
                    self.risk.amend_fill(order.ticker, status.filled_qty,
                                         status.filled_avg_price)

            if status.is_terminal:
                if status.got_nothing:
                    log.warning("reconcile %s: %s with no fill -- dropping phantom "
                                "position", order.ticker, status.status)
                    self.risk.drop_unfilled(order.ticker)
                    self.client_side_stops.discard(order.ticker)
                    self.counts["unfilled"] += 1
                return

        # Timed out still working. An entry that has not filled in 45s is stale:
        # the move it was chasing has either happened or not, and a late fill is
        # a position taken on a thesis that has already expired.
        log.warning("reconcile %s: no terminal state in %.0fs -- cancelling",
                    order.ticker, timeout_s)
        await self.broker.cancel_order(broker_order_id)
        final = await self.broker.order_status(broker_order_id)
        if final.status == "unknown":
            # A transient API error is not evidence of no fill. Dropping the
            # position here would leave a real one untracked and unstopped;
            # keeping a phantom one merely blocks a slot. Keep it and shout.
            log.critical(
                "%s: cannot determine fill status -- keeping the position and "
                "assuming it is REAL. Verify in the broker UI.", order.ticker)
            return
        if final.got_nothing:
            self.risk.drop_unfilled(order.ticker)
            self.client_side_stops.discard(order.ticker)
            self.counts["unfilled"] += 1
        elif final.filled_qty > 0:
            self.risk.amend_fill(order.ticker, final.filled_qty,
                                 final.filled_avg_price or order.limit)

    async def _close_at_broker(self, ticker: str, direction, shares: int,
                               px: float, why: str, sig: Signal) -> bool:
        """Send the closing leg for a position whose stop we were holding.

        Priced marketably (through the touch) rather than at the trigger price:
        a resting limit at the stop level in a fast tape does not get filled, and
        an unfilled stop is not a stop.
        """
        from .models import Direction as _D
        from .signal.risk import Order as _Order

        # Cross by the wider of 40bps and a full spread. A fixed 40bps only
        # reaches the far side while the quoted spread is under 80bps; on a
        # 300bps after-hours book the "protective" limit rests inside the spread
        # and never fills, which is not a stop at all.
        tape = self.market.tapes.get(ticker)
        spread_bps = tape.spread_bps() if tape is not None else float("inf")
        if not (spread_bps < 10_000.0):
            spread_bps = 100.0
        slip = max(0.004, spread_bps / 10_000.0)
        exit_dir = _D.SHORT if direction is _D.LONG else _D.LONG
        limit = px * (1 - slip) if exit_dir is _D.SHORT else px * (1 + slip)

        closing = _Order(
            ticker=ticker, direction=exit_dir, shares=shares,
            limit=round(limit, 2), stop=0.0, target=0.0,
            signal=sig, reason=f"close:{why}",
        )
        fill = await self.broker.submit(closing)
        if fill.accepted:
            log.warning("CLOSE sent %s (%s) %s", ticker, why, fill.describe())
            return True
        log.error("CLOSE FAILED %s (%s): %s", ticker, why, fill.error)
        return False

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

        self.enricher = EdgarEnricher(client, sec_bucket)
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

        # Adopt anything already open at the broker BEFORE starting the feeds.
        # There is no state persistence, so a restart otherwise leaves live
        # positions completely unmanaged: RiskManager believes it is flat, no
        # stop is ever evaluated for them, and protect_on_shutdown will not
        # flatten them because client_side_stops is empty. A restart at 16:04 on
        # an earnings day would silently orphan every position you hold.
        await self.adopt_broker_positions()

        self.build_feeds(client, ir_feeds)

        tasks = [asyncio.create_task(self._classify_loop(), name="classify"),
                 asyncio.create_task(self._signal_loop(), name="signal")]
        for feed in self.feeds:
            tasks.append(asyncio.create_task(feed.run(self.stop), name=feed.name))
        if quote_source is not None:
            self.quote_source = quote_source if hasattr(quote_source, "add_symbols") else None
            tasks.append(asyncio.create_task(self._quote_loop(quote_source), name="quotes"))
        tasks.append(asyncio.create_task(self._heartbeat(), name="heartbeat"))

        try:
            await self.stop.wait()
        finally:
            # Order matters: flatten BEFORE tearing down the loop. A position
            # whose stop lives in this process becomes unprotected the moment
            # this process exits, so leaving one open on shutdown is strictly
            # worse than taking the exit at whatever the book offers.
            await self.protect_on_shutdown()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await client.aclose()

    async def protect_on_shutdown(self) -> None:
        if not (self.cfg.live and self.broker is not None):
            return
        if not self.client_side_stops:
            return
        log.error(
            "shutting down with %d client-side-stop position(s) open: %s -- "
            "flattening, because an unprotected position outlives this process",
            len(self.client_side_stops), ", ".join(sorted(self.client_side_stops)),
        )
        try:
            n = await self.broker.flatten_all()
            log.warning("flattened %d position(s) at market", n)
            self.client_side_stops.clear()
        except Exception as exc:
            log.critical(
                "FLATTEN FAILED (%r) -- YOU HAVE OPEN UNPROTECTED POSITIONS. "
                "Close them manually in the broker UI now.", exc,
            )

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
