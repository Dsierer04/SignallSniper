"""Command line entry points.

    python -m signalsniper doctor    # verify credentials, connectivity, rate limits
    python -m signalsniper demo      # offline end-to-end proof with synthetic tape
    python -m signalsniper watch     # live: feeds + quotes, alert only
    python -m signalsniper links AAPL
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .config import load
from .market.linkage import LinkageGraph, detect_channels


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(name)-24s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------


async def cmd_doctor(args) -> int:
    import httpx

    from .feeds.base import build_client
    from .feeds.edgar import EdgarCurrentFeed, TickerResolver
    from .feeds.base import TokenBucket
    from .bus import EventBus

    cfg = load()
    print("== config ==")
    problems = cfg.validate()
    if problems:
        for p in problems:
            print(f"  [!] {p}")
    else:
        print("  ok")
    print(f"  sec_user_agent : {cfg.sec_user_agent or '(unset)'}")
    print(f"  sec_rate       : {cfg.sec_rate}/s")
    print(f"  live trading   : {cfg.live}")
    print(f"  equity         : ${cfg.equity:,.0f}")
    print(f"  watchlist      : {len(cfg.watchlist)} symbols")

    if not cfg.sec_user_agent:
        print("\nSet SEC_USER_AGENT='Your Name you@example.com' and re-run.")
        return 1

    print("\n== connectivity ==")
    client = build_client(cfg.sec_user_agent)
    bus = EventBus()
    rc = 0
    try:
        resolver = TickerResolver()
        try:
            n = await resolver.load(client)
            print(f"  sec ticker map : ok ({n} issuers)")
        except Exception as exc:
            print(f"  sec ticker map : FAIL {exc!r}")
            rc = 1

        feed = EdgarCurrentFeed(bus, client, TokenBucket(cfg.sec_rate), form="8-K")
        docs = await feed.poll_once()  # primes
        docs = await feed.poll_once()
        s = feed.stats
        print(f"  edgar 8-K      : {s.polls} polls, {s.errors} errors, "
              f"{s.latency_ms_ewma:.0f}ms rtt")
        if s.errors:
            rc = 1
    finally:
        await client.aclose()
    return rc


# ---------------------------------------------------------------------------


async def cmd_demo(args) -> int:
    """Offline end-to-end proof: synthetic AAPL drop -> CRUS/SWKS dislocation."""
    from .market.quotes import synth_walk
    from .market.tape import MarketState
    from .models import RawDoc, now_ns
    from .parse.classify import build_event
    from .signal.engine import DEFAULT_MOVE_SCALE, EngineConfig, SignalEngine
    from .signal.risk import RiskConfig, RiskManager

    market = MarketState()
    t0 = now_ns()

    # Pre-event baseline so the tapes have history to reference.
    for tick, px in [("AAPL", 232.0), ("CRUS", 104.0), ("SWKS", 78.0), ("QRVO", 92.0)]:
        for q in synth_walk(tick, px, 40, t0 - 40 * 100_000_000, drift_bps_total=0.0):
            market.on_quote(q)

    event_ns = now_ns()
    doc = RawDoc(
        source="edgar",
        doc_id="0000320193-26-000077",
        title="8-K - Apple Inc. (0000320193) (Filer)",
        url="https://www.sec.gov/Archives/edgar/data/320193/000032019326000077-index.htm",
        published=None,
        body="Item 2.02 Results of Operations and Financial Condition. "
             "The Company lowered its guidance for iPhone hardware revenue and "
             "now expects results below consensus estimates.",
        meta={"form": "8-K", "company": "APPLE INC", "cik": "320193",
              "items": ("2.02", "7.01")},
    )
    doc.t_ingest = event_ns

    # AAPL reprices hard and instantly. The suppliers barely move -- that is the
    # dislocation the engine is built to find.
    for q in synth_walk("AAPL", 232.0, 30, event_ns, drift_bps_total=-420.0):
        market.on_quote(q)
    for q in synth_walk("CRUS", 104.0, 30, event_ns, drift_bps_total=-60.0):
        market.on_quote(q)
    for q in synth_walk("SWKS", 78.0, 30, event_ns, drift_bps_total=-40.0):
        market.on_quote(q)
    for q in synth_walk("QRVO", 92.0, 30, event_ns, drift_bps_total=-330.0):
        market.on_quote(q)  # already repriced -- should be rejected

    event = build_event(doc, ("AAPL",))
    print("== classification ==")
    print(f"  kind        : {event.kind.value}")
    print(f"  materiality : {event.materiality}")
    print(f"  prior       : {event.prior.name}")
    print(f"  confidence  : {event.confidence}")
    print(f"  reasons     : {', '.join(event.reasons)}")
    print(f"  channels    : {sorted(detect_channels(doc.title + ' ' + doc.body, 'AAPL'))}")
    print(f"  classify latency: {event.ingest_latency_us:.0f}us")

    engine = SignalEngine(market, LinkageGraph(),
                          EngineConfig(move_scale=dict(DEFAULT_MOVE_SCALE)))
    risk = RiskManager(RiskConfig(equity=25_000.0))

    signals = engine.on_event(event)
    print(f"\n== signals ({len(signals)}) ==")
    for s in signals:
        print(f"  {s.describe()}")
        for n in s.notes:
            if n:
                print(f"      | {n}")
        order = risk.size_order(s)
        if order:
            print(f"      -> {order.describe()}  [{order.reason}]")
        else:
            print("      -> no order (risk gate)")

    print(f"\n== engine rejects ==\n  {engine.stats()}")
    print(f"\n== risk ==\n  {risk.summary()}")
    return 0


# ---------------------------------------------------------------------------


async def cmd_watch(args) -> int:
    from .market.quotes import AlpacaQuoteStream
    from .runner import Runner

    cfg = load()
    if args.equity:
        cfg.equity = args.equity
    cfg.live = bool(args.live)

    runner = Runner(cfg)
    source = None
    if cfg.alpaca_key and cfg.alpaca_secret:
        source = AlpacaQuoteStream(
            cfg.alpaca_key, cfg.alpaca_secret, cfg.watchlist, feed=cfg.alpaca_feed
        )
    else:
        logging.getLogger("signalsniper").error(
            "no market data credentials -- second-order signals are DISABLED "
            "(they need a tape on the linked names)"
        )

    loop = asyncio.get_running_loop()
    for sig in ("SIGINT", "SIGTERM"):
        try:
            import signal as _sig
            loop.add_signal_handler(getattr(_sig, sig), runner.shutdown)
        except (NotImplementedError, AttributeError):  # pragma: no cover
            pass

    await runner.run(quote_source=source)
    return 0


# ---------------------------------------------------------------------------


async def cmd_links(args) -> int:
    graph = LinkageGraph()
    src = args.ticker.upper()
    links = graph.neighbors(src)
    if not links:
        print(f"no links from {src}. known sources: {', '.join(graph.sources())}")
        return 1
    print(f"== second-order map from {src} ==")
    by_channel: dict[str, list] = {}
    for l in links:
        by_channel.setdefault(l.channel, []).append(l)
    for channel, group in by_channel.items():
        print(f"\n  [{channel}]")
        for l in group:
            arrow = "same" if l.polarity > 0 else "INVERSE"
            print(f"    {l.dst:<6} beta={l.beta:.2f} {arrow:<7} lag~{l.lag_s:.0f}s  {l.note}")
    return 0


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="signalsniper")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="verify credentials and connectivity")
    sub.add_parser("demo", help="offline end-to-end proof")

    w = sub.add_parser("watch", help="live feeds, alert only unless --live")
    w.add_argument("--equity", type=float, default=None)
    w.add_argument("--live", action="store_true", help="actually send orders")

    l = sub.add_parser("links", help="print the second-order map for a ticker")
    l.add_argument("ticker")

    args = p.parse_args(argv)
    _setup_logging(args.verbose)

    fn = {
        "doctor": cmd_doctor,
        "demo": cmd_demo,
        "watch": cmd_watch,
        "links": cmd_links,
    }[args.cmd]
    return asyncio.run(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
