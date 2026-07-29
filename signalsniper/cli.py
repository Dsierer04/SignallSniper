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
    from .execution.broker import AlpacaBroker
    from .market.quotes import AlpacaQuoteStream
    from .runner import Runner

    log = logging.getLogger("signalsniper")
    cfg = load()
    if args.equity:
        cfg.equity = args.equity
    # --live is an AND with the env flag, never an override. Two independent
    # switches means a stale shell export cannot arm this on its own, and
    # neither can a stray CLI flag.
    cfg.live = bool(args.live) and cfg.live

    broker = None
    if cfg.live:
        if not (cfg.alpaca_key and cfg.alpaca_secret):
            log.error("LIVE_TRADING requested without broker credentials -- refusing")
            return 1
        broker = AlpacaBroker(cfg.alpaca_key, cfg.alpaca_secret,
                              paper=cfg.alpaca_paper,
                              allow_extended=cfg.allow_extended)
        acct = await broker.account()
        blockers = acct.blockers(need_short=True)
        for b in blockers:
            log.error("account: %s", b)
        mode = "PAPER" if acct.is_paper else "*** LIVE MONEY ***"
        log.warning("broker armed [%s] equity=$%s", mode, f"{acct.equity:,.2f}")
    elif args.live:
        log.error("--live passed but LIVE_TRADING is not set in the environment; "
                  "staying alert-only")

    runner = Runner(cfg, broker=broker)
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


async def cmd_preflight(args) -> int:
    """Everything that can stop you at 16:05, checked at 15:00 instead."""
    from .execution.broker import AlpacaBroker
    from .execution.session import current_session

    cfg = load()
    rc = 0

    print("== session ==")
    s = current_session()
    print(f"  now (ET)       : {s.now_et:%Y-%m-%d %H:%M:%S}")
    print(f"  session        : {s.session.value}{' (' + s.note + ')' if s.note else ''}")
    print(f"  bracket orders : {'yes' if s.supports_bracket else 'NO'}")
    if s.is_extended:
        print("  [!] extended hours: Alpaca rejects bracket orders here, so any")
        print("      stop is enforced by this process only. If it dies, you are naked.")
        print("      Requires ALLOW_EXTENDED=1 to trade at all.")

    print("\n== market data ==")
    if not (cfg.alpaca_key and cfg.alpaca_secret):
        print("  [!] no Alpaca credentials -- second-order signals are IMPOSSIBLE")
        print("      (they need a live tape on the LINKED names, not just issuers)")
        rc = 1
    else:
        print(f"  feed           : {cfg.alpaca_feed}")
        if cfg.alpaca_feed == "iex":
            print("  [!] 'iex' is ~2% of consolidated volume. After-hours prints are")
            print("      too thin to reference against -- the 16:05 window will")
            print("      produce confident nonsense. Use 'sip' or sit that window out.")
            rc = 1
    print(f"  watchlist      : {len(cfg.watchlist)} symbols")

    print("\n== broker ==")
    if not (cfg.alpaca_key and cfg.alpaca_secret):
        print("  [!] no credentials -- cannot place orders")
        return 1

    broker = AlpacaBroker(cfg.alpaca_key, cfg.alpaca_secret, paper=cfg.alpaca_paper)
    try:
        acct = await broker.account()
    except Exception as exc:
        print(f"  [!] account fetch FAILED: {exc!r}")
        await broker.aclose()
        return 1

    print(f"  mode           : {'PAPER' if acct.is_paper else '*** LIVE ***'}")
    print(f"  account        : {acct.account_number}")
    print(f"  equity         : ${acct.equity:,.2f}")
    print(f"  buying power   : ${acct.buying_power:,.2f}")
    if acct.pdt_fields_present:
        print(f"  day trades used: {acct.daytrade_count} (legacy PDT reporting)")
    else:
        print("  day trades     : not reported -- account is on the Intraday")
        print("                   Margin Framework (PDT retired 2026-06-04);")
        print("                   buying power is the binding constraint")
    print(f"  shorting       : {'enabled' if acct.shorting_enabled else 'DISABLED'}")

    blockers = acct.blockers(need_short=True)
    if blockers:
        print()
        for b in blockers:
            print(f"  [!] {b}")
        rc = 1

    if abs(acct.equity - cfg.equity) > max(100.0, acct.equity * 0.05):
        print(f"\n  [!] EQUITY env is ${cfg.equity:,.0f} but the account holds "
              f"${acct.equity:,.0f}.")
        print("      Sizing uses the env value. Fix it or you will size wrong.")
        rc = 1

    await broker.aclose()

    print("\n== execution posture ==")
    print(f"  LIVE_TRADING   : {cfg.live}")
    if not cfg.live:
        print("  -> alert only. No orders will be sent regardless of --live.")

    print("\n" + ("PREFLIGHT FAILED -- fix the [!] items above" if rc
                  else "preflight clean"))
    return rc


async def cmd_flatten(args) -> int:
    """The panic button. Cancels every open order, closes every position."""
    from .execution.broker import AlpacaBroker

    cfg = load()
    if not (cfg.alpaca_key and cfg.alpaca_secret):
        print("no Alpaca credentials configured")
        return 1

    broker = AlpacaBroker(cfg.alpaca_key, cfg.alpaca_secret, paper=cfg.alpaca_paper)
    try:
        mode = "PAPER" if cfg.alpaca_paper else "*** LIVE ***"
        print(f"[{mode}] cancelling orders and closing positions...")
        cancelled = await broker.cancel_all()
        closed = await broker.flatten_all()
        print(f"  cancelled {cancelled} order(s)")
        print(f"  closed    {closed} position(s)")
        remaining = await broker.positions()
        if remaining:
            print(f"  [!] {len(remaining)} position(s) STILL OPEN:")
            for p in remaining:
                print(f"      {p.get('symbol')} qty={p.get('qty')}")
            print("      Close these manually in the broker UI.")
            return 1
        print("  flat.")
    finally:
        await broker.aclose()
    return 0


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
    sub.add_parser("preflight", help="check account, session and feed before trading")
    sub.add_parser("flatten", help="PANIC: cancel all orders and close all positions")

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
        "preflight": cmd_preflight,
        "flatten": cmd_flatten,
        "watch": cmd_watch,
        "links": cmd_links,
    }[args.cmd]
    return asyncio.run(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
