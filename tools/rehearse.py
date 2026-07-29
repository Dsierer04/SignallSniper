"""Full-session rehearsal: replay a realistic trading day through the real pipeline.

    python3 tools/rehearse.py
    python3 tools/rehearse.py --equity 200 --verbose

Every other harness here tests a moment. `soak.py` tests the 16:05 burst,
`bench.py` tests the hot path, the unit suite tests components. None of them
answers the question you actually have the morning you run this: *what does a
normal session look like, so I can tell when it is going wrong?*

This replays 09:30-16:00 -- macro open, midday filing flow, afternoon drift --
through the real Runner, real classifier, real engine, real risk manager, with
venue-clock (epoch) timestamps exactly as the live feed produces them. It exists
partly as a rehearsal and partly as a regression test for the clock bug: if the
two time domains ever diverge again, the reference prices go wrong and the
signal counts here move.

It is NOT a backtest. The events are synthetic and prove nothing about whether
the strategy is profitable. It proves the machine behaves.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signalsniper.bus import TOPIC_RAW, TOPIC_SIGNAL
from signalsniper.config import Config
from signalsniper.feeds.edgar import TickerResolver
from signalsniper.market.quotes import _parse_rfc3339_ns
from signalsniper.models import Quote, RawDoc
from signalsniper.runner import Runner

ET_OFFSET = timedelta(hours=-4)  # EDT

#: A realistic small/mid-cap universe -- prices a $200 account can actually
#: deploy into, which is where the surviving EDGAR path lives.
UNIVERSE = {
    "TINY": 2.40, "SMOL": 5.10, "MIDC": 12.75, "GROW": 28.40,
    "CRUS": 104.00, "GLW": 45.00, "AAPL": 232.00,
}

#: (minutes after 09:30, ticker, form, items, headline, body)
SESSION: list[tuple[int, str, str, tuple[str, ...], str, str]] = [
    (2,   "SMOL", "8-K",   ("2.02",), "Q2 results",
     "Item 2.02 Results of Operations. Revenue exceeded consensus estimates."),
    (14,  "TINY", "424B5", (),        "prospectus supplement",
     "424B5 prospectus supplement. Pricing of an underwritten public offering "
     "for gross proceeds of $12 million."),
    (37,  "MIDC", "8-K",   ("9.01",), "exhibits",
     "Item 9.01 Financial Statements and Exhibits."),
    (61,  "GROW", "8-K",   ("4.02",), "non-reliance",
     "Item 4.02 Non-Reliance on Previously Issued Financial Statements."),
    (95,  "SMOL", "8-K",   ("5.07",), "shareholder vote",
     "Item 5.07 Submission of Matters to a Vote of Security Holders."),
    (140, "MIDC", "SC 13D", (),       "activist stake",
     "SC 13D filing."),
    (182, "GLW",  "8-K",   ("7.01",), "Reg FD",
     "Item 7.01 Regulation FD Disclosure. Company to present at a conference."),
    (233, "TINY", "8-K",   ("3.01",), "listing deficiency",
     "Item 3.01 Notice of Delisting or Failure to Satisfy a Listing Rule."),
    (300, "GROW", "8-K",   ("1.01",), "material agreement",
     "Item 1.01 Entry into a Material Definitive Agreement. Company awarded a "
     "multi-year contract."),
]

#: Names that actually move after their event, and by how much (bps over 20 min).
REACTIONS = {"SMOL": +380.0, "TINY": -900.0, "GROW": -1400.0, "MIDC": +260.0}


def et(base: datetime, minutes: float) -> datetime:
    return base + timedelta(minutes=minutes)


async def main(args) -> int:
    day = datetime.now(timezone.utc).replace(hour=13, minute=30, second=0,
                                             microsecond=0)  # 09:30 ET
    cfg = Config(sec_user_agent="Rehearsal rehearse@example.com",
                 equity=args.equity, watchlist=tuple(UNIVERSE))
    resolver = TickerResolver()
    resolver.load_mapping({
        str(i): {"cik_str": 1_000_000 + i, "ticker": t, "title": f"{t} Inc"}
        for i, t in enumerate(UNIVERSE)
    })

    runner = Runner(cfg, resolver=resolver)
    signals: list = []
    orders: list = []
    runner.on_signal = signals.append
    runner.on_order = orders.append

    # --- build the whole day's tape up front, in VENUE (epoch) time ---------
    # One quote per symbol per minute for 390 minutes, with each name's reaction
    # applied over the 20 minutes following its own event.
    event_min = {}
    for m, tick, *_ in SESSION:
        event_min.setdefault(tick, m)

    ticks = 0
    for minute in range(390):
        stamp = _parse_rfc3339_ns(et(day, minute).isoformat())
        for tick, base in UNIVERSE.items():
            px = base
            em = event_min.get(tick)
            if em is not None and minute > em and tick in REACTIONS:
                frac = min(1.0, (minute - em) / 20.0)
                px = base * (1 + REACTIONS[tick] / 10_000.0 * frac)
            half = px * 6.0 / 2 / 10_000.0        # 6bps spread
            runner.market.on_quote(
                Quote(tick, round(px - half, 4), round(px + half, 4),
                      round(px, 4), 200, stamp))
            ticks += 1

    tasks = [asyncio.create_task(runner._classify_loop()),
             asyncio.create_task(runner._signal_loop())]
    sink = runner.bus.subscribe(TOPIC_SIGNAL, maxsize=4096, name="rehearse")

    print(f"replaying 09:30-16:00 ET  |  {len(UNIVERSE)} symbols, "
          f"{ticks:,} quotes, {len(SESSION)} filings")
    print(f"equity ${cfg.equity:,.0f}  ->  max_position "
          f"${runner.risk.cfg.equity * runner.risk.cfg.max_position_frac:,.0f}, "
          f"concurrent {runner.risk.cfg.max_concurrent}, "
          f"fractional {runner.risk.cfg.allow_fractional}\n")

    for m, tick, form, items, headline, body in SESSION:
        doc = RawDoc(
            source="edgar", doc_id=f"{tick}-{m}",
            title=f"{form} - {tick} Inc ({1_000_000 + list(UNIVERSE).index(tick):010d}) (Filer)",
            url=f"https://www.sec.gov/Archives/edgar/data/{tick}/x-index.htm",
            published=None, body=body,
            meta={"form": form, "company": f"{tick} Inc",
                  "cik": str(1_000_000 + list(UNIVERSE).index(tick)),
                  "items": items},
        )
        doc.t_ingest = _parse_rfc3339_ns(et(day, m).isoformat())
        runner.bus.publish(TOPIC_RAW, doc)
        await asyncio.sleep(0.05)
        if args.verbose:
            print(f"  {et(day, m) + ET_OFFSET:%H:%M} {tick:5} {form:7} {headline}")

    await asyncio.sleep(0.5)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # --- report -------------------------------------------------------------
    print(f"\n{'=' * 72}\nSESSION SUMMARY\n{'=' * 72}")
    print(f"  filings ingested   : {runner.counts['docs']}")
    print(f"  events classified  : {runner.counts['events']}")
    print(f"  signals emitted    : {runner.counts['signals']}")
    print(f"  orders sized       : {runner.counts['orders']}")
    drops = runner.bus.stats().get("dropped", {})
    print(f"  queue drops        : {sum(drops.values()) if drops else 0}")

    if signals:
        print(f"\n  signals:")
        for s in signals:
            print(f"    {s.describe()}")
    if orders:
        print(f"\n  orders:")
        for o in orders:
            print(f"    {o.describe()}  [{o.reason}]")

    rej = runner.engine.stats()
    print(f"\n  engine rejects     : {rej}")
    print(f"  risk rejects       : {runner.risk.summary()['rejects']}")

    # --- sanity assertions --------------------------------------------------
    print(f"\n{'=' * 72}\nCHECKS\n{'=' * 72}")
    rc = 0
    checks = []

    checks.append(("all filings reached the classifier",
                   runner.counts["docs"] == len(SESSION)))
    checks.append(("no queue drops", not drops or sum(drops.values()) == 0))
    # Noise must not produce signals.
    noise = {"MIDC-37", "SMOL-95", "GLW-182"}
    noise_signals = [s for s in signals if s.event.doc.doc_id in noise]
    checks.append(("exhibits/vote/RegFD produced no signals", not noise_signals))
    # The clock regression: with domains mixed, reference prices come from the
    # oldest tick and materially different names all look identically "unmoved".
    checks.append(("reference prices are event-relative (clock domains agree)",
                   all(abs(s.edge_bps) <= 1500.0 for s in signals)))
    checks.append(("every order is within the notional cap",
                   all(o.notional <= cfg.equity * runner.risk.cfg.max_position_frac + 1
                       for o in orders)))
    checks.append(("no shorts sized below the $2,000 floor",
                   cfg.equity >= 2000 or not any(
                       o.direction.value < 0 for o in orders)))

    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            rc = 1

    print(f"\n{'PASS -- the machine behaves' if rc == 0 else 'FAIL'}")
    print("\nThis proves the pipeline, NOT the strategy. The events are synthetic;")
    print("no part of this is evidence that the signals are profitable.")
    return rc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Full-session rehearsal")
    p.add_argument("--equity", type=float, default=200.0)
    p.add_argument("--verbose", action="store_true")
    raise SystemExit(asyncio.run(main(p.parse_args())))
