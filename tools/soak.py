"""Operational soak test: does the pipeline survive 16:05?

    python3 tools/soak.py
    python3 tools/soak.py --filings 800 --quote-rate 20000 --seconds 20

At 16:05 ET on an earnings day, several hundred 8-Ks hit EDGAR within seconds
while the quote stream for the whole watchlist goes from a trickle to a flood.
Unit tests feed one document at a time and prove nothing about that moment.

This harness reproduces the burst against the real runner -- real bus, real
classifier, real engine, real risk manager -- and reports what breaks:

  * queue drops (a dropped RawDoc is a missed trade, silently)
  * end-to-end latency under load vs idle
  * whether the signal that matters still gets through when 800 filings
    for names we do not care about are competing for the same event loop

The failure mode this is built to catch: the pipeline appears healthy, drops
nothing visible, and quietly delivers the one signal you needed 4 seconds late.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signalsniper.bus import TOPIC_RAW, TOPIC_SIGNAL
from signalsniper.config import Config
from signalsniper.feeds.edgar import TickerResolver
from signalsniper.market.quotes import synth_walk
from signalsniper.models import RawDoc, now_ns
from signalsniper.runner import Runner

# A realistic mix: mostly noise, because that is what the tape actually looks
# like. Exhibits-only filings, scheduling announcements, routine 13Gs.
NOISE_TEMPLATES = [
    ("8-K", ("9.01",), "Item 9.01 Financial Statements and Exhibits."),
    ("8-K", ("5.07",), "Item 5.07 Submission of Matters to a Vote of Security Holders."),
    ("SC 13G", (), "SC 13G filing."),
    ("8-K", ("7.01",), "Item 7.01 Regulation FD Disclosure. Company to present at a conference."),
    ("4", (), "Statement of changes in beneficial ownership."),
]


def make_noise(i: int) -> RawDoc:
    form, items, body = NOISE_TEMPLATES[i % len(NOISE_TEMPLATES)]
    cik = 1_000_000 + i
    return RawDoc(
        source="edgar", doc_id=f"{cik:010d}-26-{i:06d}",
        title=f"{form} - NOISE CORP {i} ({cik:010d}) (Filer)",
        url=f"https://www.sec.gov/Archives/edgar/data/{cik}/x-index.htm",
        published=None, body=body,
        meta={"form": form, "company": f"NOISE CORP {i}", "cik": str(cik),
              "items": items},
    )


def make_the_one(t_ingest: int) -> RawDoc:
    """The filing that actually matters, buried in the storm."""
    d = RawDoc(
        source="edgar", doc_id="0000320193-26-000077",
        title="8-K - Apple Inc. (0000320193) (Filer)",
        url="https://www.sec.gov/Archives/edgar/data/320193/x-index.htm",
        published=None,
        body="Item 2.02 Results of Operations and Financial Condition. The "
             "Company lowered its guidance for iPhone hardware revenue and now "
             "expects results below consensus estimates.",
        meta={"form": "8-K", "company": "Apple Inc.", "cik": "320193",
              "items": ("2.02", "9.01")},
    )
    d.t_ingest = t_ingest
    return d


#: Realistic price levels. The first version of this harness used 100.0 for
#: every symbol, which injected a wild bad tick into each tape -- and that
#: accident is what exposed the missing outlier defenses. Kept correct here so
#: the soak measures throughput; tests/test_badtick.py covers corruption.
BASE_PRICES = {"AAPL": 232.0, "CRUS": 104.0, "SWKS": 78.0,
               "QRVO": 92.0, "LITE": 60.0, "GLW": 45.0}


async def quote_flood(runner: Runner, tickers: list[str], rate: int,
                      seconds: float, stop: asyncio.Event) -> int:
    """Push quotes at `rate`/sec across the watchlist until stopped."""
    sent = 0
    per_batch = max(1, rate // 100)
    t_end = time.monotonic() + seconds
    base = {t: BASE_PRICES.get(t, 100.0) for t in tickers}
    while time.monotonic() < t_end and not stop.is_set():
        t0 = now_ns()
        for i in range(per_batch):
            tick = tickers[(sent + i) % len(tickers)]
            for q in synth_walk(tick, base[tick], 1, t0 + i * 1_000_000):
                runner.market.on_quote(q)
        sent += per_batch
        await asyncio.sleep(0.01)
    return sent


async def run(args) -> int:
    cfg = Config(sec_user_agent="Soak soak@example.com",
                 watchlist=("AAPL", "CRUS", "SWKS", "QRVO", "LITE", "GLW"))
    resolver = TickerResolver()
    resolver.load_mapping({
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    })

    runner = Runner(cfg, resolver=resolver)
    runner.on_signal = lambda s: None  # silence; we measure, not narrate

    # This harness measures pipeline THROUGHPUT under a burst, not signal
    # quality. In production verified_links_only defaults to True and the
    # shipped graph has nothing measured, so the second-order path emits
    # nothing -- correct, but it would leave this test with no signal to time.
    runner.engine.cfg.verified_links_only = False

    # Seed tapes so the second-order path has references to work from.
    t_pre = now_ns() - 8_000_000_000
    for tick, px in (("AAPL", 232.0), ("CRUS", 104.0), ("SWKS", 78.0),
                     ("QRVO", 92.0), ("LITE", 60.0), ("GLW", 45.0)):
        for q in synth_walk(tick, px, 60, t_pre):
            runner.market.on_quote(q)

    sink = runner.bus.subscribe(TOPIC_SIGNAL, maxsize=8192, name="soak")
    tasks = [asyncio.create_task(runner._classify_loop()),
             asyncio.create_task(runner._signal_loop())]

    stop = asyncio.Event()
    flood = asyncio.create_task(
        quote_flood(runner, list(cfg.watchlist), args.quote_rate, args.seconds, stop))

    await asyncio.sleep(0.3)  # let the flood establish

    # --- the burst -------------------------------------------------------
    print(f"firing {args.filings} filings in a burst, "
          f"quotes at ~{args.quote_rate}/s...")

    t_event = now_ns()
    for q in synth_walk("AAPL", 232.0, 40, t_event, drift_bps_total=-450.0):
        runner.market.on_quote(q)
    for tick, px in (("CRUS", 104.0), ("SWKS", 78.0)):
        for q in synth_walk(tick, px, 40, t_event, drift_bps_total=-50.0):
            runner.market.on_quote(q)

    burst_start = time.monotonic()
    the_one_at = args.filings // 2  # bury it in the middle
    fired_ns = 0
    for i in range(args.filings):
        if i == the_one_at:
            doc = make_the_one(t_event)
            fired_ns = now_ns()
        else:
            doc = make_noise(i)
        runner.bus.publish(TOPIC_RAW, doc)
    burst_publish_ms = (time.monotonic() - burst_start) * 1000

    # --- wait for the signal we care about -------------------------------
    found = None
    latency_ms = float("nan")
    try:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            sig = await asyncio.wait_for(sink.get(), timeout=args.timeout)
            if sig.event.doc.doc_id == "0000320193-26-000077":
                found = sig
                latency_ms = (now_ns() - fired_ns) / 1e6
                break
    except asyncio.TimeoutError:
        pass

    # Drain whatever else landed.
    await asyncio.sleep(0.5)
    extra = []
    while not sink.queue.empty():
        extra.append(sink.queue.get_nowait())

    stop.set()
    for t in tasks:
        t.cancel()
    quotes_sent = await flood
    await asyncio.gather(*tasks, return_exceptions=True)

    # --- report ----------------------------------------------------------
    stats = runner.bus.stats()
    raw_dropped = stats.get("dropped", {}).get(TOPIC_RAW, 0)
    sig_dropped = stats.get("dropped", {}).get(TOPIC_SIGNAL, 0)

    print(f"\n{'=' * 70}")
    print("SOAK RESULTS")
    print("=" * 70)
    print(f"  filings published      : {args.filings}")
    print(f"  burst publish time     : {burst_publish_ms:.1f}ms")
    print(f"  quotes pushed          : {quotes_sent:,}")
    print(f"  docs classified        : {runner.counts['docs']}")
    print(f"  events emitted         : {runner.counts['events']}")
    print(f"  signals emitted        : {runner.counts['signals']}")
    print(f"  RawDoc queue drops     : {raw_dropped}")
    print(f"  Signal queue drops     : {sig_dropped}")

    print(f"\n  the signal that mattered:")
    if found is not None:
        print(f"    FOUND  {found.describe()}")
        print(f"    wire-to-signal under load: {latency_ms:.1f}ms")
    else:
        print("    *** NOT FOUND *** -- the AAPL signal did not survive the burst")

    rc = 0
    print(f"\n{'=' * 70}")
    if found is None:
        print("FAIL: the signal was lost under load.")
        rc = 1
    elif raw_dropped:
        print(f"FAIL: {raw_dropped} RawDocs were dropped. Each is a missed filing,")
        print("      and nothing downstream can tell you which one.")
        rc = 1
    elif runner.counts["docs"] < args.filings:
        missing = args.filings - runner.counts["docs"]
        print(f"FAIL: {missing} filings never reached the classifier.")
        rc = 1
    elif latency_ms > args.max_latency_ms:
        print(f"FAIL: {latency_ms:.0f}ms exceeds the {args.max_latency_ms:.0f}ms budget.")
        print("      Being late is worse than being absent -- you cross a spread")
        print("      into a move that has already happened.")
        rc = 1
    else:
        print(f"PASS: all {args.filings} filings classified, zero drops,")
        print(f"      target signal delivered in {latency_ms:.1f}ms under load.")
    print("=" * 70)
    return rc


def main() -> int:
    p = argparse.ArgumentParser(description="16:05 burst soak test")
    p.add_argument("--filings", type=int, default=500,
                   help="filings in the burst (a heavy 16:05 is several hundred)")
    p.add_argument("--quote-rate", type=int, default=10_000, help="quotes/sec")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument("--max-latency-ms", type=float, default=2_000.0)
    return asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
