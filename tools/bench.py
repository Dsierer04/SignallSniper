"""Latency budget. Run: python3 tools/bench.py

Measures the parts we control. The parts we do not control (network RTT to SEC,
broker ack) dominate the real number and are labelled as such in the output --
the point of this bench is to prove the in-process path is not where the time
goes, so that optimisation effort stays pointed at the network.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signalsniper.feeds.edgar import EdgarCurrentFeed
from signalsniper.bus import EventBus
from signalsniper.feeds.base import TokenBucket
from signalsniper.market.linkage import LinkageGraph
from signalsniper.market.quotes import synth_walk
from signalsniper.market.tape import MarketState
from signalsniper.models import RawDoc, now_ns
from signalsniper.parse.classify import build_event, classify_doc
from signalsniper.signal.engine import DEFAULT_MOVE_SCALE, EngineConfig, SignalEngine
from signalsniper.signal.risk import RiskConfig, RiskManager

import httpx

ATOM = b"""<?xml version="1.0" encoding="ISO-8859-1"?>
<feed xmlns="http://www.w3.org/2005/Atom">
%s
</feed>
""" % b"\n".join(
    b"""<entry>
    <title>8-K - COMPANY %d INC (00012345%02d) (Filer)</title>
    <link rel="alternate" href="https://www.sec.gov/Archives/edgar/data/1/x%d-index.htm"/>
    <summary type="html">Item 2.02 Results of Operations and Financial Condition.</summary>
    <updated>2026-07-30T16:05:12-04:00</updated>
    <category scheme="https://www.sec.gov/" label="form type" term="8-K"/>
    <id>urn:tag:sec.gov,2008:accession-number=00012345%02d-26-0000%02d</id>
  </entry>""" % (i, i, i, i, i)
    for i in range(40)
)


def bench(label: str, fn, n: int = 2000, unit: str = "us") -> None:
    fn()  # warm
    samples = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    samples.sort()
    div = 1_000.0 if unit == "us" else 1_000_000.0
    p50 = samples[len(samples) // 2] / div
    p99 = samples[int(len(samples) * 0.99)] / div
    mean = statistics.mean(samples) / div
    print(f"  {label:<38} p50={p50:8.2f}{unit}  p99={p99:8.2f}{unit}  mean={mean:8.2f}{unit}")


def main() -> None:
    print("SignalSniper latency budget (in-process path)\n")

    # --- parse -----------------------------------------------------------
    feed = EdgarCurrentFeed(EventBus(), None, TokenBucket(10.0), form="8-K")
    headers = httpx.Headers()
    print("[parse] 40-entry EDGAR atom document")
    bench("xml parse + 40 docs extracted", lambda: list(feed.parse(ATOM, headers)))

    doc = RawDoc(
        source="edgar", doc_id="a",
        title="8-K - Apple Inc. (0000320193) (Filer)", url="", published=None,
        body="Item 2.02 Results of Operations. The Company lowered its iPhone "
             "hardware guidance below consensus estimates for the quarter.",
        meta={"form": "8-K", "company": "Apple Inc.", "cik": "320193",
              "items": ("2.02", "9.01")},
    )

    print("\n[classify] single document")
    bench("classify_doc", lambda: classify_doc(doc))
    bench("build_event", lambda: build_event(doc, ("AAPL",)))

    # --- engine ----------------------------------------------------------
    market = MarketState()
    t_pre = now_ns() - 8_000_000_000
    universe = ["AAPL", "CRUS", "SWKS", "QRVO", "LITE", "COHR", "GLW", "JBL",
                "QCOM", "AVGO", "TXN", "FN", "GOOGL"]
    for tick in universe:
        for q in synth_walk(tick, 100.0, 200, t_pre):
            market.on_quote(q)
    t_event = now_ns()
    for q in synth_walk("AAPL", 100.0, 50, t_event, drift_bps_total=-500.0):
        market.on_quote(q)
    for tick in universe[1:]:
        for q in synth_walk(tick, 100.0, 50, t_event, drift_bps_total=-40.0):
            market.on_quote(q)

    doc.t_ingest = t_event
    event = build_event(doc, ("AAPL",))
    engine = SignalEngine(market, LinkageGraph(),
                          EngineConfig(move_scale=dict(DEFAULT_MOVE_SCALE)))
    n_sig = len(engine.on_event(event))

    print(f"\n[engine] 13-name universe, {n_sig} signals produced")
    bench("on_event (classify->signals)", lambda: engine.on_event(event), n=1000)
    bench("tape.move_bps_since", lambda: market.tape("AAPL").move_bps_since(t_event), n=5000)
    bench("graph.neighbors(AAPL)", lambda: LinkageGraph().neighbors("AAPL"), n=1000)

    # --- risk ------------------------------------------------------------
    sigs = engine.on_event(event)
    if sigs:
        rm = RiskManager(RiskConfig(equity=25_000.0))
        print("\n[risk] sizing")
        bench("size_order", lambda: rm.size_order(sigs[0]), n=2000)

    # --- ingest ----------------------------------------------------------
    quotes = synth_walk("AAPL", 100.0, 1000, now_ns())
    tape_market = MarketState()
    print("\n[ingest] quote handling")
    bench("on_quote x1000", lambda: [tape_market.on_quote(q) for q in quotes],
          n=200, unit="ms")

    print("""
Not measured here, and dominant in production:
  SEC EDGAR poll RTT .................. 30-120ms  (network, your ISP to sec.gov)
  Alpaca/Polygon websocket delivery ... 5-50ms    (venue to you)
  Broker order ack .................... 50-300ms  (retail broker routing)

Read: the in-process path is noise against the network. Do not micro-optimise
this code -- spend effort on colocation-adjacent choices (a VPS near the data
source, SIP instead of IEX, direct IR feeds instead of aggregators).""")


if __name__ == "__main__":
    main()
