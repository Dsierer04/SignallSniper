"""Measure whether the propagation lag actually exists.

    python3 tools/validate.py --primary AAPL
    python3 tools/validate.py --primary AMZN --feed sip
    python3 tools/validate.py --primary AAPL --selftest   # no network

This is the empirical test the whole system rests on. It answers, per linked
name: after the primary printed, what fraction of the linked name's eventual
move was STILL AVAILABLE five minutes later?

  >= 50% still available, and the name actually printed  -> the edge is real
  <  50%, or no prints in the window                     -> no edge, do not trade it

A "no edge" verdict on a name is not a failure of this tool. It is the tool
doing its job before the money does.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from signalsniper.market.linkage import LinkageGraph
from signalsniper.validate.events import events_for
from signalsniper.validate.loader import AlpacaBars, BarCache, load_window
from signalsniper.validate.study import (
    estimate_beta,
    lag_verdict,
    profile_event,
)


def _selftest(primary: str) -> int:
    """Prove the study logic on constructed series -- no network, no keys.

    Two linked names by construction: one that lags (most of its move arrives
    after +5m) and one that reprices instantly. The tool must separate them.
    """
    from datetime import timedelta

    from signalsniper.validate.events import _et
    from signalsniper.validate.study import Bar, Series

    t0 = _et(2026, 4, 30, 16, 30)

    def build(ticker: str, base: float, path: list[tuple[float, float]]) -> Series:
        """path = [(minutes_from_event, cumulative_bps)]"""
        bars = []
        for minutes, bps in path:
            px = base * (1 + bps / 10_000.0)
            bars.append(Bar(t=t0 + timedelta(minutes=minutes), open=px, high=px,
                            low=px, close=px, volume=5_000))
        return Series(ticker, bars)

    flat = [(-30.0, 0.0), (-1.0, 0.0)]
    # Primary: gone in the first minute, as reality dictates.
    prim = build(primary, 230.0, flat + [(0.0, 0.0), (1.0, -400.0), (5.0, -410.0),
                                         (15.0, -415.0), (30.0, -420.0), (240.0, -420.0)])
    # LAGGER: barely moves early, arrives over the following half hour.
    lag = build("LAGGY", 100.0, flat + [(0.0, 0.0), (1.0, -20.0), (5.0, -40.0),
                                        (15.0, -180.0), (30.0, -300.0), (240.0, -340.0)])
    # INSTANT: fully repriced inside the first minute. No edge.
    inst = build("INSTA", 100.0, flat + [(0.0, 0.0), (1.0, -300.0), (5.0, -330.0),
                                         (15.0, -335.0), (30.0, -340.0), (240.0, -340.0)])

    print("== selftest (synthetic, no network) ==\n")
    ok = True
    for series, expect_edge in ((lag, True), (inst, False)):
        profiles = [profile_event(f"E{i}", prim, series, t0) for i in range(6)]
        v = lag_verdict(profiles)
        print("  " + v.describe())
        if v.edge_exists is not expect_edge:
            print(f"    !! expected edge_exists={expect_edge}, got {v.edge_exists}")
            ok = False

        b = estimate_beta(profiles, prior=0.85)
        if b:
            print("    " + b.describe())

    print("\n" + ("selftest PASSED -- the study separates lag from no-lag"
                  if ok else "selftest FAILED"))
    return 0 if ok else 1


async def run(args) -> int:
    if args.selftest:
        return _selftest(args.primary)

    key = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_SECRET_KEY", "")
    if not (key and secret):
        print("ALPACA_API_KEY / ALPACA_SECRET_KEY required.")
        print("Run with --selftest to verify the study logic without credentials.")
        return 1

    primary = args.primary.upper()
    events = events_for(primary)
    if not events:
        print(f"no known events for {primary}")
        return 1

    graph = LinkageGraph()
    links = graph.neighbors(primary, min_beta=args.min_beta)
    if not links:
        print(f"no links from {primary}")
        return 1

    linked = [l.dst for l in links]
    priors = {l.dst: l.beta * l.polarity for l in links}
    symbols = [primary] + linked

    if args.feed != "sip":
        print(f"WARNING: feed='{args.feed}'. After-hours coverage on a non-SIP feed")
        print("         is thin, and thin coverage looks exactly like 'no move'.")
        print("         A 'no edge' verdict from this run would not be trustworthy.\n")

    provider = AlpacaBars(key, secret, feed=args.feed, cache=BarCache(args.cache))
    per_linked: dict[str, list] = {t: [] for t in linked}
    skipped: list[str] = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        for ev in events[: args.max_events]:
            print(f"loading {ev.event_id} ({ev.t_release:%Y-%m-%d %H:%M %Z})...",
                  flush=True)
            series = await load_window(provider, client, symbols, ev.t_release)
            prim = series.get(primary)
            if not prim or not prim.bars:
                print(f"  no bars for {primary}; skipping")
                skipped.append(ev.event_id)
                continue

            # Sanity gate: if the primary did not move, this date is wrong or the
            # print was a non-event. Either way the linked readings are noise.
            probe = profile_event(ev.event_id, prim, prim, ev.t_release)
            if abs(probe.primary_total_bps) < args.min_primary_bps:
                print(f"  {primary} moved only {probe.primary_total_bps:+.0f}bps "
                      f"-- date likely wrong or non-event; skipping")
                skipped.append(ev.event_id)
                continue
            print(f"  {primary} {probe.primary_total_bps:+.0f}bps")

            for t in linked:
                s = series.get(t)
                if s and s.bars:
                    per_linked[t].append(profile_event(ev.event_id, prim, s, ev.t_release))

    print(f"\n{'=' * 78}")
    print(f"LAG STUDY: {primary}  ({len(events[:args.max_events]) - len(skipped)} usable events)")
    print("=" * 78)
    print("'still-available@5m' = share of the linked name's move that had NOT yet")
    print("happened 5 minutes after the print. That is what a live system can catch.\n")

    verdicts = []
    for t in linked:
        v = lag_verdict(per_linked[t])
        if v:
            verdicts.append(v)
            print("  " + v.describe())

    print(f"\n{'=' * 78}")
    print("EMPIRICAL BETAS  (regression through origin; replaces hand-set priors)")
    print("=" * 78 + "\n")
    for t in linked:
        b = estimate_beta(per_linked[t], prior=priors.get(t, 0.0))
        if b:
            print("  " + b.describe())

    edges = [v for v in verdicts if v.edge_exists]
    print(f"\n{'=' * 78}")
    print(f"VERDICT: {len(edges)}/{len(verdicts)} linked names show a tradeable lag")
    print("=" * 78)
    if edges:
        print("\nNames where the lag is real:")
        for v in sorted(edges, key=lambda x: -abs(x.median_total_bps)):
            print(f"  {v.linked:<6} median move {v.median_total_bps:+.0f}bps, "
                  f"{v.median_capture_after_5m:.0%} still available at +5m")
    else:
        print("\nNo linked name cleared the bar. Either the lag does not exist at")
        print("this horizon, or the data does not cover after-hours well enough to")
        print("see it. Check the feed before concluding the former.")

    if skipped:
        print(f"\nSkipped {len(skipped)} event(s): {', '.join(skipped)}")
        print("Verify those dates against the actual 8-K timestamps on EDGAR.")

    if args.emit:
        import json

        rows = []
        for t in linked:
            v = lag_verdict(per_linked[t])
            b = estimate_beta(per_linked[t], prior=priors.get(t, 0.0))
            if v is None:
                continue
            rows.append({
                "src": primary,
                "dst": t,
                "n_events": v.n_events,
                "lag_capture": (None if math.isnan(v.median_capture_after_5m)
                                else round(v.median_capture_after_5m, 4)),
                "dead_rate": round(v.dead_early_rate, 4),
                "median_move_bps": round(v.median_total_bps, 1),
                "beta": round(b.beta, 4) if b else None,
                "beta_r2": round(b.r_squared, 4) if b else None,
                "beta_usable": bool(b and b.usable),
                "prior_beta": round(priors.get(t, 0.0), 4),
            })

        path = Path(args.emit)
        existing = {"links": []}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except ValueError:
                pass
        # Merge rather than overwrite: an AMZN run must not erase an AAPL run.
        keep = [r for r in existing.get("links", []) if r.get("src") != primary]
        payload = {
            "generated_for": primary,
            "feed": args.feed,
            "links": keep + rows,
        }
        path.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {len(rows)} calibrated link(s) to {path}")
        print("The engine picks this up via LinkageGraph.calibrated().")

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Measure the propagation lag empirically")
    p.add_argument("--primary", default="AAPL", help="AAPL or AMZN")
    p.add_argument("--feed", default="sip", help="sip (correct) or iex (misleading after hours)")
    p.add_argument("--min-beta", type=float, default=0.15)
    p.add_argument("--max-events", type=int, default=10)
    p.add_argument("--min-primary-bps", type=float, default=100.0,
                   help="skip events where the primary barely moved")
    p.add_argument("--cache", default=".cache/bars")
    p.add_argument("--selftest", action="store_true",
                   help="verify the study logic on synthetic data, no network")
    p.add_argument("--emit", default="", metavar="PATH",
                   help="write measured betas and lag captures to a calibration "
                        "file the engine can load (e.g. calibration.json)")
    return asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
