# SignalSniper — orientation for a new session

Read this before changing anything. Several behaviours here look like bugs and
are deliberate; a few things that look settled were found wrong the hard way.

**Active branch:** `claude/trading-signal-speed-cr039h` → [PR #1](https://github.com/Dsierer04/SignallSniper/pull/1)
**Full findings:** `docs/VERIFICATION.md` — read this second.

---

## What this is

Low-latency SEC-filing event ingestion → deterministic classification → signal
generation → risk-gated execution via Alpaca. Rewritten from a Reddit sentiment
scraper (now in `legacy/`, demoted for reasons in `docs/DATA_SOURCES.md`).

```
feeds/ → parse/ → signal/engine.py → signal/risk.py → execution/broker.py
                       ↑ runner.py wires it
```

## Things that look broken and are not

**1. `python3 -m signalsniper demo` emits zero signals.**
`verified_links_only` defaults to `True` and nothing in the shipped linkage graph
has been measured, so the second-order path is silent by design. The literature
refuted the minutes-scale propagation thesis it depends on. Do not "fix" this by
flipping the default. It turns itself on when `tools/validate.py` measures a link.

**2. `preflight` reports FAILED on a small account.**
Correct. It is reporting that shorting is regulatorily impossible below $2,000
equity, which is true and unappealable.

**3. Most sessions produce no signals.**
`tools/rehearse.py` replays a realistic day: 9 filings → 2 signals. Noise
(exhibits-only 8-Ks, shareholder votes, Reg FD conference notices) is supposed to
be suppressed.

## Rules that exist because breaking them cost something

**Two clock domains. Do not mix them.**
- `epoch_ns()` — event times, anything compared **across sources** (quotes, filings)
- `mono_ns()` — durations **inside** the process (latency, age, cooldowns)

Mixing them shipped a bug where `move_bps_since` returned −385bps when the truth
was 0, on *every* event, because the tape held epoch stamps and events carried
monotonic ones (~310,000× apart). The core "already priced in" gate was broken in
production. **All 250 tests passed**, because fixtures built tapes from the same
monotonic clock they stamped events with.

> A fixture that mirrors your assumption tests the assumption, not the system.
> When adding tests that touch time or feed shapes, build the fixture from the
> **production representation** — see `tests/test_fatal_fixes.py`.

The same failure hit the EDGAR parser: item codes were scanned from the Atom
`<summary>`, which never contains them. Every 8-K silently fell below the
materiality floor. The test fixture had item codes because it was hand-written.

**Do not re-add removed linkage links.**
`market/linkage.py` records 11 removals inline with 10-K citations. Notably: FN
has *no* Apple exposure (NVIDIA 27.6%, Cisco 18.2%); ANET's top customers are
Microsoft and Meta, not AWS; the SHOP/AMZN "inverse" link is empirically
*positive* (+0.42 1y) and would have lost systematically.

**Betas need error bars.** Without `beta_stderr` the slack gate is a
beta-overstatement detector — it fires on a provably zero-edge tape whenever the
hand-set beta is ≥35% too high, and rejects understated betas as "overshot."

## Safety invariants — do not weaken

- Orders require **both** `LIVE_TRADING=1` and `--live`. Two independent switches.
- `ALLOW_EXTENDED` is separate from `LIVE_TRADING`: Alpaca rejects bracket orders
  outside regular hours, so after 16:00 **the daemon is the stop**.
- `flatten` refuses to run without an explicit `ALPACA_PAPER` — it once could
  have flattened paper while real positions stayed open.
- Send the closing order **before** booking the exit (`pending_exits()` is pure,
  `close()` mutates). Booking first deleted live positions from memory.
- Kill switch takes marks and counts **unrealized** — realized-only was blind to
  the drawdown that actually matters.
- `RiskConfig.for_equity()` scales position mechanics for small accounts but
  never widens `risk_per_trade` or `max_daily_loss_frac`.

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate   # macOS: pip is not aliased
pip install -r requirements.txt
python3 -m pytest -q                  # 293 tests, no network

python3 -m signalsniper demo          # offline proof
python3 -m signalsniper doctor        # SEC connectivity
python3 -m signalsniper preflight     # account, session, feed
python3 -m signalsniper watch         # alert-only; --live to arm
python3 -m signalsniper flatten       # panic (needs explicit ALPACA_PAPER)

python3 tools/validate.py --selftest  # lag study, no network
python3 tools/validate.py --primary AAPL --feed sip   # THE important one
python3 tools/soak.py                 # 500-filing 16:05 burst
python3 tools/rehearse.py             # full-session replay
python3 tools/bench.py                # latency budget
```

## Open threads

1. **`tools/validate.py` has never been run against real data.** It is the single
   highest-value action available and free (Alpaca's free-tier SIP restriction is
   a recency gate, not an access gate). It decides whether the second-order path
   has any basis at all.
2. **Nothing here has touched a live feed.** All 293 tests are fixtures. Expect
   first contact to find integration bugs; two of the worst ones already found
   were exactly that shape.
3. EDGAR index-page enrichment (`EdgarEnricher`) is unverified against live
   sec.gov — the build environment blocks it.
4. Consensus estimates unwired (`parse/extract.py::surprise_bps` is ready).
5. No options support.

## What not to build

Short-dated options are the only instrument where a small account reaches large
returns in a day. Deliberately unsupported: long options into an earnings print
is a well-documented negative-EV trade (IV crush), and adding it on top of a
refuted thesis would be responsive rather than useful.

## Context this repo does not contain

Two research workflows (16 agents, ~1.4M tokens) produced the findings in
`docs/VERIFICATION.md`. That document is the durable summary — the raw agent
transcripts are session-local and gone. If a claim in the code comments cites a
figure, `VERIFICATION.md` is the source, not a re-derivation.
