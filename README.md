# SignalSniper

Low-latency event ingestion and **second-order** signal generation for US equities.

The thesis in one line: **you will not beat colocated HFT to a headline, so don't
try — trade the propagation instead.**

When Apple prints at 16:05, AAPL reprices in under 50 milliseconds and you are
not in that race. But Cirrus Logic, which books ~90% of its revenue from Apple,
does *not* reprice in 50ms. It reprices when a human thinks "Services beat,
hardware light — what's that worth to the audio codec supplier?" That thought
takes two to ten minutes, and after-hours books are thin.

That gap is the trade. This system precomputes the thought.

## How it works

```
EDGAR / newswire / IR feeds          (1-15s from the source of record)
        |
        v
  classify  --- 8-K item codes, form priors, headline regex.  ~80us, no ML.
        |
        v
   engine   --- edge = materiality - what the market already took
        |         hop 0: the issuer itself (long tail only)
        |         hop 1: propagate the *realized* primary move through
        |                disclosed-exposure betas, trade the residual
        v
    risk    --- stop-distance sizing, daily-loss kill switch, concurrency caps
        |
        v
  alert (default)  |  order (explicit opt-in only)
```

The discipline that makes this a system rather than a news reader: a material
event is not a trade. A material event **where the price hasn't moved yet** is a
trade. The same event thirty seconds later is somebody else's exit liquidity.

## Quick start

```bash
pip install -r requirements.txt
export SEC_USER_AGENT="Your Name you@example.com"   # SEC requires this; anonymous polling gets blocked

python3 -m pytest -q               # 125 tests, no network needed
python3 -m signalsniper demo       # end-to-end proof against a synthetic tape
python3 -m signalsniper doctor     # verify credentials + SEC connectivity
python3 -m signalsniper links AAPL # inspect the second-order map
python3 -m signalsniper watch      # live, alert-only
```

`demo` shows the gate working — AAPL is *refused* because it already repriced
420bps, QRVO is refused as `overshot`, and CRUS/SWKS signal because the move that
belongs to them hasn't arrived:

```
SHORT CRUS [hop1] edge=297bps conf=0.67 via earnings (1.4ms)
      | AAPL moved -420bps
      | beta=0.85 pol=+1 ch=iphone_hardware
      | implied=-357bps actual=-60bps
      | slack=83%
      -> SELL SHORT 48 CRUS @ 103.38 stop 105.22 target 100.31
```

Nothing sends an order unless you pass `--live` *and* set `LIVE_TRADING=1`. The
risk manager still gets the final veto.

## Layout

| Path | What |
|---|---|
| `signalsniper/feeds/` | EDGAR current-filings + RSS, conditional GET, token-bucket rate limiting |
| `signalsniper/parse/` | Deterministic classifier (item codes, form priors, headline rules) + numeric extraction |
| `signalsniper/market/` | Linkage graph, rolling tape, quote sources |
| `signalsniper/signal/` | The gate and the risk manager |
| `signalsniper/runner.py` | Async wiring |
| `tools/bench.py` | Latency budget |
| `legacy/` | The original Reddit scraper, demoted — see `docs/DATA_SOURCES.md` |

## Docs

- **[docs/RUNBOOK_2026-07-30.md](docs/RUNBOOK_2026-07-30.md)** — the plan for
  tomorrow: what's playable, what isn't, and the kill conditions
- [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) — the latency ladder, and why
  Tier 2 (EDGAR + IR pages) is the only tier worth fighting over
- [docs/MCP_SETUP.md](docs/MCP_SETUP.md) — which MCP servers help, which don't,
  and why none of them belong in the hot path

## Latency

`python3 tools/bench.py`:

```
xml parse + 40 docs extracted          p50=  533us
classify_doc                           p50=   80us
on_event (classify->signals)           p50=  105us
tape.move_bps_since                    p50= 0.53us   (at 20k ticks)
size_order                             p50=    3us
```

Bytes-to-decision is well under a millisecond. **This is not where your time
goes** — SEC poll RTT is 30–120ms and broker ack is 50–300ms. Don't
micro-optimise this code; spend the effort on a VPS near the data source, a SIP
feed instead of IEX, and direct IR feeds instead of aggregators.

## Configuration

All via environment (see `.env.example`). The ones that matter:

| Var | Default | Note |
|---|---|---|
| `SEC_USER_AGENT` | — | **Required.** `"Name email@example.com"`. SEC blocks anonymous polling. |
| `SEC_RATE` | `6.0` | Requests/sec against SEC. Their cap is 10. Don't go near it. |
| `EQUITY` | `25000` | Set this honestly; everything sizes off it |
| `RISK_PER_TRADE` | `0.01` | Fraction of equity risked to the stop |
| `MAX_DAILY_LOSS` | `0.03` | Kill switch. Sticky — only an explicit resume clears it. |
| `ALPACA_FEED` | `iex` | **`iex` is ~2% of volume and unusable after hours.** Use `sip` for the earnings window. |
| `LIVE_TRADING` | `false` | Must be explicitly on, and `--live` passed |

## Known limits

- **Betas in `market/linkage.py` are priors, not truth.** They're read off
  disclosed revenue-concentration figures. Recalibrate against your own fills —
  this is the highest-value thing you can do after one session.
- The linkage graph is hand-built and currently covers the AAPL/AMZN complexes
  and the rates complex. Extending it to a new issuer is a data-entry job, and
  that's deliberate: an auto-derived correlation graph would produce spurious
  links with no economic story behind them.
- `evaluate_primary` is realistically only useful on the small/mid-cap long tail.
  On anything well covered it will correctly refuse to fire.
- Consensus estimates aren't wired in. `parse/extract.py::surprise_bps` is ready
  for them; you need an estimates source.
- No options support. The second-order thesis often expresses better in options
  after hours, where the underlying is thin.

## License

All rights reserved. See `LICENSE.txt`.

**This is trading software. It can lose money quickly. Default posture is
alert-only for a reason — paper it until you agree with the alerts.**
