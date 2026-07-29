# Verification results

You asked me to verify we'd make thousands tomorrow. I can't verify a future
P&L — nobody can, and any process claiming to would be lying to you. What I ran
instead was a verification of everything that *determines* whether the edge is
real: the academic basis, every encoded fact, the API contracts, and the
system's behaviour under a realistic 16:05 load.

**The headline result: the core thesis was refuted at the timescale I built it
for.** Here is the evidence, because you should be able to check my work.

---

## 1. The minutes-scale propagation thesis — REFUTED

The system was built on this claim:

> When a mega cap prints, its economically-linked names reprice on a lag of
> MINUTES, because the read-through requires human inference.

The literature does not support that. It supports a **weeks-to-months** effect,
and the work that actually measures fast timescales finds the opposite of what
this needs:

| Claim | Finding |
|---|---|
| Cohen & Frazzini (2008) documents a tradeable linked-firm lag | **REFUTED.** Monthly rebalance, one-month holding, ~150bp/month. Contains no daily analysis, let alone intraday. |
| C&F's mechanism supports an immediate post-print window | **REFUTED.** Their predictability concentrates around the *linked firm's own subsequent earnings announcement* — the news gets incorporated later, at a different event, not in the minutes after the primary's print. |
| Menzly & Ozbas supports fast cross-predictability | **REFUTED.** Monthly, industry-level, explicitly attributed to gradual diffusion among *segmented* investors. It shrinks precisely in the liquid, well-covered names a low-latency system must trade. |
| A residual fast window remains | **REFUTED.** The exploitable window for delayed participants collapsed from ~10 seconds (pre-2016) to effectively zero. A minutes-scale premise is roughly two orders of magnitude too slow. |
| The read-through requires human inference | **REFUTED.** It is machine-mediated and near-real-time; sector-ETF arbitrage propagates it mechanically. **The bottleneck the entire thesis rests on has been automated away.** |
| PEAD remains exploitable | **PARTIAL.** Largely dead in US large caps post-2006. Where it survives (microcaps, EM) it is a days-to-weeks drift. |
| These anomalies persisted post-publication | **REFUTED.** Base-rate post-publication decay is 58%, larger for high-in-sample-return predictors. Customer-momentum specifically shows loss of significance. |
| The predictability is genuine news read-through | **PARTIAL.** Only ~half is attributable to the news component; the rest is momentum commonality and contemporaneous correlation. |

**What this means concretely.** The 16:05 supplier-dislocation trade I pitched
you — AAPL prints, CRUS hasn't moved yet, short CRUS — rests on a lag that the
evidence says is not there at minutes scale. I should not have presented it as
"the actual edge in this system" without checking first. That was my error, and
it was the load-bearing claim of the whole design.

---

## 2. The linkage graph — 11 of 24 links removed, 4 betas corrected

Every link was checked against the most recent 10-K customer-concentration
disclosure. The table had been calibrated around 2021–2023 and never refreshed,
so it systematically missed the AI/datacom re-rating.

**Apple — 6 removed:**

| Link | Was | Finding |
|---|---|---|
| FN | 0.15 | **No Apple exposure at all.** NVIDIA 27.6%, Cisco 18.2% are its only >10% customers. Zero factual basis. |
| JBL | 0.25 | Apple business largely sold to BYD Electronic in 2023. Its >10% customer is in Intelligent Infrastructure. |
| LITE | 0.35 | Apple under 10% and shrinking. The whole segment holding all VCSEL/3D sensing is ~11.8% of revenue. |
| QCOM | 0.25 | Terminal customer — ~20% of iPhone modems in 2026, zero by 2027. A static beta keeps firing on an ending relationship. |
| AVGO | 0.20 | Figure is from FY2022/23, stale by two years of extreme mix shift. |
| QRVO | 0.55 | Apple is *50%*, higher than I had — but the name is pinned by a pending merger and won't trade on Apple headlines. |

Corrected: GLW 0.30→0.15, COHR 0.25→0.10, SWKS 0.70→0.65. Confirmed: CRUS 0.85, TXN 0.10.

**Amazon — only MRVL survived.** The worst two:

- **ANET 0.35 → removed.** Arista's >10% customers are **Microsoft (26%) and
  Meta (16%)**. AWS designs its own switches and runs white-box. My beta was, in
  the researcher's word, "a narrative."
- **SHOP/ETSY/W/TGT inverse links → removed.** The sign was **empirically
  backwards**. SHOP-AMZN correlation is *positive* at every horizon (+0.45 3m,
  +0.42 1y, +0.59 5y). Consumer-discretionary and market beta dominate
  competitive substitution. I described these as "where the least competition
  is." A hard-coded inverse beta there would have lost systematically.

TTD's sign was also inverted and is now correctly negative — Amazon's ad
business is TTD's biggest competitive threat, not a demand signal.

---

## 3. Broker contracts — two critical corrections

- **The PDT rule I warned you about no longer exists at Alpaca.** Replaced by an
  Intraday Margin Framework on 2026-06-04, and `daytrade_count` /
  `pattern_day_trader` were removed from `/v2/account` on 2026-07-06. My code
  read fields that aren't there and gated on a rule that no longer applies. Now
  handles both, and gates on buying power. The $2,000 margin/short minimum does
  still apply.
- **Extended hours confirmed:** limit only, `extended_hours=true`, `order_class`
  simple. Bracket/OCO rejected. 24/5 overnight is 20:00–04:00 and changes nothing
  at 16:05 — you are in after-hours, with no broker-side stop.

*Caveat:* the egress proxy here blocks `*.alpaca.markets`, so this rests on
first-party SDK source rather than the docs pages. **Confirm the PDT change in
your dashboard before relying on it.**

---

## 4. Bugs found by testing, not reasoning

**`tools/soak.py`** fires 500 filings in a burst against the real pipeline.
Throughput was fine — zero drops, 32ms under load. The *orders* were not:

```
BUY 4 SWKS @ 77.61 stop 27.56 target 161.03   (stop 6449bps)
```

A stop 64% away is not a stop. One bad tick read as a 2820bps move, the engine
turned that into an enormous "edge", and the risk manager derived the stop *from*
that edge. Nothing bounded it anywhere. Bad ticks are routine on exactly the thin
after-hours names this targets. Now defended at three independent layers (tape
outlier hold-and-confirm, engine edge/primary caps, risk stop-width refusal).

**Fill reconciliation was absent.** The system treated a submitted order as a
filled one. An unfilled limit leaves a phantom position that occupies a
concurrency slot — and when price crosses its stop, the exit path sends a
*closing* order for shares never bought, which opens a real position in the
opposite direction. Now polls order status and corrects.

---

## 4b. Calendar and sizing corrections

- **The FOMC decision has not happened.** It lands 14:00 ET *today* (Jul 29),
  Warsh's first meeting as Chair, with a genuine hike tail (6.6–27% across
  prediction markets in July). My runbook treated it as settled history. A
  hawkish surprise today reprices the whole Jul 30 tape.
- **The releases are staggered, not simultaneous.** AMZN ~16:00–16:05, AAPL
  16:30. That is *better* than I assumed — the PR cascades are sequential with
  ~30 minutes between them. What actually collides is the two 17:00 calls
  (verify AMZN's, it may be 17:30, which would remove the overlap entirely).
- **The 08:30 block is four-way, not three:** GDP + PCE + claims + ECI. Growth
  and inflation surprises land in the same tick, so single-indicator attribution
  misfires by construction.
- **AMZN's options-implied move is ~1.7x AAPL's.** `DEFAULT_MOVE_SCALE` had it
  at 1.2x, so a uniform notional cap would have taken materially more risk on
  AMZN than intended. Now 1.68x.
- Consensus, confirmed: AAPL revenue ~$108.8B / EPS ~$1.88; AMZN revenue ~$196B
  / EPS $1.82; AWS ~$40.5B at ~32% YoY (accelerating from 28%).
- MSFT / META / ARM report tonight, so Jul 30 opens on an overnight gap.

---

## 5. What survives

Not nothing:

- **The EDGAR long-tail path is untouched by this.** Being first to read an
  under-covered small-cap 8-K is a *different mechanism* — it is not
  cross-firm inference, it is document-reading speed on names with no algo
  coverage. The refutation above is specifically about second-order propagation.
  424B5 offerings, 8-K item 4.02 restatements, SC 13D activist stakes on thin
  names are all still plausible.
- **The macro/rates links** (TLT, XLRE, KRE, XHB) are index duration
  relationships, not supply-chain inference, and are not subject to the
  "automated away" critique in the same form.
- **`tools/validate.py`** is now the most valuable thing in the repo. It measures
  whether any lag exists rather than assuming one, and it would have caught this
  before I wrote 5,000 lines on top of it.
- **The engineering** — the gate, the risk manager, the bad-tick defenses, the
  reconciliation — is sound and reusable regardless of which signal feeds it.

---

## 6. My recommendation for tomorrow

**Do not put real money on the second-order trade tomorrow.** Not because of
hedging or risk-aversion, but because the specific mechanism it depends on has
been refuted and I have no measurement showing otherwise.

What I'd actually do:

1. **Run `tools/validate.py --primary AAPL --feed sip` tonight.** It measures the
   lag across 10 historical Apple prints. This is the empirical test. If the
   surviving links show real lag capture, you have evidence the literature
   doesn't — take it small tomorrow. If they don't, you've saved the money.
2. **Run alert-only tomorrow, all sessions.** Log everything. One night of honest
   data beats the entire literature review for *your* specific names.
3. **If you want to trade tomorrow anyway**, the EDGAR long-tail path is the
   defensible one — small-cap 8-Ks and 424B5 offerings where the edge is reading
   speed, not cross-firm inference.

You said you've made money on this by chance before. That's consistent with
everything above: a real distribution with real variance, where individual wins
happen and don't imply an edge. The way to find out whether there's an edge is to
measure it, and tonight you can.

The system is ready to run. The thesis is not ready to bet on.
