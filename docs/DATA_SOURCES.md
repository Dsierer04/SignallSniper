# News and market data, ranked by how fast you can actually get it

The question isn't "what's the fastest feed" — it's "what's the fastest feed
*whose consumers aren't already faster than me*." Those are different questions
and the second one is the only one that pays.

## The latency ladder

| Tier | Path | Typical delay | Who else is there |
|---|---|---|---|
| 0 | Exchange colo, direct wire feeds | microseconds | HFT only. Not you, not ever. |
| 1 | SEC PDS (paid dissemination), BusinessWire/PRNewswire direct XML | 10–500ms | Prop shops, quant funds |
| 2 | **SEC EDGAR public index, company IR pages** | **1–15s** | **Nearly empty on small caps** |
| 3 | Newswire public RSS, broker news APIs | 15–90s | Retail algos |
| 4 | Financial news sites, aggregators | 1–10min | Everyone |
| 5 | Reddit / X / Discord | minutes–hours | Lagging price entirely |

**Tier 2 is the whole game.** It's public, free, structured, timestamped, and on
the long tail of issuers there is genuinely almost nobody parsing it in real
time. That's what `feeds/edgar.py` targets.

Tier 1 is buyable if this works — SEC's PDS feed and direct newswire XML are real
products with real price tags. Don't buy them until Tier 2 has proven profitable,
because they only compress a latency that isn't currently your binding constraint.

## Primary sources (build on these)

### SEC EDGAR — the source of record

```
https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&count=40&output=atom
```

- **Rate limit: 10 req/s per IP, and a declaring `User-Agent` with real contact
  info is mandatory.** Anonymous polling gets throttled then blocked. Getting
  blocked at 16:05 on earnings day is the dumbest possible own goal — the
  `TokenBucket` in `feeds/base.py` spends 6/s and leaves headroom deliberately.
- Conditional GET (`ETag` / `If-Modified-Since`) makes an unchanged poll a
  bodiless 304. That's what makes 1s polling cheap.
- Dedupe on **accession number**, never on URL or title.
- The forms that actually pay: `8-K` (item codes are everything), `424B5`
  (dilutive offering pricing — reliably negative), `SC 13D` (activist stake —
  reliably positive), `NT 10-K/Q` (late filing = trouble), `25-NSE` (delisting).

### Company IR pages — the underrated one

A press release hits the issuer's own IR page at the same instant it hits the
wire. Almost nobody polls IR pages directly; they wait for an aggregator to pick
it up, which costs 15–90 seconds. Most IR sites expose RSS.

Wire them in pre-attributed so there's no ticker-resolution step at all:

```python
ir_feeds = {"acme-ir": ("https://ir.acme.com/rss/news-releases.xml", ("ACME",))}
await runner.run(quote_source=source, ir_feeds=ir_feeds)
```

### Government macro releases

BLS, BEA, and Treasury publish on a **published schedule to the second**. You
can't beat the algos to the number, but you can have the parser pre-armed and the
cross-asset propagation precomputed (`MACRO_RATES` links in `market/linkage.py`).

## Market data — the part people get wrong

Your reference price quality *is* your edge quality. The second-order logic
computes "how far has CRUS moved since the AAPL print" — a bad tape makes that
number meaningless.

| Feed | Cost | Coverage | Verdict |
|---|---|---|---|
| Alpaca `iex` | free | ~2% of consolidated volume | **Not usable after hours.** Thin prints, unrepresentative spreads. |
| Alpaca `sip` | paid | full consolidated tape | Minimum viable for the 16:05 window |
| Massive (ex-Polygon) | paid tiers | full tape + trades | Best free-tier historical; real-time needs a paid plan |
| Broker-embedded (IBKR etc.) | varies | good | Fine if you're already there |

**If you run the 16:05 AAPL/AMZN window on the IEX feed, the reference prices
will be garbage and the engine will produce confident nonsense.** Budget for SIP
or don't run that window.

## Why the original `main.py` was demoted

The repo started as a Reddit sentiment scraper. It's kept at
`legacy/reddit_sentiment.py` as an optional low-priority input, not as the core,
for four reasons — each of which is a general lesson:

1. **It scraped once at startup and never again.** No `while` loop. A "real-time
   sentiment tracker" that reads the world exactly one time.
2. **Substring ticker matching.** `if ticker in full_text.upper()` matches "AMC"
   inside "AMC theaters", "ALL" inside "all of it", "ON" inside "one". The
   replacement (`runner._extract_cashtags`) requires `$TICKER` or `(NASDAQ: X)`
   and filters against the configured universe.
3. **A general-purpose sentiment model on financial text.** `pipeline("sentiment-analysis")`
   loads a model trained on product reviews. "Company announces restructuring
   and impairment charge" is not a sentiment problem, it's a *structure* problem
   — and the 8-K item code already answers it, in microseconds instead of 200ms.
4. **Reddit lags price.** By the time a liquid name trends on r/wallstreetbets,
   the move is done. Social data is a *thesis generator* for pre-market work, not
   a trade trigger.

Where social data does have value: illiquid microcaps where retail flow is the
marginal buyer, and as a pre-market screen for what's about to be crowded. Both
are prep, not execution.
