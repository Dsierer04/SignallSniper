# Going live

## Read this first

**This system has never touched a live byte.** Every one of the 155 tests runs
against fixtures. It has never parsed a real EDGAR response, never held a real
websocket open, never had an order rejected by a real broker. That is not false
modesty — it's the actual state, and it determines the sequence below.

Code that passes tests and code that survives contact with a live feed are
different things. The gap is always the boring stuff: an unexpected XML
namespace, a symbol Alpaca calls `BRK.B` and SEC calls `BRK-B`, a websocket that
drops at 16:04:58, a rate limiter that was fine at 3 req/s and gets you a 429 at
16:05:01 when four hundred 8-Ks land in the same second.

You cannot find those by reading. You find them by running it.

## What I can't do for you

Not through Claude in Chrome, not through anything else:

| Step | Why it's yours |
|---|---|
| Open/fund a brokerage account | Requires your SSN, identity verification, and money movement. I should not be driving a browser through KYC on your behalf, and you shouldn't want me to. |
| Buy the market data subscription | Payment details. Same reason. |
| Generate API keys | They authenticate as *you*. Create them in the dashboard yourself. |
| Run the daemon | It has to live on your machine. This container is ephemeral, network-restricted, and dies when the session ends. |
| Flip `LIVE_TRADING=1` | Sending real orders is your decision, made with your money. |

What I *can* do: everything up to that line, plus debug whatever the first live
run throws — which is the part that's actually going to need doing.

Even if a browser-automation tool were wired into this session, the answer on
account creation and funding would be the same. Those are decisions with your
name and money attached.

---

## The blocking discovery

**Alpaca does not accept bracket orders in extended hours.** After-hours is
limit-only, DAY or GTC, `extended_hours=true`. Bracket and OCO are regular-hours
only.

The 16:05 AAPL/AMZN propagation trade — the entire reason this system exists —
happens in extended hours. So:

- In regular hours, `submit()` sends a bracket order and the stop lives at the
  broker. Your process can die and the stop still fires.
- **After 16:00, there is no broker-side stop.** The stop lives in this process.
  If it crashes, if your laptop sleeps, if your wifi drops — you are naked short
  a name that just moved 4%.

This is handled explicitly, not hidden:

- `ALLOW_EXTENDED=1` is a *separate* opt-in from `LIVE_TRADING=1`. Both required.
- `Fill.stop_is_client_side` is returned on every submission.
- The runner tracks those tickers and **flattens them on shutdown** before tearing
  down, because an unprotected position outlives the process holding its stop.
- If the flatten itself fails, it logs CRITICAL telling you to close manually.

If you trade the 16:05 window, run on a machine that will not sleep, on wired
internet, and keep the broker UI open in a tab.

---

## Sequence

### Step 1 — tonight, no money (30 min)

```bash
git checkout claude/trading-signal-speed-cr039h
pip install -r requirements.txt
export SEC_USER_AGENT="Dylan Sierer dylan@buildmind.tech"

python3 -m pytest -q          # 155 passed
python3 -m signalsniper demo  # offline proof
python3 -m signalsniper doctor
```

`doctor` is the first real network call this code has ever made. **Expect it to
find something.** If EDGAR returns a shape the parser doesn't like, that's the
point of running it now instead of at 16:04.

### Step 2 — accounts (yours, ~20 min)

1. **Alpaca** — sign up, get **paper** keys first from the dashboard.
2. **Market data.** The decision that actually matters:
   - Alpaca `iex` is free and is **~2% of consolidated volume**. After-hours
     prints are too thin to reference against. On IEX the 16:05 window produces
     confident nonsense — the engine will compute "CRUS hasn't moved" when really
     nobody printed on IEX.
   - Alpaca `sip` (paid) or a Massive real-time plan gives you the full tape.
   - **If you don't buy this, skip the 16:05 window entirely.** Trade the
     09:30–16:00 session where IEX is merely bad rather than actively misleading.
3. Check **shorting is enabled**. Half the second-order signals are shorts —
   `SHORT CRUS` in the demo is the canonical one. An account that can't short
   trades half the system.

```bash
export ALPACA_API_KEY=...
export ALPACA_SECRET_KEY=...
export ALPACA_PAPER=true
export ALPACA_FEED=sip     # or iex, and then skip the earnings window
export EQUITY=25000        # set honestly — everything sizes off this

python3 -m signalsniper preflight
```

`preflight` checks session legality, feed quality, PDT status, shorting, and
whether `EQUITY` matches the actual account. Run it again at 15:00 tomorrow.

### Step 3 — the PDT problem

**This strategy is same-day round trips.** Under $25k equity, you get 3 day
trades per rolling 5 business days before your account is locked to closing-only.

Four signals on Thursday and you're done trading until the following week. If
you're under $25k, either hold overnight (a completely different risk profile —
these are event-driven intraday theses) or accept ~3 trades and pick them
carefully. `preflight` warns you; it can't fix it.

### Step 4 — paper, live data (tomorrow, 08:00)

```bash
export LIVE_TRADING=1
python3 -m signalsniper watch --live      # PAPER keys still set
```

This is the real first test: live EDGAR, live tape, real order submission, fake
money. Watch for:

- Feed errors climbing in the heartbeat, or SEC 429s (rate limited = late = worse
  than absent)
- Signals on names where the linkage note doesn't describe a real exposure
- Broker rejections — wrong symbol format, PDT, shorting, insufficient buying power

**Trade the 08:30 macro window on paper.** It's a cheap, observable calibration:
do the `MACRO_RATES` alerts fire on names that then actually move? If they're
noise on a window you can watch for free, that tells you something important
before 16:05.

### Step 5 — real money, only if step 4 was clean

Swap to live keys, `ALPACA_PAPER=false`. Then **cut `EQUITY` to a quarter of
what you plan to trade** for the first session. Not because the math is wrong,
but because the first live session finds the integration bugs, and you want to
find them on a quarter position.

```bash
export ALPACA_PAPER=false
export EQUITY=<one quarter of real>
python3 -m signalsniper preflight    # confirm it says *** LIVE ***
python3 -m signalsniper watch --live
```

Both `LIVE_TRADING=1` and `--live` are required — two independent switches, so
neither a stale shell export nor a stray flag can arm it alone.

### The panic button

```bash
python3 -m signalsniper flatten
```

Cancels every order, closes every position, and tells you if anything is still
open. Know this command before you need it.

---

## My honest read on tomorrow

The system is sound and the thesis is real. But going from "never made a network
call" to "real money on the two biggest earnings prints of the quarter" in under
24 hours is compressing three separate risks into one session: unproven code,
unproven calibration, and the highest-volatility window of the month.

If tomorrow's the day, **paper it through 08:30 and the regular session, and make
the 16:05 call at 15:45 based on what you actually saw.** If the alerts looked
sane all day and preflight is clean, take it small. If anything was off, watch
the window and log it — the same setup recurs every quarter, and the linkage
betas you calibrate tomorrow are worth more than one session's P&L.

That's my recommendation, not a gate. Your call, your money.

## After the close, regardless

Log every alert: took it or didn't, and what the name did over the next 30
minutes. **The betas in `market/linkage.py` are priors read off disclosed
revenue-concentration figures — not calibrated truth.** One night of honest
logging beats any amount of further feature work.
