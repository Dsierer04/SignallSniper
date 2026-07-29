# macOS setup — copy/paste, in order

Apple does not alias `pip`. On macOS it is always `python3 -m pip`, or plain
`pip` **only after** a virtualenv is active. That is why the earlier commands
failed.

There is a bigger macOS gotcha underneath it, so do Step 1 first.

---

## Step 1 — check your Python version (this blocks everything)

```bash
python3 -V
```

**You need 3.10 or newer.** This code uses `X | None` type syntax in dataclass
fields, which is evaluated at import time — on Python 3.9 every module fails
immediately with a `TypeError`, not a nice message.

Older macOS ships Python 3.9. If you see 3.9 or lower:

```bash
# Install Homebrew if you don't have it (paste the whole line)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Apple Silicon (M1/M2/M3/M4) — add brew to your PATH
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"

# Intel Macs use /usr/local instead:
# echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
# eval "$(/usr/local/bin/brew shellenv)"

brew install python@3.12
python3 -V     # should now say 3.12.x
```

---

## Step 2 — get the code

```bash
cd ~
git clone https://github.com/Dsierer04/SignallSniper.git
cd SignallSniper
git checkout claude/trading-signal-speed-cr039h
```

If you already cloned it, just:

```bash
cd ~/SignallSniper
git checkout claude/trading-signal-speed-cr039h
git pull
```

---

## Step 3 — virtualenv, then `pip` works normally

```bash
cd ~/SignallSniper
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt now starts with `(.venv)`. **Inside the venv, plain `pip` works.**

The venv is not optional on modern macOS: Homebrew Python refuses system-wide
installs with `error: externally-managed-environment` (PEP 668). The venv is
the supported way around it, not a workaround.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Takes ~20 seconds. Installs `httpx`, `websockets`, `pytest`, `python-dotenv`.

**Every new Terminal tab needs the venv re-activated:**

```bash
cd ~/SignallSniper && source .venv/bin/activate
```

---

## Step 4 — prove the install (no network, no keys)

```bash
python3 -m pytest -q
```

Expect `293 passed` in about 6 seconds.

```bash
python3 tools/rehearse.py
python3 tools/validate.py --selftest
```

`rehearse.py` replays a full 09:30–16:00 session so you can see what a normal
day looks like before you have money on it.

---

## Step 5 — your `.env`

```bash
cd ~/SignallSniper
cp .env.example .env
open -e .env          # opens in TextEdit; or: nano .env
```

Set these values. **`RISK_PER_TRADE=0.05` is not optional at $200** — at the 1%
default, a $2 risk budget means most orders fall under the minimum notional and
get silently dropped. Measured:

| Price | @1% risk | @5% risk |
|---|---|---|
| $2.40 | **rejected** | 66.7 sh / $160 |
| $13.08 | **rejected** | 8.8 sh / $116 |
| $45.00 | 1.85 sh / $83 | 3.56 sh / $160 |

```bash
SEC_USER_AGENT="Dylan Sierer dylan@buildmind.tech"
ALPACA_API_KEY=PK...
ALPACA_SECRET_KEY=...
ALPACA_PAPER=true
ALPACA_FEED=iex
EQUITY=200
RISK_PER_TRADE=0.05
MAX_DAILY_LOSS=0.03
MAX_CONCURRENT=1
LIVE_TRADING=false
ALLOW_EXTENDED=false
```

`.env` is gitignored. `config.load()` reads it automatically — you do not need
to `source` it for the daemon. (`tools/validate.py` reads `os.environ` directly,
so for that one: `set -a; source .env; set +a`.)

---

## Step 6 — Alpaca keys

You already have a funded account, so KYC is done and **live keys are available
to you immediately** — no 1–3 day wait.

1. **Install an authenticator app first** (Google Authenticator / Authy / 1Password).
   MFA is mandatory before the Trading API works at all. Skip it and every call
   returns 401 with no useful explanation.
2. Go to <https://app.alpaca.markets> and log in.
3. **Top-left account switcher → set to `Paper`.** The API Keys panel is scoped
   to whichever environment is selected. This is the #1 beginner mistake.
4. Home page → right sidebar → **API Keys** → **Generate New Keys**.
5. **The Secret is shown exactly once.** Have `.env` open before you click.
   Paper Key IDs start with `PK`.

Smoke test:

```bash
set -a; source .env; set +a
curl -s -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
        -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY" \
        https://paper-api.alpaca.markets/v2/account | head -c 300
```

You want JSON with ~100000 in `cash`. A 401/403 here means either wrong
environment (step 3) or MFA incomplete (step 1).

---

## Step 7 — verify, then run

```bash
python3 -m signalsniper doctor       # first real network call this code makes
python3 -m signalsniper preflight    # will say FAILED at $200 — read why
python3 -m signalsniper watch        # alert-only
```

`preflight` reporting FAILED at $200 is expected, not broken. It is telling you
shorts are impossible below $2,000.

---

## Step 8 — keep the Mac awake (do not skip this)

**A sleeping Mac is the single worst failure mode for an unattended bot.**
macOS sleeps the CPU, the network, or both — and your daemon is holding the
stop for any open position. Asleep means no stop.

```bash
caffeinate -i python3 -m signalsniper watch
```

`caffeinate -i` prevents idle sleep for as long as that command runs. It does
**not** stop sleep if you close a laptop lid — for that, run on a Mac that stays
open, or use `caffeinate -s` while on power.

Belt and braces, in System Settings → Displays → Advanced → "Prevent automatic
sleeping on power adapter when the display is off."

To leave it running after you close Terminal:

```bash
cd ~/SignallSniper && source .venv/bin/activate
nohup caffeinate -i python3 -m signalsniper watch > ~/signalsniper.log 2>&1 &
echo $!            # the PID — write this down

tail -f ~/signalsniper.log      # watch it live; Ctrl-C just stops tailing
kill <PID>                       # stop the bot
```

---

## The panic button — learn it now

```bash
cd ~/SignallSniper && source .venv/bin/activate
ALPACA_PAPER=true python3 -m signalsniper flatten
```

The explicit `ALPACA_PAPER=` is required — `flatten` refuses to run without it,
because guessing wrong on a panic button is the worst possible failure. For a
live account it is `ALPACA_PAPER=false`.

---

## What "stacking" actually looks like at $200

Set the expectation now so a quiet day does not read as a broken install:

- **Long only.** Shorts are regulatorily impossible below $2,000 equity
  (FINRA 4210(b)/Reg T). The sizer refuses them locally with
  `short_blocked_below_2k` rather than eating a broker rejection.
- **Regular hours only.** The free IEX feed's venue does not operate past
  17:00 ET, and extended-hours orders carry no broker-side stop. Leave
  `ALLOW_EXTENDED=false`.
- **One position at a time**, ~$110–160 deployed.
- **The second-order path emits nothing** until `tools/validate.py` measures a
  link. Only the EDGAR long-tail path can fire out of the box.
- **Most days produce zero signals.** In the rehearsal, nine filings produced
  two signals. That is the system working, not failing.
- **Per well-executed event: roughly $17 on a 15% move.** The daily loss
  kill switch trips at −$6 and is sticky.

At 5% risk per trade the account survives roughly 20 consecutive losers before
it is gone. Compounding from $200 is real but slow, and the kill switch will
halt a bad day early by design — that is what stops the stack from going
backwards faster than it goes forwards.
