"""Runtime configuration, environment-driven.

SEC *requires* a declaring User-Agent with real contact info. Requests without
one get throttled or blocked, and getting your IP blocked at 16:05 on an earnings
day is a self-inflicted wound. Set SEC_USER_AGENT before you run anything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _b(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    # --- identity ---------------------------------------------------------
    sec_user_agent: str = field(
        default_factory=lambda: os.getenv("SEC_USER_AGENT", "")
    )

    # --- feeds ------------------------------------------------------------
    #: SEC permits 10 req/s per IP. We spend 6 and leave headroom.
    sec_rate: float = field(default_factory=lambda: _f("SEC_RATE", 6.0))
    edgar_forms: tuple[str, ...] = ("8-K", "424B5", "SC 13D", "6-K")
    edgar_interval: float = field(default_factory=lambda: _f("EDGAR_INTERVAL", 1.0))
    wire_interval: float = field(default_factory=lambda: _f("WIRE_INTERVAL", 2.0))
    wire_rate: float = field(default_factory=lambda: _f("WIRE_RATE", 4.0))
    enable_wires: bool = field(default_factory=lambda: _b("ENABLE_WIRES", True))

    # --- market data ------------------------------------------------------
    alpaca_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    alpaca_secret: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    alpaca_feed: str = field(default_factory=lambda: os.getenv("ALPACA_FEED", "iex"))
    alpaca_paper: bool = field(default_factory=lambda: _b("ALPACA_PAPER", True))
    polygon_key: str = field(default_factory=lambda: os.getenv("POLYGON_API_KEY", ""))

    # --- risk -------------------------------------------------------------
    equity: float = field(default_factory=lambda: _f("EQUITY", 25_000.0))
    risk_per_trade: float = field(default_factory=lambda: _f("RISK_PER_TRADE", 0.01))
    max_daily_loss: float = field(default_factory=lambda: _f("MAX_DAILY_LOSS", 0.03))
    max_concurrent: int = field(default_factory=lambda: _i("MAX_CONCURRENT", 4))

    # --- execution --------------------------------------------------------
    #: Nothing sends an order unless this is explicitly turned on.
    live: bool = field(default_factory=lambda: _b("LIVE_TRADING", False))
    #: Extended hours cannot carry a broker-side stop (Alpaca rejects bracket
    #: orders outside regular hours). Trading there means your process IS the
    #: stop, so it must be opted into separately from LIVE_TRADING.
    allow_extended: bool = field(default_factory=lambda: _b("ALLOW_EXTENDED", False))

    # --- watchlist --------------------------------------------------------
    #: Names we stream quotes for. Second-order signals are impossible without a
    #: tape on the *linked* names, so this must include the whole complex, not
    #: just the issuers you care about.
    watchlist: tuple[str, ...] = ()

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.sec_user_agent or "@" not in self.sec_user_agent:
            problems.append(
                "SEC_USER_AGENT must be set to 'Your Name your@email.com' -- "
                "SEC blocks anonymous polling"
            )
        if self.sec_rate > 9.0:
            problems.append(f"SEC_RATE {self.sec_rate} risks an IP block; keep it under 9")
        if self.live and not (self.alpaca_key and self.alpaca_secret):
            problems.append("LIVE_TRADING is on but no broker credentials are set")
        if self.live and self.alpaca_paper:
            problems.append("LIVE_TRADING is on but ALPACA_PAPER is also on -- pick one")
        if self.risk_per_trade > 0.05:
            problems.append(f"RISK_PER_TRADE {self.risk_per_trade:.0%} is above 5% per trade")
        if self.live and self.allow_extended:
            problems.append(
                "LIVE_TRADING + ALLOW_EXTENDED: extended-hours positions have NO "
                "broker-side stop. If this process dies you are unprotected."
            )
        if self.live and self.alpaca_feed == "iex":
            problems.append(
                "LIVE_TRADING on the 'iex' feed: ~2% of consolidated volume makes "
                "reference prices unreliable, especially after hours"
            )
        return problems


#: Default streaming universe for 2026-07-30: the two issuers plus every name
#: the linkage graph can propagate into. Quote coverage on the *linked* names is
#: what makes the second-order path work.
JUL30_WATCHLIST: tuple[str, ...] = (
    # issuers
    "AAPL", "AMZN",
    # Apple hardware complex (post fact-check; LITE/JBL/QCOM/AVGO/FN/QRVO removed
    # -- see market/linkage.py for the filing citations behind each removal)
    "CRUS", "SWKS", "GLW", "COHR", "TXN",
    # Apple services complex
    "GOOGL", "APP", "U",
    # AWS / datacenter complex (ANET/SMCI/DDOG/SNOW/MDB/NET removed)
    "MRVL", "MSFT", "VRT", "ETN", "GEV",
    # advertising
    "TTD", "META", "PINS",
    # macro complex for the 08:30 GDP/PCE print
    "SPY", "QQQ", "IWM", "TLT", "XLU", "XLRE", "KRE", "XHB", "GLD",
)


def load(**overrides) -> Config:
    cfg = Config(**overrides)
    if not cfg.watchlist:
        cfg.watchlist = JUL30_WATCHLIST
    return cfg
