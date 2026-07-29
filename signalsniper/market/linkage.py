"""Second-order propagation graph -- the actual edge in this system.

When AAPL prints at 16:05, AAPL itself reprices in milliseconds and you will
never be in front of that. But Cirrus Logic, which books ~90% of its revenue from
Apple, does *not* reprice in milliseconds. It reprices when a human thinks
"Services beat, iPhone units light -- what does that mean for the audio codec
supplier?" That thought takes minutes, and it is thin after hours.

That gap is the trade. This module encodes the map so the thought is precomputed.

Two design points that make this better than a naive correlation table:

1. **Channels.** An event is not a scalar. An AAPL Services beat should propagate
   to the app-economy and to Google (TAC), and should *not* propagate to RF
   front-end suppliers. Edges are tagged by channel and only fire when the event
   actually touched that channel.

2. **Polarity.** Some links are inverse. AMZN crushing retail is bad for SHOP's
   merchant story and bad for FDX/UPS parcel volume. Those inverse edges are
   where the least competition is, because the reflexive trade is to buy
   everything adjacent to a winner.

Betas are read-through sensitivity, not price correlation: "if the primary moves
1%, how much of that is *attributable* to this name's exposure". They are
starting priors from public revenue-concentration disclosures. Calibrate them
against your own fills.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Link:
    src: str
    dst: str
    beta: float          # read-through sensitivity, 0..1+
    channel: str         # which part of the primary's story this rides on
    polarity: int = 1    # +1 moves with the primary, -1 moves against
    lag_s: float = 120.0  # typical seconds before the crowd gets there
    note: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.src, self.dst, self.channel)


# ---------------------------------------------------------------------------
# Channels. Keep these stable -- the classifier maps release text onto them.
# ---------------------------------------------------------------------------

CH_IPHONE = "iphone_hardware"
CH_SERVICES = "services"
CH_AWS = "cloud_infra"
CH_ADS = "digital_ads"
CH_RETAIL = "ecommerce_retail"
CH_LOGISTICS = "logistics"
CH_CAPEX = "datacenter_capex"
CH_CONSUMER = "consumer_demand"
CH_RATES = "rates"


# ---------------------------------------------------------------------------
# The map. Betas sourced from disclosed customer-concentration figures.
# ---------------------------------------------------------------------------

LINKS: list[Link] = [
    # --- Apple hardware supply chain ---------------------------------------
    Link("AAPL", "CRUS", 0.85, CH_IPHONE, 1, 90, "~90% of revenue from Apple audio codecs"),
    Link("AAPL", "SWKS", 0.70, CH_IPHONE, 1, 90, "~65-70% Apple RF front-end"),
    Link("AAPL", "QRVO", 0.55, CH_IPHONE, 1, 100, "~45% Apple RF"),
    Link("AAPL", "LITE", 0.35, CH_IPHONE, 1, 150, "VCSEL arrays for Face ID"),
    Link("AAPL", "COHR", 0.25, CH_IPHONE, 1, 180, "optical/laser content"),
    Link("AAPL", "GLW", 0.30, CH_IPHONE, 1, 150, "cover glass, ~25% Apple"),
    Link("AAPL", "JBL", 0.25, CH_IPHONE, 1, 180, "contract manufacturing"),
    Link("AAPL", "QCOM", 0.25, CH_IPHONE, 1, 120, "modem, declining share"),
    Link("AAPL", "AVGO", 0.20, CH_IPHONE, 1, 120, "RF + wireless combo"),
    Link("AAPL", "TXN", 0.10, CH_IPHONE, 1, 240, "diversified analog, thin read"),
    Link("AAPL", "FN", 0.15, CH_IPHONE, 1, 210, "optical contract manufacturing"),

    # --- Apple services ------------------------------------------------------
    Link("AAPL", "GOOGL", 0.20, CH_SERVICES, 1, 120, "TAC: Google pays Apple for default search"),
    Link("AAPL", "MTCH", 0.15, CH_SERVICES, -1, 240, "App Store take-rate pressure"),
    Link("AAPL", "SPOT", 0.15, CH_SERVICES, -1, 240, "App Store economics"),
    Link("AAPL", "U", 0.20, CH_SERVICES, 1, 210, "mobile app/ads ecosystem read"),
    Link("AAPL", "APP", 0.20, CH_SERVICES, 1, 180, "mobile ad-tech ecosystem read"),

    # --- Amazon AWS ----------------------------------------------------------
    Link("AMZN", "ANET", 0.35, CH_AWS, 1, 120, "hyperscaler switching, AWS a top customer"),
    Link("AMZN", "AVGO", 0.30, CH_AWS, 1, 120, "custom ASIC (Trainium/Inferentia) + networking"),
    Link("AMZN", "MRVL", 0.30, CH_AWS, 1, 130, "custom silicon and optics"),
    Link("AMZN", "NVDA", 0.20, CH_AWS, 1, 90, "GPU demand read, but NVDA has its own cycle"),
    Link("AMZN", "VRT", 0.35, CH_CAPEX, 1, 150, "datacenter power and cooling"),
    Link("AMZN", "SMCI", 0.25, CH_CAPEX, 1, 150, "server buildout"),
    Link("AMZN", "ETN", 0.20, CH_CAPEX, 1, 210, "electrical infrastructure"),
    Link("AMZN", "GEV", 0.20, CH_CAPEX, 1, 210, "grid/power equipment"),
    Link("AMZN", "PWR", 0.15, CH_CAPEX, 1, 240, "electrical construction"),
    Link("AMZN", "DDOG", 0.30, CH_AWS, 1, 150, "cloud consumption read-through"),
    Link("AMZN", "SNOW", 0.25, CH_AWS, 1, 150, "cloud consumption read-through"),
    Link("AMZN", "MDB", 0.25, CH_AWS, 1, 165, "cloud consumption read-through"),
    Link("AMZN", "NET", 0.20, CH_AWS, 1, 180, "edge/cloud consumption"),
    Link("AMZN", "MSFT", 0.25, CH_AWS, 1, 90, "Azure read-across from AWS growth rate"),
    Link("AMZN", "GOOGL", 0.25, CH_AWS, 1, 90, "GCP read-across"),

    # --- Amazon retail / logistics (the inverse edges) -----------------------
    Link("AMZN", "SHOP", 0.25, CH_RETAIL, -1, 180, "merchant share competition"),
    Link("AMZN", "ETSY", 0.20, CH_RETAIL, -1, 210, "marketplace share"),
    Link("AMZN", "W", 0.20, CH_RETAIL, -1, 210, "online goods share"),
    Link("AMZN", "TGT", 0.15, CH_RETAIL, -1, 240, "general merchandise share"),
    Link("AMZN", "FDX", 0.20, CH_LOGISTICS, -1, 210, "Amazon in-housing parcel volume"),
    Link("AMZN", "UPS", 0.25, CH_LOGISTICS, -1, 200, "Amazon volume insourcing"),
    Link("AMZN", "CHRW", 0.12, CH_LOGISTICS, -1, 270, "freight brokerage"),

    # --- Amazon advertising --------------------------------------------------
    Link("AMZN", "TTD", 0.30, CH_ADS, 1, 150, "programmatic demand read"),
    Link("AMZN", "META", 0.20, CH_ADS, 1, 120, "digital ad spend read"),
    Link("AMZN", "PINS", 0.20, CH_ADS, 1, 180, "digital ad spend read"),
    Link("AMZN", "RDDT", 0.20, CH_ADS, 1, 180, "digital ad spend read"),

    # --- Macro: post-FOMC / GDP / PCE ---------------------------------------
    Link("MACRO_RATES", "XLU", 0.40, CH_RATES, -1, 60, "utilities are bond proxies"),
    Link("MACRO_RATES", "XLRE", 0.55, CH_RATES, -1, 60, "REITs are the purest duration equity"),
    Link("MACRO_RATES", "IWM", 0.45, CH_RATES, -1, 60, "small caps carry floating-rate debt"),
    Link("MACRO_RATES", "KRE", 0.50, CH_RATES, -1, 60, "regional banks: NIM and AFS marks"),
    Link("MACRO_RATES", "XHB", 0.50, CH_RATES, -1, 90, "homebuilders track mortgage rates"),
    Link("MACRO_RATES", "TLT", 0.80, CH_RATES, -1, 30, "long duration, most direct"),
    Link("MACRO_RATES", "GLD", 0.30, CH_RATES, -1, 90, "real-rate sensitivity"),
]


class LinkageGraph:
    """Adjacency with channel filtering. Lookup is a dict hit -- nanoseconds."""

    def __init__(self, links: list[Link] | None = None) -> None:
        self._out: dict[str, list[Link]] = {}
        for link in links if links is not None else LINKS:
            self._out.setdefault(link.src, []).append(link)
        for bucket in self._out.values():
            bucket.sort(key=lambda x: -x.beta)

    def neighbors(self, src: str, channels: frozenset[str] | None = None,
                  min_beta: float = 0.0) -> list[Link]:
        out = self._out.get(src.upper(), [])
        if channels is not None:
            out = [x for x in out if x.channel in channels]
        if min_beta > 0:
            out = [x for x in out if x.beta >= min_beta]
        return out

    def sources(self) -> list[str]:
        return sorted(self._out)

    def add(self, link: Link) -> None:
        self._out.setdefault(link.src, []).append(link)
        self._out[link.src].sort(key=lambda x: -x.beta)


# ---------------------------------------------------------------------------
# Channel detection: which part of the story did this release actually touch?
# ---------------------------------------------------------------------------

import re  # noqa: E402  (kept local to this section for readability)

_CHANNEL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (CH_IPHONE, re.compile(r"\biPhone\b|\bhardware\b|\bunit\s+sales\b|\bwearables\b|\bMac\b|\biPad\b", re.I)),
    (CH_SERVICES, re.compile(r"\bServices\b|\bApp\s+Store\b|\bsubscription\b|\binstalled\s+base\b", re.I)),
    (CH_AWS, re.compile(r"\bAWS\b|\bAmazon\s+Web\s+Services\b|\bcloud\b|\bAzure\b|\bGoogle\s+Cloud\b", re.I)),
    (CH_ADS, re.compile(r"\badvertis\w+\b|\bad\s+revenue\b|\bsponsored\b", re.I)),
    (CH_RETAIL, re.compile(r"\bonline\s+stores?\b|\bthird[- ]party\s+seller\b|\bretail\b|\bmarketplace\b", re.I)),
    (CH_LOGISTICS, re.compile(r"\bfulfillment\b|\bshipping\b|\blogistics\b|\bdelivery\s+speed\b", re.I)),
    (CH_CAPEX, re.compile(r"\bcapital\s+expenditure\b|\bcapex\b|\bdata\s*cent(?:er|re)\b|\binfrastructure\s+invest", re.I)),
    (CH_CONSUMER, re.compile(r"\bconsumer\s+(?:demand|spending)\b|\bdiscretionary\b", re.I)),
    (CH_RATES, re.compile(r"\bfederal\s+funds\b|\bFOMC\b|\bPCE\b|\bGDP\b|\binflation\b|\bjobless\s+claims\b", re.I)),
]

#: When a release is too terse to name a channel, fall back to the issuer's
#: dominant channels rather than propagating to everything.
DEFAULT_CHANNELS: dict[str, frozenset[str]] = {
    "AAPL": frozenset({CH_IPHONE, CH_SERVICES}),
    "AMZN": frozenset({CH_AWS, CH_RETAIL, CH_ADS, CH_CAPEX}),
    "MACRO_RATES": frozenset({CH_RATES}),
}


def detect_channels(text: str, primary: str = "") -> frozenset[str]:
    hits = {ch for ch, pat in _CHANNEL_PATTERNS if pat.search(text)}
    if hits:
        return frozenset(hits)
    return DEFAULT_CHANNELS.get(primary.upper(), frozenset())
