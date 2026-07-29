"""Second-order propagation graph.

READ THIS BEFORE TRUSTING ANYTHING BELOW.

This module was built on the thesis that when a mega cap prints, its
economically-linked names reprice on a lag of MINUTES, because the read-through
requires human inference. A literature review refuted that at this timescale:

  * Cohen & Frazzini (2008) and Menzly & Ozbas (2010) -- the papers the thesis
    leans on -- are MONTHLY-rebalance, monthly-holding studies (~150bp/month).
    Neither contains daily analysis, let alone intraday.
  * C&F's own mechanism evidence points the wrong way: the predictability
    concentrates around the LINKED firm's own subsequent earnings announcement,
    not in the minutes after the primary's.
  * The high-frequency work that does measure seconds-scale behaviour finds
    linked-firm repricing is same-session and effectively simultaneous, driven
    by machines and by sector-ETF arbitrage that propagates mechanically.
  * The delayed-participant window collapsed from ~10 seconds (pre-2016) to
    roughly zero.

So the "human inference bottleneck" this graph was built to exploit has largely
been automated away. The structure is kept because it is still the right shape
for a *measured* edge -- but every link is now a hypothesis to be tested by
`tools/validate.py`, not a claim to trade on. `lag_capture` is None on all of
them, `Link.verified` is False, and the engine discounts them accordingly.

Two design points that remain sound:

1. **Channels.** An event is not a scalar. An AAPL Services beat should propagate
   to the app-economy and to Google (TAC), and should *not* propagate to RF
   front-end suppliers. Edges only fire when the event touched their channel.

2. **Polarity.** Some links are genuinely inverse -- Amazon's advertising
   strength is a headwind for The Trade Desk. But note that the *original*
   inverse edges here (SHOP, ETSY, W, TGT on retail) were empirically backwards:
   SHOP-AMZN correlation is POSITIVE at every horizon (+0.45 3m, +0.42 1y,
   +0.59 5y). Consumer-discretionary and market beta dominate competitive
   substitution. A hard-coded inverse beta there would have lost systematically.
   That is exactly the kind of plausible-sounding link that fact-checking kills.

Betas are read-through sensitivity, not price correlation: "if the primary moves
1%, how much of that is attributable to this name's exposure". Every one below
has been checked against the most recent 10-K customer-concentration disclosure;
the removals are recorded inline so they are not casually re-added.
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

    #: Fraction of this name's post-event move that was still available 5 minutes
    #: after the primary printed, measured over historical events by
    #: `tools/validate.py`. **None means UNVERIFIED** -- the link is an economic
    #: story that has never been checked against what the tape actually did.
    #:
    #: This is deliberately separate from `beta`, because they answer different
    #: questions and a high beta does not imply a tradeable one. Two names can
    #: have identical exposure to the primary while one reprices in the first
    #: minute and the other takes half an hour. Only the second is a trade; beta
    #: alone cannot tell you which is which.
    lag_capture: float | None = None

    #: Rate at which this name had NO prints in the first 5 minutes after past
    #: events. High values mean "hasn't moved" usually meant "hasn't traded".
    dead_rate: float | None = None

    @property
    def verified(self) -> bool:
        return self.lag_capture is not None

    @property
    def tradeable_lag(self) -> bool:
        """Cleared the empirical bar. False for anything unverified."""
        if self.lag_capture is None:
            return False
        if self.dead_rate is not None and self.dead_rate > 0.35:
            return False
        return self.lag_capture >= 0.50

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
    # Betas below were fact-checked against the most recent 10-K customer
    # concentration disclosures. Five of the original eleven were refuted
    # outright and removed; the table had been calibrated around 2021-2023 and
    # missed the AI/datacom re-rating that now drives LITE, COHR, FN and AVGO.
    Link("AAPL", "CRUS", 0.85, CH_IPHONE, 1, 90,
         "~90% of revenue from Apple audio codecs (10-K confirmed)"),
    Link("AAPL", "SWKS", 0.65, CH_IPHONE, 1, 90,
         "~65-68% Apple FY2025 (10-K); content share declining, trend not static"),
    Link("AAPL", "GLW", 0.15, CH_IPHONE, 1, 150,
         "Apple ~10-16%, not the ~25% previously assumed; Optical Comms (38% of "
         "segment sales) dominates the stock"),
    Link("AAPL", "COHR", 0.10, CH_IPHONE, 1, 180,
         "Apple VCSEL agreement is real but Datacenter & Comms is ~65% of revenue; "
         "trades on AI datacom, not Apple"),
    Link("AAPL", "TXN", 0.10, CH_IPHONE, 1, 240,
         "diversified analog, thin read (confirmed)"),

    # REMOVED after fact-check, kept here as a record so they are not re-added:
    #   QRVO  Apple is 50% of revenue (FY2026 10-K, higher than the 45% assumed),
    #         but the name is pinned by a pending merger. A merger-arb target does
    #         not respond to Apple headlines with a supply-chain beta.
    #   LITE  Apple is under 10% and shrinking. The whole Industrial Tech segment
    #         (all VCSEL/3D sensing) is ~11.8% of revenue. Its two >10% customers
    #         are unnamed and neither is plausibly Apple. Driven by AI datacenter
    #         optics.
    #   JBL   Apple exposure was largely sold to BYD Electronic in 2023. The one
    #         disclosed >10% customer (16%) sits in Intelligent Infrastructure --
    #         cloud/datacenter, not consumer hardware.
    #   QCOM  Terminal, decaying customer: Qualcomm expects to supply modems for
    #         only ~20% of iPhones in 2026, heading to zero by 2027. A static beta
    #         would keep firing on a relationship that is ending.
    #   AVGO  The ~20% figure is from FY2022/23 and is stale by two years of
    #         extreme mix shift; FY2025 AI semiconductor revenue alone is $20B.
    #   FN    Apple is not a disclosed customer at all. NVIDIA 27.6% and Cisco
    #         18.2% are its only >10% customers. The link had no factual basis.

    # --- Apple services ------------------------------------------------------
    Link("AAPL", "GOOGL", 0.20, CH_SERVICES, 1, 120,
         "TAC: Google pays Apple for default search placement"),
    Link("AAPL", "APP", 0.15, CH_SERVICES, 1, 180, "mobile ad-tech ecosystem read"),
    Link("AAPL", "U", 0.15, CH_SERVICES, 1, 210, "mobile app/ads ecosystem read"),

    # --- Amazon: AWS / custom silicon ---------------------------------------
    # Of eighteen originally encoded AMZN links, MRVL was the only one that
    # survived fact-check as a documented, Amazon-specific economic exposure.
    Link("AMZN", "MRVL", 0.30, CH_AWS, 1, 130,
         "custom silicon and optics for AWS (confirmed); note Trainium3 socket "
         "share is contested with Alchip and Amazon's own Annapurna"),
    Link("AMZN", "MSFT", 0.20, CH_AWS, 1, 90,
         "Azure read-across from the AWS growth rate"),
    Link("AMZN", "GOOGL", 0.20, CH_AWS, 1, 90, "GCP read-across"),

    # REMOVED after fact-check:
    #   ANET  Arista's >10% customers are Microsoft (26% of 2025 revenue) and
    #         Meta (16%). AWS designs its own switches and runs white-box; it is
    #         not a disclosed concentration customer. The 0.35 beta was a
    #         narrative. ANET's real event exposure is MSFT and META.
    #   AVGO  Broadcom is not Amazon's Trainium partner -- that is Marvell,
    #         Alchip and Annapurna. The stated causal mechanism was simply wrong.
    #   SMCI  AWS procures ODM-direct (Foxconn, Quanta, Wistron, Inventec,
    #         Celestica). Supermicro's demand is the neocloud/AI-lab channel,
    #         structurally the segment that does NOT use ODM-direct.
    #   DDOG/SNOW/MDB/NET  The documented transmission is peer-to-peer, not from
    #         AWS: Datadog's print lifted Snowflake and MongoDB, DDOG -> SNOW/MDB.
    #         Conditioning on AMZN's return is the wrong axis entirely, since
    #         capex guidance routinely drives AMZN opposite to AWS fundamentals.

    # --- Amazon: datacenter capex -------------------------------------------
    # Sector exposure is documented; Amazon-specific attribution is not. Vertiv
    # does not disclose revenue from any individual hyperscaler. Betas cut hard
    # to reflect that these are AI-capex-complex names, not AMZN proxies.
    Link("AMZN", "VRT", 0.15, CH_CAPEX, 1, 150,
         "hyperscale + colo >45% of FY2024 revenue, but no Amazon-specific "
         "disclosure -- this is AI-capex beta, not an AMZN link"),
    Link("AMZN", "ETN", 0.10, CH_CAPEX, 1, 210,
         "electrical backlog +48% on datacenter demand; not Amazon-attributable"),
    Link("AMZN", "GEV", 0.10, CH_CAPEX, 1, 210,
         "datacenter equipment orders strong; not Amazon-attributable"),

    # --- Amazon: advertising -------------------------------------------------
    Link("AMZN", "TTD", 0.25, CH_ADS, -1, 150,
         "INVERSE: Amazon's ad business is TTD's biggest competitive threat, not "
         "a demand signal -- the original positive sign was backwards"),
    Link("AMZN", "META", 0.15, CH_ADS, 1, 120, "digital ad spend read, reduced weight"),
    Link("AMZN", "PINS", 0.15, CH_ADS, 1, 180, "digital ad spend read, reduced weight"),

    # REMOVED after fact-check:
    #   SHOP/ETSY/W/TGT  The inverse sign is empirically backwards. Measured
    #         SHOP-AMZN correlation is POSITIVE at every horizon (+0.45 3m,
    #         +0.42 1y, +0.59 5y) -- they co-move on consumer-discretionary and
    #         market beta, which dominates competitive substitution. A hard-coded
    #         inverse beta would have lost systematically.
    #   FDX/UPS  The economic exposure is real and well documented (Amazon was
    #         10.6% of UPS 2025 revenue, its largest customer, and UPS is cutting
    #         that volume >50% by June 2026). But it is mis-specified as a price
    #         beta: it should fire on Amazon logistics announcements, not on
    #         AMZN's daily return. FedEx's sign is additionally stale.

    # --- Macro: post-FOMC / GDP / PCE ---------------------------------------
    # These are index/ETF duration relationships, not supply-chain inference,
    # and are not subject to the same "already automated away" critique.
    Link("MACRO_RATES", "TLT", 0.80, CH_RATES, -1, 30, "long duration, most direct"),
    Link("MACRO_RATES", "XLRE", 0.55, CH_RATES, -1, 60, "REITs are the purest duration equity"),
    Link("MACRO_RATES", "KRE", 0.50, CH_RATES, -1, 60, "regional banks: NIM and AFS marks"),
    Link("MACRO_RATES", "XHB", 0.50, CH_RATES, -1, 90, "homebuilders track mortgage rates"),
    Link("MACRO_RATES", "IWM", 0.45, CH_RATES, -1, 60, "small caps carry floating-rate debt"),
    Link("MACRO_RATES", "XLU", 0.40, CH_RATES, -1, 60, "utilities are bond proxies"),
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
                  min_beta: float = 0.0, verified_only: bool = False) -> list[Link]:
        out = self._out.get(src.upper(), [])
        if channels is not None:
            out = [x for x in out if x.channel in channels]
        if min_beta > 0:
            out = [x for x in out if x.beta >= min_beta]
        if verified_only:
            out = [x for x in out if x.tradeable_lag]
        return out

    def coverage(self) -> dict[str, int]:
        """How much of this graph has actually been checked against the tape."""
        all_links = [l for bucket in self._out.values() for l in bucket]
        return {
            "total": len(all_links),
            "verified": sum(1 for l in all_links if l.verified),
            "tradeable": sum(1 for l in all_links if l.tradeable_lag),
        }

    def sources(self) -> list[str]:
        return sorted(self._out)

    def add(self, link: Link) -> None:
        self._out.setdefault(link.src, []).append(link)
        self._out[link.src].sort(key=lambda x: -x.beta)

    def apply_calibration(self, calib: dict) -> int:
        """Fold measurements from `tools/validate.py --emit` into the graph.

        Replaces hand-set betas with empirical ones and attaches the lag
        measurements that decide whether a link is tradeable at all. Only
        overwrites beta when the regression is `usable` -- a beta from four
        events with an r-squared of 0.1 is a number, not an improvement on a
        considered prior.

        Returns the number of links updated.
        """
        by_pair: dict[tuple[str, str], dict] = {}
        for row in calib.get("links", []):
            by_pair[(row["src"].upper(), row["dst"].upper())] = row

        updated = 0
        for src, bucket in self._out.items():
            for i, link in enumerate(bucket):
                row = by_pair.get((link.src.upper(), link.dst.upper()))
                if row is None:
                    continue
                beta = link.beta
                polarity = link.polarity
                if row.get("beta_usable") and row.get("beta") is not None:
                    measured = float(row["beta"])
                    # An empirical beta carries its own sign; keep magnitude in
                    # `beta` and sign in `polarity` so the rest of the engine's
                    # arithmetic is unchanged.
                    beta = abs(measured)
                    polarity = -1 if measured < 0 else 1
                bucket[i] = Link(
                    src=link.src, dst=link.dst, beta=beta, channel=link.channel,
                    polarity=polarity, lag_s=link.lag_s, note=link.note,
                    lag_capture=row.get("lag_capture"),
                    dead_rate=row.get("dead_rate"),
                )
                updated += 1
            bucket.sort(key=lambda x: -x.beta)
        return updated

    @classmethod
    def calibrated(cls, path: str = "calibration.json",
                   links: list[Link] | None = None) -> "LinkageGraph":
        """Build a graph, applying a calibration file if one exists."""
        import json
        import os

        g = cls(links)
        if os.path.exists(path):
            with open(path) as fh:
                g.apply_calibration(json.load(fh))
        return g


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
