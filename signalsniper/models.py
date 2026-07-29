"""Core data types.

Everything that flows through the system carries monotonic nanosecond stamps so
the latency budget is measurable end to end rather than guessed at. `t_source`
is when the venue/publisher says the thing happened; `t_ingest` is when we first
saw the bytes; `t_decide` is when a signal came out the other side.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def mono_ns() -> int:
    """Monotonic clock, for DURATIONS measured inside this process.

    Latency, signal age, position age, cooldowns. Never wall time for these --
    an NTP step mid-session would corrupt every delta.
    """
    return time.monotonic_ns()


def epoch_ns() -> int:
    """Wall-clock nanoseconds since the Unix epoch, for EVENT TIMES.

    Anything compared across sources must use this. A venue timestamps a quote
    in epoch nanoseconds; if we stamp the filing that quote is compared against
    with a monotonic clock, the two live in different number spaces (monotonic
    is ~5e12 seconds-since-boot, epoch is ~1.8e18) and every comparison silently
    returns garbage.

    That exact bug shipped here: `move_bps_since(t_event)` bisected a tape full
    of epoch stamps with a monotonic event time, found index -1 every single
    time, and returned the OLDEST price in the tape as the reference. The whole
    "how much is already priced in" gate was computed against a stale price on
    every event. All 236 tests passed because the fixtures built tapes from the
    same monotonic clock they stamped events with -- the mismatch only existed
    once a real venue quote entered the tape.

    THE RULE: cross-source comparison -> epoch_ns(). In-process duration ->
    mono_ns(). Do not mix them, and do not add a timestamp field without
    deciding which of the two it is.
    """
    return time.time_ns()


#: Backwards-compatible alias. Prefer the explicit names above -- the ambiguity
#: of "now" is what allowed the two clocks to be mixed in the first place.
now_ns = mono_ns


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Direction(Enum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0

    def flip(self) -> "Direction":
        if self is Direction.LONG:
            return Direction.SHORT
        if self is Direction.SHORT:
            return Direction.LONG
        return Direction.NEUTRAL


class EventKind(Enum):
    """What actually happened. Ordering is not significance -- see materiality."""

    EARNINGS = "earnings"
    GUIDANCE = "guidance"
    MERGER = "merger"
    MATERIAL_AGREEMENT = "material_agreement"
    OFFERING = "offering"  # 424B5 / dilution
    ACTIVIST_STAKE = "activist_stake"  # SC 13D
    PASSIVE_STAKE = "passive_stake"  # SC 13G
    RESTATEMENT = "restatement"  # 8-K 4.02 -- rare and violent
    AUDITOR_CHANGE = "auditor_change"  # 8-K 4.01
    IMPAIRMENT = "impairment"
    DELISTING = "delisting"
    EXEC_CHANGE = "exec_change"
    BANKRUPTCY = "bankruptcy"
    BUYBACK = "buyback"
    DIVIDEND = "dividend"
    REG_FD = "reg_fd"
    INSIDER_TRADE = "insider_trade"
    LEGAL = "legal"
    CLINICAL = "clinical"
    REGULATORY_APPROVAL = "regulatory_approval"
    MACRO = "macro"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class RawDoc:
    """A document as it came off the wire, before any interpretation."""

    source: str  # "edgar", "globenewswire", ...
    doc_id: str  # accession number / guid -- the dedupe key
    title: str
    url: str
    published: datetime | None
    body: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    #: EVENT TIME, epoch ns. This is compared against quote timestamps, so it
    #: must share their clock. See epoch_ns() for why this is not monotonic.
    t_ingest: int = field(default_factory=epoch_ns)
    #: The same instant on the monotonic clock, for latency deltas only.
    t_mono: int = field(default_factory=mono_ns)


@dataclass(slots=True)
class Event:
    """A classified, tradeable-or-not occurrence attached to an issuer."""

    doc: RawDoc
    kind: EventKind
    tickers: tuple[str, ...]
    materiality: float  # 0..1 -- how much this *should* move the name
    prior: Direction  # sign we expect before looking at price
    confidence: float  # 0..1 -- how sure the classifier is
    reasons: tuple[str, ...] = ()
    #: Monotonic -- paired with doc.t_mono, never with doc.t_ingest.
    t_classified: int = field(default_factory=mono_ns)

    @property
    def ingest_latency_us(self) -> float:
        # Monotonic minus monotonic. Subtracting t_ingest (epoch) here would
        # produce a number ~1.8e18 microseconds and look like nothing at all.
        return (self.t_classified - self.doc.t_mono) / 1_000.0

    @property
    def primary(self) -> str:
        return self.tickers[0] if self.tickers else ""


@dataclass(slots=True)
class Quote:
    ticker: str
    bid: float
    ask: float
    last: float
    volume: int = 0
    #: EVENT TIME, epoch ns -- as stamped by the venue.
    t_source: int = field(default_factory=epoch_ns)

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last

    @property
    def spread_bps(self) -> float:
        m = self.mid
        if m <= 0 or self.bid <= 0 or self.ask <= 0:
            return float("inf")
        return (self.ask - self.bid) / m * 10_000.0


@dataclass(slots=True)
class Signal:
    """The output. Carries its own expiry -- a stale edge is a liability."""

    ticker: str
    direction: Direction
    edge_bps: float  # expected move still available, in basis points
    confidence: float
    event: Event
    hop: int = 0  # 0 = the issuer itself, 1 = second-order via linkage
    ref_price: float = 0.0
    notes: tuple[str, ...] = ()
    ttl_s: float = 300.0
    #: Monotonic -- age and expiry are in-process durations.
    t_emit: int = field(default_factory=mono_ns)
    #: Monotonic snapshot of the event, so latency stays a monotonic delta.
    t_event_mono: int = 0

    @property
    def age_s(self) -> float:
        return (mono_ns() - self.t_emit) / 1e9

    @property
    def expired(self) -> bool:
        return self.age_s > self.ttl_s

    @property
    def total_latency_ms(self) -> float:
        """Wire-to-decision. Monotonic minus monotonic."""
        return (self.t_emit - self.event.doc.t_mono) / 1e6

    def describe(self) -> str:
        arrow = {Direction.LONG: "LONG", Direction.SHORT: "SHORT", Direction.NEUTRAL: "FLAT"}[
            self.direction
        ]
        tag = "primary" if self.hop == 0 else f"hop{self.hop}"
        return (
            f"{arrow} {self.ticker} [{tag}] edge={self.edge_bps:.0f}bps "
            f"conf={self.confidence:.2f} via {self.event.kind.value} "
            f"({self.total_latency_ms:.1f}ms)"
        )
