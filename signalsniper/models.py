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


def now_ns() -> int:
    """Monotonic clock for latency math. Never wall time -- NTP steps ruin deltas."""
    return time.monotonic_ns()


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
    t_ingest: int = field(default_factory=now_ns)


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
    t_classified: int = field(default_factory=now_ns)

    @property
    def ingest_latency_us(self) -> float:
        return (self.t_classified - self.doc.t_ingest) / 1_000.0

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
    t_source: int = field(default_factory=now_ns)

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
    t_emit: int = field(default_factory=now_ns)

    @property
    def age_s(self) -> float:
        return (now_ns() - self.t_emit) / 1e9

    @property
    def expired(self) -> bool:
        return self.age_s > self.ttl_s

    @property
    def total_latency_ms(self) -> float:
        """Wire-to-decision. This is the number that matters."""
        return (self.t_emit - self.event.doc.t_ingest) / 1e6

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
