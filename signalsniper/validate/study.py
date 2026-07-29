"""Empirical validation of the propagation thesis.

This module exists because the entire system rests on one falsifiable claim:

    When a primary reprices on an earnings print, its economically-linked names
    reprice on a LAG of minutes, and that lag is tradeable.

If linked names actually move simultaneously with the primary, there is no edge
and the system is an elaborate way to pay spreads. That is a measurable question,
not a matter of opinion, and this module measures it.

Two outputs:

1. **Lag profile.** For each linked name, what fraction of its eventual move
   happened AFTER the first K minutes? High fraction => the lag is real. Low
   fraction => it repriced with the primary and there was never anything to
   catch.

2. **Empirical beta.** Regress the linked name's move on the primary's move
   across historical events. This replaces the hand-set priors in linkage.py
   with numbers derived from what actually happened, which is the single
   highest-value calibration available.

Everything here is pure functions over bar series so it can be unit tested
without a data provider. `loader.py` supplies the bars.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass(frozen=True, slots=True)
class Bar:
    t: datetime          # bar open time, timezone-aware
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(slots=True)
class Series:
    ticker: str
    bars: list[Bar] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.bars.sort(key=lambda b: b.t)

    def price_at(self, when: datetime) -> float:
        """Last close at or before `when`. 0.0 if we have nothing that early."""
        best = 0.0
        for b in self.bars:
            if b.t <= when:
                best = b.close
            else:
                break
        return best

    def first_price_after(self, when: datetime) -> float:
        for b in self.bars:
            if b.t >= when:
                return b.close
        return 0.0

    def volume_between(self, start: datetime, end: datetime) -> int:
        return sum(b.volume for b in self.bars if start <= b.t < end)

    def bars_between(self, start: datetime, end: datetime) -> list[Bar]:
        return [b for b in self.bars if start <= b.t < end]


def move_bps(ref: float, px: float) -> float:
    if ref <= 0 or px <= 0:
        return 0.0
    return (px - ref) / ref * 10_000.0


#: Windows measured from the event timestamp. The 0-1 and 1-5 minute buckets are
#: the ones the strategy cannot compete in; everything after is the claimed edge.
DEFAULT_WINDOWS: tuple[tuple[str, float, float], ...] = (
    ("0-1m", 0.0, 1.0),
    ("1-5m", 1.0, 5.0),
    ("5-15m", 5.0, 15.0),
    ("15-30m", 15.0, 30.0),
    ("30-60m", 30.0, 60.0),
    ("60-240m", 60.0, 240.0),
)


@dataclass(slots=True)
class WindowMove:
    label: str
    start_min: float
    end_min: float
    primary_bps: float
    linked_bps: float
    linked_volume: int
    bars_seen: int

    @property
    def had_liquidity(self) -> bool:
        """No prints means the 'no move' reading is absence of data, not of news."""
        return self.bars_seen > 0


@dataclass(slots=True)
class EventProfile:
    """One linked name's behaviour around one event."""

    event_id: str
    primary: str
    linked: str
    t_event: datetime
    primary_total_bps: float
    linked_total_bps: float
    windows: list[WindowMove]
    linked_bars_total: int
    overnight_gap_bps: float = 0.0
    next_day_bps: float = 0.0

    @property
    def realized_beta(self) -> float:
        """Linked move over primary move for this single event.

        Meaningless on its own -- a single ratio with a small denominator is
        mostly noise. Aggregated across events by `estimate_beta` it becomes the
        empirical replacement for a hand-set prior.
        """
        if abs(self.primary_total_bps) < 1e-9:
            return 0.0
        return self.linked_total_bps / self.primary_total_bps

    def captured_after(self, minutes: float) -> float:
        """Fraction of the linked name's total move that arrived after `minutes`.

        This is THE number. It is the share of the move a system that reacts in
        `minutes` could still have caught. Near 1.0 means the lag is real and
        large; near 0.0 means the name repriced instantly alongside the primary
        and there was never an edge to capture.

        Returns nan when the total move is too small to form a ratio -- a 3bps
        move split 50/50 is not evidence of anything.
        """
        if abs(self.linked_total_bps) < 25.0:
            return float("nan")
        early = 0.0
        for w in self.windows:
            if w.end_min <= minutes:
                early += w.linked_bps
        return (self.linked_total_bps - early) / self.linked_total_bps

    @property
    def dead_early(self) -> bool:
        """No prints in the first 5 minutes -- 'unpriced' was really 'untraded'.

        This is the failure mode the live engine cannot distinguish: a linked
        name with no after-hours prints looks exactly like one that has not
        repriced yet.
        """
        return not any(w.had_liquidity for w in self.windows if w.end_min <= 5.0)


def profile_event(
    event_id: str,
    primary: Series,
    linked: Series,
    t_event: datetime,
    horizon_min: float = 240.0,
    windows: tuple[tuple[str, float, float], ...] = DEFAULT_WINDOWS,
) -> EventProfile:
    """Measure one (event, linked name) pair."""
    p_ref = primary.price_at(t_event)
    l_ref = linked.price_at(t_event)
    end = t_event + timedelta(minutes=horizon_min)

    p_total = move_bps(p_ref, primary.price_at(end))
    l_total = move_bps(l_ref, linked.price_at(end))

    out: list[WindowMove] = []
    for label, a, b in windows:
        ta = t_event + timedelta(minutes=a)
        tb = t_event + timedelta(minutes=b)
        p_a, p_b = primary.price_at(ta), primary.price_at(tb)
        l_a, l_b = linked.price_at(ta), linked.price_at(tb)
        out.append(WindowMove(
            label=label, start_min=a, end_min=b,
            primary_bps=move_bps(p_ref, p_b) - move_bps(p_ref, p_a) if p_a > 0 else 0.0,
            linked_bps=move_bps(l_ref, l_b) - move_bps(l_ref, l_a) if l_a > 0 else 0.0,
            linked_volume=linked.volume_between(ta, tb),
            bars_seen=len(linked.bars_between(ta, tb)),
        ))

    return EventProfile(
        event_id=event_id, primary=primary.ticker, linked=linked.ticker,
        t_event=t_event, primary_total_bps=p_total, linked_total_bps=l_total,
        windows=out, linked_bars_total=len(linked.bars_between(t_event, end)),
    )


@dataclass(slots=True)
class BetaEstimate:
    linked: str
    primary: str
    n: int
    beta: float           # slope of linked_bps on primary_bps, through origin
    r_squared: float
    residual_bps: float   # stdev of residuals -- the error bar that matters
    prior_beta: float = 0.0

    @property
    def prior_error(self) -> float:
        return self.beta - self.prior_beta

    @property
    def usable(self) -> bool:
        """A beta from four events with r2 of 0.1 is a number, not a signal."""
        return self.n >= 6 and self.r_squared >= 0.30

    def describe(self) -> str:
        flag = "" if self.usable else "  [INSUFFICIENT]"
        return (f"{self.linked:<6} n={self.n:<3} beta={self.beta:+.2f} "
                f"(prior {self.prior_beta:+.2f}, err {self.prior_error:+.2f})  "
                f"r2={self.r_squared:.2f} resid={self.residual_bps:.0f}bps{flag}")


def estimate_beta(profiles: list[EventProfile], prior: float = 0.0) -> BetaEstimate | None:
    """Regress linked move on primary move through the origin.

    Through the origin deliberately: the claim is proportional read-through, not
    an unconditional drift. Fitting an intercept would let a name with a constant
    post-earnings tendency masquerade as having exposure to the primary.
    """
    pts = [(p.primary_total_bps, p.linked_total_bps) for p in profiles
           if abs(p.primary_total_bps) > 50.0]
    if len(pts) < 3:
        return None

    sxy = sum(x * y for x, y in pts)
    sxx = sum(x * x for x, _ in pts)
    if sxx <= 0:
        return None
    beta = sxy / sxx

    resid = [y - beta * x for x, y in pts]
    ss_res = sum(r * r for r in resid)
    ss_tot = sum(y * y for _, y in pts)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return BetaEstimate(
        linked=profiles[0].linked, primary=profiles[0].primary, n=len(pts),
        beta=beta, r_squared=max(0.0, r2),
        residual_bps=statistics.pstdev(resid) if len(resid) > 1 else 0.0,
        prior_beta=prior,
    )


@dataclass(slots=True)
class LagVerdict:
    linked: str
    n_events: int
    median_capture_after_5m: float
    median_capture_after_15m: float
    dead_early_rate: float
    median_total_bps: float

    @property
    def edge_exists(self) -> bool:
        """The bar: a majority of the move must still be available at +5min,
        and the name must actually print in that window often enough to trade."""
        if math.isnan(self.median_capture_after_5m):
            return False
        return (self.median_capture_after_5m >= 0.50
                and self.dead_early_rate <= 0.35
                and self.n_events >= 4)

    def describe(self) -> str:
        cap5 = ("n/a" if math.isnan(self.median_capture_after_5m)
                else f"{self.median_capture_after_5m:+.0%}")
        cap15 = ("n/a" if math.isnan(self.median_capture_after_15m)
                 else f"{self.median_capture_after_15m:+.0%}")
        mark = "EDGE" if self.edge_exists else "no edge"
        return (f"{self.linked:<6} n={self.n_events:<3} "
                f"move={self.median_total_bps:+6.0f}bps  "
                f"still-available@5m={cap5:>6}  @15m={cap15:>6}  "
                f"no-early-prints={self.dead_early_rate:.0%}  -> {mark}")


def _median_ignoring_nan(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return statistics.median(clean) if clean else float("nan")


def lag_verdict(profiles: list[EventProfile]) -> LagVerdict | None:
    if not profiles:
        return None
    return LagVerdict(
        linked=profiles[0].linked,
        n_events=len(profiles),
        median_capture_after_5m=_median_ignoring_nan([p.captured_after(5.0) for p in profiles]),
        median_capture_after_15m=_median_ignoring_nan([p.captured_after(15.0) for p in profiles]),
        dead_early_rate=sum(1 for p in profiles if p.dead_early) / len(profiles),
        median_total_bps=statistics.median([p.linked_total_bps for p in profiles]),
    )
