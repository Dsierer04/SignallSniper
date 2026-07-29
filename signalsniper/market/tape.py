"""Rolling market state -- the "how much is already priced in" side of the edge.

This is the discipline that turns a news reader into a trading system. A material
event is not a trade. A material event where *the price has not moved yet* is a
trade. The same event thirty seconds later, after the name has already run 8%, is
somebody else's exit liquidity.

So the tape's job is to answer one question fast: given an event stamped at time
T, how many basis points has this name already travelled since T, and how much
liquidity is there to get filled against?

Everything is O(1) amortised on the hot path. No pandas, no numpy allocation per
tick -- a deque of primitives and integer index math.
"""

from __future__ import annotations

import bisect
import statistics
from collections import deque
from dataclasses import dataclass, field

from ..models import Quote, epoch_ns, mono_ns


@dataclass(slots=True)
class Bar:
    t_ns: int
    price: float
    size: int


class TickerTape:
    """Bounded rolling history for one symbol."""

    __slots__ = ("ticker", "_t", "_px", "_sz", "maxlen", "last_quote",
                 "_vol_buckets", "_bucket_s", "outlier_bps", "_pending",
                 "rejected_ticks")

    def __init__(self, ticker: str, maxlen: int = 20_000, bucket_s: float = 1.0,
                 outlier_bps: float = 1_500.0) -> None:
        self.ticker = ticker.upper()
        # Plain lists, not deques. bisect needs random access into a sorted
        # sequence; a deque gives O(n) indexing, and converting one to a list per
        # lookup copies 20k ints on every link evaluation -- which turns a 2us
        # read into a 200us one exactly when the event storm hits. Lists are
        # trimmed in amortised batches instead of per-append.
        self._t: list[int] = []
        self._px: list[float] = []
        self._sz: list[int] = []
        self.maxlen = maxlen
        self.last_quote: Quote | None = None
        self._bucket_s = bucket_s
        self._vol_buckets: deque[tuple[int, int]] = deque(maxlen=1800)
        #: A single print further than this from the last accepted one is held
        #: rather than trusted. Bad ticks are routine on thin after-hours names,
        #: and one of them reading as a 2800bps move is enough to manufacture a
        #: signal out of nothing -- and then a stop derived from that phantom
        #: edge is placed so far away it is not a stop at all.
        self.outlier_bps = outlier_bps
        self._pending: Quote | None = None
        self.rejected_ticks = 0

    def _trim(self) -> None:
        """Drop the oldest quarter once we exceed maxlen. Amortised O(1)/append."""
        if len(self._t) <= self.maxlen:
            return
        cut = self.maxlen // 4
        del self._t[:cut]
        del self._px[:cut]
        del self._sz[:cut]

    # ---- ingest --------------------------------------------------------

    def _is_outlier(self, px: float) -> bool:
        if not self._px:
            return False
        last = self._px[-1]
        if last <= 0:
            return False
        return abs(px - last) / last * 10_000.0 > self.outlier_bps

    def on_quote(self, q: Quote) -> None:
        px = q.mid
        if px <= 0:
            return

        # Outlier gate with confirmation. A genuine earnings gap arrives as a
        # sequence of prints that agree with each other; a bad tick is a single
        # print that nothing corroborates. So hold the first surprising print,
        # and only accept it once a second one lands near it. This costs one
        # tick of latency on a real gap and rejects the fabricated move entirely.
        if self._is_outlier(px):
            if self._pending is not None and self._pending.mid > 0:
                drift = abs(px - self._pending.mid) / self._pending.mid * 10_000.0
                if drift <= self.outlier_bps:
                    confirmed = self._pending
                    self._pending = None
                    self.rejected_ticks -= 1  # it was real after all
                    self._accept(confirmed)
                    self._accept(q)
                    return
            self._pending = q
            self.rejected_ticks += 1
            return

        self._pending = None
        self._accept(q)

    def _accept(self, q: Quote) -> None:
        self.last_quote = q
        px = q.mid
        if px <= 0:
            return
        # Quotes can arrive fractionally out of order across venues. bisect
        # assumes sorted timestamps, so clamp a late print to the last one we
        # hold rather than corrupting the ordering.
        if self._t and q.t_source < self._t[-1]:
            self._t.append(self._t[-1])
        else:
            self._t.append(q.t_source)
        self._px.append(px)
        self._sz.append(q.volume)
        self._trim()
        bucket = int(q.t_source // int(self._bucket_s * 1e9))
        if self._vol_buckets and self._vol_buckets[-1][0] == bucket:
            last_b, last_v = self._vol_buckets[-1]
            self._vol_buckets[-1] = (last_b, last_v + q.volume)
        else:
            self._vol_buckets.append((bucket, q.volume))

    # ---- reads ---------------------------------------------------------

    @property
    def last(self) -> float:
        return self._px[-1] if self._px else 0.0

    @property
    def ticks(self) -> int:
        return len(self._px)

    def price_at(self, t_ns: int) -> float:
        """Price as of t_ns, using the last print at or before it.

        If t_ns predates our history we return the oldest price we have. That is
        deliberately conservative: it makes `move_bps_since` report the *full*
        move we can see, so an under-observed name looks more priced-in, not
        less, and the gate errs toward not trading.
        """
        if not self._t:
            return 0.0
        idx = bisect.bisect_right(self._t, t_ns) - 1
        if idx < 0:
            return self._px[0]
        return self._px[idx]

    def move_bps_since(self, t_ns: int) -> float:
        ref = self.price_at(t_ns)
        cur = self.last
        if ref <= 0 or cur <= 0:
            return 0.0
        return (cur - ref) / ref * 10_000.0

    def realized_vol_bps(self, lookback_s: float = 300.0) -> float:
        """Stdev of per-tick returns over the window, in bps. The unit of 'normal'."""
        if len(self._px) < 3:
            return 0.0
        cutoff = epoch_ns() - int(lookback_s * 1e9)
        start = bisect.bisect_left(self._t, cutoff)
        px = self._px[start:]
        if len(px) < 3:
            return 0.0
        rets = [
            (px[i] - px[i - 1]) / px[i - 1] * 10_000.0
            for i in range(1, len(px))
            if px[i - 1] > 0
        ]
        if len(rets) < 2:
            return 0.0
        try:
            return statistics.pstdev(rets)
        except statistics.StatisticsError:  # pragma: no cover
            return 0.0

    def rvol(self, window_s: float = 60.0, baseline_s: float = 900.0) -> float:
        """Recent volume rate over baseline rate. >3 means something is happening."""
        if not self._vol_buckets:
            return 0.0
        now_b = int(epoch_ns() // int(self._bucket_s * 1e9))
        win_b = int(window_s / self._bucket_s)
        base_b = int(baseline_s / self._bucket_s)
        recent = sum(v for b, v in self._vol_buckets if b > now_b - win_b)
        base = sum(v for b, v in self._vol_buckets if b > now_b - base_b)
        if base <= 0 or win_b <= 0 or base_b <= 0:
            return 0.0
        recent_rate = recent / win_b
        base_rate = base / base_b
        if base_rate <= 0:
            return 0.0
        return recent_rate / base_rate

    def spread_bps(self) -> float:
        return self.last_quote.spread_bps if self.last_quote else float("inf")

    def tradeable(self, max_spread_bps: float = 60.0, min_ticks: int = 5) -> tuple[bool, str]:
        """Liquidity gate. A 300bps spread eats the entire edge on entry."""
        if self.ticks < min_ticks:
            return False, f"only {self.ticks} ticks observed"
        s = self.spread_bps()
        if s > max_spread_bps:
            return False, f"spread {s:.0f}bps > {max_spread_bps:.0f}bps"
        if self.last <= 0:
            return False, "no price"
        return True, "ok"


class MarketState:
    """All tapes, plus event-time reference snapshots."""

    def __init__(self) -> None:
        self.tapes: dict[str, TickerTape] = {}

    def tape(self, ticker: str) -> TickerTape:
        t = ticker.upper()
        tp = self.tapes.get(t)
        if tp is None:
            tp = TickerTape(t)
            self.tapes[t] = tp
        return tp

    def on_quote(self, q: Quote) -> None:
        self.tape(q.ticker).on_quote(q)

    def already_priced_bps(self, ticker: str, t_event_ns: int, expected_direction: int) -> float:
        """Signed bps already travelled *in the expected direction* since the event.

        Returns 0 for a move against the thesis -- a name that dropped on news we
        read as bullish has not "used up" any of the bullish move. Whether that
        counts as a better entry or a broken thesis is the engine's call, not the
        tape's.
        """
        tp = self.tapes.get(ticker.upper())
        if tp is None:
            return 0.0
        moved = tp.move_bps_since(t_event_ns)
        signed = moved * (1 if expected_direction >= 0 else -1)
        return max(0.0, signed)

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            t: {
                "last": tp.last,
                "ticks": tp.ticks,
                "spread_bps": round(tp.spread_bps(), 1) if tp.last_quote else -1.0,
                "rvol": round(tp.rvol(), 2),
                "rv_bps": round(tp.realized_vol_bps(), 1),
            }
            for t, tp in self.tapes.items()
        }
