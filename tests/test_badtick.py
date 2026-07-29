"""Defenses against corrupt market data.

Found by tools/soak.py, not by reasoning: a single bad print on a thin name read
as a 2820bps move, the engine turned that into a huge "edge", and the risk
manager derived the stop FROM that edge -- producing a 6449bps stop. A stop 64%
away is not a stop. The position had no floor under it.

Bad ticks are routine, especially after hours on the exact thin names this
strategy targets. Three independent layers now bound this, so no single one is
load-bearing:

  1. tape    -- outlier prints are held until a second print corroborates them
  2. engine  -- an implied move beyond a sane bound is a bug, not an opportunity
  3. risk    -- a stop wider than a sane signal can produce refuses the trade
"""

from __future__ import annotations

import pytest

from signalsniper.market.quotes import synth_walk
from signalsniper.market.tape import MarketState, TickerTape
from signalsniper.models import Direction, Event, EventKind, Quote, RawDoc, Signal, now_ns
from signalsniper.signal.engine import DEFAULT_MOVE_SCALE, EngineConfig, SignalEngine
from signalsniper.signal.risk import RiskConfig, RiskManager


def feed(tape: TickerTape, prices: list[float], t0: int, step_ns: int = 1_000_000) -> None:
    for i, px in enumerate(prices):
        tape.on_quote(Quote(tape.ticker, px - 0.01, px + 0.01, px, 100, t0 + i * step_ns))


class TestTapeOutlierRejection:
    def test_single_bad_tick_is_rejected(self):
        """The exact failure the soak test found."""
        tape = TickerTape("SWKS", outlier_bps=1500.0)
        feed(tape, [78.0, 78.1, 77.9, 100.0], now_ns())  # 100.0 is garbage
        assert tape.last == pytest.approx(77.9)
        assert tape.rejected_ticks == 1

    def test_bad_tick_does_not_corrupt_the_move_reading(self):
        tape = TickerTape("SWKS", outlier_bps=1500.0)
        t0 = now_ns()
        feed(tape, [78.0, 78.1, 77.9, 100.0], t0)
        # Without the gate this read as +2820bps.
        assert abs(tape.move_bps_since(t0)) < 100.0

    def test_a_confirmed_gap_is_accepted(self):
        """A real earnings gap arrives as prints that agree with each other."""
        tape = TickerTape("CRUS", outlier_bps=1500.0)
        t0 = now_ns()
        feed(tape, [104.0, 104.1, 85.0, 85.2, 84.8], t0)
        assert tape.last == pytest.approx(84.8)
        assert tape.move_bps_since(t0) < -1500.0  # the real move survives
        assert tape.rejected_ticks == 0           # retroactively accepted

    def test_gap_costs_exactly_one_tick_of_latency(self):
        tape = TickerTape("CRUS", outlier_bps=1500.0)
        t0 = now_ns()
        feed(tape, [104.0, 85.0], t0)
        assert tape.last == pytest.approx(104.0)   # held, not yet trusted
        feed(tape, [85.1], t0 + 5_000_000)
        assert tape.last == pytest.approx(85.1)    # confirmed, both accepted

    def test_two_unrelated_outliers_are_both_rejected(self):
        """Garbage that does not agree with itself never gets in."""
        tape = TickerTape("X", outlier_bps=1500.0)
        feed(tape, [50.0, 50.1, 200.0, 5.0, 50.2], now_ns())
        assert tape.last == pytest.approx(50.2)
        assert tape.rejected_ticks == 2

    def test_first_print_is_never_an_outlier(self):
        tape = TickerTape("X", outlier_bps=1500.0)
        feed(tape, [999.0], now_ns())
        assert tape.last == pytest.approx(999.0)

    def test_normal_volatility_passes_untouched(self):
        tape = TickerTape("X", outlier_bps=1500.0)
        for q in synth_walk("X", 100.0, 200, now_ns(), drift_bps_total=-800.0):
            tape.on_quote(q)
        assert tape.rejected_ticks == 0
        assert tape.ticks == 200


class TestEngineSanityBounds:
    def _corrupt_market(self):
        """AAPL reads an absurd move -- the shape a broken series produces."""
        market = MarketState()
        t_pre = now_ns() - 8_000_000_000
        for tick, px in (("AAPL", 232.0), ("CRUS", 104.0)):
            for q in synth_walk(tick, px, 40, t_pre):
                market.on_quote(q)
        t_event = now_ns()
        # Bypass the tape gate to simulate corruption that got through anyway --
        # the engine bound must not depend on the tape bound holding.
        tape = market.tape("AAPL")
        for q in synth_walk("AAPL", 232.0, 30, t_event, drift_bps_total=-5000.0):
            tape._accept(q)
        for q in synth_walk("CRUS", 104.0, 30, t_event):
            market.on_quote(q)
        return market, t_event

    def _event(self, t_event):
        from signalsniper.parse.classify import build_event
        doc = RawDoc(
            source="edgar", doc_id="d", title="8-K - Apple Inc. (0000320193) (Filer)",
            url="", published=None,
            body="Item 2.02 Results of Operations. iPhone hardware guidance "
                 "lowered below consensus estimates.",
            meta={"form": "8-K", "company": "Apple Inc.", "cik": "320193",
                  "items": ("2.02",)},
        )
        doc.t_ingest = t_event
        return build_event(doc, ("AAPL",))

    def test_implausible_primary_move_blocks_all_propagation(self):
        """One corrupt primary would otherwise corrupt every linked name at once."""
        market, t_event = self._corrupt_market()
        engine = SignalEngine(market, None, EngineConfig(
            move_scale=dict(DEFAULT_MOVE_SCALE), max_primary_bps=3000.0))
        assert engine.evaluate_second_order(self._event(t_event)) == []
        assert engine.rejected.get("primary_implausible")

    def test_implausible_edge_is_rejected_not_traded(self):
        market = MarketState()
        t_pre = now_ns() - 8_000_000_000
        for q in synth_walk("TINY", 4.0, 40, t_pre):
            market.on_quote(q)
        t_event = now_ns()
        for q in synth_walk("TINY", 4.0, 20, t_event):
            market.on_quote(q)

        from signalsniper.parse.classify import build_event
        doc = RawDoc(source="edgar", doc_id="d",
                     title="424B5 - Tiny Inc (0001800001) (Filer)", url="",
                     published=None, body="424B5.",
                     meta={"form": "424B5", "company": "Tiny Inc",
                           "cik": "1800001", "items": ()})
        doc.t_ingest = t_event

        engine = SignalEngine(market, None, EngineConfig(
            move_scale=dict(DEFAULT_MOVE_SCALE),
            max_event_bps=50_000.0,   # force an absurd expected move
            max_edge_bps=1_500.0))
        assert engine.evaluate_primary(build_event(doc, ("TINY",))) is None
        assert engine.rejected.get("edge_implausible")


class TestRiskStopBackstop:
    def _signal(self, edge_bps: float, price: float = 100.0):
        doc = RawDoc(source="t", doc_id="d", title="t", url="", published=None)
        ev = Event(doc=doc, kind=EventKind.EARNINGS, tickers=("X",),
                   materiality=0.8, prior=Direction.SHORT, confidence=0.8)
        return Signal(ticker="X", direction=Direction.SHORT, edge_bps=edge_bps,
                      confidence=0.7, event=ev, ref_price=price)

    def test_absurd_edge_is_refused_rather_than_clamped(self):
        """The soak case: edge ~10700bps produced a 6449bps stop."""
        rm = RiskManager(RiskConfig(equity=25_000.0))
        assert rm.size_order(self._signal(10_700.0)) is None
        assert rm.rejects.get("stop_too_wide") == 1

    def test_backstop_matches_the_engine_ceiling(self):
        """A signal at the engine's own max must still be sizeable, or the
        backstop is vetoing legitimate trades instead of catching corrupt ones."""
        rm = RiskManager(RiskConfig(equity=200_000.0))
        order = rm.size_order(self._signal(1_500.0))
        assert order is not None
        assert abs(order.limit - order.stop) / order.limit * 10_000 <= 900.0 + 1

    def test_loss_at_stop_stays_bounded_at_the_ceiling(self):
        rm = RiskManager(RiskConfig(equity=25_000.0, risk_per_trade=0.01))
        order = rm.size_order(self._signal(1_500.0))
        if order is not None:
            loss = abs(order.limit - order.stop) * order.shares
            assert loss <= 25_000.0 * 0.01 + 1.0
