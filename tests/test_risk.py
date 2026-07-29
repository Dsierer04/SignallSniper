"""Risk gate and tape mechanics.

These tests are about survival, not edge. Every one of them describes a way an
account gets destroyed and asserts that the code refuses to go there.
"""

from __future__ import annotations

import pytest

from signalsniper.market.quotes import synth_walk
from signalsniper.market.tape import MarketState, TickerTape
from signalsniper.models import Direction, Event, EventKind, Quote, RawDoc, Signal, now_ns
from signalsniper.signal.risk import RiskConfig, RiskManager


def make_signal(ticker="CRUS", direction=Direction.SHORT, edge_bps=300.0,
                price=100.0, ttl_s=300.0):
    doc = RawDoc(source="test", doc_id="d", title="t", url="", published=None)
    ev = Event(doc=doc, kind=EventKind.EARNINGS, tickers=(ticker,),
               materiality=0.8, prior=direction, confidence=0.8)
    return Signal(ticker=ticker, direction=direction, edge_bps=edge_bps,
                  confidence=0.7, event=ev, ref_price=price, ttl_s=ttl_s)


class TestSizing:
    def test_risk_per_trade_is_respected(self):
        rm = RiskManager(RiskConfig(equity=25_000.0, risk_per_trade=0.01))
        order = rm.size_order(make_signal(edge_bps=300.0, price=100.0))
        assert order is not None
        # stop = max(40, 300*0.6) = 180bps = $1.80. $250 risk / $1.80 = 138 shares.
        loss_at_stop = abs(order.limit - order.stop) * order.shares
        assert loss_at_stop <= 25_000.0 * 0.01 + 1.0

    def test_volatile_name_gets_a_smaller_position(self):
        """Same dollar risk, wider stop, fewer shares. This is the whole point."""
        rm = RiskManager(RiskConfig(equity=25_000.0, risk_per_trade=0.01))
        tight = rm.size_order(make_signal(ticker="A", edge_bps=200.0, price=100.0))
        rm2 = RiskManager(RiskConfig(equity=25_000.0, risk_per_trade=0.01))
        wide = rm2.size_order(make_signal(ticker="B", edge_bps=1200.0, price=100.0))
        assert tight.shares > wide.shares

    def test_notional_cap_binds_before_risk_budget(self):
        rm = RiskManager(RiskConfig(equity=10_000.0, risk_per_trade=0.02,
                                    max_position_frac=0.20))
        order = rm.size_order(make_signal(edge_bps=100.0, price=10.0))
        assert order is not None
        assert order.notional <= 10_000.0 * 0.20 + 10.0

    def test_min_stop_floor_prevents_noise_stopouts(self):
        rm = RiskManager(RiskConfig(equity=25_000.0, min_stop_bps=40.0))
        order = rm.size_order(make_signal(edge_bps=10.0, price=100.0))
        if order is not None:
            assert abs(order.limit - order.stop) >= 100.0 * 40.0 / 10_000.0 - 1e-9

    def test_sub_minimum_notional_rejected(self):
        # LONG, because a SHORT at $500 equity is refused earlier by the
        # $2,000 regulatory floor and would test the wrong gate.
        rm = RiskManager(RiskConfig(equity=500.0, risk_per_trade=0.01,
                                    min_notional=200.0))
        assert rm.size_order(
            make_signal(direction=Direction.LONG, edge_bps=300.0, price=100.0)) is None
        assert rm.rejects.get("below_min_notional") or rm.rejects.get("size_zero")

    def test_stop_and_target_on_the_correct_sides(self):
        rm = RiskManager(RiskConfig(equity=50_000.0))
        long = rm.size_order(make_signal("AAA", Direction.LONG, 300.0, 100.0))
        assert long.stop < long.limit < long.target
        rm2 = RiskManager(RiskConfig(equity=50_000.0))
        short = rm2.size_order(make_signal("BBB", Direction.SHORT, 300.0, 100.0))
        assert short.stop > short.limit > short.target


class TestGates:
    def test_max_concurrent_enforced(self):
        rm = RiskManager(RiskConfig(equity=200_000.0, max_concurrent=2))
        for t in ("AAA", "BBB"):
            rm.open(rm.size_order(make_signal(ticker=t)))
        assert rm.size_order(make_signal(ticker="CCC")) is None
        assert rm.rejects.get("max_concurrent") == 1

    def test_no_duplicate_position_in_same_name(self):
        rm = RiskManager(RiskConfig(equity=200_000.0))
        rm.open(rm.size_order(make_signal(ticker="AAA")))
        assert rm.size_order(make_signal(ticker="AAA")) is None
        assert rm.rejects.get("already_open") == 1

    def test_cooldown_after_exit(self):
        rm = RiskManager(RiskConfig(equity=200_000.0, cooldown_s=600.0))
        rm.open(rm.size_order(make_signal(ticker="AAA", price=100.0)))
        rm.close("AAA", 100.0)
        assert rm.size_order(make_signal(ticker="AAA")) is None
        assert rm.rejects.get("cooldown") == 1

    def test_expired_signal_rejected(self):
        rm = RiskManager(RiskConfig(equity=200_000.0))
        sig = make_signal(ttl_s=0.0)
        sig.t_emit = now_ns() - 10_000_000_000
        assert rm.size_order(sig) is None
        assert rm.rejects.get("expired") == 1

    def test_neutral_direction_rejected(self):
        rm = RiskManager(RiskConfig(equity=200_000.0))
        assert rm.size_order(make_signal(direction=Direction.NEUTRAL)) is None


class TestKillSwitch:
    def test_trips_at_daily_loss_limit_and_is_sticky(self):
        rm = RiskManager(RiskConfig(equity=10_000.0, max_daily_loss_frac=0.03))
        rm.realized_pnl = -301.0
        assert rm.check_kill_switch() is True
        assert rm.halted

        # Even a subsequent profit does not un-halt it. Only an explicit resume.
        rm.realized_pnl = 5_000.0
        assert rm.check_kill_switch() is True
        assert rm.size_order(make_signal()) is None
        assert rm.rejects.get("halted") == 1

        rm.resume()
        assert rm.check_kill_switch() is False

    def test_not_tripped_just_under_the_limit(self):
        rm = RiskManager(RiskConfig(equity=10_000.0, max_daily_loss_frac=0.03))
        rm.realized_pnl = -299.0
        assert rm.check_kill_switch() is False

    def test_closing_a_loser_can_trip_it(self):
        rm = RiskManager(RiskConfig(equity=10_000.0, max_daily_loss_frac=0.03,
                                    risk_per_trade=0.05, max_position_frac=1.0))
        order = rm.size_order(make_signal(ticker="AAA", direction=Direction.LONG,
                                          edge_bps=500.0, price=100.0))
        rm.open(order)
        rm.close("AAA", 80.0)  # -20/share
        assert rm.halted

    def test_manual_halt(self):
        rm = RiskManager(RiskConfig())
        rm.halt("fat finger")
        assert rm.check_kill_switch() and "fat finger" in rm.halt_reason


class TestExits:
    def test_long_stop_and_target(self):
        rm = RiskManager(RiskConfig(equity=200_000.0))
        rm.open(rm.size_order(make_signal("AAA", Direction.LONG, 300.0, 100.0)))
        pos = rm.positions["AAA"]
        assert rm.check_exits({"AAA": pos.stop - 0.01})[0][2] == "stop"

        rm2 = RiskManager(RiskConfig(equity=200_000.0))
        rm2.open(rm2.size_order(make_signal("BBB", Direction.LONG, 300.0, 100.0)))
        assert rm2.check_exits({"BBB": rm2.positions["BBB"].target + 0.01})[0][2] == "target"

    def test_short_stop_is_above_entry(self):
        rm = RiskManager(RiskConfig(equity=200_000.0))
        rm.open(rm.size_order(make_signal("AAA", Direction.SHORT, 300.0, 100.0)))
        pos = rm.positions["AAA"]
        assert rm.check_exits({"AAA": pos.stop + 0.01})[0][2] == "stop"
        assert "AAA" not in rm.positions

    def test_short_pnl_sign(self):
        rm = RiskManager(RiskConfig(equity=200_000.0))
        rm.open(rm.size_order(make_signal("AAA", Direction.SHORT, 300.0, 100.0)))
        assert rm.close("AAA", 90.0) > 0   # short into a fall = profit
        rm.resume()
        rm.last_exit_ns.clear()
        rm.open(rm.size_order(make_signal("BBB", Direction.SHORT, 300.0, 100.0)))
        assert rm.close("BBB", 110.0) < 0

    def test_missing_price_does_not_exit(self):
        rm = RiskManager(RiskConfig(equity=200_000.0))
        rm.open(rm.size_order(make_signal("AAA", Direction.LONG, 300.0, 100.0)))
        assert rm.check_exits({}) == []
        assert rm.check_exits({"AAA": 0.0}) == []
        assert "AAA" in rm.positions


class TestTape:
    def test_price_at_finds_the_print_at_or_before(self):
        tape = TickerTape("AAA")
        t0 = 1_000_000_000
        for i, px in enumerate([10.0, 11.0, 12.0, 13.0]):
            tape.on_quote(Quote("AAA", px - 0.01, px + 0.01, px, 100, t0 + i * 1_000_000_000))
        assert tape.price_at(t0 + 1_500_000_000) == pytest.approx(11.0)
        assert tape.price_at(t0 + 2_000_000_000) == pytest.approx(12.0)

    def test_price_before_history_returns_oldest(self):
        tape = TickerTape("AAA")
        tape.on_quote(Quote("AAA", 9.99, 10.01, 10.0, 100, 5_000_000_000))
        assert tape.price_at(1_000_000_000) == pytest.approx(10.0)

    def test_move_bps_since(self):
        tape = TickerTape("AAA")
        t0 = 1_000_000_000
        tape.on_quote(Quote("AAA", 99.99, 100.01, 100.0, 100, t0))
        tape.on_quote(Quote("AAA", 104.99, 105.01, 105.0, 100, t0 + 1_000_000_000))
        assert tape.move_bps_since(t0) == pytest.approx(500.0, abs=1.0)

    def test_already_priced_ignores_moves_against_the_thesis(self):
        market = MarketState()
        t0 = now_ns()
        for q in synth_walk("AAA", 100.0, 20, t0, drift_bps_total=+300.0):
            market.on_quote(q)
        # We are short. The name went UP, so none of our expected move is used.
        assert market.already_priced_bps("AAA", t0, -1) == 0.0
        # We are long. 300bps of the move is already gone.
        assert market.already_priced_bps("AAA", t0, +1) == pytest.approx(300.0, abs=5.0)

    def test_unknown_ticker_is_zero_not_an_error(self):
        assert MarketState().already_priced_bps("NOPE", now_ns(), 1) == 0.0

    def test_tradeable_gates(self):
        tape = TickerTape("AAA")
        ok, why = tape.tradeable()
        assert not ok and "ticks" in why

        for q in synth_walk("AAA", 100.0, 20, now_ns(), spread_bps=500.0):
            tape.on_quote(q)
        ok, why = tape.tradeable(max_spread_bps=60.0)
        assert not ok and "spread" in why

        tape2 = TickerTape("BBB")
        for q in synth_walk("BBB", 100.0, 20, now_ns(), spread_bps=5.0):
            tape2.on_quote(q)
        assert tape2.tradeable(max_spread_bps=60.0)[0]

    def test_bounded_memory(self):
        tape = TickerTape("AAA", maxlen=100)
        for q in synth_walk("AAA", 100.0, 500, now_ns()):
            tape.on_quote(q)
        assert tape.ticks <= 100

    def test_trim_keeps_the_newest_prints(self):
        tape = TickerTape("AAA", maxlen=100)
        quotes = synth_walk("AAA", 100.0, 400, now_ns(), drift_bps_total=1000.0)
        for q in quotes:
            tape.on_quote(q)
        assert tape.last == pytest.approx(quotes[-1].mid, abs=0.01)
        # Parallel arrays must stay the same length or bisect returns garbage.
        assert len(tape._t) == len(tape._px) == len(tape._sz)

    def test_out_of_order_quote_preserves_sorted_timestamps(self):
        """Cross-venue prints can arrive late. bisect assumes sorted order."""
        tape = TickerTape("AAA")
        t0 = now_ns()
        tape.on_quote(Quote("AAA", 99.99, 100.01, 100.0, 100, t0))
        tape.on_quote(Quote("AAA", 100.99, 101.01, 101.0, 100, t0 + 2_000_000_000))
        tape.on_quote(Quote("AAA", 101.99, 102.01, 102.0, 100, t0 + 1_000_000_000))
        assert tape._t == sorted(tape._t)
        assert tape.ticks == 3
        assert tape.price_at(t0 + 5_000_000_000) == pytest.approx(102.0)

    def test_price_at_stays_fast_on_a_deep_tape(self):
        """Regression guard: this was O(n) copy-per-call before."""
        import time
        tape = TickerTape("AAA", maxlen=20_000)
        t0 = now_ns()
        for q in synth_walk("AAA", 100.0, 20_000, t0, step_ns=1_000_000):
            tape.on_quote(q)
        assert tape.ticks == 20_000
        mid = t0 + 10_000 * 1_000_000
        start = time.perf_counter()
        for _ in range(1000):
            tape.move_bps_since(mid)
        per_call_us = (time.perf_counter() - start) / 1000 * 1e6
        assert per_call_us < 50.0, f"{per_call_us:.1f}us/call -- lookup went linear again"

    def test_zero_price_quotes_ignored(self):
        tape = TickerTape("AAA")
        tape.on_quote(Quote("AAA", 0.0, 0.0, 0.0, 0, now_ns()))
        assert tape.ticks == 0
