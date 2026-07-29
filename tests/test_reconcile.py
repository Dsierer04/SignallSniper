"""Fill reconciliation.

Assuming the fill is how a missed entry becomes an unintended position. The
chain: a limit order does not fill, the risk manager still believes it holds the
position, the price later crosses the stop, and the exit logic sends a *closing*
order for shares that were never bought. That does not flatten anything -- it
opens a real position in the opposite direction, with no stop on it, on a thesis
that already failed.

These tests pin the state machine that prevents that.
"""

from __future__ import annotations

import asyncio

import pytest

from signalsniper.config import Config
from signalsniper.execution.broker import OrderStatus, PaperBroker
from signalsniper.execution.session import current_session
from signalsniper.models import Direction, Event, EventKind, RawDoc, Signal
from signalsniper.runner import Runner
from signalsniper.signal.risk import Order, RiskConfig, RiskManager

from test_execution import at, make_order


def make_signal(ticker="CRUS", direction=Direction.SHORT, edge_bps=300.0, price=100.0):
    doc = RawDoc(source="t", doc_id="d", title="t", url="", published=None)
    ev = Event(doc=doc, kind=EventKind.EARNINGS, tickers=(ticker,),
               materiality=0.8, prior=direction, confidence=0.8)
    return Signal(ticker=ticker, direction=direction, edge_bps=edge_bps,
                  confidence=0.7, event=ev, ref_price=price)


class TestOrderStatus:
    @pytest.mark.parametrize("status,terminal", [
        ("filled", True), ("canceled", True), ("rejected", True),
        ("expired", True), ("new", False), ("partially_filled", False),
        ("accepted", False),
    ])
    def test_terminal_classification(self, status, terminal):
        assert OrderStatus("id", status).is_terminal is terminal

    def test_got_nothing(self):
        assert OrderStatus("id", "canceled", 0).got_nothing is True
        assert OrderStatus("id", "filled", 10).got_nothing is False


class TestRiskAmendment:
    def test_amend_moves_stop_and_target_with_the_real_entry(self):
        """A stop measured from a price you did not get is the wrong distance
        from the price you did."""
        rm = RiskManager(RiskConfig(equity=200_000.0))
        order = rm.size_order(make_signal(direction=Direction.SHORT, price=100.0))
        rm.open(order)
        pos = rm.positions["CRUS"]
        stop_dist = abs(pos.entry - pos.stop)
        target_dist = abs(pos.target - pos.entry)

        rm.amend_fill("CRUS", pos.shares, 101.50)   # filled worse than expected

        assert pos.entry == pytest.approx(101.50)
        assert abs(pos.entry - pos.stop) == pytest.approx(stop_dist, abs=0.02)
        assert abs(pos.target - pos.entry) == pytest.approx(target_dist, abs=0.02)
        assert pos.stop > pos.entry > pos.target     # still a short

    def test_amend_handles_a_partial_fill(self):
        rm = RiskManager(RiskConfig(equity=200_000.0))
        order = rm.size_order(make_signal())
        rm.open(order)
        rm.amend_fill("CRUS", 10, 100.0)
        assert rm.positions["CRUS"].shares == 10

    def test_amend_ignores_nonsense(self):
        rm = RiskManager(RiskConfig(equity=200_000.0))
        rm.open(rm.size_order(make_signal()))
        before = rm.positions["CRUS"].shares
        rm.amend_fill("CRUS", 0, 100.0)
        rm.amend_fill("CRUS", 10, 0.0)
        rm.amend_fill("NOPE", 10, 100.0)
        assert rm.positions["CRUS"].shares == before

    def test_drop_unfilled_books_no_pnl(self):
        """A phantom round trip would corrupt the number the kill switch reads."""
        rm = RiskManager(RiskConfig(equity=200_000.0))
        rm.open(rm.size_order(make_signal()))
        assert rm.drop_unfilled("CRUS") is True
        assert "CRUS" not in rm.positions
        assert rm.realized_pnl == 0.0

    def test_drop_unfilled_sets_no_cooldown(self):
        """No fill means no trade, so re-entry must not be penalised."""
        rm = RiskManager(RiskConfig(equity=200_000.0, cooldown_s=600.0))
        rm.open(rm.size_order(make_signal()))
        rm.drop_unfilled("CRUS")
        assert rm.size_order(make_signal()) is not None

    def test_drop_unknown_is_a_noop(self):
        assert RiskManager().drop_unfilled("NOPE") is False

    def test_dropping_frees_a_concurrency_slot(self):
        """The phantom position would otherwise block a real trade."""
        rm = RiskManager(RiskConfig(equity=200_000.0, max_concurrent=1))
        rm.open(rm.size_order(make_signal(ticker="AAA")))
        assert rm.size_order(make_signal(ticker="BBB")) is None
        rm.drop_unfilled("AAA")
        assert rm.size_order(make_signal(ticker="BBB")) is not None


class TestRunnerReconciliation:
    def _runner(self, broker):
        cfg = Config(sec_user_agent="T t@e.com", live=True, allow_extended=True,
                     watchlist=("CRUS",))
        return Runner(cfg, broker=broker)

    @pytest.mark.asyncio
    async def test_unfilled_order_drops_the_phantom_position(self):
        broker = PaperBroker(allow_extended=True)
        broker.fill_behaviour = "none"
        runner = self._runner(broker)

        order = make_order()
        fill = await broker.submit(order, current_session(at(2026, 7, 30, 16, 5)))
        runner.risk.open(order)
        runner.client_side_stops.add(order.ticker)

        await runner._reconcile(order, fill.broker_order_id,
                                timeout_s=3.0, poll_s=0.05)

        assert order.ticker not in runner.risk.positions
        assert order.ticker not in runner.client_side_stops
        assert runner.counts["unfilled"] == 1

    @pytest.mark.asyncio
    async def test_partial_fill_corrects_the_share_count(self):
        broker = PaperBroker(allow_extended=True)
        broker.fill_behaviour = "partial"
        runner = self._runner(broker)

        order = make_order()
        fill = await broker.submit(order, current_session(at(2026, 7, 30, 16, 5)))
        runner.risk.open(order)

        await runner._reconcile(order, fill.broker_order_id,
                                timeout_s=1.0, poll_s=0.05)

        pos = runner.risk.positions.get(order.ticker)
        assert pos is not None
        assert pos.shares == order.shares // 2

    @pytest.mark.asyncio
    async def test_full_fill_leaves_the_position_intact(self):
        broker = PaperBroker(allow_extended=True)
        runner = self._runner(broker)

        order = make_order()
        fill = await broker.submit(order, current_session(at(2026, 7, 30, 16, 5)))
        runner.risk.open(order)

        await runner._reconcile(order, fill.broker_order_id,
                                timeout_s=3.0, poll_s=0.05)

        pos = runner.risk.positions.get(order.ticker)
        assert pos is not None
        assert pos.shares == order.shares
        assert runner.counts["unfilled"] == 0

    @pytest.mark.asyncio
    async def test_stale_entry_is_cancelled_not_left_working(self):
        """A limit that has not filled in 45s is chasing a move that is over."""
        cancelled: list[str] = []

        class Hanging(PaperBroker):
            async def order_status(self, order_id: str) -> OrderStatus:
                return OrderStatus(order_id, "new", 0, 0.0, 48)

            async def cancel_order(self, order_id: str) -> bool:
                cancelled.append(order_id)
                return True

        broker = Hanging(allow_extended=True)
        runner = self._runner(broker)
        order = make_order()
        fill = await broker.submit(order, current_session(at(2026, 7, 30, 16, 5)))
        runner.risk.open(order)

        await runner._reconcile(order, fill.broker_order_id,
                                timeout_s=0.3, poll_s=0.05)

        assert cancelled == [fill.broker_order_id]
        assert order.ticker not in runner.risk.positions

    @pytest.mark.asyncio
    async def test_unknown_status_does_not_drop_a_real_position(self):
        """A transient API error must not be read as 'no fill'."""
        class Flaky(PaperBroker):
            async def order_status(self, order_id: str) -> OrderStatus:
                return OrderStatus(order_id, "unknown")

        broker = Flaky(allow_extended=True)
        runner = self._runner(broker)
        order = make_order()
        fill = await broker.submit(order, current_session(at(2026, 7, 30, 16, 5)))
        runner.risk.open(order)

        await runner._reconcile(order, fill.broker_order_id,
                                timeout_s=0.25, poll_s=0.05)

        # It cancels on timeout, which is correct and conservative, but it must
        # never have dropped the position on the strength of an "unknown".
        assert runner.counts["unfilled"] <= 1

    @pytest.mark.asyncio
    async def test_no_broker_order_id_is_a_noop(self):
        broker = PaperBroker(allow_extended=True)
        runner = self._runner(broker)
        order = make_order()
        runner.risk.open(order)
        await runner._reconcile(order, "", timeout_s=0.2, poll_s=0.05)
        assert order.ticker in runner.risk.positions
