"""Regression tests for defects found by adversarial review of the live path.

Every one of these bugs existed while 250 tests passed, because the fixtures
shared the defect with the code. The clock bug is the clearest case: fixtures
built tapes from the same monotonic clock they stamped events with, so the
mismatch only existed once a real venue quote entered the tape.

The lesson these tests encode: a fixture that mirrors your assumption tests the
assumption, not the system. Where possible below, the fixture deliberately uses
the PRODUCTION representation (epoch stamps from a parsed RFC3339 string) rather
than a convenient in-process one.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from signalsniper.config import Config
from signalsniper.execution.broker import PaperBroker, to_broker_symbol
from signalsniper.execution.session import current_session
from signalsniper.market.quotes import AlpacaQuoteStream, _parse_rfc3339_ns
from signalsniper.market.tape import MarketState, TickerTape
from signalsniper.models import (
    Direction, Event, EventKind, Quote, RawDoc, Signal, epoch_ns, mono_ns,
)
from signalsniper.runner import Runner
from signalsniper.signal.risk import RiskConfig, RiskManager

from test_execution import at, make_order


def venue_quote(ticker: str, px: float, when: datetime) -> Quote:
    """A quote stamped the way the live feed stamps one -- epoch ns from RFC3339."""
    return Quote(ticker, px - 0.01, px + 0.01, px, 100,
                 _parse_rfc3339_ns(when.isoformat()))


class TestClockDomains:
    """The two clocks are 300,000x apart. Mixing them silently returns garbage."""

    def test_the_two_clocks_are_not_interchangeable(self):
        assert epoch_ns() / mono_ns() > 1000

    def test_event_time_shares_the_clock_of_venue_quotes(self):
        doc = RawDoc(source="edgar", doc_id="d", title="t", url="", published=None)
        now = _parse_rfc3339_ns(datetime.now(timezone.utc).isoformat())
        assert abs(doc.t_ingest - now) < 5e9      # same domain, seconds apart
        assert doc.t_mono < doc.t_ingest / 1000   # different domain entirely

    def test_reference_price_is_correct_against_venue_stamps(self):
        """The original bug: bisect found index -1 and returned the OLDEST price
        in the tape for every event, so 'already priced in' was always wrong."""
        tape = TickerTape("CRUS")
        t0 = datetime.now(timezone.utc) - timedelta(seconds=10)
        for i, px in enumerate([104.0, 103.0, 102.0, 101.0, 100.0]):
            tape.on_quote(venue_quote("CRUS", px, t0 + timedelta(seconds=i)))

        mid = _parse_rfc3339_ns((t0 + timedelta(seconds=2)).isoformat())
        assert tape.price_at(mid) == pytest.approx(102.0)
        assert tape.move_bps_since(mid) == pytest.approx(-196.0, abs=2.0)

    def test_event_now_reports_no_move_not_the_whole_history(self):
        tape = TickerTape("CRUS")
        t0 = datetime.now(timezone.utc) - timedelta(seconds=10)
        for i, px in enumerate([104.0, 103.0, 102.0, 101.0, 100.0]):
            tape.on_quote(venue_quote("CRUS", px, t0 + timedelta(seconds=i)))

        doc = RawDoc(source="edgar", doc_id="d", title="t", url="", published=None)
        # The event is NOW; the tape is entirely in the past. Nothing has moved
        # since. The bug reported -385bps here.
        assert tape.move_bps_since(doc.t_ingest) == pytest.approx(0.0, abs=1.0)

    def test_latency_stays_a_monotonic_delta(self):
        doc = RawDoc(source="edgar", doc_id="d", title="t", url="", published=None)
        ev = Event(doc=doc, kind=EventKind.EARNINGS, tickers=("X",),
                   materiality=0.8, prior=Direction.SHORT, confidence=0.8)
        assert 0 <= ev.ingest_latency_us < 1_000_000   # microseconds, not 1e18

    def test_signal_age_stays_a_monotonic_delta(self):
        doc = RawDoc(source="t", doc_id="d", title="t", url="", published=None)
        ev = Event(doc=doc, kind=EventKind.EARNINGS, tickers=("X",),
                   materiality=0.8, prior=Direction.SHORT, confidence=0.8)
        sig = Signal(ticker="X", direction=Direction.SHORT, edge_bps=300.0,
                     confidence=0.7, event=ev, ref_price=100.0)
        assert 0 <= sig.age_s < 5.0
        assert not sig.expired


class TestKillSwitchSeesOpenPositions:
    """Positions opened off one event are one bet. While they are all open the
    realized figure is zero, so a realized-only switch sits at zero through the
    entire drawdown."""

    def _rm(self):
        rm = RiskManager(RiskConfig(equity=10_000.0, max_daily_loss_frac=0.03,
                                    max_concurrent=4, risk_per_trade=0.01))
        for t in ("CRUS", "GLW", "TXN", "COHR"):
            doc = RawDoc(source="t", doc_id="d", title="t", url="", published=None)
            ev = Event(doc=doc, kind=EventKind.EARNINGS, tickers=(t,),
                       materiality=0.8, prior=Direction.LONG, confidence=0.8)
            sig = Signal(ticker=t, direction=Direction.LONG, edge_bps=300.0,
                         confidence=0.7, event=ev, ref_price=100.0)
            o = rm.size_order(sig)
            if o:
                rm.open(o)
        return rm

    def test_realized_only_check_is_blind_to_an_open_drawdown(self):
        rm = self._rm()
        assert rm.check_kill_switch() is False       # nothing realized yet
        assert len(rm.positions) >= 2

    def test_unrealized_loss_trips_the_switch(self):
        rm = self._rm()
        marks = {t: 50.0 for t in rm.positions}      # every position halved
        assert rm.check_kill_switch(marks) is True
        assert "incl. open" in rm.halt_reason

    def test_unrealized_profit_does_not_trip_it(self):
        rm = self._rm()
        assert rm.check_kill_switch({t: 110.0 for t in rm.positions}) is False

    def test_stays_sticky_once_tripped(self):
        rm = self._rm()
        rm.check_kill_switch({t: 50.0 for t in rm.positions})
        assert rm.check_kill_switch({t: 200.0 for t in rm.positions}) is True


class TestExitOrderingAndSafety:
    def _runner(self, broker, live=True):
        cfg = Config(sec_user_agent="T t@e.com", live=live, allow_extended=True,
                     watchlist=("CRUS",))
        return Runner(cfg, broker=broker)

    def _open(self, runner, ticker="CRUS", entry=103.38):
        order = make_order(ticker=ticker, limit=entry)
        runner.risk.open(order)
        runner.client_side_stops.add(ticker)
        return order

    def test_pending_exits_does_not_mutate(self):
        rm = RiskManager(RiskConfig(equity=200_000.0))
        order = make_order(direction=Direction.LONG, limit=100.0)
        rm.open(order)
        pos = rm.positions["CRUS"]
        triggered = rm.pending_exits({"CRUS": pos.stop - 1.0})
        assert triggered and triggered[0][2] == "stop"
        assert "CRUS" in rm.positions      # still there
        assert rm.realized_pnl == 0.0      # nothing booked

    @pytest.mark.asyncio
    async def test_rejected_close_keeps_the_position(self):
        """Booking the fill first would delete a still-live position from memory,
        leaving it untracked and unstopped at the broker."""
        class Rejecting(PaperBroker):
            async def submit(self, order, session=None):
                from signalsniper.execution.broker import Fill
                return Fill(accepted=False, ticker=order.ticker,
                            error="insufficient buying power")

        broker = Rejecting(allow_extended=True)
        runner = self._runner(broker)
        self._open(runner)
        pos = runner.risk.positions["CRUS"]

        sent = await runner._close_at_broker(
            "CRUS", pos.direction, pos.shares, 105.5, "stop", pos.signal)

        assert sent is False
        # The caller must not book the exit when send failed.
        assert "CRUS" in runner.risk.positions
        assert runner.risk.realized_pnl == 0.0

    @pytest.mark.asyncio
    async def test_accepted_close_reports_success(self):
        broker = PaperBroker(allow_extended=True)
        runner = self._runner(broker)
        self._open(runner)
        pos = runner.risk.positions["CRUS"]
        sent = await runner._close_at_broker(
            "CRUS", pos.direction, pos.shares, 105.5, "stop", pos.signal)
        assert sent is True

    @pytest.mark.asyncio
    async def test_exit_limit_crosses_a_wide_book(self):
        """A fixed 40bps only reaches the far side while the spread is under
        80bps. On a 300bps after-hours book it rests inside the spread forever."""
        broker = PaperBroker(allow_extended=True)
        runner = self._runner(broker)
        self._open(runner)

        t0 = datetime.now(timezone.utc)
        tape = runner.market.tape("CRUS")
        for i in range(5):
            mid = 100.0
            half = mid * 300.0 / 2 / 10_000.0     # 300bps spread
            tape.on_quote(Quote("CRUS", mid - half, mid + half, mid, 100,
                                _parse_rfc3339_ns((t0 + timedelta(seconds=i)).isoformat())))

        pos = runner.risk.positions["CRUS"]
        await runner._close_at_broker("CRUS", pos.direction, pos.shares,
                                      100.0, "stop", pos.signal)
        sent = broker.submitted[-1]
        # Covering a short: must be priced at or above the ask (100 + 1.5).
        assert sent.direction is Direction.LONG
        assert sent.limit >= 101.4, f"limit {sent.limit} does not cross a 300bps book"


class TestSymbolNormalization:
    @pytest.mark.parametrize("sec,alpaca", [
        ("BRK-B", "BRK.B"), ("BF-B", "BF.B"), ("HEI-A", "HEI.A"),
        ("AAPL", "AAPL"), ("crus", "CRUS"), (" GLW ", "GLW"),
    ])
    def test_sec_hyphen_becomes_alpaca_dot(self, sec, alpaca):
        assert to_broker_symbol(sec) == alpaca

    @pytest.mark.asyncio
    async def test_submitted_payload_uses_the_broker_form(self):
        import httpx

        captured = {}

        class FakeClient:
            async def post(self, url, headers=None, json=None):
                captured.update(json)
                return httpx.Response(200, json={"id": "abc"})

        from signalsniper.execution.broker import AlpacaBroker
        b = AlpacaBroker("k", "s", paper=True, client=FakeClient())
        await b.submit(make_order(ticker="BRK-B"),
                       current_session(at(2026, 7, 30, 10, 30)))
        assert captured["symbol"] == "BRK.B"


class TestWebsocketAuth:
    """A wrong key produced a socket that connected, yielded nothing, and looked
    exactly like a quiet market."""

    class FakeWS:
        def __init__(self, frames): self.frames = list(frames)
        async def recv(self):
            if not self.frames:
                raise RuntimeError("closed")
            return self.frames.pop(0)

    @pytest.mark.asyncio
    async def test_reads_past_the_connect_frame_to_the_auth_ack(self):
        st = AlpacaQuoteStream("k", "s", ["AAPL"])
        assert await st._await_auth(self.FakeWS([
            json.dumps([{"T": "success", "msg": "connected"}]),
            json.dumps([{"T": "success", "msg": "authenticated"}]),
        ])) is True

    @pytest.mark.asyncio
    async def test_explicit_error_is_a_failure(self):
        st = AlpacaQuoteStream("k", "s", ["AAPL"])
        assert await st._await_auth(self.FakeWS([
            json.dumps([{"T": "success", "msg": "connected"}]),
            json.dumps([{"T": "error", "code": 402, "msg": "auth failed"}]),
        ])) is False

    @pytest.mark.asyncio
    async def test_connect_without_auth_ack_is_a_failure(self):
        st = AlpacaQuoteStream("k", "s", ["AAPL"])
        assert await st._await_auth(self.FakeWS([
            json.dumps([{"T": "success", "msg": "connected"}]),
        ])) is False


class TestReconcileUnknownStatus:
    @pytest.mark.asyncio
    async def test_unknown_on_timeout_keeps_the_position(self):
        """A transient API error is not evidence of no fill. Dropping the
        position leaves a real one untracked; keeping a phantom blocks a slot."""
        from signalsniper.execution.broker import OrderStatus

        class Opaque(PaperBroker):
            async def order_status(self, order_id):
                return OrderStatus(order_id, "unknown")
            async def cancel_order(self, order_id):
                return True

        broker = Opaque(allow_extended=True)
        cfg = Config(sec_user_agent="T t@e.com", live=True, allow_extended=True,
                     watchlist=("CRUS",))
        runner = Runner(cfg, broker=broker)
        order = make_order()
        fill = await broker.submit(order, current_session(at(2026, 7, 30, 16, 5)))
        runner.risk.open(order)

        await runner._reconcile(order, fill.broker_order_id,
                                timeout_s=0.2, poll_s=0.05)

        assert "CRUS" in runner.risk.positions
        assert runner.counts["unfilled"] == 0


class TestStartupAdoption:
    """A restart with positions open leaves them unmanaged: RiskManager thinks
    it is flat, no stop is evaluated, and shutdown will not flatten them."""

    def _runner(self, broker):
        cfg = Config(sec_user_agent="T t@e.com", live=True, allow_extended=True,
                     watchlist=("CRUS",))
        return Runner(cfg, broker=broker)

    @pytest.mark.asyncio
    async def test_existing_positions_are_detected_and_reported(self):
        class WithPositions(PaperBroker):
            async def positions(self):
                return [{"symbol": "CRUS", "qty": "48"},
                        {"symbol": "GLW", "qty": "10"}]

        runner = self._runner(WithPositions())
        assert await runner.adopt_broker_positions() == 2
        assert runner.counts["adopted"] == 2

    @pytest.mark.asyncio
    async def test_flat_account_is_silent(self):
        class Flat(PaperBroker):
            async def positions(self):
                return []

        runner = self._runner(Flat())
        assert await runner.adopt_broker_positions() == 0
        assert runner.counts["adopted"] == 0

    @pytest.mark.asyncio
    async def test_broker_error_does_not_crash_startup(self):
        class Broken(PaperBroker):
            async def positions(self):
                raise RuntimeError("API down")

        runner = self._runner(Broken())
        assert await runner.adopt_broker_positions() == 0
