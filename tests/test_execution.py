"""Execution layer: session legality and where the stop lives.

The extended-hours cases are the important ones. Alpaca rejects bracket orders
outside regular hours, so a 16:05 earnings trade cannot carry a broker-side stop
-- and a stop that only exists in a process that can crash is a different risk
profile than one held at the broker. These tests pin that behaviour down.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from signalsniper.execution.broker import AccountSnapshot, PaperBroker
from signalsniper.execution.session import Session, current_session
from signalsniper.models import Direction, Event, EventKind, RawDoc, Signal
from signalsniper.signal.risk import Order

ET = ZoneInfo("America/New_York")


def at(y, m, d, hh, mm) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=ET).astimezone(timezone.utc)


def make_order(ticker="CRUS", direction=Direction.SHORT, shares=48, limit=103.38):
    doc = RawDoc(source="t", doc_id="d", title="t", url="", published=None)
    ev = Event(doc=doc, kind=EventKind.EARNINGS, tickers=(ticker,),
               materiality=0.8, prior=direction, confidence=0.8)
    sig = Signal(ticker=ticker, direction=direction, edge_bps=300.0,
                 confidence=0.7, event=ev, ref_price=limit)
    return Order(ticker=ticker, direction=direction, shares=shares, limit=limit,
                 stop=105.22, target=100.31, signal=sig)


class TestSessionBoundaries:
    @pytest.mark.parametrize("hh,mm,expected", [
        (3, 59, Session.CLOSED),
        (4, 0, Session.PREMARKET),
        (9, 29, Session.PREMARKET),
        (9, 30, Session.REGULAR),
        (15, 59, Session.REGULAR),
        (16, 0, Session.AFTERHOURS),      # the earnings window starts here
        (16, 5, Session.AFTERHOURS),
        (19, 59, Session.AFTERHOURS),
        (20, 0, Session.CLOSED),
    ])
    def test_thursday_boundaries(self, hh, mm, expected):
        # 2026-07-30 is a Thursday -- the day this system was built for.
        assert current_session(at(2026, 7, 30, hh, mm)).session is expected

    def test_only_regular_hours_supports_bracket(self):
        assert current_session(at(2026, 7, 30, 10, 0)).supports_bracket
        assert not current_session(at(2026, 7, 30, 16, 5)).supports_bracket
        assert not current_session(at(2026, 7, 30, 8, 0)).supports_bracket

    def test_weekend_closed(self):
        assert current_session(at(2026, 8, 1, 12, 0)).session is Session.CLOSED  # Sat
        assert current_session(at(2026, 8, 2, 12, 0)).session is Session.CLOSED  # Sun

    def test_holiday_closed(self):
        s = current_session(at(2026, 7, 3, 12, 0))  # Independence Day observed
        assert s.session is Session.CLOSED
        assert "holiday" in s.note

    def test_half_day_closes_early(self):
        # Christmas Eve 2026: regular close 13:00, extended close 17:00.
        assert current_session(at(2026, 12, 24, 12, 59)).session is Session.REGULAR
        assert current_session(at(2026, 12, 24, 13, 1)).session is Session.AFTERHOURS
        assert current_session(at(2026, 12, 24, 17, 1)).session is Session.CLOSED

    def test_naive_datetime_rejected(self):
        with pytest.raises(ValueError, match="naive"):
            current_session(datetime(2026, 7, 30, 16, 5))


class TestPaperBrokerSubmission:
    @pytest.mark.asyncio
    async def test_regular_hours_gets_broker_side_stop(self):
        broker = PaperBroker()
        fill = await broker.submit(make_order(),
                                   current_session(at(2026, 7, 30, 10, 30)))
        assert fill.accepted
        assert fill.stop_is_client_side is False
        assert fill.session == "regular"

    @pytest.mark.asyncio
    async def test_extended_hours_stop_is_client_side(self):
        """The 16:05 trade. This flag is the whole reason the runner tracks it."""
        broker = PaperBroker(allow_extended=True)
        fill = await broker.submit(make_order(),
                                   current_session(at(2026, 7, 30, 16, 5)))
        assert fill.accepted
        assert fill.stop_is_client_side is True
        assert fill.session == "afterhours"

    @pytest.mark.asyncio
    async def test_extended_hours_refused_without_optin(self):
        broker = PaperBroker(allow_extended=False)
        fill = await broker.submit(make_order(),
                                   current_session(at(2026, 7, 30, 16, 5)))
        assert not fill.accepted
        assert "extended" in fill.error

    @pytest.mark.asyncio
    async def test_closed_market_refused(self):
        broker = PaperBroker(allow_extended=True)
        fill = await broker.submit(make_order(),
                                   current_session(at(2026, 7, 30, 22, 0)))
        assert not fill.accepted
        assert "closed" in fill.error

    @pytest.mark.asyncio
    async def test_flatten_clears(self):
        broker = PaperBroker()
        await broker.submit(make_order(), current_session(at(2026, 7, 30, 10, 30)))
        assert await broker.flatten_all() == 1
        assert broker.submitted == []


class TestAccountBlockers:
    def test_pdt_under_25k_is_flagged(self):
        acct = AccountSnapshot(equity=10_000.0, pattern_day_trader=True)
        assert any("PDT" in b for b in acct.blockers())

    def test_under_25k_warns_about_the_three_day_trade_limit(self):
        """This strategy is same-day round trips. Sub-$25k accounts hit PDT fast."""
        acct = AccountSnapshot(equity=10_000.0, pattern_day_trader=False)
        blockers = acct.blockers()
        assert any("3 day trades" in b for b in blockers)

    def test_shorting_disabled_is_flagged_when_needed(self):
        acct = AccountSnapshot(equity=50_000.0, shorting_enabled=False)
        assert acct.blockers(need_short=False) == []
        assert any("horting" in b for b in acct.blockers(need_short=True))

    def test_trading_blocked_is_flagged(self):
        acct = AccountSnapshot(equity=50_000.0, trading_blocked=True)
        assert any("blocked" in b for b in acct.blockers())

    def test_healthy_account_is_clean(self):
        acct = AccountSnapshot(equity=50_000.0, shorting_enabled=True)
        assert acct.blockers(need_short=True) == []


class TestRunnerShutdownProtection:
    """An unprotected position outlives the process that was holding its stop."""

    def _runner(self, live: bool, broker):
        from signalsniper.config import Config
        from signalsniper.runner import Runner
        cfg = Config(sec_user_agent="T t@e.com", live=live, allow_extended=True,
                     watchlist=("CRUS",))
        return Runner(cfg, broker=broker)

    @pytest.mark.asyncio
    async def test_flattens_client_side_stops_on_shutdown(self):
        broker = PaperBroker(allow_extended=True)
        runner = self._runner(live=True, broker=broker)
        await broker.submit(make_order(), current_session(at(2026, 7, 30, 16, 5)))
        runner.client_side_stops.add("CRUS")

        await runner.protect_on_shutdown()

        assert runner.client_side_stops == set()
        assert broker.submitted == []

    @pytest.mark.asyncio
    async def test_no_flatten_when_all_stops_are_broker_side(self):
        """Bracket legs live at the broker and must survive our shutdown."""
        broker = PaperBroker()
        runner = self._runner(live=True, broker=broker)
        await broker.submit(make_order(), current_session(at(2026, 7, 30, 10, 30)))

        await runner.protect_on_shutdown()

        assert len(broker.submitted) == 1  # untouched

    @pytest.mark.asyncio
    async def test_alert_only_never_touches_the_broker(self):
        broker = PaperBroker(allow_extended=True)
        runner = self._runner(live=False, broker=broker)
        runner.client_side_stops.add("CRUS")

        await runner.protect_on_shutdown()

        assert broker.submitted == []

    @pytest.mark.asyncio
    async def test_flatten_failure_does_not_raise(self):
        """If flattening fails we must log CRITICAL, not crash the shutdown path."""
        class Exploding(PaperBroker):
            async def flatten_all(self) -> int:
                raise RuntimeError("broker unreachable")

        broker = Exploding(allow_extended=True)
        runner = self._runner(live=True, broker=broker)
        runner.client_side_stops.add("CRUS")

        await runner.protect_on_shutdown()  # must not raise
        assert runner.client_side_stops == {"CRUS"}  # still flagged as unprotected


class TestConfigGuards:
    def test_live_plus_extended_is_flagged(self):
        from signalsniper.config import Config
        cfg = Config(sec_user_agent="A a@b.com", live=True, allow_extended=True,
                     alpaca_key="k", alpaca_secret="s", alpaca_paper=False,
                     alpaca_feed="sip")
        assert any("broker-side stop" in p for p in cfg.validate())

    def test_live_on_iex_is_flagged(self):
        from signalsniper.config import Config
        cfg = Config(sec_user_agent="A a@b.com", live=True, alpaca_key="k",
                     alpaca_secret="s", alpaca_paper=False, alpaca_feed="iex")
        assert any("iex" in p for p in cfg.validate())
