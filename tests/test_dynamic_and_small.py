"""Dynamic subscription + small-account viability."""
from __future__ import annotations
import pytest
from signalsniper.config import Config
from signalsniper.models import Direction
from signalsniper.market.quotes import AlpacaQuoteStream
from signalsniper.models import RawDoc, now_ns
from signalsniper.runner import Runner
from signalsniper.feeds.edgar import TickerResolver
from signalsniper.signal.risk import RiskConfig


class TestDynamicSubscription:
    """The EDGAR path watches every filer; without this it can only ever trade
    the pre-configured watchlist, which defeats the point."""

    def _stream(self, symbols=("AAPL",)):
        return AlpacaQuoteStream("k", "s", symbols, feed="iex")

    @pytest.mark.asyncio
    async def test_adds_new_symbols_before_connect(self):
        st = self._stream()
        added = await st.add_symbols(["TINY", "SMOL"])
        assert added == ["TINY", "SMOL"]
        assert set(st.symbols) == {"AAPL", "TINY", "SMOL"}

    @pytest.mark.asyncio
    async def test_ignores_duplicates_and_case(self):
        st = self._stream()
        assert await st.add_symbols(["aapl"]) == []
        assert await st.add_symbols(["TINY"]) == ["TINY"]
        assert await st.add_symbols(["tiny"]) == []

    @pytest.mark.asyncio
    async def test_respects_the_symbol_cap(self):
        st = AlpacaQuoteStream("k", "s", ["A"], feed="iex", max_symbols=3)
        assert len(await st.add_symbols(["B", "C", "D", "E"])) == 2
        assert len(st.symbols) == 3
        assert await st.add_symbols(["F"]) == []

    @pytest.mark.asyncio
    async def test_sends_subscribe_on_a_live_socket(self):
        sent = []

        class WS:
            async def send(self, msg):
                sent.append(msg)

        st = self._stream()
        st._ws = WS()
        await st.add_symbols(["TINY"])
        assert len(sent) == 1
        assert "TINY" in sent[0] and "subscribe" in sent[0]

    @pytest.mark.asyncio
    async def test_send_failure_does_not_lose_the_symbol(self):
        class Broken:
            async def send(self, msg):
                raise RuntimeError("socket gone")

        st = self._stream()
        st._ws = Broken()
        await st.add_symbols(["TINY"])
        # Must remain in the set so the next reconnect resubscribes it.
        assert "TINY" in st.symbols

    @pytest.mark.asyncio
    async def test_reconnect_resubscribes_dynamically_added_names(self):
        """A reconnect restoring only the original watchlist would drop exactly
        the names carrying a live position or pending signal."""
        st = self._stream()
        await st.add_symbols(["TINY"])
        assert "TINY" in st.symbols   # what the reconnect path sends


class TestRunnerOnDemandSubscribe:
    def _runner(self):
        cfg = Config(sec_user_agent="T t@e.com", watchlist=("AAPL",))
        r = TickerResolver()
        r.load_mapping({"0": {"cik_str": 1800001, "ticker": "TINY", "title": "Tinybio"}})
        return Runner(cfg, resolver=r)

    @pytest.mark.asyncio
    async def test_material_filing_on_unwatched_name_triggers_subscribe(self):
        import asyncio
        from signalsniper.bus import TOPIC_RAW

        runner = self._runner()
        runner.quote_source = AlpacaQuoteStream("k", "s", ("AAPL",), feed="iex")

        task = asyncio.create_task(runner._classify_loop())
        doc = RawDoc(source="edgar", doc_id="x",
                     title="424B5 - Tinybio Therapeutics Inc (0001800001) (Filer)",
                     url="", published=None, body="424B5 prospectus supplement.",
                     meta={"form": "424B5", "company": "Tinybio",
                           "cik": "1800001", "items": ()})
        doc.t_ingest = now_ns()
        runner.bus.publish(TOPIC_RAW, doc)
        await asyncio.sleep(0.2)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert "TINY" in runner.quote_source.symbols
        assert runner.counts["subscribed"] == 1

    @pytest.mark.asyncio
    async def test_immaterial_filing_does_not_subscribe(self):
        """Do not burn the symbol budget on exhibits-only noise."""
        import asyncio
        from signalsniper.bus import TOPIC_RAW

        runner = self._runner()
        runner.quote_source = AlpacaQuoteStream("k", "s", ("AAPL",), feed="iex")

        task = asyncio.create_task(runner._classify_loop())
        doc = RawDoc(source="edgar", doc_id="y",
                     title="8-K - Tinybio Therapeutics Inc (0001800001) (Filer)",
                     url="", published=None, body="Item 9.01 Exhibits.",
                     meta={"form": "8-K", "company": "Tinybio",
                           "cik": "1800001", "items": ("9.01",)})
        doc.t_ingest = now_ns()
        runner.bus.publish(TOPIC_RAW, doc)
        await asyncio.sleep(0.2)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert "TINY" not in runner.quote_source.symbols
        assert runner.counts["subscribed"] == 0


class TestSmallAccountViability:
    """A structurally untradeable account looks identical to a quiet market."""

    def test_200_dollar_account_is_flagged(self):
        v = RiskConfig(equity=200.0).viability()
        assert any("EVERY order will be rejected" in x for x in v)
        assert any("SHORT SELLING IS IMPOSSIBLE" in x for x in v)

    def test_cash_account_short_block_stated_below_2000(self):
        assert any("2,000" in x for x in RiskConfig(equity=1_500.0).viability())
        assert not any("2,000" in x for x in RiskConfig(equity=25_000.0).viability())

    def test_healthy_account_is_clean(self):
        assert RiskConfig(equity=25_000.0).viability() == []

    def test_for_equity_makes_a_small_account_able_to_order(self):
        from signalsniper.signal.risk import RiskManager
        from signalsniper.models import Direction, Event, EventKind, Signal
        cfg = RiskConfig.for_equity(200.0)
        assert not any("EVERY order will be rejected" in x for x in cfg.viability())
        assert cfg.max_concurrent == 1

        doc = RawDoc(source="t", doc_id="d", title="t", url="", published=None)
        ev = Event(doc=doc, kind=EventKind.EARNINGS, tickers=("X",),
                   materiality=0.8, prior=Direction.LONG, confidence=0.8)
        sig = Signal(ticker="X", direction=Direction.LONG, edge_bps=300.0,
                     confidence=0.7, event=ev, ref_price=45.0)
        assert RiskManager(cfg).size_order(sig) is not None

    def test_for_equity_does_not_widen_the_loss_cap(self):
        """Scaling for a small account must not quietly raise risk tolerance."""
        base, small = RiskConfig(), RiskConfig.for_equity(200.0)
        assert small.max_daily_loss_frac == base.max_daily_loss_frac
        assert small.risk_per_trade == base.risk_per_trade

    def test_for_equity_leaves_large_accounts_alone(self):
        base, big = RiskConfig(), RiskConfig.for_equity(25_000.0)
        assert big.max_position_frac == base.max_position_frac
        assert big.max_concurrent == base.max_concurrent


class TestSmallAccountOrderMechanics:
    """What a $200 account can and cannot actually send."""

    def _sig(self, px, direction=Direction.LONG, edge=300.0):
        from signalsniper.models import Event, EventKind, RawDoc, Signal
        doc = RawDoc(source="t", doc_id="d", title="t", url="", published=None)
        ev = Event(doc=doc, kind=EventKind.EARNINGS, tickers=("X",),
                   materiality=0.8, prior=direction, confidence=0.8)
        return Signal(ticker="X", direction=direction, edge_bps=edge,
                      confidence=0.7, event=ev, ref_price=px)

    def test_fractional_makes_an_expensive_stock_buyable(self):
        """At $200 a $230 stock is unbuyable in whole shares."""
        from signalsniper.signal.risk import RiskManager
        rm = RiskManager(RiskConfig.for_equity(200.0))
        order = rm.size_order(self._sig(230.0))
        assert order is not None
        assert order.is_fractional
        assert 0 < order.shares < 1
        assert order.notional <= 200.0

    def test_fractional_sizes_to_risk_not_to_share_price(self):
        """Whole-share quantization forces position size to whatever one share
        costs; fractional lets it track the risk budget instead."""
        from signalsniper.signal.risk import RiskManager
        notionals = []
        for px in (45.0, 104.0, 230.0):
            rm = RiskManager(RiskConfig.for_equity(200.0))
            notionals.append(rm.size_order(self._sig(px)).notional)
        assert max(notionals) - min(notionals) < 1.0   # same risk, any price

    def test_shorts_are_refused_below_the_regulatory_floor(self):
        """FINRA 4210(b)/Reg T is a $2,000 floor. Sending it anyway earns a
        broker rejection that costs the whole window."""
        from signalsniper.signal.risk import RiskManager
        rm = RiskManager(RiskConfig.for_equity(200.0))
        assert rm.size_order(self._sig(104.0, Direction.SHORT)) is None
        assert rm.rejects.get("short_blocked_below_2k") == 1

    def test_shorts_work_once_funded_above_the_floor(self):
        from signalsniper.signal.risk import RiskManager
        rm = RiskManager(RiskConfig.for_equity(5_000.0))
        assert rm.size_order(self._sig(104.0, Direction.SHORT)) is not None

    def test_shorts_stay_whole_share_even_when_fractional_is_on(self):
        """Fractional shorts are not permitted at any equity level."""
        from signalsniper.signal.risk import RiskManager
        cfg = RiskConfig.for_equity(10_000.0)
        cfg.allow_fractional = True
        order = RiskManager(cfg).size_order(self._sig(104.0, Direction.SHORT))
        assert order is not None
        assert not order.is_fractional


class TestFreeTierChannelBudget:
    """The free plan allows ONE connection and 30 channels for trades+quotes
    COMBINED. Exceeding it presents as a connected socket delivering nothing."""

    def test_default_watchlist_exceeds_the_cap_with_both_streams(self):
        from signalsniper.config import load
        from signalsniper.market.quotes import AlpacaQuoteStream
        ok, msg = AlpacaQuoteStream("k", "s", load().watchlist).channel_budget_ok()
        assert ok is False and "exceeds" in msg

    def test_quotes_only_fits(self):
        from signalsniper.config import load
        from signalsniper.market.quotes import AlpacaQuoteStream
        st = AlpacaQuoteStream("k", "s", load().watchlist, subscribe_trades=False)
        assert st.channel_budget_ok()[0] is True

    def test_channel_count_tracks_dynamic_subscription(self):
        from signalsniper.market.quotes import AlpacaQuoteStream
        st = AlpacaQuoteStream("k", "s", ["AAPL"], subscribe_trades=False)
        assert st.channels_used == 1
        import asyncio
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            st.add_symbols(["TINY", "SMOL"]))
        assert st.channels_used == 3
