"""Bus semantics, ticker attribution, and a full async pipeline pass."""

from __future__ import annotations

import asyncio

import pytest

from signalsniper.bus import TOPIC_RAW, TOPIC_SIGNAL, EventBus
from signalsniper.config import Config
from signalsniper.feeds.base import TokenBucket
from signalsniper.feeds.edgar import TickerResolver
from signalsniper.market.quotes import ReplaySource, synth_walk
from signalsniper.models import RawDoc, now_ns
from signalsniper.runner import Runner, _extract_cashtags


class TestBus:
    def test_fanout_to_every_subscriber(self):
        bus = EventBus()
        a = bus.subscribe("t", name="a")
        b = bus.subscribe("t", name="b")
        assert bus.publish("t", 1) == 2
        assert a.queue.get_nowait() == 1
        assert b.queue.get_nowait() == 1

    def test_publish_with_no_subscribers_is_fine(self):
        assert EventBus().publish("nobody", 1) == 0

    def test_overflow_drops_oldest_and_keeps_newest(self):
        """A stalled consumer must never block the feed reader."""
        bus = EventBus()
        sub = bus.subscribe("t", maxsize=2, name="slow")
        for i in range(5):
            bus.publish("t", i)
        assert sub.dropped == 3
        drained = [sub.queue.get_nowait() for _ in range(sub.queue.qsize())]
        assert 4 in drained          # newest survived
        assert 0 not in drained      # oldest evicted

    def test_unsubscribe(self):
        bus = EventBus()
        sub = bus.subscribe("t")
        sub.close()
        assert bus.publish("t", 1) == 0


class TestCashtagExtraction:
    """Bare substring matching is what makes naive scrapers worthless."""

    def test_exchange_prefixed(self):
        assert _extract_cashtags("Acme Corp (NASDAQ: ACME) reports", set()) == ("ACME",)
        assert _extract_cashtags("Beta (NYSE: BET) reports", set()) == ("BET",)

    def test_cashtag(self):
        assert _extract_cashtags("watching $AAPL and $CRUS today", set()) == ("AAPL", "CRUS")

    @pytest.mark.parametrize("text", [
        "AMC theaters reopened downtown",
        "the CEO said ALL of the guidance stands",
        "reported under GAAP and non-GAAP measures",
        "ON Semiconductor was not mentioned here as a ticker",
    ])
    def test_no_bare_substring_false_positives(self, text):
        assert _extract_cashtags(text, {"AMC", "ALL", "GAAP", "ON"}) == ()

    def test_universe_filter_kills_common_word_tickers(self):
        text = "we improved $IT spend and $ALL of the margins"
        assert _extract_cashtags(text, {"AAPL", "CRUS"}) == ()
        assert _extract_cashtags(text, {"IT", "ALL"}) == ("IT", "ALL")

    def test_deduplicates_preserving_order(self):
        assert _extract_cashtags("$AAPL $CRUS $AAPL", set()) == ("AAPL", "CRUS")


class TestTokenBucket:
    @pytest.mark.asyncio
    async def test_allows_a_burst_up_to_capacity(self):
        bucket = TokenBucket(rate=100.0, capacity=5.0)
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        for _ in range(5):
            await bucket.take()
        assert loop.time() - t0 < 0.05

    @pytest.mark.asyncio
    async def test_throttles_past_capacity(self):
        bucket = TokenBucket(rate=20.0, capacity=2.0)
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        for _ in range(6):
            await bucket.take()
        # 4 tokens beyond capacity at 20/s is ~0.2s of enforced waiting.
        assert loop.time() - t0 >= 0.15


class TestRunnerAttribution:
    def _runner(self):
        cfg = Config(sec_user_agent="Test test@example.com",
                     watchlist=("AAPL", "CRUS", "ACME"))
        resolver = TickerResolver()
        resolver.load_mapping({
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        })
        return Runner(cfg, resolver=resolver)

    def test_edgar_doc_resolves_via_cik(self):
        r = self._runner()
        doc = RawDoc(source="edgar", doc_id="a", title="8-K", url="", published=None,
                     meta={"cik": "0000320193", "company": "Apple Inc."})
        assert r.tickers_for(doc) == ("AAPL",)

    def test_unknown_cik_yields_nothing(self):
        r = self._runner()
        doc = RawDoc(source="edgar", doc_id="a", title="8-K", url="", published=None,
                     meta={"cik": "999999", "company": "Who Knows Inc"})
        assert r.tickers_for(doc) == ()

    def test_ir_feed_preattribution_wins(self):
        r = self._runner()
        doc = RawDoc(source="newswire", doc_id="a", title="anything", url="",
                     published=None, meta={"tickers": ("CRUS",)})
        assert r.tickers_for(doc) == ("CRUS",)

    def test_newswire_uses_explicit_tickers_only(self):
        r = self._runner()
        doc = RawDoc(source="newswire", doc_id="a",
                     title="Acme Corp (NASDAQ: ACME) Raises Guidance",
                     url="", published=None, meta={})
        assert r.tickers_for(doc) == ("ACME",)


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_doc_to_signal_end_to_end(self):
        """Publish a raw EDGAR doc, assert a second-order signal comes out."""
        cfg = Config(sec_user_agent="Test test@example.com",
                     watchlist=("AAPL", "CRUS"))
        resolver = TickerResolver()
        resolver.load_mapping({
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        })
        runner = Runner(cfg, resolver=resolver)
        # Testing pipeline mechanics, not the gating policy.
        runner.engine.cfg.verified_links_only = False

        # Prime both tapes, then move AAPL hard and leave CRUS behind.
        t_pre = now_ns() - 8_000_000_000
        for tick, px in (("AAPL", 232.0), ("CRUS", 104.0)):
            for q in synth_walk(tick, px, 40, t_pre):
                runner.market.on_quote(q)
        t_event = now_ns()
        for q in synth_walk("AAPL", 232.0, 30, t_event, drift_bps_total=-450.0):
            runner.market.on_quote(q)
        for q in synth_walk("CRUS", 104.0, 30, t_event, drift_bps_total=-50.0):
            runner.market.on_quote(q)

        got: list = []
        runner.on_signal = got.append
        sink = runner.bus.subscribe(TOPIC_SIGNAL, name="test")

        tasks = [asyncio.create_task(runner._classify_loop()),
                 asyncio.create_task(runner._signal_loop())]

        doc = RawDoc(
            source="edgar", doc_id="0000320193-26-000077",
            title="8-K - Apple Inc. (0000320193) (Filer)", url="", published=None,
            body="Item 2.02 Results of Operations. The Company lowered iPhone "
                 "hardware guidance below consensus estimates.",
            meta={"form": "8-K", "company": "Apple Inc.", "cik": "320193",
                  "items": ("2.02",)},
        )
        doc.t_ingest = t_event
        runner.bus.publish(TOPIC_RAW, doc)

        sig = await asyncio.wait_for(sink.get(), timeout=2.0)
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        assert sig.ticker == "CRUS"
        assert sig.hop == 1
        assert runner.counts["docs"] == 1
        assert runner.counts["signals"] >= 1
        assert got, "signal sink should have fired"

    @pytest.mark.asyncio
    async def test_replay_source_yields_in_order(self):
        quotes = synth_walk("AAA", 100.0, 10, now_ns())
        seen = [q async for q in ReplaySource(quotes, speed=0.0).stream()]
        assert len(seen) == 10
        assert [q.t_source for q in seen] == sorted(q.t_source for q in seen)


class TestConfigValidation:
    def test_missing_user_agent_is_flagged(self):
        assert any("SEC_USER_AGENT" in p for p in Config(sec_user_agent="").validate())

    def test_excessive_sec_rate_is_flagged(self):
        cfg = Config(sec_user_agent="A a@b.com", sec_rate=15.0)
        assert any("IP block" in p for p in cfg.validate())

    def test_live_without_credentials_is_flagged(self):
        cfg = Config(sec_user_agent="A a@b.com", live=True, alpaca_key="", alpaca_secret="")
        assert any("credentials" in p for p in cfg.validate())

    def test_live_and_paper_together_is_flagged(self):
        cfg = Config(sec_user_agent="A a@b.com", live=True, alpaca_key="k",
                     alpaca_secret="s", alpaca_paper=True)
        assert any("pick one" in p for p in cfg.validate())

    def test_clean_config_passes(self):
        assert Config(sec_user_agent="Dylan dylan@example.com").validate() == []
