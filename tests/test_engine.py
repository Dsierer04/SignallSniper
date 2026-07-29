"""The gate. These tests are the ones that matter -- they encode when we DON'T trade."""

from __future__ import annotations

import pytest

from signalsniper.market.linkage import LinkageGraph, Link, detect_channels
from signalsniper.market.quotes import synth_walk
from signalsniper.market.tape import MarketState
from signalsniper.models import Direction, RawDoc, now_ns
from signalsniper.parse.classify import build_event
from signalsniper.signal.engine import DEFAULT_MOVE_SCALE, EngineConfig, SignalEngine


def make_engine(market, **cfg_kw):
    cfg = EngineConfig(move_scale=dict(DEFAULT_MOVE_SCALE), **cfg_kw)
    return SignalEngine(market, LinkageGraph(), cfg)


def seed(market, ticker, price, t0, n=40, drift=0.0, spread_bps=8.0):
    for q in synth_walk(ticker, price, n, t0, drift_bps_total=drift, spread_bps=spread_bps):
        market.on_quote(q)


def aapl_doc(t_ingest, body=None):
    d = RawDoc(
        source="edgar", doc_id="acc-1",
        title="8-K - Apple Inc. (0000320193) (Filer)",
        url="", published=None,
        body=body or ("Item 2.02 Results of Operations. The Company lowered its "
                      "iPhone hardware guidance below consensus estimates."),
        meta={"form": "8-K", "company": "Apple Inc.", "cik": "320193", "items": ("2.02",)},
    )
    d.t_ingest = t_ingest
    return d


@pytest.fixture
def scenario():
    """AAPL drops 420bps on a guidance cut; suppliers have barely moved."""
    market = MarketState()
    t_pre = now_ns() - 8_000_000_000
    for tick, px in [("AAPL", 232.0), ("CRUS", 104.0), ("SWKS", 78.0), ("QRVO", 92.0)]:
        seed(market, tick, px, t_pre, n=40, drift=0.0)
    t_event = now_ns()
    return market, t_event


class TestSecondOrder:
    def test_finds_the_lagging_supplier(self, scenario):
        market, t_event = scenario
        seed(market, "AAPL", 232.0, t_event, n=30, drift=-420.0)
        seed(market, "CRUS", 104.0, t_event, n=30, drift=-60.0)

        event = build_event(aapl_doc(t_event), ("AAPL",))
        sigs = make_engine(market).on_event(event)

        crus = [s for s in sigs if s.ticker == "CRUS"]
        assert crus, "CRUS should signal: implied -357bps, actual -60bps"
        s = crus[0]
        assert s.direction is Direction.SHORT
        assert s.hop == 1
        # implied = -420 * 0.85 = -357; residual = -357 - (-60) = -297
        assert 280 < s.edge_bps < 315

    def test_rejects_a_name_that_already_repriced(self, scenario):
        market, t_event = scenario
        seed(market, "AAPL", 232.0, t_event, n=30, drift=-420.0)
        # QRVO beta 0.55 -> implied -231bps. It has already done -330.
        seed(market, "QRVO", 92.0, t_event, n=30, drift=-330.0)

        event = build_event(aapl_doc(t_event), ("AAPL",))
        sigs = make_engine(market).on_event(event)

        assert not [s for s in sigs if s.ticker == "QRVO"]

    def test_inverse_polarity_link_signals_the_other_way(self):
        """AMZN crushing retail is bad for SHOP. The reflexive trade is wrong."""
        market = MarketState()
        t_pre = now_ns() - 8_000_000_000
        seed(market, "AMZN", 190.0, t_pre, n=40)
        seed(market, "SHOP", 105.0, t_pre, n=40)
        t_event = now_ns()
        seed(market, "AMZN", 190.0, t_event, n=30, drift=+600.0)
        seed(market, "SHOP", 105.0, t_event, n=30, drift=0.0)

        doc = RawDoc(
            source="edgar", doc_id="acc-2",
            title="8-K - Amazon.com Inc. (0001018724) (Filer)",
            url="", published=None,
            body="Item 2.02 Results of Operations. Online stores and third-party "
                 "seller retail revenue exceeded consensus estimates.",
            meta={"form": "8-K", "company": "Amazon.com Inc.", "cik": "1018724",
                  "items": ("2.02",)},
        )
        doc.t_ingest = t_event
        sigs = make_engine(market).on_event(build_event(doc, ("AMZN",)))

        shop = [s for s in sigs if s.ticker == "SHOP"]
        assert shop, "SHOP should signal on the inverse retail link"
        assert shop[0].direction is Direction.SHORT

    def test_channel_gating_blocks_irrelevant_links(self):
        """A Services-only story must not propagate to RF front-end suppliers."""
        market = MarketState()
        t_pre = now_ns() - 8_000_000_000
        for tick, px in [("AAPL", 232.0), ("CRUS", 104.0), ("GOOGL", 180.0)]:
            seed(market, tick, px, t_pre, n=40)
        t_event = now_ns()
        seed(market, "AAPL", 232.0, t_event, n=30, drift=+500.0)
        seed(market, "CRUS", 104.0, t_event, n=30, drift=0.0)
        seed(market, "GOOGL", 180.0, t_event, n=30, drift=0.0)

        doc = aapl_doc(t_event, body=(
            "Item 2.02 Results of Operations. Services revenue and App Store "
            "growth exceeded consensus estimates on a record installed base."
        ))
        sigs = make_engine(market).on_event(build_event(doc, ("AAPL",)))
        tickers = {s.ticker for s in sigs}

        assert "GOOGL" in tickers, "services channel should reach GOOGL via TAC"
        assert "CRUS" not in tickers, "hardware supplier must not fire on a services story"

    def test_no_second_order_signal_when_primary_has_not_moved(self, scenario):
        """Hop 1 reads magnitude off the primary's tape. No move, no magnitude,
        no propagation -- we do not invent one from the classifier's prior."""
        market, t_event = scenario
        seed(market, "AAPL", 232.0, t_event, n=30, drift=-20.0)  # noise only
        seed(market, "CRUS", 104.0, t_event, n=30, drift=0.0)

        engine = make_engine(market)
        sigs = engine.on_event(build_event(aapl_doc(t_event), ("AAPL",)))

        assert [s for s in sigs if s.hop == 1] == []
        assert engine.rejected.get("primary_not_moved")

    def test_wide_spread_blocks_the_trade(self, scenario):
        market, t_event = scenario
        seed(market, "AAPL", 232.0, t_event, n=30, drift=-420.0)
        seed(market, "CRUS", 104.0, t_event, n=30, drift=-60.0, spread_bps=400.0)

        sigs = make_engine(market).on_event(build_event(aapl_doc(t_event), ("AAPL",)))
        assert not [s for s in sigs if s.ticker == "CRUS"]

    def test_confidence_scales_with_beta(self, scenario):
        market, t_event = scenario
        seed(market, "AAPL", 232.0, t_event, n=30, drift=-800.0)
        seed(market, "CRUS", 104.0, t_event, n=30, drift=0.0)   # beta 0.85
        seed(market, "SWKS", 78.0, t_event, n=30, drift=0.0)    # beta 0.70

        sigs = {s.ticker: s for s in make_engine(market).on_event(
            build_event(aapl_doc(t_event), ("AAPL",)))}
        assert sigs["CRUS"].confidence > sigs["SWKS"].confidence


class TestPrimaryPath:
    def test_mega_cap_that_already_moved_is_refused(self, scenario):
        """The whole point: we do not chase AAPL after it has repriced."""
        market, t_event = scenario
        seed(market, "AAPL", 232.0, t_event, n=30, drift=-420.0)

        engine = make_engine(market)
        event = build_event(aapl_doc(t_event), ("AAPL",))
        assert engine.evaluate_primary(event) is None
        assert engine.rejected.get("residual_below_min")

    def test_small_cap_that_has_not_moved_signals(self):
        market = MarketState()
        t_pre = now_ns() - 8_000_000_000
        seed(market, "TINY", 4.20, t_pre, n=40)
        t_event = now_ns()
        seed(market, "TINY", 4.20, t_event, n=20, drift=0.0)

        doc = RawDoc(
            source="edgar", doc_id="acc-3",
            title="424B5 - Tinybio Therapeutics Inc (0001800001) (Filer)",
            url="", published=None, body="424B5 prospectus supplement.",
            meta={"form": "424B5", "company": "Tinybio Therapeutics Inc",
                  "cik": "1800001", "items": ()},
        )
        doc.t_ingest = t_event

        sig = make_engine(market).evaluate_primary(build_event(doc, ("TINY",)))
        assert sig is not None
        assert sig.direction is Direction.SHORT
        assert sig.hop == 0
        assert sig.edge_bps > 500  # 0.75 materiality * 1800bps, undamped

    def test_neutral_prior_never_guesses_a_direction(self):
        market = MarketState()
        t_pre = now_ns() - 8_000_000_000
        seed(market, "MIDC", 30.0, t_pre, n=40)
        t_event = now_ns()
        seed(market, "MIDC", 30.0, t_event, n=20)

        doc = RawDoc(
            source="edgar", doc_id="acc-4", title="8-K - Midcap Inc (0001234567) (Filer)",
            url="", published=None, body="Item 2.02 Results of Operations.",
            meta={"form": "8-K", "company": "Midcap Inc", "cik": "1234567",
                  "items": ("2.02",)},
        )
        doc.t_ingest = t_event

        engine = make_engine(market)
        assert engine.evaluate_primary(build_event(doc, ("MIDC",))) is None
        assert engine.rejected.get("no_direction")


class TestThresholds:
    def test_low_materiality_is_dropped_before_any_work(self, scenario):
        market, t_event = scenario
        seed(market, "AAPL", 232.0, t_event, n=30, drift=-420.0)
        seed(market, "CRUS", 104.0, t_event, n=30, drift=0.0)

        doc = RawDoc(
            source="edgar", doc_id="acc-5", title="8-K - Apple Inc. (0000320193) (Filer)",
            url="", published=None, body="Item 9.01 Financial Statements and Exhibits.",
            meta={"form": "8-K", "company": "Apple Inc.", "cik": "320193",
                  "items": ("9.01",)},
        )
        doc.t_ingest = t_event

        engine = make_engine(market)
        assert engine.on_event(build_event(doc, ("AAPL",))) == []
        assert engine.rejected.get("materiality")

    def test_min_slack_frac_is_enforced(self, scenario):
        market, t_event = scenario
        seed(market, "AAPL", 232.0, t_event, n=30, drift=-420.0)
        # implied -357; already -300 leaves 16% slack, under the 35% floor.
        seed(market, "CRUS", 104.0, t_event, n=30, drift=-300.0)

        sigs = make_engine(market).on_event(build_event(aapl_doc(t_event), ("AAPL",)))
        assert not [s for s in sigs if s.ticker == "CRUS"]


class TestChannelDetection:
    def test_detects_named_channels(self):
        assert "iphone_hardware" in detect_channels("iPhone unit sales fell", "AAPL")
        assert "cloud_infra" in detect_channels("AWS revenue grew 19%", "AMZN")
        assert "digital_ads" in detect_channels("advertising revenue rose", "AMZN")

    def test_falls_back_to_issuer_defaults(self):
        ch = detect_channels("Acme reports results", "AAPL")
        assert ch == frozenset({"iphone_hardware", "services"})

    def test_unknown_issuer_with_no_language_yields_nothing(self):
        assert detect_channels("Acme reports results", "ZZZZ") == frozenset()


class TestLinkageGraph:
    def test_neighbors_sorted_by_beta_descending(self):
        links = LinkageGraph().neighbors("AAPL")
        betas = [l.beta for l in links]
        assert betas == sorted(betas, reverse=True)

    def test_min_beta_filter(self):
        assert all(l.beta >= 0.5 for l in LinkageGraph().neighbors("AAPL", min_beta=0.5))

    def test_custom_graph(self):
        g = LinkageGraph([Link("XYZ", "ABC", 0.9, "test")])
        assert [l.dst for l in g.neighbors("XYZ")] == ["ABC"]
        assert g.neighbors("AAPL") == []
