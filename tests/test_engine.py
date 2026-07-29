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
    """Engine for testing propagation MECHANICS.

    verified_links_only defaults to True in production so an unmeasured graph
    emits nothing. These tests are about the arithmetic of propagation, not the
    gating policy, so they opt out explicitly. TestVerificationGating covers the
    policy itself, including that the real default is silent.
    """
    cfg_kw.setdefault("verified_links_only", False)
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
        """Amazon ad strength is a headwind for The Trade Desk, not a tailwind.

        The original inverse edges were SHOP/ETSY/W/TGT on retail. Fact-check
        killed those: SHOP-AMZN correlation is POSITIVE at every horizon
        (+0.45 3m, +0.42 1y, +0.59 5y), so a hard-coded inverse beta would have
        lost systematically. TTD is the inverse link that survived -- Amazon's
        ad business is TTD's biggest competitive threat.
        """
        market = MarketState()
        t_pre = now_ns() - 8_000_000_000
        seed(market, "AMZN", 190.0, t_pre, n=40)
        seed(market, "TTD", 105.0, t_pre, n=40)
        t_event = now_ns()
        seed(market, "AMZN", 190.0, t_event, n=30, drift=+600.0)
        seed(market, "TTD", 105.0, t_event, n=30, drift=0.0)

        doc = RawDoc(
            source="edgar", doc_id="acc-2",
            title="8-K - Amazon.com Inc. (0001018724) (Filer)",
            url="", published=None,
            body="Item 2.02 Results of Operations. Advertising services revenue "
                 "exceeded consensus estimates.",
            meta={"form": "8-K", "company": "Amazon.com Inc.", "cik": "1018724",
                  "items": ("2.02",)},
        )
        doc.t_ingest = t_event
        sigs = make_engine(market).on_event(build_event(doc, ("AMZN",)))

        ttd = [s for s in sigs if s.ticker == "TTD"]
        assert ttd, "TTD should signal on the inverse advertising link"
        assert ttd[0].direction is Direction.SHORT

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


class TestVerificationGating:
    """A plausible economic story and a story measured against the tape are
    not the same asset, and the engine must not price them the same."""

    def _market(self):
        market = MarketState()
        t_pre = now_ns() - 8_000_000_000
        for tick, px in (("AAPL", 232.0), ("CRUS", 104.0)):
            seed(market, tick, px, t_pre, n=40)
        t_event = now_ns()
        seed(market, "AAPL", 232.0, t_event, n=30, drift=-420.0)
        seed(market, "CRUS", 104.0, t_event, n=30, drift=-60.0)
        return market, t_event

    def _graph(self, **link_kw):
        return LinkageGraph([
            Link("AAPL", "CRUS", 0.85, "iphone_hardware", 1, 90, "test", **link_kw)
        ])

    def test_shipped_graph_is_entirely_unverified(self):
        """Honesty check: nothing in the default graph has been measured."""
        cov = LinkageGraph().coverage()
        assert cov["total"] > 0
        assert cov["verified"] == 0, (
            "a link claims verification without a validate.py run behind it"
        )

    def test_unverified_link_is_confidence_penalised(self):
        market, t_event = self._market()
        event = build_event(aapl_doc(t_event), ("AAPL",))

        unver = SignalEngine(market, self._graph(), EngineConfig(
            move_scale=dict(DEFAULT_MOVE_SCALE), unverified_penalty=0.6,
            verified_links_only=False))
        ver = SignalEngine(market, self._graph(lag_capture=0.8, dead_rate=0.1),
                           EngineConfig(move_scale=dict(DEFAULT_MOVE_SCALE),
                                        verified_links_only=False))

        a = unver.evaluate_second_order(event)[0]
        b = ver.evaluate_second_order(event)[0]
        assert a.edge_bps == pytest.approx(b.edge_bps)   # same edge
        assert a.confidence < b.confidence                # different trust
        assert a.confidence == pytest.approx(b.confidence * 0.6, abs=0.01)

    def test_signal_notes_state_verification_status(self):
        market, t_event = self._market()
        event = build_event(aapl_doc(t_event), ("AAPL",))
        sig = SignalEngine(market, self._graph(), EngineConfig(
            move_scale=dict(DEFAULT_MOVE_SCALE),
            verified_links_only=False)).evaluate_second_order(event)[0]
        assert any("UNVERIFIED" in n for n in sig.notes)

    def test_the_default_is_silent_on_an_unmeasured_graph(self):
        """The shipped default must not trade a thesis the evidence refutes.

        verified_links_only defaults to True and nothing in the shipped graph is
        measured, so the second-order path emits nothing until validate.py
        produces evidence. That silence is the intended behaviour.
        """
        market, t_event = self._market()
        event = build_event(aapl_doc(t_event), ("AAPL",))
        engine = SignalEngine(market, self._graph(),
                              EngineConfig(move_scale=dict(DEFAULT_MOVE_SCALE)))
        assert engine.cfg.verified_links_only is True
        assert engine.evaluate_second_order(event) == []
        assert engine.rejected.get("no_verified_links")

    def test_shipped_engine_and_graph_together_emit_nothing(self):
        """End to end on the real defaults: no config, no graph override."""
        market, t_event = self._market()
        event = build_event(aapl_doc(t_event), ("AAPL",))
        engine = SignalEngine(market)   # real graph, real defaults
        assert [s for s in engine.on_event(event) if s.hop == 1] == []

    def test_verified_only_mode_passes_a_measured_link(self):
        market, t_event = self._market()
        event = build_event(aapl_doc(t_event), ("AAPL",))
        engine = SignalEngine(market, self._graph(lag_capture=0.8, dead_rate=0.1),
                              EngineConfig(move_scale=dict(DEFAULT_MOVE_SCALE)))
        assert len(engine.evaluate_second_order(event)) == 1

    def test_measured_but_instant_repricer_is_not_tradeable(self):
        """High beta, measured, and still no edge -- it repriced with the primary."""
        link = Link("AAPL", "CRUS", 0.85, "iphone_hardware", 1, 90, "t",
                    lag_capture=0.05, dead_rate=0.0)
        assert link.verified is True
        assert link.tradeable_lag is False

    def test_illiquid_name_is_not_tradeable_even_with_a_good_lag(self):
        """'Hasn't moved' meant 'hasn't traded' on most past events."""
        link = Link("AAPL", "CRUS", 0.85, "iphone_hardware", 1, 90, "t",
                    lag_capture=0.90, dead_rate=0.60)
        assert link.tradeable_lag is False


class TestCalibration:
    """Measurements from tools/validate.py must flow into the graph correctly."""

    CALIB = {"links": [
        # lags and trades -> tradeable
        {"src": "AAPL", "dst": "CRUS", "lag_capture": 0.72, "dead_rate": 0.12,
         "beta": 0.62, "beta_r2": 0.55, "beta_usable": True},
        # reprices instantly -> not tradeable despite a fine beta
        {"src": "AAPL", "dst": "SWKS", "lag_capture": 0.18, "dead_rate": 0.10,
         "beta": 0.51, "beta_r2": 0.48, "beta_usable": True},
        # lags but barely trades -> not tradeable
        {"src": "AAPL", "dst": "COHR", "lag_capture": 0.80, "dead_rate": 0.55,
         "beta": 0.40, "beta_r2": 0.40, "beta_usable": True},
        # measured, but the regression is junk -> keep the considered prior
        {"src": "AAPL", "dst": "GLW", "lag_capture": 0.60, "dead_rate": 0.10,
         "beta": -0.02, "beta_r2": 0.01, "beta_usable": False},
    ]}

    def _calibrated(self):
        g = LinkageGraph()
        g.apply_calibration(self.CALIB)
        return {l.dst: l for l in g.neighbors("AAPL")}

    def test_coverage_reflects_what_was_measured(self):
        g = LinkageGraph()
        assert g.coverage()["verified"] == 0
        g.apply_calibration(self.CALIB)
        cov = g.coverage()
        assert cov["verified"] == 4
        # CRUS and GLW both clear the lag bar. GLW's beta regression was junk,
        # but beta usability and lag tradeability are independent questions --
        # a name can have a measurable lag and an unmeasurable exposure.
        assert cov["tradeable"] == 2

    def test_empirical_beta_replaces_the_prior(self):
        links = self._calibrated()
        assert links["CRUS"].beta == pytest.approx(0.62)  # was 0.85

    def test_unusable_regression_does_not_overwrite_the_prior(self):
        """r2 of 0.01 is a number, not an improvement on a considered prior."""
        links = self._calibrated()
        assert links["GLW"].beta == pytest.approx(0.15)  # original prior kept
        assert links["GLW"].verified is True             # but lag was measured

    def test_instant_repricer_is_excluded_from_verified_only(self):
        g = LinkageGraph()
        g.apply_calibration(self.CALIB)
        passing = {l.dst for l in g.neighbors("AAPL", verified_only=True)}
        assert "CRUS" in passing          # lags and trades
        assert "SWKS" not in passing      # repriced instantly
        assert "QRVO" not in passing      # lags but barely trades

    def test_illiquid_name_excluded_despite_good_lag(self):
        links = self._calibrated()
        assert links["COHR"].lag_capture == 0.80
        assert links["COHR"].tradeable_lag is False

    def test_negative_measured_beta_becomes_inverse_polarity(self):
        g = LinkageGraph([Link("X", "Y", 0.5, "test", 1, 60, "n")])
        g.apply_calibration({"links": [
            {"src": "X", "dst": "Y", "lag_capture": 0.7, "dead_rate": 0.0,
             "beta": -0.35, "beta_usable": True}]})
        link = g.neighbors("X")[0]
        assert link.beta == pytest.approx(0.35)
        assert link.polarity == -1

    def test_unlisted_links_are_untouched(self):
        links = self._calibrated()
        assert links["TXN"].verified is False
        assert links["TXN"].beta == pytest.approx(0.10)

    def test_missing_calibration_file_is_not_an_error(self):
        g = LinkageGraph.calibrated("/nonexistent/path/calibration.json")
        assert g.coverage()["verified"] == 0


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


class TestOwnEarningsBlackout:
    """A name repricing on its OWN guidance is not a read-through from someone
    else's print, and reading its move as 'hasn't repriced yet' inverts the
    signal's meaning.

    Not hypothetical: SWKS and QRVO both reported 2026-07-28, two days before
    Apple's 2026-07-30 print. They were two of the three headline supplier names.
    """

    def _market(self):
        market = MarketState()
        t_pre = now_ns() - 8_000_000_000
        for tick, px in (("AAPL", 232.0), ("CRUS", 104.0), ("SWKS", 78.0)):
            seed(market, tick, px, t_pre, n=40)
        t_event = now_ns()
        seed(market, "AAPL", 232.0, t_event, n=30, drift=-420.0)
        seed(market, "CRUS", 104.0, t_event, n=30, drift=-60.0)
        seed(market, "SWKS", 78.0, t_event, n=30, drift=-40.0)
        return market, t_event

    def _engine(self, blackout):
        return SignalEngine(market=self._m, graph=LinkageGraph(), config=EngineConfig(
            move_scale=dict(DEFAULT_MOVE_SCALE), verified_links_only=False,
            blackout_tickers=frozenset(blackout)))

    def test_blacked_out_name_does_not_signal(self):
        market, t_event = self._market()
        self._m = market
        event = build_event(aapl_doc(t_event), ("AAPL",))

        without = {s.ticker for s in self._engine(set()).evaluate_second_order(event)}
        assert "SWKS" in without

        engine = self._engine({"SWKS"})
        with_bl = {s.ticker for s in engine.evaluate_second_order(event)}
        assert "SWKS" not in with_bl
        assert "CRUS" in with_bl          # unrelated names unaffected
        assert engine.rejected.get("own_earnings_blackout") == 1

    def test_jul30_config_blacks_out_the_names_that_already_reported(self):
        from signalsniper.config import load
        assert {"SWKS", "QRVO"} <= set(load().blackout)

    def test_runner_wires_the_blackout_into_the_engine(self):
        from signalsniper.config import load
        from signalsniper.runner import Runner
        cfg = load(sec_user_agent="T t@e.com")
        assert "SWKS" in Runner(cfg).engine.cfg.blackout_tickers
