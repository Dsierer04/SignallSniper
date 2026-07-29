"""The lag study. These tests protect the tool that decides whether to trade at all."""

from __future__ import annotations

import math
from datetime import timedelta

import pytest

from signalsniper.validate.events import _et, events_for
from signalsniper.validate.study import (
    Bar,
    Series,
    estimate_beta,
    lag_verdict,
    move_bps,
    profile_event,
)

T0 = _et(2026, 4, 30, 16, 30)


def build(ticker: str, base: float, path: list[tuple[float, float]],
          volume: int = 5_000) -> Series:
    """path = [(minutes_from_event, cumulative_bps_from_base)]"""
    return Series(ticker, [
        Bar(t=T0 + timedelta(minutes=m), open=base * (1 + b / 10_000),
            high=base * (1 + b / 10_000), low=base * (1 + b / 10_000),
            close=base * (1 + b / 10_000), volume=volume)
        for m, b in path
    ])


FLAT = [(-30.0, 0.0), (-1.0, 0.0), (0.0, 0.0)]

PRIMARY = build("AAPL", 230.0, FLAT + [
    (1.0, -400.0), (5.0, -410.0), (15.0, -415.0), (30.0, -420.0), (240.0, -420.0)])

LAGGER = build("LAGGY", 100.0, FLAT + [
    (1.0, -20.0), (5.0, -40.0), (15.0, -180.0), (30.0, -300.0), (240.0, -340.0)])

INSTANT = build("INSTA", 100.0, FLAT + [
    (1.0, -300.0), (5.0, -330.0), (15.0, -335.0), (30.0, -340.0), (240.0, -340.0)])


class TestMoveBps:
    def test_basic(self):
        assert move_bps(100.0, 105.0) == pytest.approx(500.0)
        assert move_bps(100.0, 95.0) == pytest.approx(-500.0)

    def test_degenerate_inputs_are_zero_not_errors(self):
        assert move_bps(0.0, 100.0) == 0.0
        assert move_bps(100.0, 0.0) == 0.0
        assert move_bps(-5.0, 100.0) == 0.0


class TestSeries:
    def test_price_at_uses_last_bar_at_or_before(self):
        assert PRIMARY.price_at(T0 + timedelta(minutes=3)) == pytest.approx(
            230.0 * (1 - 400 / 10_000))

    def test_price_before_history_is_zero(self):
        assert PRIMARY.price_at(T0 - timedelta(days=5)) == 0.0

    def test_bars_are_sorted_on_construction(self):
        s = Series("X", [
            Bar(T0 + timedelta(minutes=5), 1, 1, 1, 1, 1),
            Bar(T0, 2, 2, 2, 2, 2),
        ])
        assert s.bars[0].t < s.bars[1].t

    def test_volume_between_is_half_open(self):
        s = build("X", 100.0, [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)], volume=10)
        assert s.volume_between(T0, T0 + timedelta(minutes=2)) == 20


class TestLagDetection:
    """The core claim: a lagging name and an instant name must be separable."""

    def test_lagger_shows_move_still_available(self):
        p = profile_event("E1", PRIMARY, LAGGER, T0)
        assert p.captured_after(5.0) > 0.80
        assert p.captured_after(15.0) > 0.40

    def test_instant_repricer_shows_nothing_available(self):
        p = profile_event("E1", PRIMARY, INSTANT, T0)
        assert p.captured_after(5.0) < 0.10

    def test_verdict_separates_them(self):
        lag = lag_verdict([profile_event(f"E{i}", PRIMARY, LAGGER, T0) for i in range(6)])
        inst = lag_verdict([profile_event(f"E{i}", PRIMARY, INSTANT, T0) for i in range(6)])
        assert lag.edge_exists is True
        assert inst.edge_exists is False

    def test_identical_beta_different_tradeability(self):
        """The insight that matters: exposure and tradeability are different things.

        Both names end at -340bps against the same -420bps primary, so their
        betas are identical. Only one of them was catchable.
        """
        lag_p = [profile_event(f"E{i}", PRIMARY, LAGGER, T0) for i in range(6)]
        inst_p = [profile_event(f"E{i}", PRIMARY, INSTANT, T0) for i in range(6)]

        b_lag = estimate_beta(lag_p)
        b_inst = estimate_beta(inst_p)
        assert b_lag.beta == pytest.approx(b_inst.beta, abs=0.01)

        assert lag_verdict(lag_p).edge_exists
        assert not lag_verdict(inst_p).edge_exists


class TestLiquidityHonesty:
    """'Hasn't moved' and 'hasn't traded' look identical to the live engine."""

    def test_no_early_prints_is_flagged(self):
        silent = Series("QUIET", [
            Bar(T0 - timedelta(minutes=30), 100.0, 100.0, 100.0, 100.0, 100),
            Bar(T0 + timedelta(minutes=45), 96.0, 96.0, 96.0, 96.0, 100),
        ])
        p = profile_event("E1", PRIMARY, silent, T0)
        assert p.dead_early is True

    def test_dead_early_names_fail_the_verdict(self):
        silent = Series("QUIET", [
            Bar(T0 - timedelta(minutes=30), 100.0, 100.0, 100.0, 100.0, 100),
            Bar(T0 + timedelta(minutes=45), 96.0, 96.0, 96.0, 96.0, 100),
        ])
        v = lag_verdict([profile_event(f"E{i}", PRIMARY, silent, T0) for i in range(6)])
        assert v.dead_early_rate == 1.0
        assert v.edge_exists is False

    def test_tiny_moves_return_nan_not_a_ratio(self):
        """A 3bps move split 50/50 is not evidence of a lag."""
        noise = build("NOISE", 100.0, FLAT + [
            (1.0, -2.0), (5.0, -3.0), (15.0, -4.0), (30.0, -5.0), (240.0, -5.0)])
        p = profile_event("E1", PRIMARY, noise, T0)
        assert math.isnan(p.captured_after(5.0))

    def test_verdict_needs_enough_events(self):
        v = lag_verdict([profile_event(f"E{i}", PRIMARY, LAGGER, T0) for i in range(3)])
        assert v.edge_exists is False  # n < 4


class TestBetaEstimation:
    def test_recovers_a_known_beta(self):
        prim = build("P", 100.0, FLAT + [(240.0, -1000.0)])
        half = build("H", 50.0, FLAT + [(240.0, -500.0)])
        profiles = [profile_event(f"E{i}", prim, half, T0) for i in range(5)]
        b = estimate_beta(profiles, prior=0.5)
        assert b.beta == pytest.approx(0.5, abs=0.02)
        assert b.r_squared > 0.99

    def test_reports_error_against_the_prior(self):
        prim = build("P", 100.0, FLAT + [(240.0, -1000.0)])
        weak = build("W", 50.0, FLAT + [(240.0, -200.0)])
        profiles = [profile_event(f"E{i}", prim, weak, T0) for i in range(5)]
        b = estimate_beta(profiles, prior=0.85)
        assert b.beta == pytest.approx(0.2, abs=0.02)
        assert b.prior_error < -0.5  # the hand-set prior was badly too high

    def test_negative_beta_for_an_inverse_link(self):
        prim = build("P", 100.0, FLAT + [(240.0, 1000.0)])
        inverse = build("I", 50.0, FLAT + [(240.0, -300.0)])
        b = estimate_beta([profile_event(f"E{i}", prim, inverse, T0) for i in range(5)])
        assert b.beta < 0

    def test_too_few_points_returns_none(self):
        assert estimate_beta([]) is None
        assert estimate_beta([profile_event("E1", PRIMARY, LAGGER, T0)]) is None

    def test_events_with_a_flat_primary_are_excluded(self):
        """A primary that did not move carries no information about exposure."""
        flat_prim = build("P", 100.0, FLAT + [(240.0, 5.0)])
        assert estimate_beta(
            [profile_event(f"E{i}", flat_prim, LAGGER, T0) for i in range(5)]) is None

    def test_usable_requires_sample_and_fit(self):
        prim = build("P", 100.0, FLAT + [(240.0, -1000.0)])
        half = build("H", 50.0, FLAT + [(240.0, -500.0)])
        assert not estimate_beta(
            [profile_event(f"E{i}", prim, half, T0) for i in range(4)]).usable
        assert estimate_beta(
            [profile_event(f"E{i}", prim, half, T0) for i in range(8)]).usable


class TestEventCatalog:
    def test_events_are_filtered_by_primary(self):
        assert all(e.primary == "AAPL" for e in events_for("AAPL"))
        assert all(e.primary == "AMZN" for e in events_for("amzn"))
        assert events_for("ZZZZ") == ()

    def test_events_carry_timezone(self):
        for e in events_for("AAPL"):
            assert e.t_release.tzinfo is not None
