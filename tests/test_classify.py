"""Classifier: structure beats language, and noise gets suppressed."""

from __future__ import annotations

import pytest

from signalsniper.models import Direction, EventKind, RawDoc
from signalsniper.parse.classify import classify_doc
from signalsniper.parse.extract import dilution_severity, extract, surprise_bps


def doc(title="", body="", form="", items=(), source="edgar", **meta):
    return RawDoc(
        source=source, doc_id="x", title=title, url="", published=None, body=body,
        meta={"form": form, "items": items, **meta},
    )


class TestItemCodes:
    @pytest.mark.parametrize("item,kind,direction", [
        ("4.02", EventKind.RESTATEMENT, Direction.SHORT),
        ("1.03", EventKind.BANKRUPTCY, Direction.SHORT),
        ("3.01", EventKind.DELISTING, Direction.SHORT),
        ("4.01", EventKind.AUDITOR_CHANGE, Direction.SHORT),
        ("2.02", EventKind.EARNINGS, Direction.NEUTRAL),
        ("2.01", EventKind.MERGER, Direction.LONG),
    ])
    def test_item_maps_to_kind_and_direction(self, item, kind, direction):
        c = classify_doc(doc(title="8-K", form="8-K", items=(item,)))
        assert c.kind is kind
        assert c.prior is direction
        assert c.confidence >= 0.7

    def test_highest_materiality_item_wins(self):
        # 9.01 (exhibits, noise) alongside 4.02 (restatement) -- 4.02 must win.
        c = classify_doc(doc(title="8-K", form="8-K", items=("9.01", "4.02")))
        assert c.kind is EventKind.RESTATEMENT
        assert c.materiality > 0.9

    def test_exhibits_only_is_near_zero(self):
        c = classify_doc(doc(title="8-K", form="8-K", items=("9.01",)))
        assert c.materiality < 0.1

    def test_shareholder_vote_is_noise(self):
        c = classify_doc(doc(title="8-K", form="8-K", items=("5.07",)))
        assert c.materiality < 0.1


class TestFormPriors:
    def test_424b5_is_a_short(self):
        c = classify_doc(doc(title="424B5 - Tiny Inc", form="424B5"))
        assert c.kind is EventKind.OFFERING
        assert c.prior is Direction.SHORT
        assert c.materiality > 0.7

    def test_13d_is_activist_long_and_13g_is_not(self):
        d = classify_doc(doc(title="SC 13D", form="SC 13D"))
        g = classify_doc(doc(title="SC 13G", form="SC 13G"))
        assert d.kind is EventKind.ACTIVIST_STAKE and d.prior is Direction.LONG
        assert g.kind is EventKind.PASSIVE_STAKE
        assert d.materiality > g.materiality * 3

    def test_amendment_suffix_falls_back_to_base_form(self):
        c = classify_doc(doc(title="424B5/A", form="424B5/A"))
        assert c.kind is EventKind.OFFERING


class TestHeadlineLanguage:
    @pytest.mark.parametrize("title,kind,direction", [
        ("Acme Raises Full-Year Guidance", EventKind.GUIDANCE, Direction.LONG),
        ("Acme Cuts Full-Year Outlook", EventKind.GUIDANCE, Direction.SHORT),
        ("Acme Withdraws Guidance", EventKind.GUIDANCE, Direction.SHORT),
        ("Acme Agrees to Acquire Beta Corp", EventKind.MERGER, Direction.LONG),
        ("Acme Announces Pricing of Public Offering", EventKind.OFFERING, Direction.SHORT),
        ("Acme Suspends Its Quarterly Dividend", EventKind.DIVIDEND, Direction.SHORT),
        ("Bio Inc Receives FDA Approval for XYZ", EventKind.REGULATORY_APPROVAL, Direction.LONG),
        ("Bio Inc Receives Complete Response Letter", EventKind.REGULATORY_APPROVAL, Direction.SHORT),
        ("Bio Inc Met Its Primary Endpoint", EventKind.CLINICAL, Direction.LONG),
        ("Bio Inc Failed to Meet Primary Endpoint", EventKind.CLINICAL, Direction.SHORT),
        ("Acme CEO Resigns Effective Immediately", EventKind.EXEC_CHANGE, Direction.SHORT),
        ("Acme Announces Chapter 11 Filing", EventKind.BANKRUPTCY, Direction.SHORT),
    ])
    def test_language_rules(self, title, kind, direction):
        c = classify_doc(doc(title=title, source="newswire"))
        assert c.kind is kind, f"{title} -> {c.kind}"
        assert c.prior is direction, f"{title} -> {c.prior}"

    def test_language_supplies_sign_when_structure_is_neutral(self):
        # 8-K 2.02 alone has no direction; the release text provides it.
        c = classify_doc(doc(
            title="Acme Reports Q2 Results and Cuts Full-Year Guidance",
            form="8-K", items=("2.02",),
        ))
        assert c.prior is Direction.SHORT
        assert c.materiality > 0.8

    def test_corroborating_language_raises_confidence(self):
        one = classify_doc(doc(title="Acme Cuts Full-Year Guidance", source="newswire"))
        two = classify_doc(doc(
            title="Acme Cuts Full-Year Guidance; Results Below Consensus Estimates",
            source="newswire",
        ))
        assert two.confidence > one.confidence


class TestNoiseSuppression:
    @pytest.mark.parametrize("title", [
        "Acme to Present at the Whatever Healthcare Conference",
        "Acme Announces Date of Second Quarter 2026 Earnings Call",
        "Acme to Report Second Quarter Results on August 5, 2026",
        "Acme Announces Annual Meeting of Stockholders",
    ])
    def test_scheduling_announcements_suppressed(self, title):
        c = classify_doc(doc(title=title, source="newswire"))
        assert c.materiality < 0.2, f"{title} scored {c.materiality}"

    def test_suppression_does_not_kill_a_genuine_event(self):
        # Mentions a call but the substance is a guidance cut.
        c = classify_doc(doc(
            title="Acme Cuts Full-Year Guidance; Conference Call Scheduled",
            source="newswire",
        ))
        assert c.materiality > 0.7

    def test_unmatched_doc_scores_zero(self):
        c = classify_doc(doc(title="Acme Publishes Sustainability Report"))
        assert c.materiality == 0.0
        assert "unmatched" in c.reasons


class TestExtraction:
    def test_money_units(self):
        ex = extract("raised $60 million and also $1.2 billion plus $500")
        assert 60e6 in ex.money
        assert 1.2e9 in ex.money
        assert 500.0 in ex.money
        assert ex.largest_money == 1.2e9

    def test_percentages_signed(self):
        ex = extract("revenue grew 12.5% while margin fell -3%")
        assert 12.5 in ex.percents
        assert -3.0 in ex.percents

    def test_eps(self):
        assert extract("Diluted EPS of $1.42 for the quarter").eps == 1.42

    def test_shares(self):
        assert extract("offering of 12,000,000 shares").shares == 12_000_000

    def test_dilution_severity_scales_with_raise_over_cap(self):
        text = "pricing of a public offering for gross proceeds of $50 million"
        small = dilution_severity(text, market_cap=80e6)     # 62% of the company
        large = dilution_severity(text, market_cap=3e9)      # 1.7%
        assert small > 0.9
        assert large < 0.25
        assert small > large

    def test_dilution_unsized_returns_zero(self):
        assert dilution_severity("announces an offering", market_cap=1e9) == 0.0
        assert dilution_severity("raised $10 million", market_cap=0) == 0.0

    def test_surprise_bps_handles_negative_consensus(self):
        assert surprise_bps(-0.10, -0.20) > 0     # lost less than feared
        assert surprise_bps(-0.30, -0.20) < 0
        assert surprise_bps(1.0, 0.0) == 0.0
        assert surprise_bps(None, 1.0) == 0.0

    def test_surprise_is_clipped(self):
        assert surprise_bps(100.0, 0.01) == 3000.0
