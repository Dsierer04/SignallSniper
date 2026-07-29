"""Numeric extraction from release text.

Classification says *what* happened. Extraction says *how big*, which is what
separates a 40bps drift from a 2000bps gap. The highest-value single number here
is offering size as a fraction of market cap -- a $50m raise against a $3bn cap
is noise; against a $80m cap it is a 60% dilution event and the stock is going to
crater regardless of what the classifier thinks about sentiment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_MULT = {"": 1.0, "K": 1e3, "THOUSAND": 1e3, "M": 1e6, "MM": 1e6, "MILLION": 1e6,
         "B": 1e9, "BN": 1e9, "BILLION": 1e9, "T": 1e12, "TRILLION": 1e12}

_MONEY_RE = re.compile(
    r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(billion|million|thousand|trillion|bn|mm|[kmbt])?\b",
    re.IGNORECASE,
)
_PCT_RE = re.compile(r"([+-]?[0-9]+(?:\.[0-9]+)?)\s*%")
_EPS_RE = re.compile(
    r"\b(?:diluted\s+|adjusted\s+|non[- ]GAAP\s+)*(?:EPS|earnings\s+per\s+share)\b"
    r"[^.$\n]{0,40}\$\s*\(?([0-9]+(?:\.[0-9]+)?)\)?",
    re.IGNORECASE,
)
_SHARES_RE = re.compile(
    r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(million|billion|mm)?\s+shares\b", re.IGNORECASE
)


def _to_float(num: str, unit: str | None) -> float:
    try:
        base = float(num.replace(",", ""))
    except ValueError:
        return 0.0
    return base * _MULT.get((unit or "").upper(), 1.0)


@dataclass(slots=True)
class Extracted:
    money: tuple[float, ...] = ()
    percents: tuple[float, ...] = ()
    eps: float | None = None
    shares: float | None = None

    @property
    def largest_money(self) -> float:
        return max(self.money) if self.money else 0.0


def extract(text: str) -> Extracted:
    money = tuple(_to_float(m.group(1), m.group(2)) for m in _MONEY_RE.finditer(text))
    pcts: list[float] = []
    for m in _PCT_RE.finditer(text):
        try:
            pcts.append(float(m.group(1)))
        except ValueError:
            continue

    eps = None
    em = _EPS_RE.search(text)
    if em:
        try:
            eps = float(em.group(1))
            # "$(0.42)" is accounting negative notation.
            if "(" in text[em.start(): em.end()]:
                eps = -eps
        except ValueError:
            eps = None

    shares = None
    sm = _SHARES_RE.search(text)
    if sm:
        shares = _to_float(sm.group(1), sm.group(2))

    return Extracted(money=money, percents=tuple(pcts), eps=eps, shares=shares)


def dilution_severity(text: str, market_cap: float) -> float:
    """0..1 severity of an offering, as raise size over market cap.

    Returns 0.0 when we cannot size it -- an unsized offering should fall back to
    the classifier's generic prior rather than fabricate a magnitude.
    """
    if market_cap <= 0:
        return 0.0
    ex = extract(text)
    raise_amt = ex.largest_money
    if raise_amt <= 0:
        return 0.0
    ratio = raise_amt / market_cap
    # 5% dilution is a nuisance; 40%+ is an existential repricing.
    if ratio <= 0.02:
        return 0.15
    if ratio >= 0.40:
        return 1.0
    return round(0.15 + (ratio - 0.02) / (0.40 - 0.02) * 0.85, 4)


def surprise_bps(actual: float | None, consensus: float | None, cap_bps: float = 3000.0) -> float:
    """Signed surprise in bps, clipped. Negative consensus is handled by |denom|."""
    if actual is None or consensus is None or consensus == 0:
        return 0.0
    raw = (actual - consensus) / abs(consensus) * 10_000.0
    return max(-cap_bps, min(cap_bps, raw))
