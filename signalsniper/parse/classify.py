"""Deterministic event classification. No ML in the hot path.

A transformer forward pass is 20-200ms and gives you a sentiment label that means
nothing for a filing. What actually predicts a move is *structure*: the form type
and the 8-K item code. Those are regulator-defined, unambiguous, and match in
microseconds. This module is the reason the pipeline can be sub-millisecond from
bytes to decision.

The numbers below are directional priors, not truth. `materiality` is "how much
should this move the name at all" (0-1) and `prior` is the expected sign. They
encode well-documented filing-response regularities:

  * 8-K 4.02 (non-reliance / restatement) is the most violent single item code
    in the corpus -- it says previously issued financials are wrong.
  * 424B5 intraday means an offering is pricing. Dilution. Reliably negative.
  * SC 13D is an activist declaring intent -- reliably positive, and distinct
    from 13G, which is a passive index holder and is near-noise.
  * 3.01 (listing-rule noncompliance) and 25-NSE (delisting) are terminal-risk.

Tune these against your own fills. They are a starting prior, not a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import Direction, Event, EventKind, RawDoc

# --------------------------------------------------------------------------
# 8-K item codes: (kind, materiality, directional prior)
# --------------------------------------------------------------------------

ITEM_MAP: dict[str, tuple[EventKind, float, Direction]] = {
    "1.01": (EventKind.MATERIAL_AGREEMENT, 0.55, Direction.LONG),
    "1.02": (EventKind.MATERIAL_AGREEMENT, 0.55, Direction.SHORT),
    "1.03": (EventKind.BANKRUPTCY, 0.95, Direction.SHORT),
    "2.01": (EventKind.MERGER, 0.75, Direction.LONG),
    "2.02": (EventKind.EARNINGS, 0.80, Direction.NEUTRAL),  # sign comes from the numbers
    "2.03": (EventKind.OFFERING, 0.35, Direction.SHORT),    # new debt obligation
    "2.04": (EventKind.BANKRUPTCY, 0.80, Direction.SHORT),  # acceleration trigger = distress
    "2.05": (EventKind.IMPAIRMENT, 0.45, Direction.SHORT),  # exit/disposal costs
    "2.06": (EventKind.IMPAIRMENT, 0.65, Direction.SHORT),  # material impairment
    "3.01": (EventKind.DELISTING, 0.85, Direction.SHORT),
    "3.02": (EventKind.OFFERING, 0.60, Direction.SHORT),    # unregistered equity sale
    "3.03": (EventKind.OTHER, 0.40, Direction.NEUTRAL),
    "4.01": (EventKind.AUDITOR_CHANGE, 0.70, Direction.SHORT),
    "4.02": (EventKind.RESTATEMENT, 0.95, Direction.SHORT),  # the big one
    "5.01": (EventKind.MERGER, 0.70, Direction.LONG),        # change in control
    "5.02": (EventKind.EXEC_CHANGE, 0.45, Direction.NEUTRAL),
    "5.03": (EventKind.OTHER, 0.15, Direction.NEUTRAL),
    "5.07": (EventKind.OTHER, 0.05, Direction.NEUTRAL),      # shareholder vote -- noise
    "7.01": (EventKind.REG_FD, 0.50, Direction.NEUTRAL),
    "8.01": (EventKind.OTHER, 0.40, Direction.NEUTRAL),      # catch-all, read the text
    "9.01": (EventKind.OTHER, 0.05, Direction.NEUTRAL),      # exhibits only
}

# --------------------------------------------------------------------------
# Form-level priors for non-8-K filings
# --------------------------------------------------------------------------

FORM_MAP: dict[str, tuple[EventKind, float, Direction]] = {
    "424B5": (EventKind.OFFERING, 0.75, Direction.SHORT),
    "424B4": (EventKind.OFFERING, 0.65, Direction.SHORT),
    "424B3": (EventKind.OFFERING, 0.45, Direction.SHORT),
    "SC 13D": (EventKind.ACTIVIST_STAKE, 0.80, Direction.LONG),
    "SC 13D/A": (EventKind.ACTIVIST_STAKE, 0.60, Direction.LONG),
    "SC 13G": (EventKind.PASSIVE_STAKE, 0.20, Direction.NEUTRAL),
    "SC 13G/A": (EventKind.PASSIVE_STAKE, 0.10, Direction.NEUTRAL),
    "SC TO-T": (EventKind.MERGER, 0.90, Direction.LONG),     # tender offer
    "DEFM14A": (EventKind.MERGER, 0.60, Direction.LONG),
    "25-NSE": (EventKind.DELISTING, 0.90, Direction.SHORT),
    "NT 10-K": (EventKind.RESTATEMENT, 0.65, Direction.SHORT),  # late filing = trouble
    "NT 10-Q": (EventKind.RESTATEMENT, 0.60, Direction.SHORT),
    "4": (EventKind.INSIDER_TRADE, 0.25, Direction.NEUTRAL),
    "S-3ASR": (EventKind.OFFERING, 0.25, Direction.SHORT),   # shelf -- capacity, not an act
    "S-1": (EventKind.OFFERING, 0.20, Direction.NEUTRAL),
    "8-K": (EventKind.OTHER, 0.35, Direction.NEUTRAL),       # fallback when items absent
}

# --------------------------------------------------------------------------
# Headline patterns. Compiled once; matched against title + body.
# Each entry: (regex, kind, materiality floor, direction, weight)
# --------------------------------------------------------------------------

_P = re.IGNORECASE

HEADLINE_RULES: list[tuple[re.Pattern[str], EventKind, float, Direction]] = [
    # --- guidance: the single most reliable mover in an earnings release ------
    (re.compile(r"\brais(?:es|ed|ing)\s+(?:full[- ]year\s+|FY\s*\d*\s*)?(?:guidance|outlook|forecast)", _P),
     EventKind.GUIDANCE, 0.85, Direction.LONG),
    (re.compile(r"\b(?:cuts?|lowers?|lowered|reduces?|reduced|slash(?:es|ed)?)\s+(?:full[- ]year\s+|FY\s*\d*\s*)?(?:guidance|outlook|forecast)", _P),
     EventKind.GUIDANCE, 0.88, Direction.SHORT),
    (re.compile(r"\b(?:withdraws?|withdrew|suspends?|suspended)\s+(?:full[- ]year\s+)?(?:guidance|outlook)", _P),
     EventKind.GUIDANCE, 0.92, Direction.SHORT),
    (re.compile(r"\b(?:above|exceeds?|beats?)\s+(?:consensus|estimates|expectations)", _P),
     EventKind.EARNINGS, 0.70, Direction.LONG),
    (re.compile(r"\b(?:below|misses?|missed|falls?\s+short\s+of)\s+(?:consensus|estimates|expectations)", _P),
     EventKind.EARNINGS, 0.72, Direction.SHORT),

    # --- M&A -----------------------------------------------------------------
    (re.compile(r"\b(?:to\s+be\s+acquired|definitive\s+(?:merger\s+)?agreement|agrees?\s+to\s+acquire|"
                r"to\s+acquire|merger\s+agreement|all[- ]cash\s+transaction)\b", _P),
     EventKind.MERGER, 0.88, Direction.LONG),
    (re.compile(r"\b(?:terminates?|terminated)\s+(?:the\s+)?(?:merger|acquisition|definitive)\b", _P),
     EventKind.MERGER, 0.85, Direction.SHORT),
    (re.compile(r"\btender\s+offer\b", _P), EventKind.MERGER, 0.85, Direction.LONG),

    # --- capital structure ---------------------------------------------------
    (re.compile(r"\b(?:pricing\s+of|prices?|announces?)\s+(?:an?\s+)?(?:underwritten\s+)?"
                r"(?:public\s+)?offering\b", _P),
     EventKind.OFFERING, 0.78, Direction.SHORT),
    (re.compile(r"\b(?:registered\s+direct|private\s+placement|at[- ]the[- ]market|ATM\s+program|"
                r"convertible\s+(?:senior\s+)?notes?\s+offering)\b", _P),
     EventKind.OFFERING, 0.72, Direction.SHORT),
    (re.compile(r"\b(?:share\s+repurchase|stock\s+buyback|buyback\s+program|repurchase\s+authoriz)", _P),
     EventKind.BUYBACK, 0.60, Direction.LONG),
    (re.compile(r"\breverse\s+(?:stock\s+)?split\b", _P), EventKind.OTHER, 0.65, Direction.SHORT),
    (re.compile(r"\b(?:increases?|raises?)\s+(?:quarterly\s+)?dividend\b", _P),
     EventKind.DIVIDEND, 0.45, Direction.LONG),
    (re.compile(r"\b(?:suspends?|cuts?|eliminates?)\s+(?:its\s+)?(?:quarterly\s+)?dividend\b", _P),
     EventKind.DIVIDEND, 0.80, Direction.SHORT),

    # --- life sciences: binary, violent, and heavily retail-traded ------------
    (re.compile(r"\bFDA\s+(?:approv\w+|grants?\s+(?:full\s+)?approval)\b", _P),
     EventKind.REGULATORY_APPROVAL, 0.90, Direction.LONG),
    (re.compile(r"\b(?:complete\s+response\s+letter|CRL\b|FDA\s+reject\w*|refuse\s+to\s+file)", _P),
     EventKind.REGULATORY_APPROVAL, 0.92, Direction.SHORT),
    (re.compile(r"\b(?:breakthrough\s+therapy|fast\s+track|orphan\s+drug)\s+designation\b", _P),
     EventKind.REGULATORY_APPROVAL, 0.55, Direction.LONG),
    (re.compile(r"\bmet\s+(?:its\s+)?primary\s+endpoint\b|\bstatistically\s+significant\b", _P),
     EventKind.CLINICAL, 0.88, Direction.LONG),
    (re.compile(r"\b(?:failed?\s+to\s+meet|did\s+not\s+meet|missed)\s+(?:its\s+)?primary\s+endpoint\b|"
                r"\bclinical\s+hold\b|\bdiscontinu\w+\s+(?:the\s+)?(?:trial|study|program)\b", _P),
     EventKind.CLINICAL, 0.92, Direction.SHORT),

    # --- distress ------------------------------------------------------------
    (re.compile(r"\bchapter\s+11\b|\bbankrupt\w*\b|\bgoing\s+concern\b", _P),
     EventKind.BANKRUPTCY, 0.95, Direction.SHORT),
    (re.compile(r"\b(?:restat\w+|non[- ]reliance)\b.{0,40}\bfinancial\s+statements?\b", _P),
     EventKind.RESTATEMENT, 0.92, Direction.SHORT),
    (re.compile(r"\b(?:SEC|DOJ)\s+(?:investigation|subpoena|inquiry)\b|\bWells\s+notice\b", _P),
     EventKind.LEGAL, 0.80, Direction.SHORT),
    (re.compile(r"\b(?:notice|notification)\s+of\s+(?:non)?compliance\b.{0,40}\blisting\b|"
                r"\bdelisting\b", _P),
     EventKind.DELISTING, 0.82, Direction.SHORT),

    # --- leadership ----------------------------------------------------------
    (re.compile(r"\b(?:CEO|CFO|Chief\s+Executive|Chief\s+Financial)\b.{0,40}"
                r"\b(?:resign\w*|steps?\s+down|departure|terminated|dismissed)\b", _P),
     EventKind.EXEC_CHANGE, 0.70, Direction.SHORT),
    (re.compile(r"\bappoints?\b.{0,30}\b(?:CEO|CFO|Chief\s+Executive|Chief\s+Financial)\b", _P),
     EventKind.EXEC_CHANGE, 0.35, Direction.NEUTRAL),

    # --- commercial ----------------------------------------------------------
    (re.compile(r"\b(?:awarded|wins?|secures?)\b.{0,30}\b(?:contract|order|award)\b", _P),
     EventKind.MATERIAL_AGREEMENT, 0.55, Direction.LONG),
    (re.compile(r"\b(?:strategic\s+partnership|collaboration\s+agreement|licensing\s+agreement)\b", _P),
     EventKind.MATERIAL_AGREEMENT, 0.50, Direction.LONG),
]

#: Words that mean "this is a scheduled, pre-announced non-event". Filings that
#: match only these are suppressed -- trading a conference-attendance notice is
#: how you bleed commissions.
NOISE_RE = re.compile(
    r"\b(?:to\s+(?:present|participate|attend)\s+at|will\s+(?:present|participate)\s+at|"
    r"conference\s+call\s+(?:and\s+webcast\s+)?(?:to\s+be\s+held|scheduled)|"
    r"announces?\s+(?:date|timing)\s+(?:of|for)|to\s+(?:report|announce)\s+"
    r"(?:its\s+)?(?:first|second|third|fourth|Q[1-4])[\w\s-]*results\s+on|"
    r"annual\s+meeting\s+of\s+(?:stock|share)holders)\b",
    _P,
)


@dataclass(slots=True)
class Classification:
    kind: EventKind
    materiality: float
    prior: Direction
    confidence: float
    reasons: tuple[str, ...]


def classify_doc(doc: RawDoc) -> Classification:
    """Structure first, then language. Pure function, no I/O, microseconds."""
    text = f"{doc.title} {doc.body}"
    reasons: list[str] = []

    kind = EventKind.UNKNOWN
    materiality = 0.0
    prior = Direction.NEUTRAL
    confidence = 0.0

    form = str(doc.meta.get("form", "")).strip().upper()
    items: tuple[str, ...] = tuple(doc.meta.get("items", ()) or ())

    # --- 1. 8-K item codes: the strongest structural signal available --------
    matched_item = False
    if items:
        best = None
        for code in items:
            hit = ITEM_MAP.get(code)
            if hit and (best is None or hit[1] > best[1][1]):
                best = (code, hit)
        if best is not None:
            code, (k, m, d) = best
            kind, materiality, prior = k, m, d
            confidence = 0.80
            matched_item = True
            reasons.append(f"8-K item {code}")

    # --- 2. form-level prior -------------------------------------------------
    # The generic "8-K" entry is a *fallback* for filings whose item codes we
    # could not read. Once a real item code has spoken it must not be overridden:
    # an exhibits-only 8-K (item 9.01) is noise, and letting the 0.35 generic
    # prior win would promote every routine filing into a tradeable event.
    if form and not (matched_item and form == "8-K"):
        hit = FORM_MAP.get(form)
        if hit is None and "/" in form:  # 424B5/A, SC 13D/A ...
            hit = FORM_MAP.get(form.split("/")[0].strip())
        if hit is not None:
            k, m, d = hit
            if m > materiality:
                kind, materiality, prior = k, m, d
                confidence = max(confidence, 0.75)
                reasons.append(f"form {form}")
            elif not reasons:
                reasons.append(f"form {form}")

    # --- 3. headline language ------------------------------------------------
    lang_hits = 0
    for pattern, k, m, d in HEADLINE_RULES:
        if pattern.search(text):
            lang_hits += 1
            reasons.append(f"lang:{k.value}:{d.name.lower()}")
            if m > materiality:
                kind, materiality, prior = k, m, d
            elif m >= materiality - 0.10 and prior is Direction.NEUTRAL and d is not Direction.NEUTRAL:
                # Structure said "earnings" without a sign; language supplies it.
                prior = d
            confidence = max(confidence, 0.70)

    if lang_hits > 1:
        # Corroborating language raises confidence but never past the ceiling
        # that a real read of the document would earn.
        confidence = min(0.92, confidence + 0.06 * (lang_hits - 1))

    # --- 4. noise suppression ------------------------------------------------
    if NOISE_RE.search(doc.title) and materiality < 0.75:
        materiality *= 0.15
        confidence *= 0.5
        reasons.append("noise:scheduling")

    if kind is EventKind.UNKNOWN and materiality == 0.0:
        reasons.append("unmatched")

    return Classification(
        kind=kind,
        materiality=round(min(1.0, materiality), 4),
        prior=prior,
        confidence=round(min(1.0, confidence), 4),
        reasons=tuple(reasons),
    )


def build_event(doc: RawDoc, tickers: tuple[str, ...]) -> Event:
    c = classify_doc(doc)
    return Event(
        doc=doc,
        kind=c.kind,
        tickers=tickers,
        materiality=c.materiality,
        prior=c.prior,
        confidence=c.confidence,
        reasons=c.reasons,
    )
