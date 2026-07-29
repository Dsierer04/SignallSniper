"""Signal generation: edge = materiality - what the market already took.

Two distinct paths, and it matters which is which.

**Primary (hop 0).** We read a filing on an issuer and form a view before the
name has repriced. Realistic only for the long tail -- small and mid caps whose
8-Ks nobody is parsing with a machine. On a mega cap you will lose this race
every single time, which is why `min_slack_frac` will veto it automatically once
the name has already moved.

**Second-order (hop 1).** This is the good one. We do *not* forecast the primary's
move -- we read it off the tape, where the market has already told us how big the
news was. We then propagate that realized move through a disclosed-exposure beta
and trade the difference against what the linked name has actually done. It is a
relative-value dislocation, not a prediction, and the window is minutes wide
because the propagation is a human inference nobody has precomputed.

Worked example. AAPL prints; the tape says AAPL -420bps and the text names the
iPhone hardware channel. CRUS has a 0.85 beta on that channel, so the implied
CRUS move is -357bps. CRUS has actually done -90bps so far. Residual is -267bps
of move that belongs there and has not arrived. That is the trade, and it exists
because "Apple guided hardware down, therefore the audio codec supplier with 90%
Apple concentration is worth less" takes a person a few minutes to think.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..market.linkage import LinkageGraph, detect_channels
from ..market.tape import MarketState
from ..models import Direction, Event, Signal

log = logging.getLogger("signalsniper.signal")


@dataclass(slots=True)
class EngineConfig:
    #: A materiality of 1.0 maps to this much expected move on a typical name.
    max_event_bps: float = 1800.0
    #: Reject anything under this much remaining edge -- it will not clear costs.
    min_edge_bps: float = 80.0
    #: Reject anything OVER this much too. An implied move of 3000bps is not an
    #: opportunity, it is a corrupt tape: a stale reference, a bad tick that
    #: survived the outlier gate, or a symbol whose price series has a gap in it.
    #: Real edges are bounded; a spectacular one is a bug until proven otherwise.
    max_edge_bps: float = 1_500.0
    #: Sanity bound on the primary's observed move. Beyond this we are reading a
    #: broken series, not a repricing.
    max_primary_bps: float = 3_000.0
    #: At least this fraction of the expected move must still be unpriced.
    min_slack_frac: float = 0.35
    #: Below this classifier confidence we do not act at all.
    min_confidence: float = 0.55
    #: Below this materiality it is not an event.
    min_materiality: float = 0.40
    #: Second-order: the primary must have moved at least this much to bother.
    min_primary_bps: float = 150.0
    #: Second-order: only follow links with at least this exposure beta.
    min_beta: float = 0.15
    #: Widest spread we will cross.
    max_spread_bps: float = 60.0
    #: How long a signal stays actionable.
    ttl_s: float = 300.0
    #: Per-ticker damping. Mega caps move less on the same news.
    move_scale: dict[str, float] | None = None
    #: Only propagate through links whose lag has been MEASURED against history
    #: by tools/validate.py and found tradeable.
    #:
    #: Defaults to True, and on the shipped graph -- where nothing is measured --
    #: that means the second-order path emits NOTHING. That silence is the point.
    #: The literature review refuted the minutes-scale propagation thesis this
    #: path depends on (see docs/VERIFICATION.md), so trading an unmeasured link
    #: is acting on a claim the evidence contradicts. Run the validation, and
    #: links that clear the bar start firing on their own.
    #:
    #: Set False to trade unverified links anyway -- an explicit choice to act
    #: without evidence, not a default you back into.
    verified_links_only: bool = True
    #: Confidence multiplier applied to links that have never been checked
    #: against the tape. An unchecked economic story is not worth the same as a
    #: measured one.
    unverified_penalty: float = 0.60
    #: Linked names excluded from propagation because they have had their OWN
    #: earnings event recently. Such a name is repricing on its own guidance, not
    #: on the primary's read-through, and treating its move as "hasn't repriced
    #: yet" is reading the wrong signal entirely.
    #:
    #: This is not hypothetical for 2026-07-30: Skyworks and Qorvo both reported
    #: on 2026-07-28, two days BEFORE Apple. They were two of the three headline
    #: names in this graph and are structurally invalid as Apple read-throughs
    #: this week.
    blackout_tickers: frozenset[str] = frozenset()

    def scale_for(self, ticker: str) -> float:
        if not self.move_scale:
            return 1.0
        return self.move_scale.get(ticker.upper(), 1.0)


#: Mega caps do not move 18% on an 8-K. Damp the expected-move prior for names
#: where the float is enormous and the coverage is total.
#:
#: These are calibrated against the OPTIONS-IMPLIED move into the print, which is
#: the market's own estimate of event magnitude and a far better anchor than a
#: guess. For 2026-07-30 the options market prices **AMZN at roughly 1.7x AAPL's
#: expected move**. The earlier table had AMZN at only 1.2x AAPL, which would
#: have taken materially more risk on AMZN than intended -- a uniform notional
#: cap across two names with different expected moves is not uniform risk.
#:
#: Re-derive these from the implied move before any event you actually trade;
#: they are event-specific, not permanent properties of the ticker.
DEFAULT_MOVE_SCALE: dict[str, float] = {
    "AAPL": 0.25, "AMZN": 0.42,   # 0.42/0.25 = 1.68x, matching implied
    "MSFT": 0.25, "GOOGL": 0.30, "META": 0.35,
    "NVDA": 0.40, "AVGO": 0.40, "TSLA": 0.50, "JPM": 0.25, "XOM": 0.25,
    "SPY": 0.15, "QQQ": 0.18, "IWM": 0.20, "TLT": 0.15,
}


class SignalEngine:
    def __init__(
        self,
        market: MarketState,
        graph: LinkageGraph | None = None,
        config: EngineConfig | None = None,
    ) -> None:
        self.market = market
        self.graph = graph or LinkageGraph()
        self.cfg = config or EngineConfig(move_scale=dict(DEFAULT_MOVE_SCALE))
        if self.cfg.move_scale is None:
            self.cfg.move_scale = dict(DEFAULT_MOVE_SCALE)
        self.rejected: dict[str, int] = {}

    def _reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    # -----------------------------------------------------------------
    # entry point
    # -----------------------------------------------------------------

    def on_event(self, event: Event) -> list[Signal]:
        if event.materiality < self.cfg.min_materiality:
            self._reject("materiality")
            return []
        if event.confidence < self.cfg.min_confidence:
            self._reject("confidence")
            return []
        if not event.tickers:
            self._reject("no_ticker")
            return []

        out: list[Signal] = []
        primary = self.evaluate_primary(event)
        if primary is not None:
            out.append(primary)
        out.extend(self.evaluate_second_order(event))
        return out

    # -----------------------------------------------------------------
    # hop 0
    # -----------------------------------------------------------------

    def evaluate_primary(self, event: Event) -> Signal | None:
        ticker = event.primary
        if event.prior is Direction.NEUTRAL:
            # We know something happened but not which way. Do not guess a
            # direction on a coin flip -- the second-order path will read the
            # sign off the tape instead.
            self._reject("no_direction")
            return None

        cfg = self.cfg
        expected = event.materiality * cfg.max_event_bps * cfg.scale_for(ticker)
        if expected < cfg.min_edge_bps:
            self._reject("expected_too_small")
            return None

        t_event = event.doc.t_ingest
        already = self.market.already_priced_bps(ticker, t_event, event.prior.value)
        residual = expected - already
        slack = residual / expected if expected > 0 else 0.0

        if residual < cfg.min_edge_bps:
            self._reject("residual_below_min")
            return None
        if residual > cfg.max_edge_bps:
            self._reject("edge_implausible")
            return None
        if slack < cfg.min_slack_frac:
            self._reject("already_priced")
            return None

        tape = self.market.tape(ticker)
        ok, why = tape.tradeable(cfg.max_spread_bps)
        if not ok:
            self._reject(f"illiquid:{why.split()[0]}")
            return None

        return Signal(
            ticker=ticker,
            direction=event.prior,
            edge_bps=round(residual, 1),
            confidence=round(event.confidence * slack, 4),
            event=event,
            hop=0,
            ref_price=tape.last,
            ttl_s=cfg.ttl_s,
            notes=(
                f"expected={expected:.0f}bps",
                f"already={already:.0f}bps",
                f"slack={slack:.0%}",
                *event.reasons,
            ),
        )

    # -----------------------------------------------------------------
    # hop 1 -- the edge
    # -----------------------------------------------------------------

    def evaluate_second_order(self, event: Event) -> list[Signal]:
        cfg = self.cfg
        src = event.primary
        src_tape = self.market.tapes.get(src.upper())
        if src_tape is None:
            self._reject("no_primary_tape")
            return []

        t_event = event.doc.t_ingest
        primary_move = src_tape.move_bps_since(t_event)
        if abs(primary_move) < cfg.min_primary_bps:
            self._reject("primary_not_moved")
            return []
        if abs(primary_move) > cfg.max_primary_bps:
            # Propagating a corrupt primary move multiplies the corruption
            # across every linked name at once.
            self._reject("primary_implausible")
            return []

        channels = detect_channels(f"{event.doc.title} {event.doc.body}", src)
        links = self.graph.neighbors(src, channels or None, cfg.min_beta,
                                     verified_only=cfg.verified_links_only)
        if not links:
            self._reject("no_verified_links" if cfg.verified_links_only else "no_links")
            return []

        signals: list[Signal] = []
        for link in links:
            if link.dst in cfg.blackout_tickers:
                # Its own earnings dominate; the primary read-through is noise
                # against that, and "hasn't moved yet" means something else here.
                self._reject("own_earnings_blackout")
                continue
            implied = primary_move * link.beta * link.polarity
            if abs(implied) < cfg.min_edge_bps:
                self._reject("implied_too_small")
                continue

            dst_tape = self.market.tapes.get(link.dst)
            if dst_tape is None:
                self._reject("no_dst_tape")
                continue

            actual = dst_tape.move_bps_since(t_event)
            residual = implied - actual

            # Only trade when the gap points the same way as the implied move.
            # If the linked name has already overshot, that is a fade setup and a
            # different thesis with a different risk profile -- not this one.
            if implied > 0 and residual <= 0:
                self._reject("overshot")
                continue
            if implied < 0 and residual >= 0:
                self._reject("overshot")
                continue
            if abs(residual) < cfg.min_edge_bps:
                self._reject("residual_below_min")
                continue
            if abs(residual) > cfg.max_edge_bps:
                self._reject("edge_implausible")
                continue

            slack = abs(residual) / abs(implied)
            if slack < cfg.min_slack_frac:
                self._reject("already_priced")
                continue

            ok, why = dst_tape.tradeable(cfg.max_spread_bps)
            if not ok:
                self._reject(f"illiquid:{why.split()[0]}")
                continue

            direction = Direction.LONG if residual > 0 else Direction.SHORT
            # Confidence discounts by beta: a 0.85-beta link is a far tighter
            # economic claim than a 0.15-beta "read-through". It discounts again
            # if the link's lag has never been measured -- a plausible economic
            # story and a story checked against the tape are not the same asset.
            conf = event.confidence * slack * min(1.0, link.beta + 0.15)
            if not link.verified:
                conf *= cfg.unverified_penalty

            signals.append(
                Signal(
                    ticker=link.dst,
                    direction=direction,
                    edge_bps=round(abs(residual), 1),
                    confidence=round(conf, 4),
                    event=event,
                    hop=1,
                    ref_price=dst_tape.last,
                    ttl_s=max(cfg.ttl_s, link.lag_s * 2),
                    notes=(
                        f"{src} moved {primary_move:+.0f}bps",
                        f"beta={link.beta:.2f} pol={link.polarity:+d} ch={link.channel}",
                        f"implied={implied:+.0f}bps actual={actual:+.0f}bps",
                        f"slack={slack:.0%}",
                        (f"lag VERIFIED: {link.lag_capture:.0%} available @5m"
                         if link.verified else
                         "lag UNVERIFIED -- never measured against history"),
                        link.note,
                    ),
                )
            )

        signals.sort(key=lambda s: -(s.edge_bps * s.confidence))
        return signals

    def stats(self) -> dict[str, int]:
        return dict(sorted(self.rejected.items(), key=lambda kv: -kv[1]))
