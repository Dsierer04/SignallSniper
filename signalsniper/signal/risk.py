"""Risk gate. This module exists to survive being wrong.

The engine's job is to find edge. This module's job is to assume the engine is
wrong more often than it thinks, and to make sure that being wrong is survivable.
Every constant here binds *before* an order goes out, and the kill switch is
checked on every single call rather than on a timer -- a timer can be starved by
the event loop at exactly the moment you need it.

Sizing is stop-distance based, not notional based. "Risk 1% of the account" means
the distance to the stop is 1% of the account, so a volatile name gets a smaller
position for the same risk. Sizing by notional instead is how a single gap turns
a good week into a blown account.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..models import Direction, Signal, mono_ns

log = logging.getLogger("signalsniper.risk")


@dataclass(slots=True)
class RiskConfig:
    equity: float = 25_000.0
    #: Fraction of equity risked per trade (distance to stop).
    risk_per_trade: float = 0.01
    #: Hard ceiling on any single position's notional as a fraction of equity.
    max_position_frac: float = 0.20
    #: Cumulative realised loss that halts trading for the day.
    max_daily_loss_frac: float = 0.03
    #: Concurrent open positions.
    max_concurrent: int = 4
    #: Same-name re-entry cooldown.
    cooldown_s: float = 180.0
    #: Stop distance as a multiple of the signal's edge. <1 means the stop is
    #: tighter than the target, which is the only way a sub-50% hit rate pays.
    stop_frac_of_edge: float = 0.60
    #: Never place a stop tighter than this -- noise would take you out.
    min_stop_bps: float = 40.0
    #: And never wider than this. The stop distance is derived from the signal's
    #: edge, so a corrupt edge produces a corrupt stop -- and a stop placed 60%
    #: away is not a stop, it is a position with no floor under it. This cap is
    #: the last line: even if every upstream sanity check fails, the loss on any
    #: single trade stays bounded.
    #:
    #: Set to match the engine's own ceiling rather than undercut it: the engine
    #: caps edge at 1500bps and stop_frac_of_edge is 0.60, so 900bps is the
    #: widest stop a *sane* signal can produce. A backstop tighter than that
    #: would silently veto legitimate trades instead of catching corrupt ones.
    max_stop_bps: float = 900.0
    #: Absolute floor on order size, below which commissions dominate.
    min_shares: int = 1
    min_notional: float = 200.0
    #: Allow fractional share quantities. Essential on a small account: at $200
    #: a $230 stock is simply unbuyable in whole shares, and whole-share
    #: quantization on a $104 name means one share is 52% of the account with no
    #: room to size to risk. Alpaca supports 5,000+ fractionable US securities,
    #: $1 minimum per order.
    #:
    #: Fractional SHORTS are never permitted at any equity level, so this only
    #: ever applies to longs -- enforced in size_order.
    allow_fractional: bool = False
    #: Smallest fractional quantity worth sending.
    min_fractional_qty: float = 0.01
    #: Equity floor below which short orders are refused locally. FINRA
    #: 4210(b)/Reg T set $2,000 for any margin or short activity and Alpaca
    #: enforces it, so sending a short below this earns a broker rejection --
    #: which costs the whole window while you work out why. Refuse it here, at
    #: sizing time, where the reason is legible.
    short_min_equity: float = 2_000.0

    def viability(self, typical_price: float = 100.0) -> list[str]:
        """Can an account this size actually place a trade at all?

        Worth checking explicitly because the failure is silent otherwise: the
        sizing path just returns None with `size_zero` on every signal, which
        looks identical to "no opportunities today". An account that structurally
        cannot trade should say so at startup, not look quiet.
        """
        out: list[str] = []
        max_pos = self.equity * self.max_position_frac
        if max_pos < self.min_notional:
            out.append(
                f"max position ${max_pos:,.0f} ({self.max_position_frac:.0%} of "
                f"${self.equity:,.0f}) is below the ${self.min_notional:,.0f} "
                "minimum notional -- EVERY order will be rejected before it is sent"
            )
        risk_budget = self.equity * self.risk_per_trade
        if risk_budget < 5.0:
            out.append(
                f"risk budget is ${risk_budget:.2f} per trade -- at a typical "
                f"{self.min_stop_bps:.0f}bps minimum stop that sizes to well "
                "under one share on most names"
            )
        if self.equity < typical_price:
            out.append(
                f"${self.equity:,.0f} equity cannot buy a single share of a "
                f"${typical_price:,.0f} stock"
            )
        if self.equity < 2_000:
            # Alpaca does not offer cash accounts -- every account is a margin
            # account, and sub-$2,000 ones are "limited margin": 1x buying power
            # with shorting disabled, but unsettled funds ARE floated. So T+1
            # settlement and good-faith violations do NOT apply here; only the
            # short block does. (An earlier version of this warning claimed the
            # opposite and was wrong.)
            out.append(
                f"${self.equity:,.0f} is below the $2,000 FINRA 4210(b)/Reg T "
                "minimum -- SHORT SELLING IS IMPOSSIBLE (regulatory, not "
                "appealable), and buying power is 1x equity with no leverage. "
                "Roughly half of this system's signals are shorts."
            )
        return out

    @classmethod
    def for_equity(cls, equity: float, **overrides) -> "RiskConfig":
        """Scale the guards to the account rather than leaving them nonsensical.

        The defaults assume a five-figure account. Applied to $200 they produce a
        config that cannot trade. This keeps the *proportions* sane at any size
        while leaving the hard safety limits alone.
        """
        cfg = cls(equity=equity, **overrides)
        if equity < 5_000:
            # Whole-share quantization dominates a small account: it forces
            # position size to whatever one share costs rather than to risk.
            cfg.allow_fractional = True
            # One position at a time, most of the account, and a notional floor
            # low enough that an order can exist at all.
            cfg.max_position_frac = min(0.90, cfg.max_position_frac * 4)
            cfg.max_concurrent = 1
            cfg.min_notional = min(cfg.min_notional, equity * 0.25)
            # The loss cap stays a percentage, so it scales down with the account
            # automatically -- do not widen it to make the numbers look better.
        return cfg


@dataclass(slots=True)
class Position:
    ticker: str
    direction: Direction
    shares: float
    entry: float
    stop: float
    target: float
    signal: Signal
    opened_ns: int = field(default_factory=mono_ns)

    def unrealized(self, price: float) -> float:
        return (price - self.entry) * self.shares * self.direction.value


@dataclass(slots=True)
class Order:
    ticker: str
    direction: Direction
    shares: float
    limit: float
    stop: float
    target: float
    signal: Signal
    reason: str = ""

    @property
    def notional(self) -> float:
        return self.shares * self.limit

    @property
    def is_fractional(self) -> bool:
        return abs(self.shares - round(self.shares)) > 1e-9

    def describe(self) -> str:
        side = "BUY" if self.direction is Direction.LONG else "SELL SHORT"
        qty = f"{self.shares:g}" if self.is_fractional else f"{int(self.shares)}"
        return (
            f"{side} {qty} {self.ticker} @ {self.limit:.2f} "
            f"stop {self.stop:.2f} target {self.target:.2f} "
            f"(${self.notional:,.0f} notional)"
        )


class RiskManager:
    def __init__(self, config: RiskConfig | None = None) -> None:
        self.cfg = config or RiskConfig()
        self.positions: dict[str, Position] = {}
        self.realized_pnl: float = 0.0
        self.last_exit_ns: dict[str, int] = {}
        self.halted: bool = False
        self.halt_reason: str = ""
        self.rejects: dict[str, int] = {}

    # ---- state ---------------------------------------------------------

    def _reject(self, reason: str) -> None:
        self.rejects[reason] = self.rejects.get(reason, 0) + 1

    @property
    def daily_loss_limit(self) -> float:
        return -abs(self.cfg.equity * self.cfg.max_daily_loss_frac)

    def check_kill_switch(self, prices: dict[str, float] | None = None) -> bool:
        """True when trading is halted. Sticky: only `resume()` clears it.

        Counts UNREALIZED loss when prices are supplied. Reading realized P&L
        alone cannot see the loss that actually blows an account: several
        positions opened off the same event are the same bet, they move
        together, and while they are all open the realized figure is still zero.
        The switch would sit at zero right through the drawdown and only fire
        after the damage was booked.
        """
        if self.halted:
            return True

        total = self.realized_pnl
        if prices:
            total += sum(p.unrealized(prices.get(t, p.entry))
                         for t, p in self.positions.items())

        if total <= self.daily_loss_limit:
            self.halted = True
            kind = "total (incl. open)" if prices else "realized"
            self.halt_reason = (
                f"{kind} P&L {total:,.2f} hit limit {self.daily_loss_limit:,.2f}"
            )
            log.error("KILL SWITCH: %s", self.halt_reason)
            return True
        return False

    def halt(self, reason: str) -> None:
        self.halted = True
        self.halt_reason = reason
        log.error("KILL SWITCH (manual): %s", reason)

    def resume(self) -> None:
        self.halted = False
        self.halt_reason = ""

    # ---- sizing --------------------------------------------------------

    def size_order(self, signal: Signal) -> Order | None:
        cfg = self.cfg

        if self.check_kill_switch():
            self._reject("halted")
            return None
        if signal.expired:
            self._reject("expired")
            return None
        if signal.direction is Direction.NEUTRAL:
            self._reject("neutral")
            return None
        if (signal.direction is Direction.SHORT
                and cfg.equity < cfg.short_min_equity):
            self._reject("short_blocked_below_2k")
            return None
        if signal.ref_price <= 0:
            self._reject("no_price")
            return None
        if signal.ticker in self.positions:
            self._reject("already_open")
            return None
        if len(self.positions) >= cfg.max_concurrent:
            self._reject("max_concurrent")
            return None

        last_exit = self.last_exit_ns.get(signal.ticker)
        if last_exit is not None and (mono_ns() - last_exit) / 1e9 < cfg.cooldown_s:
            self._reject("cooldown")
            return None

        px = signal.ref_price
        stop_bps = max(cfg.min_stop_bps, signal.edge_bps * cfg.stop_frac_of_edge)
        if stop_bps > cfg.max_stop_bps:
            # Do not silently clamp: a stop this wide means the edge that
            # produced it is not trustworthy, so the trade should not happen.
            self._reject("stop_too_wide")
            return None
        stop_dist = px * stop_bps / 10_000.0
        if stop_dist <= 0:
            self._reject("bad_stop")
            return None

        risk_dollars = cfg.equity * cfg.risk_per_trade

        # Fractional quantities are longs only -- fractional shorts are not
        # permitted at any equity level.
        fractional = cfg.allow_fractional and signal.direction is Direction.LONG
        if fractional:
            shares = risk_dollars / stop_dist
        else:
            shares = float(int(risk_dollars // stop_dist))

        # Notional cap can bind before the risk budget does on a low-vol name.
        max_notional = cfg.equity * cfg.max_position_frac
        if shares * px > max_notional:
            shares = (max_notional / px) if fractional else float(int(max_notional // px))

        if fractional:
            shares = round(shares, 4)
            if shares < cfg.min_fractional_qty:
                self._reject("size_zero")
                return None
        elif shares < cfg.min_shares:
            self._reject("size_zero")
            return None
        if shares * px < cfg.min_notional:
            self._reject("below_min_notional")
            return None

        sign = signal.direction.value
        stop = px - sign * stop_dist
        target = px + sign * (px * signal.edge_bps / 10_000.0)

        return Order(
            ticker=signal.ticker,
            direction=signal.direction,
            shares=shares,
            limit=px,
            stop=round(stop, 2),
            target=round(target, 2),
            signal=signal,
            reason=f"risk ${risk_dollars:,.0f} / stop {stop_bps:.0f}bps",
        )

    # ---- lifecycle -----------------------------------------------------

    def open(self, order: Order) -> Position:
        pos = Position(
            ticker=order.ticker,
            direction=order.direction,
            shares=order.shares,
            entry=order.limit,
            stop=order.stop,
            target=order.target,
            signal=order.signal,
        )
        self.positions[order.ticker] = pos
        log.info("OPEN %s", order.describe())
        return pos

    def amend_fill(self, ticker: str, shares: int, avg_price: float) -> None:
        """Correct a position to the broker's actual fill.

        Stop and target are recomputed from the real entry rather than kept at
        the levels derived from the assumed one -- a stop measured from a price
        you did not get is the wrong distance from the price you did.
        """
        pos = self.positions.get(ticker)
        if pos is None or shares <= 0 or avg_price <= 0:
            return
        stop_dist = abs(pos.entry - pos.stop)
        target_dist = abs(pos.target - pos.entry)
        sign = pos.direction.value
        pos.shares = shares
        pos.entry = avg_price
        pos.stop = round(avg_price - sign * stop_dist, 2)
        pos.target = round(avg_price + sign * target_dist, 2)

    def drop_unfilled(self, ticker: str) -> bool:
        """Remove a position that the broker never actually gave us.

        Deliberately not `close()`: there was no fill, so there is no P&L to
        realise and no cooldown to serve. Booking a phantom round trip would
        corrupt the daily loss figure the kill switch reads.
        """
        if ticker not in self.positions:
            return False
        del self.positions[ticker]
        log.info("dropped unfilled position %s", ticker)
        return True

    def close(self, ticker: str, price: float, why: str = "") -> float:
        pos = self.positions.pop(ticker, None)
        if pos is None:
            return 0.0
        pnl = pos.unrealized(price)
        self.realized_pnl += pnl
        self.last_exit_ns[ticker] = mono_ns()
        log.info("CLOSE %s @ %.2f pnl=%+.2f (%s) day=%+.2f",
                 ticker, price, pnl, why, self.realized_pnl)
        self.check_kill_switch()
        return pnl

    def pending_exits(self, prices: dict[str, float]) -> list[tuple[str, float, str]]:
        """Which positions have triggered, WITHOUT mutating anything.

        Separated from `close()` so a caller that must send an order can send it
        first and only book the exit once the broker accepts. Booking the fill
        before the order exists deletes the position from our memory while it is
        still live at the broker -- an untracked, unstopped position that nothing
        will ever close.
        """
        out: list[tuple[str, float, str]] = []
        for ticker, pos in self.positions.items():
            px = prices.get(ticker)
            if px is None or px <= 0:
                continue
            if pos.direction is Direction.LONG:
                if px <= pos.stop:
                    out.append((ticker, px, "stop"))
                elif px >= pos.target:
                    out.append((ticker, px, "target"))
            else:
                if px >= pos.stop:
                    out.append((ticker, px, "stop"))
                elif px <= pos.target:
                    out.append((ticker, px, "target"))
        return out

    def check_exits(self, prices: dict[str, float]) -> list[tuple[str, float, str]]:
        """Stops and targets, booked immediately. Only safe when nothing has to
        be sent to a broker -- i.e. alert-only, or a broker-side bracket that has
        already done the closing for us."""
        exits: list[tuple[str, float, str]] = []
        for ticker, pos in list(self.positions.items()):
            px = prices.get(ticker)
            if px is None or px <= 0:
                continue
            if pos.direction is Direction.LONG:
                if px <= pos.stop:
                    exits.append((ticker, px, "stop"))
                elif px >= pos.target:
                    exits.append((ticker, px, "target"))
            else:
                if px >= pos.stop:
                    exits.append((ticker, px, "stop"))
                elif px <= pos.target:
                    exits.append((ticker, px, "target"))
        for ticker, px, why in exits:
            self.close(ticker, px, why)
        return exits

    def exposure(self, prices: dict[str, float] | None = None) -> float:
        prices = prices or {}
        return sum(
            abs(p.shares * prices.get(t, p.entry)) for t, p in self.positions.items()
        )

    def summary(self, prices: dict[str, float] | None = None) -> dict[str, object]:
        prices = prices or {}
        unreal = sum(p.unrealized(prices.get(t, p.entry)) for t, p in self.positions.items())
        return {
            "equity": self.cfg.equity,
            "realized": round(self.realized_pnl, 2),
            "unrealized": round(unreal, 2),
            "open": len(self.positions),
            "exposure": round(self.exposure(prices), 2),
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "rejects": dict(sorted(self.rejects.items(), key=lambda kv: -kv[1])),
        }
