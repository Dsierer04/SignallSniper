"""Broker execution.

Two implementations behind one interface: `AlpacaBroker` (real REST, paper or
live) and `PaperBroker` (in-process, no network, for dry runs and tests).

The design point that matters most here is **where the stop lives**.

In regular hours we submit a bracket order: entry plus a stop leg and a target
leg, all held at the broker. If this process crashes, the stop is still there.

In extended hours Alpaca does not accept bracket orders -- limit only, DAY or
GTC, `extended_hours=true`. So the protective leg cannot be placed at the broker,
and the stop necessarily lives in this process. That makes the process a single
point of failure on precisely the 16:05 earnings trade this system exists for.
`submit()` returns that fact in `Fill.stop_is_client_side` rather than hiding it,
and the runner refuses to open extended-hours positions unless you have opted in
explicitly.

Never crossing the spread blindly: entries are limit orders priced at the signal
reference, not market orders. A market order into a thin after-hours book is how
a 300bps edge becomes a 300bps loss on the fill alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from ..models import Direction
from ..signal.risk import Order
from .session import SessionInfo, current_session

log = logging.getLogger("signalsniper.execution")

PAPER_BASE = "https://paper-api.alpaca.markets"
LIVE_BASE = "https://api.alpaca.markets"


@dataclass(slots=True)
class Fill:
    """Result of a submission. `accepted` means the broker took it, not that it filled."""

    accepted: bool
    broker_order_id: str = ""
    ticker: str = ""
    shares: int = 0
    limit: float = 0.0
    stop_is_client_side: bool = False
    session: str = ""
    error: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        if not self.accepted:
            return f"REJECTED {self.ticker}: {self.error}"
        guard = "CLIENT-SIDE STOP" if self.stop_is_client_side else "broker-side stop"
        return (f"accepted {self.ticker} x{self.shares} @ {self.limit:.2f} "
                f"[{self.session}, {guard}] id={self.broker_order_id}")


@dataclass(slots=True)
class AccountSnapshot:
    equity: float = 0.0
    buying_power: float = 0.0
    cash: float = 0.0
    daytrade_count: int = 0
    pattern_day_trader: bool = False
    trading_blocked: bool = False
    shorting_enabled: bool = True
    account_number: str = ""
    is_paper: bool = True

    def blockers(self, need_short: bool = False) -> list[str]:
        out: list[str] = []
        if self.trading_blocked:
            out.append("account is trading-blocked")
        if need_short and not self.shorting_enabled:
            out.append("shorting is not enabled -- every SHORT signal will reject")
        if self.pattern_day_trader and self.equity < 25_000:
            out.append(
                f"flagged PDT with ${self.equity:,.0f} equity (<$25k) -- "
                "day trades will be rejected"
            )
        elif not self.pattern_day_trader and self.equity < 25_000:
            out.append(
                f"${self.equity:,.0f} equity is under $25k -- you get 3 day trades "
                "per rolling 5 days before PDT lockout. This strategy is "
                "same-day round trips; you will hit it fast."
            )
        return out


class Broker(Protocol):
    async def account(self) -> AccountSnapshot: ...
    async def submit(self, order: Order, session: SessionInfo | None = None) -> Fill: ...
    async def flatten_all(self) -> int: ...
    async def cancel_all(self) -> int: ...


class AlpacaBroker:
    def __init__(self, key: str, secret: str, paper: bool = True,
                 client: httpx.AsyncClient | None = None,
                 allow_extended: bool = False) -> None:
        if not key or not secret:
            raise ValueError("Alpaca credentials required")
        self.base = PAPER_BASE if paper else LIVE_BASE
        self.paper = paper
        self.allow_extended = allow_extended
        self._own_client = client is None
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(8.0, connect=3.0),
            limits=httpx.Limits(max_keepalive_connections=8),
        )
        self._headers = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
            "Content-Type": "application/json",
        }

    async def aclose(self) -> None:
        if self._own_client:
            await self.client.aclose()

    # ---- reads ---------------------------------------------------------

    async def account(self) -> AccountSnapshot:
        r = await self.client.get(f"{self.base}/v2/account", headers=self._headers)
        r.raise_for_status()
        d = r.json()
        return AccountSnapshot(
            equity=float(d.get("equity") or 0.0),
            buying_power=float(d.get("buying_power") or 0.0),
            cash=float(d.get("cash") or 0.0),
            daytrade_count=int(d.get("daytrade_count") or 0),
            pattern_day_trader=bool(d.get("pattern_day_trader")),
            trading_blocked=bool(d.get("trading_blocked")),
            shorting_enabled=bool(d.get("shorting_enabled", True)),
            account_number=str(d.get("account_number", "")),
            is_paper=self.paper,
        )

    async def positions(self) -> list[dict[str, Any]]:
        r = await self.client.get(f"{self.base}/v2/positions", headers=self._headers)
        r.raise_for_status()
        return r.json()

    # ---- writes --------------------------------------------------------

    async def submit(self, order: Order, session: SessionInfo | None = None) -> Fill:
        sess = session or current_session()

        if not sess.session.is_open:
            return Fill(accepted=False, ticker=order.ticker,
                        session=sess.session.value,
                        error=f"market closed ({sess.note or 'outside session'})")

        if sess.is_extended and not self.allow_extended:
            return Fill(
                accepted=False, ticker=order.ticker, session=sess.session.value,
                error="extended-hours trading not enabled: no broker-side stop is "
                      "possible here, so this must be an explicit opt-in",
            )

        side = "buy" if order.direction is Direction.LONG else "sell"
        payload: dict[str, Any] = {
            "symbol": order.ticker,
            "qty": str(order.shares),
            "side": side,
            "type": "limit",
            "limit_price": f"{order.limit:.2f}",
            "time_in_force": "day",
        }

        if sess.supports_bracket:
            # Regular hours: protective legs live at the broker and survive a
            # process crash. Always prefer this.
            payload["order_class"] = "bracket"
            payload["take_profit"] = {"limit_price": f"{order.target:.2f}"}
            payload["stop_loss"] = {"stop_price": f"{order.stop:.2f}"}
            client_side_stop = False
        else:
            # Extended hours: Alpaca rejects bracket/OCO here. Limit only.
            payload["extended_hours"] = True
            client_side_stop = True
            log.warning(
                "%s submitted in %s WITHOUT a broker-side stop -- the stop at "
                "%.2f is enforced by this process only. If it dies, you are naked.",
                order.ticker, sess.session.value, order.stop,
            )

        try:
            r = await self.client.post(f"{self.base}/v2/orders",
                                       headers=self._headers, json=payload)
        except Exception as exc:
            return Fill(accepted=False, ticker=order.ticker,
                        session=sess.session.value, error=f"network: {exc!r}")

        if r.status_code >= 400:
            detail = ""
            try:
                detail = r.json().get("message", r.text)
            except ValueError:
                detail = r.text
            return Fill(accepted=False, ticker=order.ticker,
                        session=sess.session.value,
                        error=f"HTTP {r.status_code}: {detail}")

        d = r.json()
        return Fill(
            accepted=True,
            broker_order_id=str(d.get("id", "")),
            ticker=order.ticker,
            shares=order.shares,
            limit=order.limit,
            stop_is_client_side=client_side_stop,
            session=sess.session.value,
            raw=d,
        )

    async def cancel_all(self) -> int:
        r = await self.client.delete(f"{self.base}/v2/orders", headers=self._headers)
        if r.status_code >= 400:
            log.error("cancel_all failed: HTTP %d %s", r.status_code, r.text)
            return 0
        try:
            return len(r.json())
        except ValueError:
            return 0

    async def flatten_all(self) -> int:
        """Close every position at market. The panic button."""
        r = await self.client.delete(f"{self.base}/v2/positions",
                                     headers=self._headers,
                                     params={"cancel_orders": "true"})
        if r.status_code >= 400:
            log.error("flatten_all failed: HTTP %d %s", r.status_code, r.text)
            return 0
        try:
            return len(r.json())
        except ValueError:
            return 0


class PaperBroker:
    """In-process broker. No network. Records everything for inspection."""

    def __init__(self, equity: float = 25_000.0, allow_extended: bool = True,
                 shorting_enabled: bool = True) -> None:
        self.equity = equity
        self.allow_extended = allow_extended
        self.shorting_enabled = shorting_enabled
        self.submitted: list[Order] = []
        self.fills: list[Fill] = []
        self._seq = 0

    async def account(self) -> AccountSnapshot:
        return AccountSnapshot(equity=self.equity, buying_power=self.equity * 2,
                               cash=self.equity, shorting_enabled=self.shorting_enabled,
                               account_number="PAPER", is_paper=True)

    async def submit(self, order: Order, session: SessionInfo | None = None) -> Fill:
        sess = session or current_session()
        if not sess.session.is_open:
            return Fill(accepted=False, ticker=order.ticker,
                        session=sess.session.value, error="market closed")
        if sess.is_extended and not self.allow_extended:
            return Fill(accepted=False, ticker=order.ticker,
                        session=sess.session.value,
                        error="extended-hours not enabled")
        self._seq += 1
        self.submitted.append(order)
        fill = Fill(accepted=True, broker_order_id=f"paper-{self._seq}",
                    ticker=order.ticker, shares=order.shares, limit=order.limit,
                    stop_is_client_side=not sess.supports_bracket,
                    session=sess.session.value)
        self.fills.append(fill)
        return fill

    async def cancel_all(self) -> int:
        return 0

    async def flatten_all(self) -> int:
        n = len(self.submitted)
        self.submitted.clear()
        return n
