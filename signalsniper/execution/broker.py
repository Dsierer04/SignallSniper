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


def to_broker_symbol(ticker: str) -> str:
    """SEC's ticker file uses hyphens for share classes (BRK-B, BF-B, HEI-A);
    Alpaca expects dots (BRK.B). Passing the SEC form straight through produces
    an order for a symbol that does not exist, which the broker rejects -- so
    the failure is loud, but only after the moment has passed."""
    return ticker.strip().upper().replace("-", ".")


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
class OrderStatus:
    """What the broker says actually happened, as opposed to what we asked for.

    The distinction is not pedantic. A limit order that never filled leaves the
    risk manager believing it holds a position it does not have -- it will then
    decline other trades against `max_concurrent`, and on a stop trigger it will
    send a *closing* order for shares that were never bought, opening a real
    position in the opposite direction. Assuming the fill is how a missed entry
    becomes an unintended short.
    """

    order_id: str
    status: str = ""           # new, partially_filled, filled, canceled, rejected...
    filled_qty: int = 0
    filled_avg_price: float = 0.0
    requested_qty: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in {"filled", "canceled", "expired", "rejected", "done_for_day"}

    @property
    def is_open(self) -> bool:
        return self.status in {"new", "accepted", "pending_new", "partially_filled",
                               "accepted_for_bidding"}

    @property
    def got_nothing(self) -> bool:
        return self.filled_qty <= 0

    def describe(self) -> str:
        return (f"{self.order_id[:12]} {self.status} "
                f"{self.filled_qty}/{self.requested_qty} @ {self.filled_avg_price:.2f}")


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

    #: True when the account response actually carried PDT fields. Alpaca removed
    #: `daytrade_count` and `pattern_day_trader` from /v2/account on 2026-07-06
    #: when the PDT rule was replaced by the Intraday Margin Framework
    #: (2026-06-04). Absent fields are not the same as zero, so we track whether
    #: they were present rather than inferring "0 day trades used" from silence.
    pdt_fields_present: bool = False

    def blockers(self, need_short: bool = False) -> list[str]:
        out: list[str] = []
        if self.trading_blocked:
            out.append("account is trading-blocked")

        if need_short:
            if not self.shorting_enabled:
                out.append("shorting is not enabled -- every SHORT signal will reject")
            elif self.equity < 2_000:
                # Half the second-order signals are shorts; the margin minimum
                # still applies even though the $25k PDT threshold does not.
                out.append(
                    f"${self.equity:,.0f} equity is under the $2,000 margin/short "
                    "minimum -- short orders will reject"
                )

        # Only evaluate the legacy PDT rule if the broker still reports it. On
        # accounts under the Intraday Margin Framework these fields are gone and
        # gating on them would either crash or invent a limit that no longer
        # exists. Buying power is the real constraint there.
        if self.pdt_fields_present and self.pattern_day_trader and self.equity < 25_000:
            out.append(
                f"flagged PDT with ${self.equity:,.0f} equity (<$25k) -- "
                "day trades will be rejected"
            )

        if self.buying_power > 0 and self.buying_power < self.equity * 0.5:
            out.append(
                f"buying power ${self.buying_power:,.0f} is well below equity "
                f"${self.equity:,.0f} -- intraday capacity may bind before the "
                "risk manager's own limits do"
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
        # equity / buying_power / cash come back as JSON *strings*, not numbers.
        # float() handles both, so this stays correct whichever Alpaca sends.
        return AccountSnapshot(
            equity=float(d.get("equity") or 0.0),
            buying_power=float(d.get("buying_power") or 0.0),
            cash=float(d.get("cash") or 0.0),
            daytrade_count=int(d.get("daytrade_count") or 0),
            pattern_day_trader=bool(d.get("pattern_day_trader")),
            pdt_fields_present=("pattern_day_trader" in d or "daytrade_count" in d),
            trading_blocked=bool(d.get("trading_blocked")),
            shorting_enabled=bool(d.get("shorting_enabled", True)),
            account_number=str(d.get("account_number", "")),
            is_paper=self.paper,
        )

    async def positions(self) -> list[dict[str, Any]]:
        r = await self.client.get(f"{self.base}/v2/positions", headers=self._headers)
        r.raise_for_status()
        return r.json()

    async def order_status(self, order_id: str) -> OrderStatus:
        try:
            r = await self.client.get(f"{self.base}/v2/orders/{order_id}",
                                      headers=self._headers)
        except Exception as exc:
            log.error("order_status(%s) failed: %r", order_id, exc)
            return OrderStatus(order_id=order_id, status="unknown")
        if r.status_code >= 400:
            return OrderStatus(order_id=order_id, status="unknown")
        d = r.json()
        return OrderStatus(
            order_id=order_id,
            status=str(d.get("status", "")),
            filled_qty=int(float(d.get("filled_qty") or 0)),
            filled_avg_price=float(d.get("filled_avg_price") or 0.0),
            requested_qty=int(float(d.get("qty") or 0)),
            raw=d,
        )

    async def cancel_order(self, order_id: str) -> bool:
        try:
            r = await self.client.delete(f"{self.base}/v2/orders/{order_id}",
                                         headers=self._headers)
        except Exception as exc:
            log.error("cancel_order(%s) failed: %r", order_id, exc)
            return False
        return r.status_code < 400

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
        # qty: the API coerces either a JSON string or number. The documented
        # form is a string; both official SDKs emit numbers. String is the safer
        # of the two since it matches the docs and never loses precision.
        # Short orders must be whole shares -- fractional shorting is rejected.
        payload: dict[str, Any] = {
            "symbol": to_broker_symbol(order.ticker),
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
            # NOTE: time_in_force stays "day", so this order dies at the 20:00
            # extended-hours close. An unfilled protective exit therefore ceases
            # to exist exactly when the session ends -- the runner must flatten
            # before 20:00 rather than rely on a resting order surviving it.
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

    #: Override in tests to simulate an order that never fills, or fills short.
    fill_behaviour: str = "full"   # "full" | "none" | "partial"

    async def account(self) -> AccountSnapshot:
        return AccountSnapshot(equity=self.equity, buying_power=self.equity * 2,
                               cash=self.equity, shorting_enabled=self.shorting_enabled,
                               account_number="PAPER", is_paper=True)

    async def order_status(self, order_id: str) -> OrderStatus:
        order = next((o for f, o in zip(self.fills, self.submitted)
                      if f.broker_order_id == order_id), None)
        qty = order.shares if order else 0
        if self.fill_behaviour == "none":
            return OrderStatus(order_id, "new", 0, 0.0, qty)
        if self.fill_behaviour == "partial":
            return OrderStatus(order_id, "partially_filled", max(1, qty // 2),
                               order.limit if order else 0.0, qty)
        return OrderStatus(order_id, "filled", qty,
                           order.limit if order else 0.0, qty)

    async def cancel_order(self, order_id: str) -> bool:
        return True

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
