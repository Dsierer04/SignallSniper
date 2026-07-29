"""US equity session detection.

This exists because the session changes what orders are *legal*, not just what
is wise. Alpaca rejects market orders and bracket/OCO orders outside regular
hours -- extended hours accepts limit orders only, with `extended_hours=true`
and time-in-force DAY or GTC.

The operational consequence is the important one: **in extended hours there is no
server-side stop.** A bracket order places the protective leg at the broker, so
it survives your process dying. After hours you cannot do that, so the stop lives
in your process, and your process becomes a single point of failure on exactly
the trade this system is designed for (16:05 earnings propagation).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from enum import Enum
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

PREMARKET_OPEN = time(4, 0)
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
AFTERHOURS_CLOSE = time(20, 0)


class Session(Enum):
    CLOSED = "closed"
    PREMARKET = "premarket"
    REGULAR = "regular"
    AFTERHOURS = "afterhours"

    @property
    def is_extended(self) -> bool:
        return self in (Session.PREMARKET, Session.AFTERHOURS)

    @property
    def is_open(self) -> bool:
        return self is not Session.CLOSED

    @property
    def supports_bracket(self) -> bool:
        """Only regular hours accepts bracket orders -- i.e. server-side stops."""
        return self is Session.REGULAR


#: Full-day closures. Half days (early 13:00 close) are handled separately since
#: the extended session also truncates to 17:00 on those dates.
MARKET_HOLIDAYS_2026: frozenset[date] = frozenset({
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # MLK Day
    date(2026, 2, 16),   # Presidents' Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7, 3),    # Independence Day (observed)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas
})

HALF_DAYS_2026: frozenset[date] = frozenset({
    date(2026, 11, 27),  # day after Thanksgiving
    date(2026, 12, 24),  # Christmas Eve
})


@dataclass(frozen=True, slots=True)
class SessionInfo:
    session: Session
    now_et: datetime
    note: str = ""

    @property
    def is_extended(self) -> bool:
        return self.session.is_extended

    @property
    def supports_bracket(self) -> bool:
        return self.session.supports_bracket


def current_session(now: datetime | None = None) -> SessionInfo:
    """Classify the moment. Pass an aware datetime to test a specific instant."""
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("naive datetime -- pass an aware one, timezone bugs cost money")

    et = now.astimezone(ET)
    today = et.date()

    if et.weekday() >= 5:
        return SessionInfo(Session.CLOSED, et, "weekend")
    if today in MARKET_HOLIDAYS_2026:
        return SessionInfo(Session.CLOSED, et, "market holiday")

    half_day = today in HALF_DAYS_2026
    regular_close = time(13, 0) if half_day else REGULAR_CLOSE
    ext_close = time(17, 0) if half_day else AFTERHOURS_CLOSE
    note = "half day" if half_day else ""

    t = et.time()
    if t < PREMARKET_OPEN:
        return SessionInfo(Session.CLOSED, et, note or "before premarket")
    if t < REGULAR_OPEN:
        return SessionInfo(Session.PREMARKET, et, note)
    if t < regular_close:
        return SessionInfo(Session.REGULAR, et, note)
    if t < ext_close:
        return SessionInfo(Session.AFTERHOURS, et, note)
    return SessionInfo(Session.CLOSED, et, note or "after extended close")
