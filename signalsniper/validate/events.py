"""Known historical earnings events to validate against.

Times are the press-release drop, not the call. Apple has released at ~16:30 ET
for years; Amazon at ~16:00-16:05 ET. Both hold calls at 17:00 ET.

These dates are the input to the study, and a wrong date silently produces a
"no edge" result -- measuring a window where nothing happened looks identical to
measuring a window where the market was efficient. `verify_dates()` is provided
so a run can sanity-check that the primary actually moved on each date before
drawing any conclusion from the linked names.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


@dataclass(frozen=True, slots=True)
class EarningsEvent:
    event_id: str
    primary: str
    t_release: datetime
    note: str = ""


def _et(y: int, m: int, d: int, hh: int, mm: int) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=ET)


#: Apple releases at approximately 16:30 ET. Verify each against the actual 8-K
#: acceptance timestamp on EDGAR before trusting a marginal result.
AAPL_EVENTS: tuple[EarningsEvent, ...] = (
    EarningsEvent("AAPL-2026Q2", "AAPL", _et(2026, 4, 30, 16, 30)),
    EarningsEvent("AAPL-2026Q1", "AAPL", _et(2026, 1, 29, 16, 30)),
    EarningsEvent("AAPL-2025Q4", "AAPL", _et(2025, 10, 30, 16, 30)),
    EarningsEvent("AAPL-2025Q3", "AAPL", _et(2025, 7, 31, 16, 30)),
    EarningsEvent("AAPL-2025Q2", "AAPL", _et(2025, 5, 1, 16, 30)),
    EarningsEvent("AAPL-2025Q1", "AAPL", _et(2025, 1, 30, 16, 30)),
    EarningsEvent("AAPL-2024Q4", "AAPL", _et(2024, 10, 31, 16, 30)),
    EarningsEvent("AAPL-2024Q3", "AAPL", _et(2024, 8, 1, 16, 30)),
    EarningsEvent("AAPL-2024Q2", "AAPL", _et(2024, 5, 2, 16, 30)),
    EarningsEvent("AAPL-2024Q1", "AAPL", _et(2024, 2, 1, 16, 30)),
)

#: Amazon releases at approximately 16:00-16:05 ET.
AMZN_EVENTS: tuple[EarningsEvent, ...] = (
    EarningsEvent("AMZN-2026Q1", "AMZN", _et(2026, 4, 30, 16, 5)),
    EarningsEvent("AMZN-2025Q4", "AMZN", _et(2026, 2, 5, 16, 5)),
    EarningsEvent("AMZN-2025Q3", "AMZN", _et(2025, 10, 30, 16, 5)),
    EarningsEvent("AMZN-2025Q2", "AMZN", _et(2025, 7, 31, 16, 5)),
    EarningsEvent("AMZN-2025Q1", "AMZN", _et(2025, 5, 1, 16, 5)),
    EarningsEvent("AMZN-2024Q4", "AMZN", _et(2025, 2, 6, 16, 5)),
    EarningsEvent("AMZN-2024Q3", "AMZN", _et(2024, 10, 31, 16, 5)),
    EarningsEvent("AMZN-2024Q2", "AMZN", _et(2024, 8, 1, 16, 5)),
    EarningsEvent("AMZN-2024Q1", "AMZN", _et(2024, 4, 30, 16, 5)),
)

ALL_EVENTS: tuple[EarningsEvent, ...] = AAPL_EVENTS + AMZN_EVENTS


def events_for(primary: str) -> tuple[EarningsEvent, ...]:
    p = primary.upper()
    return tuple(e for e in ALL_EVENTS if e.primary == p)
