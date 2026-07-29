"""SignalSniper -- low-latency event ingestion and second-order signal generation.

The thesis in one line: you will not beat colocated HFT to a headline, so do not
try. Trade the propagation instead -- the minutes between a primary repricing
instantly and the names economically downstream of it repricing at human speed.
"""

from .bus import EventBus
from .config import Config, load
from .models import Direction, Event, EventKind, Quote, RawDoc, Signal
from .runner import Runner

__version__ = "0.2.0"

__all__ = [
    "EventBus",
    "Config",
    "load",
    "Direction",
    "Event",
    "EventKind",
    "Quote",
    "RawDoc",
    "Signal",
    "Runner",
    "__version__",
]
