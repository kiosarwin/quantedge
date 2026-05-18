from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

WITA = ZoneInfo("Asia/Makassar")


def active_market_session_key(now: datetime | None = None) -> str:
    current = (now or datetime.now(WITA)).astimezone(WITA)
    hour = current.hour
    if hour < 7:
        return "ny"
    if hour < 15:
        return "asia"
    if hour < 20:
        return "london"
    return "overlap_london_ny"


def active_market_session_label(now: datetime | None = None) -> str:
    return {
        "asia": "Asia",
        "london": "London",
        "overlap_london_ny": "NY+London",
        "ny": "NY",
    }[active_market_session_key(now)]
