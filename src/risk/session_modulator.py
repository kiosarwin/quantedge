"""
Session-aware size modulator.

Crypto futures volatility and edge profile differ markedly across the three
major TradFi sessions:

    Asia        20:00–08:00 UTC  (low vol, mean-reverting wicks dominate)
    London      08:00–13:00 UTC  (vol expansion, BTC/ETH lead)
    NY          13:00–22:00 UTC  (highest vol, news-driven, trends extend)
    Overlap     13:00–17:00 UTC  (London + NY — best risk/reward window)

Rather than expressing this through new strategies, we apply a multiplicative
size scalar to the FundManager output so the *same* signal travels with more
or less notional depending on when it fires.

Defaults are derived from rolling Sharpe-by-session studies on 2024–2025
BTC/ETH futures: NY+overlap > London > NY > Asia.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from src.session_clock import active_market_session_key

log = logging.getLogger(__name__)

__all__ = ["SessionModulator", "SessionScale"]


@dataclass
class SessionScale:
    session: str
    multiplier: float
    rationale: str


class SessionModulator:
    DEFAULTS = {
        "asia": 0.85,
        "london": 1.10,
        "ny": 1.10,
        "overlap_london_ny": 1.20,
        "unknown": 1.00,
    }

    def __init__(self, cfg: dict):
        cfg = cfg or {}
        safety_cfg = cfg.get("safety", {}) or {}
        overrides = safety_cfg.get("session_size_multipliers", {}) or {}
        self._enabled = bool(safety_cfg.get("session_aware_sizing", True))
        self._multipliers = {**self.DEFAULTS, **{k: float(v) for k, v in overrides.items()}}
        # Hard guard rail — never let a session amplify beyond 1.5x or compress below 0.5x.
        self._lo_cap = 0.5
        self._hi_cap = 1.5

    def is_enabled(self) -> bool:
        return self._enabled

    def current(self) -> SessionScale:
        session = active_market_session_key()
        if not self._enabled:
            return SessionScale(session=session, multiplier=1.0, rationale="session sizing disabled")
        mult = float(self._multipliers.get(session, 1.0))
        mult = max(self._lo_cap, min(self._hi_cap, mult))
        rationale = f"session={session} mult={mult:.2f}"
        return SessionScale(session=session, multiplier=mult, rationale=rationale)
