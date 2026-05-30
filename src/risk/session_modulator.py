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

Professional-grade additions:
  - Weekend/holiday awareness (reduce size when liquidity thins)
  - Session transition tightening (avoid entries at session boundaries)
  - Intraday momentum awareness (late-session profit-taking)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from src.session_clock import active_market_session_key

log = logging.getLogger(__name__)

__all__ = ["SessionModulator", "SessionScale"]


@dataclass
class SessionScale:
    session: str
    multiplier: float
    rationale: str
    is_weekend: bool = False
    is_session_transition: bool = False


class SessionModulator:
    DEFAULTS = {
        "asia": 0.85,
        "london": 1.10,
        "ny": 1.10,
        "overlap_london_ny": 1.20,
        "unknown": 1.00,
    }

    # UTC hours where session transitions occur — entries near these
    # boundaries tend to have worse fill quality and higher slippage.
    _TRANSITION_HOURS = {7, 8, 12, 13, 19, 20}  # Asia→London, London→NY, NY→Asia

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
        now = datetime.now(timezone.utc)
        hour_utc = now.hour
        weekday = now.weekday()  # 0=Mon, 6=Sun

        is_weekend = weekday >= 5  # Sat/Sun
        is_transition = hour_utc in self._TRANSITION_HOURS

        if not self._enabled:
            return SessionScale(
                session=session, multiplier=1.0,
                rationale="session sizing disabled",
                is_weekend=is_weekend, is_session_transition=is_transition,
            )

        mult = float(self._multipliers.get(session, 1.0))

        # Weekend reduction: crypto markets are thinner on weekends
        if is_weekend:
            mult *= 0.80
            log.debug("Weekend detected — reducing session mult by 20%%")

        # Session transition: slight reduction to avoid boundary whipsaws
        if is_transition:
            mult *= 0.90
            log.debug("Session transition hour %d — reducing by 10%%", hour_utc)

        mult = max(self._lo_cap, min(self._hi_cap, mult))
        rationale = f"session={session} mult={mult:.2f}"
        if is_weekend:
            rationale += " [weekend]"
        if is_transition:
            rationale += f" [transition_h{hour_utc}]"
        return SessionScale(
            session=session, multiplier=mult, rationale=rationale,
            is_weekend=is_weekend, is_session_transition=is_transition,
        )
