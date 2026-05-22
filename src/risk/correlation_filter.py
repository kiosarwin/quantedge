"""
Correlation filter — block opening highly-correlated trades simultaneously.

Why: BTC, ETH, SOL move together 80%+ of the time on perp futures. If the bot
opens long BTC + long ETH + long SOL on the same momentum impulse, the *real*
exposure is one beta-1 bet at 3R, not three independent positions.  When that
single bet draws down, all three trades stop out together.

Algorithm
---------
1. For each open trade, take the last `lookback` closes on the primary
   timeframe and convert to log-returns.
2. For the candidate symbol, do the same.
3. Compute Pearson correlation of returns vs each open trade.  If max abs
   correlation exceeds `corr_max` AND the candidate direction would compound
   the existing exposure (same direction with positive corr, or opposite
   direction with negative corr), block the trade.
4. Anti-correlated trades in opposing directions are FINE — those hedge.

The filter is an additive layer; existing exposure caps still apply.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["CorrelationFilter", "CorrelationDecision"]


@dataclass
class CorrelationDecision:
    allowed: bool
    reason: str
    max_corr: float = 0.0
    against_symbol: str = ""


class CorrelationFilter:
    def __init__(self, cfg: dict):
        risk_cfg = (cfg or {}).get("risk", {}) or {}
        corr_cfg = risk_cfg.get("correlation_filter", {}) or {}
        self._enabled = bool(corr_cfg.get("enabled", True))
        self._lookback = int(corr_cfg.get("lookback_bars", 48) or 48)
        self._corr_max = float(corr_cfg.get("corr_max", 0.85) or 0.85)
        self._min_overlap = int(corr_cfg.get("min_overlap_bars", 24) or 24)

    def is_enabled(self) -> bool:
        return self._enabled

    def evaluate(
        self,
        candidate_symbol: str,
        candidate_direction: str,
        candidate_returns: pd.Series,
        open_trades: dict[str, dict],
    ) -> CorrelationDecision:
        """Decide whether to allow opening a new trade given existing book.

        Args:
            candidate_symbol:    proposed pair
            candidate_direction: 'long' | 'short'
            candidate_returns:   pd.Series of recent log-returns on primary tf
            open_trades:         {symbol: {"direction": str, "returns": pd.Series}}
        """
        if not self._enabled:
            return CorrelationDecision(True, "correlation filter disabled")
        if not open_trades:
            return CorrelationDecision(True, "no open trades")

        if candidate_returns is None or len(candidate_returns) < self._min_overlap:
            return CorrelationDecision(True, "insufficient candidate history — pass-through")

        cand = candidate_returns.dropna().tail(self._lookback)
        worst_abs = 0.0
        worst_sym = ""
        worst_corr = 0.0

        for sym, info in open_trades.items():
            if sym == candidate_symbol:
                continue
            other_returns = info.get("returns")
            if other_returns is None:
                continue
            other = other_returns.dropna().tail(self._lookback)
            # Align on the shorter overlap.
            n = min(len(cand), len(other))
            if n < self._min_overlap:
                continue
            a = cand.values[-n:]
            b = other.values[-n:]
            if np.std(a) == 0 or np.std(b) == 0:
                continue
            corr = float(np.corrcoef(a, b)[0, 1])
            if math.isnan(corr):
                continue
            same_side = (info.get("direction") == candidate_direction)
            # Same direction × positive corr → compounding exposure.
            # Opposite direction × negative corr → also compounding (both bet
            # on the same factor in the same direction net).
            compounds = (same_side and corr > 0) or (not same_side and corr < 0)
            effective = abs(corr) if compounds else 0.0
            if effective > worst_abs:
                worst_abs = effective
                worst_sym = sym
                worst_corr = corr

        if worst_abs >= self._corr_max:
            return CorrelationDecision(
                allowed=False,
                reason=(
                    f"correlation {worst_corr:+.2f} vs open {worst_sym} "
                    f"exceeds cap {self._corr_max:.2f}"
                ),
                max_corr=worst_abs,
                against_symbol=worst_sym,
            )
        return CorrelationDecision(
            allowed=True,
            reason=f"max corr {worst_abs:.2f} < cap {self._corr_max:.2f}",
            max_corr=worst_abs,
            against_symbol=worst_sym,
        )
