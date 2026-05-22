"""
ADX (Average Directional Index) — Wilder's trend-strength indicator.

This file historically also held a 3-state regime classifier (TRENDING /
SIDEWAYS / HIGH_VOLATILITY) that was superseded by the 4-state
``models.regime_classifier`` (TRENDING_EXPANSION / ACCUMULATION_COMPRESSION
/ DISTRIBUTION / HIGH_VOLATILITY_CHAOS).  The legacy classifier produced a
different taxonomy than the rest of the system, so it was removed during
the 2026-05 audit to eliminate ambiguity.  Only the ADX helper survives —
that's what the four-state classifier consumes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def adx(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder's Average Directional Index — measures trend strength (0-100).

    Values above ~25 signal a trending market. Falls back to 0.0 when the
    smoothed series is undefined (insufficient history or perfectly flat
    bars where +DI and -DI both collapse to zero).
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    atr_s = tr.ewm(com=period - 1, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(com=period - 1, adjust=False).mean() / atr_s.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(com=period - 1, adjust=False).mean() / atr_s.replace(0, np.nan)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx_series = dx.ewm(com=period - 1, adjust=False).mean()

    val = float(adx_series.iloc[-1])
    return val if not np.isnan(val) else 0.0
