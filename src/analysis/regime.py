"""
Market Regime Detector

Classifies the current market state into one of three regimes:
  TRENDING     → strong directional move; trading allowed
  SIDEWAYS     → range-bound / choppy; NO TRADE
  HIGH_VOLATILITY → extreme moves; cautious / skip

Classification uses:
  - ADX (directional strength)
  - ATR % of price (volatility normalised)
  - EMA50 vs EMA200 slope
"""
from __future__ import annotations

import logging
from enum import Enum

import numpy as np
import pandas as pd

from src.analysis.indicators import ema, atr

log = logging.getLogger(__name__)


class Regime(str, Enum):
    TRENDING = "trending"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"


def adx(df: pd.DataFrame, period: int = 14) -> float:
    """
    Wilder's Average Directional Index — measures trend strength (0–100).
    Values > 25 indicate a trending market.
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


def classify_regime(df: pd.DataFrame, cfg: dict) -> Regime:
    """
    Classify the market regime for a given OHLCV DataFrame.

    Args:
        df:  OHLCV DataFrame (recommend 4H timeframe, ≥ 100 bars)
        cfg: Full config dict

    Returns:
        Regime enum value
    """
    regime_cfg = cfg.get("regime", {})
    ind_cfg = cfg.get("indicators", {})

    adx_threshold = regime_cfg.get("adx_trending_threshold", 25)
    extreme_mult = regime_cfg.get("atr_extreme_vol_multiplier", 2.5)
    sideways_atr_max = regime_cfg.get("sideways_atr_pct_max", 0.5)
    atr_period = ind_cfg.get("atr_period", 14)
    ema_fast = ind_cfg.get("ema_fast", 50)
    ema_slow = ind_cfg.get("ema_slow", 200)

    if len(df) < max(ema_slow, atr_period * 3, 30):
        return Regime.SIDEWAYS   # not enough data → assume sideways

    current_atr = atr(df, atr_period)
    price = float(df["close"].iloc[-1])
    atr_pct = current_atr / price * 100

    # ── Extreme volatility check ──────────────────────────────────────
    avg_atr = float(df["high"].sub(df["low"]).rolling(atr_period * 2).mean().iloc[-1])
    if current_atr > avg_atr * extreme_mult:
        log.debug("Regime=HIGH_VOLATILITY  atr_pct=%.2f%%  extreme_mult=%.1f", atr_pct, extreme_mult)
        return Regime.HIGH_VOLATILITY

    # ── ADX-based trending check ──────────────────────────────────────
    adx_val = adx(df, ind_cfg.get("adx_period", 14))

    ema_f = float(ema(df["close"], ema_fast).iloc[-1])
    ema_s = float(ema(df["close"], ema_slow).iloc[-1])

    # Trending: ADX strong AND EMAs in right order AND price above EMA200
    is_bullish_trend = (price > ema_s) and (ema_f > ema_s)
    is_bearish_trend = (price < ema_s) and (ema_f < ema_s)
    has_direction = is_bullish_trend or is_bearish_trend

    if adx_val >= adx_threshold and has_direction:
        log.debug(
            "Regime=TRENDING  adx=%.1f  ema_f=%.4f  ema_s=%.4f  price=%.4f",
            adx_val, ema_f, ema_s, price,
        )
        return Regime.TRENDING

    # ── Sideways: low ATR OR weak ADX ────────────────────────────────
    log.debug(
        "Regime=SIDEWAYS  adx=%.1f (need %.0f)  atr_pct=%.2f%%",
        adx_val, adx_threshold, atr_pct,
    )
    return Regime.SIDEWAYS


def regime_score_modifier(regime: Regime) -> float:
    """
    Multiplier applied to the final score based on regime.
    TRENDING → full score
    SIDEWAYS → block trade (returns 0.0 multiplier)
    HIGH_VOLATILITY → heavy penalty
    """
    return {
        Regime.TRENDING: 1.0,
        Regime.SIDEWAYS: 0.0,       # no trade in sideways
        Regime.HIGH_VOLATILITY: 0.5,
    }[regime]
