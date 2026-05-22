"""
Four-state market regime classifier (futures pipeline).

States
------
TRENDING_EXPANSION       — directional move with healthy participation; trade WITH
ACCUMULATION_COMPRESSION — quiet range, low ATR, ADX flat; squeeze setup
DISTRIBUTION             — exhaustion / topping; long-only sleeves should pause,
                           short-reversal sleeves may activate explicitly
HIGH_VOLATILITY_CHAOS    — non-tradeable; whipsaw / liquidation cascade

Each state exposes:
    .value      stable string used by config + tests
    .label      short human-friendly label
    .tradeable  bool — whether the scorer should pass through to the EV gate
                       (DISTRIBUTION is False by default, but the scorer can
                       upgrade it when StrategyRouter detects a reversal candidate)
"""
from __future__ import annotations

import logging
from enum import Enum

import numpy as np
import pandas as pd

from src.analysis.indicators import atr, ema_value
from src.analysis.regime import adx

log = logging.getLogger(__name__)


class RegimeState(Enum):
    TRENDING_EXPANSION = ("trending_expansion", "Trending Expansion", True)
    ACCUMULATION_COMPRESSION = ("accumulation_compression", "Accumulation / Compression", True)
    DISTRIBUTION = ("distribution", "Distribution / Exhaustion", False)
    HIGH_VOLATILITY_CHAOS = ("chaos", "High-Volatility Chaos", False)

    def __init__(self, value: str, label: str, tradeable: bool):
        self._value_ = value
        self.label = label
        self.tradeable = tradeable


def _atr_pct(df: pd.DataFrame, period: int) -> float:
    price = float(df["close"].iloc[-1])
    if price <= 0:
        return 0.0
    return atr(df, period) / price * 100.0


def _avg_true_range(df: pd.DataFrame, period: int) -> float:
    return float(df["high"].sub(df["low"]).rolling(period * 2).mean().iloc[-1])


def _slope_pct(series: pd.Series, lookback: int) -> float:
    """Linear-regression-free slope: pct change of the smoothed tail vs head."""
    if len(series) < lookback:
        return 0.0
    head = float(series.iloc[-lookback])
    tail = float(series.iloc[-1])
    if head == 0:
        return 0.0
    return (tail - head) / abs(head) * 100.0


def classify_four_state(
    df: pd.DataFrame,
    oi_change_pct: float | None,
    funding_rate: float | None,
    cfg: dict,
) -> RegimeState:
    """
    Classify the market regime using ATR%, ADX, EMA alignment, OI change, and funding.

    The decision tree (in order):
      1. CHAOS    — current ATR > extreme_mult × rolling avg, OR ADX > 60 with
                    range expansion above a hard ceiling.
      2. TRENDING — ADX ≥ trending threshold AND EMAs aligned AND |EMA21 slope|
                    above a small floor.  OI rising in trend direction is a
                    bonus — never required.
      3. DISTRIBUTION — uptrend losing momentum: price near/below EMA50 from
                        above, ADX rolling over, OI falling while funding still
                        very positive (or vice versa for downtrend).
      4. ACCUMULATION_COMPRESSION — low ATR%, ADX < 22, EMAs flat, no
                                     directional break.
      5. fall-back: closer to TRENDING if ADX is moderate and ATR healthy,
                    otherwise CHAOS.
    """
    regime_cfg = cfg.get("regime", {}) or {}
    ind_cfg = cfg.get("indicators", {}) or {}

    atr_period = int(ind_cfg.get("atr_period", 14) or 14)
    adx_period = int(ind_cfg.get("adx_period", 14) or 14)
    ema_fast = int(ind_cfg.get("ema_fast", 50) or 50)
    ema_slow = int(ind_cfg.get("ema_slow", 200) or 200)

    extreme_mult = float(regime_cfg.get("atr_extreme_vol_multiplier", 2.5) or 2.5)
    trending_adx = float(regime_cfg.get("adx_trending_threshold", 25) or 25)
    chaos_adx = float(regime_cfg.get("adx_chaos_threshold", 60) or 60)
    compression_atr_pct_max = float(regime_cfg.get("compression_atr_pct_max", 0.9) or 0.9)
    compression_adx_max = float(regime_cfg.get("compression_adx_max", 22) or 22)
    distribution_funding_threshold = float(regime_cfg.get("distribution_funding_pct", 0.04) or 0.04)
    min_history = max(ema_slow, atr_period * 3, 30)

    if df is None or df.empty or len(df) < min_history:
        return RegimeState.HIGH_VOLATILITY_CHAOS

    # --- raw measurements -------------------------------------------------
    cur_atr = atr(df, atr_period)
    avg_true = _avg_true_range(df, atr_period)
    atr_pct = _atr_pct(df, atr_period)
    adx_val = adx(df, adx_period)
    price = float(df["close"].iloc[-1])
    ema_f = ema_value(df, ema_fast)
    ema_s = ema_value(df, ema_slow)
    ema_21 = ema_value(df, 21)

    is_bullish_align = price > ema_s and ema_f > ema_s
    is_bearish_align = price < ema_s and ema_f < ema_s
    has_direction = is_bullish_align or is_bearish_align
    ema21_slope = _slope_pct(df["close"].ewm(span=21, adjust=False).mean(), lookback=8)
    oi_change = float(oi_change_pct or 0.0)
    fr = float(funding_rate or 0.0)
    fr_pct = fr * 100.0

    # --- 1. CHAOS ---------------------------------------------------------
    if avg_true > 0 and cur_atr > avg_true * extreme_mult:
        log.debug(
            "regime=CHAOS: cur_atr=%.4f > %.4f×avg(%.4f)",
            cur_atr, extreme_mult, avg_true,
        )
        return RegimeState.HIGH_VOLATILITY_CHAOS

    if adx_val > chaos_adx and not has_direction:
        log.debug("regime=CHAOS: adx=%.1f > %.1f without direction", adx_val, chaos_adx)
        return RegimeState.HIGH_VOLATILITY_CHAOS

    # --- 2. TRENDING_EXPANSION -------------------------------------------
    if adx_val >= trending_adx and has_direction and abs(ema21_slope) >= 0.4:
        log.debug(
            "regime=TRENDING: adx=%.1f, slope=%.2f%%, oi=%.2f%%, fr=%.4f%%",
            adx_val, ema21_slope, oi_change, fr_pct,
        )
        return RegimeState.TRENDING_EXPANSION

    # --- 3. DISTRIBUTION --------------------------------------------------
    # Uptrend exhaustion: was bullish, now price cutting back below EMA50,
    # ADX rolling off, AND (funding still elevated long OR OI dumping fast)
    closes = df["close"]
    recent_high = float(closes.iloc[-50:].max())
    if (
        is_bullish_align
        and price < ema_f
        and ema21_slope < 0.0
        and adx_val < trending_adx
        and price <= recent_high * 0.985
        and (fr_pct >= distribution_funding_threshold or oi_change <= -2.5)
    ):
        return RegimeState.DISTRIBUTION

    # Downtrend exhaustion (mirror)
    recent_low = float(closes.iloc[-50:].min())
    if (
        is_bearish_align
        and price > ema_f
        and ema21_slope > 0.0
        and adx_val < trending_adx
        and price >= recent_low * 1.015
        and (fr_pct <= -distribution_funding_threshold or oi_change >= 2.5)
    ):
        # Distribution after a downtrend = potential bottom.  We still flag it
        # as DISTRIBUTION; long-reversal sleeves can opt in.
        return RegimeState.DISTRIBUTION

    # --- 4. ACCUMULATION_COMPRESSION --------------------------------------
    if (
        atr_pct <= compression_atr_pct_max
        and adx_val < compression_adx_max
        and abs(ema21_slope) < 0.6
    ):
        return RegimeState.ACCUMULATION_COMPRESSION

    # --- 5. Fallback ------------------------------------------------------
    if adx_val >= max(20.0, trending_adx - 5.0) and has_direction:
        return RegimeState.TRENDING_EXPANSION
    if atr_pct <= compression_atr_pct_max * 1.4:
        return RegimeState.ACCUMULATION_COMPRESSION
    return RegimeState.HIGH_VOLATILITY_CHAOS
