"""
Market structure analysis — swing H/L detection, Break-of-Structure (BOS),
liquidity sweeps, and support/resistance levels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def find_swing_highs(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """
    Vectorized swing-high detection using rolling windows.
    A bar is a swing high if its 'high' equals the max over the
    [i-lookback … i+lookback] window — computed with two rolling passes.
    """
    highs = df["high"]
    window = 2 * lookback + 1
    # rolling max centred: shift forward so the window is [i-lookback, i+lookback]
    roll_max = highs.rolling(window, min_periods=window).max().shift(-lookback)
    return (highs == roll_max) & highs.notna()


def find_swing_lows(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """
    Vectorized swing-low detection using rolling windows.
    """
    lows = df["low"]
    window = 2 * lookback + 1
    roll_min = lows.rolling(window, min_periods=window).min().shift(-lookback)
    return (lows == roll_min) & lows.notna()


def get_recent_swing_levels(df: pd.DataFrame, lookback: int = 10, count: int = 3) -> dict:
    """
    Returns recent swing highs and lows as price levels.
    """
    sh = find_swing_highs(df, lookback)
    sl = find_swing_lows(df, lookback)
    highs = df.loc[sh, "high"].iloc[-count:].tolist()
    lows = df.loc[sl, "low"].iloc[-count:].tolist()
    return {"swing_highs": highs, "swing_lows": lows}


def detect_bos(df: pd.DataFrame, lookback: int = 10, confirm_candles: int = 2) -> str:
    """
    Detects Break-of-Structure.
    Returns 'bullish_bos', 'bearish_bos', or 'none'.
    """
    if len(df) < lookback * 3:
        return "none"

    sl = find_swing_lows(df, lookback)
    sh = find_swing_highs(df, lookback)
    close = df["close"]

    # Bullish BOS: price closes above the most recent confirmed swing high
    recent_highs = df.loc[sh, "high"]
    if not recent_highs.empty:
        last_swing_high = float(recent_highs.iloc[-1])
        confirmed_closes = close.iloc[-confirm_candles:]
        if all(c > last_swing_high for c in confirmed_closes):
            return "bullish_bos"

    # Bearish BOS: price closes below the most recent confirmed swing low
    recent_lows = df.loc[sl, "low"]
    if not recent_lows.empty:
        last_swing_low = float(recent_lows.iloc[-1])
        confirmed_closes = close.iloc[-confirm_candles:]
        if all(c < last_swing_low for c in confirmed_closes):
            return "bearish_bos"

    return "none"


def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 10, sweep_pct: float = 0.003) -> str:
    """
    Detects if price recently swept above a swing high (bull trap / liquidity grab)
    or below a swing low (bear trap) and then reversed.
    Returns 'swept_highs', 'swept_lows', or 'none'.
    """
    if len(df) < lookback * 2 + 5:
        return "none"

    recent = df.iloc[-5:]
    prev = df.iloc[:-5]

    sh = find_swing_highs(prev, lookback)
    sl = find_swing_lows(prev, lookback)

    recent_highs = prev.loc[sh, "high"]
    recent_lows = prev.loc[sl, "low"]

    if not recent_highs.empty:
        last_high = float(recent_highs.iloc[-1])
        # Wick pierced above but closed below
        max_wick = recent["high"].max()
        last_close = float(recent["close"].iloc[-1])
        if max_wick > last_high * (1 + sweep_pct) and last_close < last_high:
            return "swept_highs"

    if not recent_lows.empty:
        last_low = float(recent_lows.iloc[-1])
        min_wick = recent["low"].min()
        last_close = float(recent["close"].iloc[-1])
        if min_wick < last_low * (1 - sweep_pct) and last_close > last_low:
            return "swept_lows"

    return "none"


def structure_quality_score(df: pd.DataFrame, cfg: dict) -> float:
    """
    Score 0-100 based on BOS + liquidity sweep signals.
    """
    struct_cfg = cfg["structure"]
    lookback = struct_cfg["swing_lookback"]
    confirm = struct_cfg["bos_confirmation_candles"]
    sweep_pct = struct_cfg["liquidity_sweep_pct"] / 100

    bos = detect_bos(df, lookback, confirm)
    sweep = detect_liquidity_sweep(df, lookback, sweep_pct)

    score = 0.0

    if bos != "none":
        score += 60.0

    # Sweep in the direction of the BOS is strongest signal
    if bos == "bullish_bos" and sweep == "swept_lows":
        score += 40.0
    elif bos == "bearish_bos" and sweep == "swept_highs":
        score += 40.0
    elif sweep != "none":
        score += 20.0

    return min(100.0, score)


def trade_direction_from_structure(df: pd.DataFrame, cfg: dict) -> str:
    """Returns 'long', 'short', or 'none' based on structure."""
    struct_cfg = cfg["structure"]
    bos = detect_bos(df, struct_cfg["swing_lookback"], struct_cfg["bos_confirmation_candles"])
    sweep = detect_liquidity_sweep(
        df,
        struct_cfg["swing_lookback"],
        struct_cfg["liquidity_sweep_pct"] / 100,
    )
    if bos == "bullish_bos" or sweep == "swept_lows":
        return "long"
    if bos == "bearish_bos" or sweep == "swept_highs":
        return "short"
    return "none"
