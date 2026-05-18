"""
Technical indicator calculations on OHLCV DataFrames.
All functions return floats or small dicts — no side-effects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
#  Moving averages
# ──────────────────────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def ema_value(df: pd.DataFrame, period: int) -> float:
    return float(ema(df["close"], period).iloc[-1])


# ──────────────────────────────────────────────────────────────────────────────
#  RSI
# ──────────────────────────────────────────────────────────────────────────────

def rsi(df: pd.DataFrame, period: int = 14) -> float:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    return float(rsi_series.iloc[-1])


# ──────────────────────────────────────────────────────────────────────────────
#  ATR
# ──────────────────────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(com=period - 1, adjust=False).mean().iloc[-1])


# ──────────────────────────────────────────────────────────────────────────────
#  Volume
# ──────────────────────────────────────────────────────────────────────────────

def volume_spike(df: pd.DataFrame, lookback: int = 20, multiplier: float = 2.0) -> bool:
    avg = df["volume"].iloc[-lookback - 1:-1].mean()
    current = df["volume"].iloc[-1]
    return bool(current > avg * multiplier)


def volume_ratio(df: pd.DataFrame, lookback: int = 20) -> float:
    avg = df["volume"].iloc[-lookback - 1:-1].mean()
    current = df["volume"].iloc[-1]
    if avg == 0:
        return 1.0
    return float(current / avg)


def buy_volume_ratio(df: pd.DataFrame, lookback: int = 20) -> float:
    """Approximate buy-side pressure: candles where close > open counted as buy volume."""
    recent = df.iloc[-lookback:]
    buy_vol = recent.loc[recent["close"] >= recent["open"], "volume"].sum()
    total_vol = recent["volume"].sum()
    if total_vol == 0:
        return 0.5
    return float(buy_vol / total_vol)


# ──────────────────────────────────────────────────────────────────────────────
#  Trend
# ──────────────────────────────────────────────────────────────────────────────

def trend_direction(df: pd.DataFrame, fast: int = 21, slow: int = 55, trend: int = 200) -> str:
    """Returns 'long', 'short', or 'neutral'."""
    if len(df) < trend:
        return "neutral"
    ema_f = ema_value(df, fast)
    ema_s = ema_value(df, slow)
    ema_t = ema_value(df, trend)
    price = float(df["close"].iloc[-1])
    if price > ema_t and ema_f > ema_s:
        return "long"
    if price < ema_t and ema_f < ema_s:
        return "short"
    return "neutral"


def trend_strength_score(df: pd.DataFrame, cfg: dict) -> float:
    """
    Score 0-100 based on EMA alignment + RSI momentum.
    """
    ind = cfg["indicators"]
    fast, slow, trend_p = ind["ema_fast"], ind["ema_slow"], ind["ema_trend"]
    rsi_v = rsi(df, ind["rsi_period"])
    direction = trend_direction(df, fast, slow, trend_p)

    if direction == "neutral":
        return 0.0

    # RSI in favour of direction
    if direction == "long":
        rsi_score = max(0.0, min(1.0, (rsi_v - 50) / 30))   # 50→0, 80→1
    else:
        rsi_score = max(0.0, min(1.0, (50 - rsi_v) / 30))

    # EMA separation (normalised)
    price = float(df["close"].iloc[-1])
    ema_f = ema_value(df, fast)
    ema_s = ema_value(df, slow)
    separation = abs(ema_f - ema_s) / price
    sep_score = min(1.0, separation / 0.02)   # 2% sep = perfect score

    raw = (rsi_score * 0.5 + sep_score * 0.5) * 100
    return round(raw, 2)


# ──────────────────────────────────────────────────────────────────────────────
#  Volatility
# ──────────────────────────────────────────────────────────────────────────────

def is_extreme_volatility(df: pd.DataFrame, atr_period: int = 14, mult: float = 3.0) -> bool:
    if len(df) < atr_period * 2:
        return False
    current_atr = atr(df, atr_period)
    avg_atr = float(df["high"].sub(df["low"]).rolling(atr_period * 2).mean().iloc[-1])
    return current_atr > avg_atr * mult


def volatility_score(df: pd.DataFrame, cfg: dict) -> float:
    """
    Score 0-100: moderate volatility is ideal.  Too low or too high = lower score.
    """
    ind = cfg["indicators"]
    current_atr = atr(df, ind["atr_period"])
    price = float(df["close"].iloc[-1])
    atr_pct = current_atr / price * 100   # ATR as % of price
    # Ideal: 0.5% – 3%.  Outside of that, linearly decay.
    if 0.5 <= atr_pct <= 3.0:
        return 100.0
    if atr_pct < 0.5:
        return round(atr_pct / 0.5 * 100, 2)
    # > 3%: decay
    return round(max(0.0, 100 - (atr_pct - 3.0) * 20), 2)
