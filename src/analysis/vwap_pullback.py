"""VWAP pullback continuation detector."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.analysis.indicators import atr, ema, volume_ratio, vwap


@dataclass
class VWAPPullbackSignal:
    is_valid: bool = False
    label: str = "vwap_pullback_continuation"
    direction: str = "none"
    quality_score: float = 0.0
    vwap_value: float = 0.0
    volume_ratio: float = 0.0
    reclaim_type: str = "none"
    invalidation: str = "vwap_reclaim_failed"
    notes: str = ""


def _ema_trend(df: pd.DataFrame, direction: str, fast: int = 21, slow: int = 55) -> bool:
    if df.empty or len(df) < slow + 5:
        return False
    close = df["close"].astype(float)
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    price = float(close.iloc[-1])
    if direction == "long":
        return price > float(ema_fast.iloc[-1]) > float(ema_slow.iloc[-1])
    if direction == "short":
        return price < float(ema_fast.iloc[-1]) < float(ema_slow.iloc[-1])
    return False


def detect_vwap_pullback_continuation(
    df_primary: pd.DataFrame,
    df_higher: pd.DataFrame,
    cfg: dict,
    *,
    direction_hint: str = "none",
) -> VWAPPullbackSignal:
    strategy_cfg = cfg.get("strategy", {}) or {}
    vwap_cfg = strategy_cfg.get("vwap_pullback_continuation", {}) or {}
    if not bool(vwap_cfg.get("enabled", False)):
        return VWAPPullbackSignal()
    direction = direction_hint if direction_hint in {"long", "short"} else "none"
    if direction == "none":
        return VWAPPullbackSignal(notes="no direction")

    period = int(vwap_cfg.get("vwap_period", 20) or 20)
    pullback_lookback = int(vwap_cfg.get("pullback_lookback", 6) or 6)
    min_vol_ratio = float(vwap_cfg.get("min_volume_ratio", 1.15) or 1.15)
    max_atr_distance = float(vwap_cfg.get("max_atr_distance", 0.85) or 0.85)
    if df_primary.empty or len(df_primary) < max(80, period + pullback_lookback + 5):
        return VWAPPullbackSignal(direction=direction, notes="insufficient primary candles")
    if not _ema_trend(df_primary, direction) or (not df_higher.empty and not _ema_trend(df_higher, direction)):
        return VWAPPullbackSignal(direction=direction, notes="ema trend mismatch")

    current_vwap = vwap(df_primary, period=period)
    current_atr = max(atr(df_primary, 14), 1e-12)
    vol_r = volume_ratio(df_primary, 20)
    latest = df_primary.iloc[-1]
    close = float(latest["close"])
    open_ = float(latest["open"])
    high = float(latest["high"])
    low = float(latest["low"])
    previous = df_primary.iloc[-(pullback_lookback + 1):-1]

    if direction == "long":
        touched_vwap = float(previous["low"].min()) <= current_vwap + current_atr * 0.25
        reclaim = low <= current_vwap + current_atr * max_atr_distance and close > current_vwap and close > open_
        not_extended = (close - current_vwap) <= current_atr * max_atr_distance
        reclaim_type = "vwap_bull_reclaim"
    else:
        touched_vwap = float(previous["high"].max()) >= current_vwap - current_atr * 0.25
        reclaim = high >= current_vwap - current_atr * max_atr_distance and close < current_vwap and close < open_
        not_extended = (current_vwap - close) <= current_atr * max_atr_distance
        reclaim_type = "vwap_bear_reclaim"

    if not touched_vwap:
        return VWAPPullbackSignal(direction=direction, vwap_value=current_vwap, volume_ratio=vol_r, notes="no vwap pullback")
    if not reclaim or not not_extended:
        return VWAPPullbackSignal(direction=direction, vwap_value=current_vwap, volume_ratio=vol_r, notes="no controlled vwap reclaim")
    if vol_r < min_vol_ratio:
        return VWAPPullbackSignal(direction=direction, vwap_value=current_vwap, volume_ratio=vol_r, notes="volume below confirmation floor")

    quality = 74.0
    quality += min(10.0, max(0.0, (vol_r - min_vol_ratio) * 8.0))
    quality += 6.0 if not df_higher.empty and _ema_trend(df_higher, direction) else 0.0
    quality += 4.0 if not_extended else 0.0

    return VWAPPullbackSignal(
        is_valid=True,
        direction=direction,
        quality_score=round(min(100.0, quality), 2),
        vwap_value=round(float(current_vwap), 8),
        volume_ratio=round(float(vol_r), 4),
        reclaim_type=reclaim_type,
        invalidation="loses_vwap_and_pullback_extreme",
        notes="trend aligned pullback to VWAP with volume-backed reclaim",
    )
