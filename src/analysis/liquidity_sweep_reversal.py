"""Liquidity sweep reversal detector.

This models the common SMC/ICT-style setup: price raids a recent liquidity
pool, rejects back inside the prior range, then confirms with a directional
close. It is deliberately stricter than a generic smart-money sweep phase.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.analysis.indicators import atr, volume_ratio


@dataclass
class LiquiditySweepReversalSignal:
    is_valid: bool = False
    label: str = "liquidity_sweep_reversal"
    direction: str = "none"
    quality_score: float = 0.0
    swept_level: float = 0.0
    reclaim_close: float = 0.0
    volume_ratio: float = 0.0
    trigger: str = "none"
    invalidation: str = "sweep_extreme_reclaimed_against_trade"
    notes: str = ""


def _body_direction(row: pd.Series) -> str:
    close = float(row["close"])
    open_ = float(row["open"])
    if close > open_:
        return "long"
    if close < open_:
        return "short"
    return "none"


def detect_liquidity_sweep_reversal(
    df_primary: pd.DataFrame,
    cfg: dict,
    *,
    direction_hint: str = "none",
) -> LiquiditySweepReversalSignal:
    strategy_cfg = cfg.get("strategy", {}) or {}
    sweep_cfg = strategy_cfg.get("liquidity_sweep_reversal", {}) or {}
    if not bool(sweep_cfg.get("enabled", False)):
        return LiquiditySweepReversalSignal()
    if df_primary.empty or len(df_primary) < 80:
        return LiquiditySweepReversalSignal(notes="insufficient candles")

    lookback = int(sweep_cfg.get("lookback_bars", 24) or 24)
    confirm_bars = max(1, int(sweep_cfg.get("confirm_bars", 2) or 2))
    pierce_min_pct = float(sweep_cfg.get("pierce_min_pct", 0.0015) or 0.0015)
    reclaim_buffer_atr = float(sweep_cfg.get("reclaim_buffer_atr", 0.10) or 0.10)
    min_vol_ratio = float(sweep_cfg.get("min_volume_ratio", 1.20) or 1.20)

    if len(df_primary) < lookback + confirm_bars + 5:
        return LiquiditySweepReversalSignal(notes="insufficient lookback")

    recent = df_primary.iloc[-(lookback + confirm_bars):-confirm_bars]
    trigger_window = df_primary.iloc[-confirm_bars:]
    latest = trigger_window.iloc[-1]
    prior_high = float(recent["high"].max())
    prior_low = float(recent["low"].min())
    current_atr = max(float(atr(df_primary, 14)), 1e-12)
    vol_r = float(volume_ratio(df_primary, 20))

    high = float(trigger_window["high"].max())
    low = float(trigger_window["low"].min())
    close = float(latest["close"])
    body_dir = _body_direction(latest)

    swept_high = high >= prior_high * (1.0 + pierce_min_pct)
    swept_low = low <= prior_low * (1.0 - pierce_min_pct)
    bearish_reclaim = swept_high and close < (prior_high - current_atr * reclaim_buffer_atr)
    bullish_reclaim = swept_low and close > (prior_low + current_atr * reclaim_buffer_atr)

    direction = "none"
    trigger = "none"
    swept_level = 0.0
    if bearish_reclaim and body_dir == "short":
        direction = "short"
        trigger = "buy_side_sweep_reversal"
        swept_level = prior_high
    elif bullish_reclaim and body_dir == "long":
        direction = "long"
        trigger = "sell_side_sweep_reversal"
        swept_level = prior_low

    if direction == "none":
        return LiquiditySweepReversalSignal(volume_ratio=round(vol_r, 4), notes="no sweep reclaim")
    if direction_hint in {"long", "short"} and direction_hint != direction:
        return LiquiditySweepReversalSignal(
            direction=direction,
            volume_ratio=round(vol_r, 4),
            notes="direction hint mismatch",
        )
    if vol_r < min_vol_ratio:
        return LiquiditySweepReversalSignal(
            direction=direction,
            swept_level=round(swept_level, 8),
            reclaim_close=round(close, 8),
            volume_ratio=round(vol_r, 4),
            trigger=trigger,
            notes="volume below confirmation floor",
        )

    pierce_pct = ((high - prior_high) / max(prior_high, 1e-12)) if direction == "short" else ((prior_low - low) / max(prior_low, 1e-12))
    reclaim_distance = abs(close - swept_level) / current_atr
    quality = 76.0
    quality += min(8.0, max(0.0, (vol_r - min_vol_ratio) * 6.0))
    quality += min(8.0, max(0.0, pierce_pct / max(pierce_min_pct, 1e-12) * 2.0))
    quality += min(8.0, max(0.0, reclaim_distance * 2.0))

    return LiquiditySweepReversalSignal(
        is_valid=True,
        direction=direction,
        quality_score=round(min(100.0, quality), 2),
        swept_level=round(swept_level, 8),
        reclaim_close=round(close, 8),
        volume_ratio=round(vol_r, 4),
        trigger=trigger,
        invalidation="sweep_extreme_reclaimed_against_trade",
        notes="liquidity raid + reclaim close + volume confirmation",
    )
