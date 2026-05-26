"""Multi-timeframe price-action continuation detector."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class MTFPriceActionSignal:
    is_valid: bool = False
    label: str = "mtf_price_action_continuation"
    direction: str = "none"
    quality_score: float = 0.0
    htf_trend: str = "none"
    primary_trigger: str = "none"
    lower_tf_confirmed: bool = False
    invalidation: str = "breaks_pullback_structure"
    notes: str = ""


def _close(df: pd.DataFrame) -> pd.Series:
    return df["close"].astype(float)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _trend_state(df: pd.DataFrame) -> str:
    if df.empty or len(df) < 55:
        return "none"
    close = _close(df)
    ema21 = _ema(close, 21)
    ema55 = _ema(close, 55)
    price = float(close.iloc[-1])
    if price > float(ema21.iloc[-1]) > float(ema55.iloc[-1]):
        return "long"
    if price < float(ema21.iloc[-1]) < float(ema55.iloc[-1]):
        return "short"
    return "none"


def _lower_confirms(df: pd.DataFrame, direction: str) -> bool:
    if df.empty or len(df) < 8:
        return False
    latest = df.iloc[-1]
    prior_close = float(df["close"].iloc[-2])
    close = float(latest["close"])
    open_ = float(latest["open"])
    if direction == "long":
        return close > open_ and close >= prior_close
    if direction == "short":
        return close < open_ and close <= prior_close
    return False


def detect_mtf_price_action_continuation(
    df_primary: pd.DataFrame,
    df_higher: pd.DataFrame,
    df_lower: pd.DataFrame | None,
    cfg: dict,
    *,
    direction_hint: str = "none",
) -> MTFPriceActionSignal:
    strategy_cfg = cfg.get("strategy", {}) or {}
    mtf_cfg = strategy_cfg.get("mtf_price_action_continuation", {}) or {}
    if not bool(mtf_cfg.get("enabled", False)):
        return MTFPriceActionSignal()
    if df_primary.empty or len(df_primary) < 80 or df_higher.empty or len(df_higher) < 55:
        return MTFPriceActionSignal(notes="insufficient candles")

    htf_trend = _trend_state(df_higher)
    direction = direction_hint if direction_hint in {"long", "short"} else htf_trend
    if direction not in {"long", "short"} or direction != htf_trend:
        return MTFPriceActionSignal(direction=direction, htf_trend=htf_trend, notes="htf direction mismatch")

    lookback = int(mtf_cfg.get("breakout_lookback", 20) or 20)
    pullback_lookback = int(mtf_cfg.get("pullback_lookback", 8) or 8)
    close = _close(df_primary)
    ema21 = _ema(close, 21)
    recent = df_primary.iloc[-(lookback + 1):-1]
    pullback = df_primary.iloc[-(pullback_lookback + 1):-1]
    if recent.empty or pullback.empty:
        return MTFPriceActionSignal(direction=direction, htf_trend=htf_trend, notes="insufficient structure")

    latest_close = float(close.iloc[-1])
    latest_open = float(df_primary["open"].iloc[-1])
    latest_ema21 = float(ema21.iloc[-1])
    lower_confirmed = _lower_confirms(df_lower if df_lower is not None else pd.DataFrame(), direction)

    if direction == "long":
        breakout_level = float(recent["high"].max())
        held_pullback = float(pullback["low"].min()) <= latest_ema21 <= latest_close
        impulse = latest_close > latest_open and latest_close > breakout_level
        primary_trigger = "breakout_reclaim_high"
    else:
        breakout_level = float(recent["low"].min())
        held_pullback = float(pullback["high"].max()) >= latest_ema21 >= latest_close
        impulse = latest_close < latest_open and latest_close < breakout_level
        primary_trigger = "breakdown_reclaim_low"

    if not (held_pullback and impulse):
        return MTFPriceActionSignal(
            direction=direction,
            htf_trend=htf_trend,
            lower_tf_confirmed=lower_confirmed,
            notes="no pullback-continuation trigger",
        )

    quality = 72.0
    if lower_confirmed:
        quality += 8.0
    if direction == "long" and latest_close > latest_ema21:
        quality += 5.0
    if direction == "short" and latest_close < latest_ema21:
        quality += 5.0

    return MTFPriceActionSignal(
        is_valid=True,
        direction=direction,
        quality_score=round(min(100.0, quality), 2),
        htf_trend=htf_trend,
        primary_trigger=primary_trigger,
        lower_tf_confirmed=lower_confirmed,
        invalidation="breaks_pullback_structure",
        notes="htf trend + pullback hold + continuation break",
    )
