"""
Entry Strategy Classifier — Futures Edition (Long + Short)

Scientifically-backed entry strategies for Binance USDT-M Perpetual Futures.
Both LONG and SHORT entries are first-class citizens.

Academic References:
─────────────────────
1. Jegadeesh & Titman (1993) — Momentum persistence: winners continue winning,
   losers continue losing over 3-12 month horizons.
2. Dobrynskaya (2023, SSRN:3913263) — Crypto momentum on short horizons (2-4 weeks)
   with reversal beyond 1 month.
3. Zarattini & Aziz (2023, SSRN:4631351) — VWAP as directional bias: long above VWAP,
   short below VWAP yields significant alpha.
4. arXiv:2307.15599 — Order book volume imbalance predicts short-term price direction.
5. Nakagawa & Sakemoto (2024, SSRN:5001299) — Cross-sectional reversal portfolios
   in crypto generate higher returns than conventional momentum.
6. arXiv:2602.00776 — Order flow imbalance, bid-ask spreads, and depth explain
   substantial fraction of return variation at short horizons.

Strategy Catalogue:
─────────────────────
1. MOMENTUM_BREAKOUT     — Trend continuation on BOS + volume + EMA alignment (L/S)
2. VWAP_RECLAIM          — Price reclaims VWAP with volume confirmation (L/S)
3. LIQUIDITY_SWEEP       — Fake breakdown/breakup (stop hunt) then reversal (L/S)
4. FUNDING_EXTREME       — Contrarian entry when funding is extremely one-sided (L/S)
5. OI_DIVERGENCE         — Price makes new high/low but OI diverges (L/S)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from src.analysis.indicators import ema, atr, rsi, volume_ratio, buy_volume_ratio
from src.analysis.structure import (
    find_swing_highs, find_swing_lows, detect_bos, detect_liquidity_sweep,
)

log = logging.getLogger(__name__)



class EntryStrategy(str, Enum):
    MOMENTUM_BREAKOUT = "momentum_breakout"
    VWAP_RECLAIM = "vwap_reclaim"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    FUNDING_EXTREME = "funding_extreme"
    OI_DIVERGENCE = "oi_divergence"
    NONE = "none"


@dataclass
class EntrySignal:
    strategy: EntryStrategy
    direction: str              # 'long' | 'short'
    confidence: float           # 0.0 – 1.0
    entry_price: float
    stop_loss: float
    notes: str = ""

    @property
    def is_valid(self) -> bool:
        return self.strategy != EntryStrategy.NONE and self.confidence > 0.45



def _vwap(df: pd.DataFrame, period: int = 20) -> float:
    """
    Session VWAP approximation using rolling typical price × volume.
    Per Zarattini & Aziz (SSRN:4631351): price vs VWAP is a robust
    directional filter for intraday/swing strategies.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (tp * df["volume"]).rolling(period).sum()
    cum_vol = df["volume"].rolling(period).sum()
    vwap_series = cum_tp_vol / cum_vol.replace(0, np.nan)
    return float(vwap_series.iloc[-1]) if not vwap_series.empty else float(df["close"].iloc[-1])


def _sell_volume_ratio(df: pd.DataFrame, lookback: int = 20) -> float:
    """Sell-side pressure: candles where close < open counted as sell volume."""
    recent = df.iloc[-lookback:]
    sell_vol = recent.loc[recent["close"] < recent["open"], "volume"].sum()
    total_vol = recent["volume"].sum()
    if total_vol == 0:
        return 0.5
    return float(sell_vol / total_vol)



def detect_entry(
    df: pd.DataFrame,
    cfg: dict,
    direction_hint: str = "both",
    funding_rate: float = 0.0,
    oi_change_pct: float = 0.0,
    taker_buy_ratio: float = 0.5,
) -> EntrySignal:
    """
    Run all strategy detectors for BOTH long and short, return highest-confidence.

    Args:
        df:              OHLCV DataFrame (primary TF, at least 100 bars)
        cfg:             Full config dict
        direction_hint:  'long', 'short', or 'both' — from structure analysis
        funding_rate:    Current funding rate (fraction, e.g. 0.0001)
        oi_change_pct:   OI change % vs previous snapshot
        taker_buy_ratio: Taker buy volume / total taker volume

    Returns:
        Best EntrySignal (check signal.is_valid before using)
    """
    ind_cfg = cfg.get("indicators", {})
    entry_cfg = cfg.get("entry", {})
    struct_cfg = cfg.get("structure", {})

    if len(df) < 60:
        return EntrySignal(EntryStrategy.NONE, "none", 0.0, 0.0, 0.0)

    signals: list[EntrySignal] = []
    directions = ["long", "short"] if direction_hint == "both" else [direction_hint]

    for direction in directions:
        # Strategy 1: Momentum Breakout (Jegadeesh & Titman)
        sig = _detect_momentum_breakout(df, ind_cfg, struct_cfg, entry_cfg, direction)
        if sig.is_valid:
            signals.append(sig)

        # Strategy 2: VWAP Reclaim (Zarattini & Aziz)
        sig = _detect_vwap_reclaim(df, ind_cfg, entry_cfg, direction)
        if sig.is_valid:
            signals.append(sig)

        # Strategy 3: Liquidity Sweep (Smart Money Concepts)
        sig = _detect_liquidity_sweep(df, ind_cfg, struct_cfg, entry_cfg, direction)
        if sig.is_valid:
            signals.append(sig)

    # Strategy 4: Funding Extreme (contrarian — direction from funding)
    sig = _detect_funding_extreme(df, ind_cfg, entry_cfg, funding_rate, taker_buy_ratio)
    if sig.is_valid:
        signals.append(sig)

    # Strategy 5: OI Divergence
    sig = _detect_oi_divergence(df, ind_cfg, entry_cfg, oi_change_pct, direction_hint)
    if sig.is_valid:
        signals.append(sig)

    if not signals:
        return EntrySignal(EntryStrategy.NONE, "none", 0.0, 0.0, 0.0)

    best = max(signals, key=lambda s: s.confidence)
    log.debug(
        "Entry signal: %s %s  conf=%.2f  entry=%.4f  sl=%.4f",
        best.direction, best.strategy.value, best.confidence, best.entry_price, best.stop_loss,
    )
    return best



# ══════════════════════════════════════════════════════════════════════════════
#  Strategy 1: Momentum Breakout (L/S)
#  Ref: Jegadeesh & Titman (1993), Dobrynskaya (2023)
#  Logic: BOS confirmed + EMA alignment + volume expansion + RSI momentum
# ══════════════════════════════════════════════════════════════════════════════

def _detect_momentum_breakout(
    df: pd.DataFrame,
    ind_cfg: dict,
    struct_cfg: dict,
    entry_cfg: dict,
    direction: str,
) -> EntrySignal:
    lookback = struct_cfg.get("swing_lookback", 10)
    atr_period = ind_cfg.get("atr_period", 14)
    ema_fast = ind_cfg.get("ema_fast", 21)
    ema_slow = ind_cfg.get("ema_slow", 55)
    min_vol_ratio = entry_cfg.get("min_volume_ratio", 1.3)
    null = EntrySignal(EntryStrategy.NONE, direction, 0.0, 0.0, 0.0)

    if len(df) < max(lookback * 3, ema_slow + 10):
        return null

    # BOS detection
    bos = detect_bos(df, lookback, struct_cfg.get("bos_confirmation_candles", 2))
    if direction == "long" and bos != "bullish_bos":
        return null
    if direction == "short" and bos != "bearish_bos":
        return null

    # EMA alignment check
    ema_f = float(ema(df["close"], ema_fast).iloc[-1])
    ema_s = float(ema(df["close"], ema_slow).iloc[-1])
    price = float(df["close"].iloc[-1])
    current_atr = atr(df, atr_period)

    if direction == "long":
        if not (price > ema_f > ema_s):
            return null
    else:
        if not (price < ema_f < ema_s):
            return null

    # Volume confirmation
    vol_r = volume_ratio(df, ind_cfg.get("volume_lookback", 20))
    has_volume = vol_r >= min_vol_ratio

    # RSI momentum filter (avoid overbought/oversold entries)
    rsi_v = rsi(df, ind_cfg.get("rsi_period", 14))
    if direction == "long":
        rsi_ok = 45 < rsi_v < 78  # bullish but not exhausted
    else:
        rsi_ok = 22 < rsi_v < 55  # bearish but not capitulated

    # Confidence scoring
    confidence = 0.0
    if has_volume:
        confidence += 0.35
    if rsi_ok:
        confidence += 0.30
    # EMA separation bonus
    sep = abs(ema_f - ema_s) / price
    if sep > 0.01:
        confidence += 0.20
    # Candle body strength
    last = df.iloc[-1]
    body = abs(float(last["close"]) - float(last["open"]))
    rng = float(last["high"]) - float(last["low"])
    if rng > 0 and body / rng > 0.6:
        confidence += 0.15

    if confidence < 0.45:
        return null

    # Stop loss placement: 1.5 ATR beyond entry
    if direction == "long":
        sl = price - current_atr * 1.5
    else:
        sl = price + current_atr * 1.5

    return EntrySignal(
        strategy=EntryStrategy.MOMENTUM_BREAKOUT,
        direction=direction,
        confidence=min(1.0, confidence),
        entry_price=price,
        stop_loss=sl,
        notes=f"bos={bos} ema_sep={sep:.4f} vol_r={vol_r:.2f} rsi={rsi_v:.1f}",
    )



# ══════════════════════════════════════════════════════════════════════════════
#  Strategy 2: VWAP Reclaim (L/S)
#  Ref: Zarattini & Aziz (2023, SSRN:4631351)
#  Logic: Price crosses VWAP with volume → directional continuation bias
# ══════════════════════════════════════════════════════════════════════════════

def _detect_vwap_reclaim(
    df: pd.DataFrame,
    ind_cfg: dict,
    entry_cfg: dict,
    direction: str,
) -> EntrySignal:
    atr_period = ind_cfg.get("atr_period", 14)
    min_vol_ratio = entry_cfg.get("min_volume_ratio", 1.3)
    null = EntrySignal(EntryStrategy.NONE, direction, 0.0, 0.0, 0.0)

    if len(df) < 30:
        return null

    vwap = _vwap(df, period=20)
    current_atr = atr(df, atr_period)
    price = float(df["close"].iloc[-1])
    prev_close = float(df["close"].iloc[-2])

    if direction == "long":
        # Price crossed above VWAP from below
        crossed = prev_close < vwap and price > vwap
        # Must be within 1 ATR of VWAP (not a gap-up far away)
        near_vwap = abs(price - vwap) < current_atr * 1.0
    else:
        # Price crossed below VWAP from above
        crossed = prev_close > vwap and price < vwap
        near_vwap = abs(price - vwap) < current_atr * 1.0

    if not crossed or not near_vwap:
        return null

    # Volume confirmation
    vol_r = volume_ratio(df, ind_cfg.get("volume_lookback", 20))
    has_volume = vol_r >= min_vol_ratio * 0.9

    # Directional candle body
    last = df.iloc[-1]
    last_close = float(last["close"])
    last_open = float(last["open"])
    if direction == "long":
        bullish_candle = last_close > last_open
    else:
        bullish_candle = last_close < last_open  # bearish candle

    confidence = 0.0
    if crossed:
        confidence += 0.40
    if has_volume:
        confidence += 0.25
    if bullish_candle:
        confidence += 0.20
    if near_vwap and abs(price - vwap) < current_atr * 0.5:
        confidence += 0.15

    if confidence < 0.45:
        return null

    if direction == "long":
        sl = vwap - current_atr * 1.2
    else:
        sl = vwap + current_atr * 1.2

    return EntrySignal(
        strategy=EntryStrategy.VWAP_RECLAIM,
        direction=direction,
        confidence=min(1.0, confidence),
        entry_price=price,
        stop_loss=sl,
        notes=f"vwap={vwap:.4f} vol_r={vol_r:.2f} cross={'up' if direction == 'long' else 'down'}",
    )



# ══════════════════════════════════════════════════════════════════════════════
#  Strategy 3: Liquidity Sweep (L/S)
#  Ref: Smart Money Concepts + arXiv:2602.00776 (order flow imbalance)
#  Logic: Fake breakdown/breakup (stop hunt) then strong reversal candle
# ══════════════════════════════════════════════════════════════════════════════

def _detect_liquidity_sweep(
    df: pd.DataFrame,
    ind_cfg: dict,
    struct_cfg: dict,
    entry_cfg: dict,
    direction: str,
) -> EntrySignal:
    lookback = struct_cfg.get("swing_lookback", 10)
    sweep_pct = struct_cfg.get("liquidity_sweep_pct", 0.3) / 100
    atr_period = ind_cfg.get("atr_period", 14)
    min_vol_ratio = entry_cfg.get("min_volume_ratio", 1.3)
    null = EntrySignal(EntryStrategy.NONE, direction, 0.0, 0.0, 0.0)

    if len(df) < lookback * 2 + 10:
        return null

    current_atr = atr(df, atr_period)
    last = df.iloc[-1]
    last_close = float(last["close"])
    last_open = float(last["open"])
    last_low = float(last["low"])
    last_high = float(last["high"])

    if direction == "long":
        # Sweep below swing low then recover above
        sl_mask = find_swing_lows(df.iloc[:-3], lookback)
        swing_lows = df.iloc[:-3].loc[sl_mask, "low"]
        if swing_lows.empty:
            return null
        target_level = float(swing_lows.iloc[-1])

        recent = df.iloc[-3:]
        swept = any(float(row["low"]) < target_level * (1 - sweep_pct) for _, row in recent.iterrows())
        if not swept:
            return null
        recovered = last_close > target_level
        if not recovered:
            return null

        # Strong bullish reversal candle
        body = last_close - last_open
        candle_range = last_high - last_low
        body_ratio = body / candle_range if candle_range > 0 else 0
        is_reversal_candle = last_close > last_open and body_ratio > 0.5
        sl = last_low - current_atr * 0.5

    else:  # SHORT
        # Sweep above swing high then dump below
        sh_mask = find_swing_highs(df.iloc[:-3], lookback)
        swing_highs = df.iloc[:-3].loc[sh_mask, "high"]
        if swing_highs.empty:
            return null
        target_level = float(swing_highs.iloc[-1])

        recent = df.iloc[-3:]
        swept = any(float(row["high"]) > target_level * (1 + sweep_pct) for _, row in recent.iterrows())
        if not swept:
            return null
        recovered = last_close < target_level
        if not recovered:
            return null

        # Strong bearish reversal candle
        body = last_open - last_close
        candle_range = last_high - last_low
        body_ratio = body / candle_range if candle_range > 0 else 0
        is_reversal_candle = last_close < last_open and body_ratio > 0.5
        sl = last_high + current_atr * 0.5

    # Volume spike (sweep candles need HIGH volume)
    vol_r = volume_ratio(df, ind_cfg.get("volume_lookback", 20))
    has_volume_spike = vol_r >= min_vol_ratio * 1.1

    confidence = 0.0
    if is_reversal_candle:
        confidence += 0.45
    if has_volume_spike:
        confidence += 0.30
    # Deep sweep bonus
    if direction == "long":
        sweep_depth = target_level - last_low
    else:
        sweep_depth = last_high - target_level
    if sweep_depth > current_atr * 0.5:
        confidence += 0.15
    if body_ratio > 0.7:
        confidence += 0.10

    if confidence < 0.45:
        return null

    return EntrySignal(
        strategy=EntryStrategy.LIQUIDITY_SWEEP,
        direction=direction,
        confidence=min(1.0, confidence),
        entry_price=last_close,
        stop_loss=sl,
        notes=f"level={target_level:.4f} vol_r={vol_r:.2f} body_ratio={body_ratio:.2f}",
    )



# ══════════════════════════════════════════════════════════════════════════════
#  Strategy 4: Funding Rate Extreme (Contrarian L/S)
#  Ref: Crypto-specific mean-reversion; crowded positioning unwinds
#  Logic: Extreme funding signals overcrowding → contrarian entry
# ══════════════════════════════════════════════════════════════════════════════

def _detect_funding_extreme(
    df: pd.DataFrame,
    ind_cfg: dict,
    entry_cfg: dict,
    funding_rate: float,
    taker_buy_ratio: float,
) -> EntrySignal:
    atr_period = ind_cfg.get("atr_period", 14)
    null = EntrySignal(EntryStrategy.NONE, "none", 0.0, 0.0, 0.0)

    if len(df) < 30 or funding_rate == 0.0:
        return null

    current_atr = atr(df, atr_period)
    price = float(df["close"].iloc[-1])

    # Extreme thresholds (funding is per 8h, typical range -0.01% to +0.01%)
    extreme_long_threshold = 0.0005    # 0.05% = heavily long-crowded
    extreme_short_threshold = -0.0003  # -0.03% = heavily short-crowded

    if funding_rate >= extreme_long_threshold:
        # Crowded longs → SHORT signal
        direction = "short"
        # Confirm with taker flow: sell pressure increasing
        flow_confirms = taker_buy_ratio < 0.48
        # RSI should be elevated (overbought territory)
        rsi_v = rsi(df, ind_cfg.get("rsi_period", 14))
        rsi_confirms = rsi_v > 65
    elif funding_rate <= extreme_short_threshold:
        # Crowded shorts → LONG signal
        direction = "long"
        flow_confirms = taker_buy_ratio > 0.52
        rsi_v = rsi(df, ind_cfg.get("rsi_period", 14))
        rsi_confirms = rsi_v < 35
    else:
        return null

    confidence = 0.0
    # Funding extremity: more extreme = higher confidence
    funding_severity = abs(funding_rate) / 0.001  # normalized to 0.1%
    confidence += min(0.35, funding_severity * 0.25)
    if flow_confirms:
        confidence += 0.25
    if rsi_confirms:
        confidence += 0.25
    # Recent price stall (consolidation before reversal)
    recent_range = (df["high"].iloc[-5:].max() - df["low"].iloc[-5:].min()) / price
    if recent_range < 0.02:
        confidence += 0.15

    if confidence < 0.45:
        return null

    if direction == "long":
        sl = price - current_atr * 1.8
    else:
        sl = price + current_atr * 1.8

    return EntrySignal(
        strategy=EntryStrategy.FUNDING_EXTREME,
        direction=direction,
        confidence=min(1.0, confidence),
        entry_price=price,
        stop_loss=sl,
        notes=f"funding={funding_rate:.6f} taker_buy={taker_buy_ratio:.3f} rsi={rsi_v:.1f}",
    )



# ══════════════════════════════════════════════════════════════════════════════
#  Strategy 5: OI Divergence (L/S)
#  Ref: Derivatives microstructure — when price rises but OI drops, the move
#  is driven by short-covering (weak) not new buying (strong). Vice versa.
# ══════════════════════════════════════════════════════════════════════════════

def _detect_oi_divergence(
    df: pd.DataFrame,
    ind_cfg: dict,
    entry_cfg: dict,
    oi_change_pct: float,
    direction_hint: str,
) -> EntrySignal:
    atr_period = ind_cfg.get("atr_period", 14)
    null = EntrySignal(EntryStrategy.NONE, "none", 0.0, 0.0, 0.0)

    if len(df) < 30 or oi_change_pct == 0.0:
        return null

    current_atr = atr(df, atr_period)
    price = float(df["close"].iloc[-1])

    # Price momentum over recent 10 bars
    price_change_pct = (price - float(df["close"].iloc[-10])) / float(df["close"].iloc[-10]) * 100

    # Bearish divergence: price UP but OI DOWN → short-covering rally, will fade
    if price_change_pct > 1.0 and oi_change_pct < -1.5:
        direction = "short"
        divergence_strength = abs(price_change_pct) + abs(oi_change_pct)
    # Bullish divergence: price DOWN but OI UP → new longs accumulating, bounce
    elif price_change_pct < -1.0 and oi_change_pct > 1.5:
        direction = "long"
        divergence_strength = abs(price_change_pct) + abs(oi_change_pct)
    # Strong trend confirmation: price UP + OI UP → new money entering direction
    elif price_change_pct > 1.5 and oi_change_pct > 2.0 and direction_hint == "long":
        direction = "long"
        divergence_strength = abs(oi_change_pct)
    elif price_change_pct < -1.5 and oi_change_pct > 2.0 and direction_hint == "short":
        direction = "short"
        divergence_strength = abs(oi_change_pct)
    else:
        return null

    # RSI filter to avoid picking tops/bottoms too aggressively
    rsi_v = rsi(df, ind_cfg.get("rsi_period", 14))
    if direction == "long" and rsi_v > 70:
        return null
    if direction == "short" and rsi_v < 30:
        return null

    confidence = 0.0
    confidence += min(0.40, divergence_strength * 0.08)
    # Volume profile support
    vol_r = volume_ratio(df, ind_cfg.get("volume_lookback", 20))
    if vol_r > 1.2:
        confidence += 0.20
    # Candle pattern confirmation
    last = df.iloc[-1]
    if direction == "long" and float(last["close"]) > float(last["open"]):
        confidence += 0.20
    elif direction == "short" and float(last["close"]) < float(last["open"]):
        confidence += 0.20
    # Stronger divergence bonus
    if divergence_strength > 5.0:
        confidence += 0.15

    if confidence < 0.45:
        return null

    if direction == "long":
        sl = price - current_atr * 1.5
    else:
        sl = price + current_atr * 1.5

    return EntrySignal(
        strategy=EntryStrategy.OI_DIVERGENCE,
        direction=direction,
        confidence=min(1.0, confidence),
        entry_price=price,
        stop_loss=sl,
        notes=f"price_chg={price_change_pct:.2f}% oi_chg={oi_change_pct:.2f}% div={divergence_strength:.1f}",
    )
