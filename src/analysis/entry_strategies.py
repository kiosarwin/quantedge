"""
Entry Strategy Classifier

Three distinct entry strategies for Spot Swing Trading:

1. BREAKOUT_RETEST
   ─────────────────
   - Price breaks above a confirmed resistance / swing high
   - Pulls back (retest) toward the broken level
   - Shows bullish rejection (close > open, volume spike)
   - Enter on the rejection candle

2. PULLBACK_EMA
   ──────────────
   - Uptrend confirmed (price > EMA200, EMA50 > EMA200)
   - Price retraces to the EMA50 support zone (±1 ATR)
   - Shows bullish reversal candle at the EMA
   - Enter on bullish reaction

3. LIQUIDITY_SWEEP
   ─────────────────
   - Price makes a fake breakdown below a swing low (stop hunt)
   - Recovers strongly above the swept level in the same or next candle
   - Enter LONG after the strong recovery confirms buyer control
   - This is the highest R:R setup — stops are tight below the sweep wick
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from src.analysis.indicators import ema, atr, volume_ratio
from src.analysis.structure import find_swing_highs, find_swing_lows

log = logging.getLogger(__name__)


class EntryStrategy(str, Enum):
    BREAKOUT_RETEST = "breakout_retest"
    PULLBACK_EMA = "pullback_ema"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    NONE = "none"


@dataclass
class EntrySignal:
    strategy: EntryStrategy
    confidence: float               # 0.0 – 1.0
    entry_price: float              # suggested entry level
    stop_loss: float                # suggested SL below structure
    notes: str = ""

    @property
    def is_valid(self) -> bool:
        return self.strategy != EntryStrategy.NONE and self.confidence > 0.5


def detect_entry(df: pd.DataFrame, cfg: dict) -> EntrySignal:
    """
    Run all three strategy detectors and return the highest-confidence signal.
    Only LONG entries (Spot — no shorting).

    Args:
        df:  OHLCV DataFrame (entry timeframe, e.g. 15m — at least 100 bars)
        cfg: Full config dict

    Returns:
        Best EntrySignal (check signal.is_valid before using)
    """
    ind_cfg = cfg.get("indicators", {})
    entry_cfg = cfg.get("entry", {})
    struct_cfg = cfg.get("structure", {})
    enabled = set(entry_cfg.get("strategies", [
        "breakout_retest", "pullback_ema", "liquidity_sweep"
    ]))

    signals: list[EntrySignal] = []

    if len(df) < 50:
        return EntrySignal(EntryStrategy.NONE, 0.0, 0.0, 0.0)

    if "breakout_retest" in enabled:
        sig = _detect_breakout_retest(df, ind_cfg, struct_cfg, entry_cfg)
        if sig.is_valid:
            signals.append(sig)

    if "pullback_ema" in enabled:
        sig = _detect_pullback_ema(df, ind_cfg, entry_cfg)
        if sig.is_valid:
            signals.append(sig)

    if "liquidity_sweep" in enabled:
        sig = _detect_liquidity_sweep(df, ind_cfg, struct_cfg, entry_cfg)
        if sig.is_valid:
            signals.append(sig)

    if not signals:
        return EntrySignal(EntryStrategy.NONE, 0.0, 0.0, 0.0)

    best = max(signals, key=lambda s: s.confidence)
    log.debug("Entry signal: %s  confidence=%.2f  entry=%.4f  sl=%.4f",
              best.strategy, best.confidence, best.entry_price, best.stop_loss)
    return best


# ──────────────────────────────────────────────────────────────────────────────
#  Strategy 1: Breakout + Retest
# ──────────────────────────────────────────────────────────────────────────────

def _detect_breakout_retest(
    df: pd.DataFrame,
    ind_cfg: dict,
    struct_cfg: dict,
    entry_cfg: dict,
) -> EntrySignal:
    """
    Identify: recent resistance broken → price pulled back near it → bullish rejection.

    Steps:
      1. Find last significant swing high (resistance)
      2. Check if price closed above it at some point in recent bars (breakout)
      3. Check if price has since pulled back to within 1 ATR of that level
      4. Check for bullish rejection candle (close > open, long lower wick)
      5. Confirm volume above average
    """
    lookback = struct_cfg.get("swing_lookback", 10)
    atr_period = ind_cfg.get("atr_period", 14)
    min_vol_ratio = entry_cfg.get("min_volume_ratio", 1.5)

    if len(df) < lookback * 3 + 5:
        return EntrySignal(EntryStrategy.NONE, 0.0, 0.0, 0.0)

    sh_mask = find_swing_highs(df.iloc[:-5], lookback)
    swing_highs = df.iloc[:-5].loc[sh_mask, "high"]
    if swing_highs.empty:
        return EntrySignal(EntryStrategy.NONE, 0.0, 0.0, 0.0)

    resistance = float(swing_highs.iloc[-1])
    current_atr = atr(df, atr_period)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    last_close = float(last["close"])
    last_open = float(last["open"])
    last_low = float(last["low"])

    # Was there a breakout? Any close above resistance in last 20 bars
    recent = df.iloc[-20:]
    breakout_occurred = (recent["close"] > resistance).any()
    if not breakout_occurred:
        return EntrySignal(EntryStrategy.NONE, 0.0, 0.0, 0.0)

    # Is price near the resistance (now support) level — within 1.5 ATR?
    distance = last_close - resistance
    if not (0 <= distance <= current_atr * 1.5):
        return EntrySignal(EntryStrategy.NONE, 0.0, 0.0, 0.0)

    # Bullish rejection candle: close > open, lower wick > body
    body = last_close - last_open
    lower_wick = last_open - last_low if last_close > last_open else last_close - last_low
    is_bullish_candle = last_close > last_open and lower_wick > abs(body) * 0.5

    # Volume confirmation
    vol_r = volume_ratio(df, ind_cfg.get("volume_lookback", 20))
    has_volume = vol_r >= min_vol_ratio

    confidence = 0.0
    if is_bullish_candle:
        confidence += 0.5
    if has_volume:
        confidence += 0.3
    if distance < current_atr * 0.5:   # tight retest = higher confidence
        confidence += 0.2

    if confidence < 0.5:
        return EntrySignal(EntryStrategy.NONE, 0.0, 0.0, 0.0)

    stop_loss = resistance - current_atr * ind_cfg.get("atr_sl_multiplier", 2.0)
    return EntrySignal(
        strategy=EntryStrategy.BREAKOUT_RETEST,
        confidence=min(1.0, confidence),
        entry_price=last_close,
        stop_loss=stop_loss,
        notes=f"resistance={resistance:.4f}  dist={distance:.4f}  vol_ratio={vol_r:.2f}",
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Strategy 2: Pullback to EMA50
# ──────────────────────────────────────────────────────────────────────────────

def _detect_pullback_ema(
    df: pd.DataFrame,
    ind_cfg: dict,
    entry_cfg: dict,
) -> EntrySignal:
    """
    Price is in an uptrend, retraces to EMA50 zone, shows bullish bounce.

    Steps:
      1. Confirm uptrend: price > EMA200 and EMA50 > EMA200
      2. Price has touched or come within 0.5 ATR of EMA50 in recent 5 bars
      3. Current candle closes bullishly above EMA50
      4. Volume at or above average
    """
    ema_fast = ind_cfg.get("ema_fast", 50)
    ema_slow = ind_cfg.get("ema_slow", 200)
    atr_period = ind_cfg.get("atr_period", 14)
    min_vol_ratio = entry_cfg.get("min_volume_ratio", 1.5)

    if len(df) < ema_slow + 10:
        return EntrySignal(EntryStrategy.NONE, 0.0, 0.0, 0.0)

    ema50_series = ema(df["close"], ema_fast)
    ema200_series = ema(df["close"], ema_slow)
    ema50 = float(ema50_series.iloc[-1])
    ema200 = float(ema200_series.iloc[-1])

    last_close = float(df["close"].iloc[-1])
    last_open = float(df["open"].iloc[-1])
    current_atr = atr(df, atr_period)

    # Uptrend structure check
    in_uptrend = (last_close > ema200) and (ema50 > ema200)
    if not in_uptrend:
        return EntrySignal(EntryStrategy.NONE, 0.0, 0.0, 0.0)

    # Price touched EMA50 zone in last 5 bars (low came within 0.5 ATR)
    recent_lows = df["low"].iloc[-5:]
    touched_ema = any(abs(low - ema50) <= current_atr * 0.5 for low in recent_lows)
    if not touched_ema:
        return EntrySignal(EntryStrategy.NONE, 0.0, 0.0, 0.0)

    # Current candle bullish AND closing above EMA50
    is_bullish = last_close > last_open and last_close > ema50

    vol_r = volume_ratio(df, ind_cfg.get("volume_lookback", 20))
    has_volume = vol_r >= min_vol_ratio * 0.8  # slightly looser for pullback entries

    confidence = 0.0
    if is_bullish:
        confidence += 0.5
    if has_volume:
        confidence += 0.25
    # Bonus: strong bounce (close well above EMA50)
    if last_close > ema50 * 1.005:
        confidence += 0.25

    if confidence < 0.5:
        return EntrySignal(EntryStrategy.NONE, 0.0, 0.0, 0.0)

    stop_loss = ema50 - current_atr * ind_cfg.get("atr_sl_multiplier", 2.0)
    return EntrySignal(
        strategy=EntryStrategy.PULLBACK_EMA,
        confidence=min(1.0, confidence),
        entry_price=last_close,
        stop_loss=stop_loss,
        notes=f"ema50={ema50:.4f}  ema200={ema200:.4f}  vol_ratio={vol_r:.2f}",
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Strategy 3: Liquidity Sweep (fake breakdown)
# ──────────────────────────────────────────────────────────────────────────────

def _detect_liquidity_sweep(
    df: pd.DataFrame,
    ind_cfg: dict,
    struct_cfg: dict,
    entry_cfg: dict,
) -> EntrySignal:
    """
    Price dips below a swing low (stop-hunt / liquidity grab) then recovers
    strongly, signalling a reversal and high-probability long entry.

    Steps:
      1. Identify recent swing low
      2. Recent wick broke below it (but close stayed near or above)
      3. Current candle is strongly bullish (engulfing or pinbar)
      4. Volume spike confirms genuine buying
    """
    lookback = struct_cfg.get("swing_lookback", 10)
    sweep_pct = struct_cfg.get("liquidity_sweep_pct", 0.3) / 100
    atr_period = ind_cfg.get("atr_period", 14)
    min_vol_ratio = entry_cfg.get("min_volume_ratio", 1.5)

    if len(df) < lookback * 2 + 10:
        return EntrySignal(EntryStrategy.NONE, 0.0, 0.0, 0.0)

    # Look for swing lows in data excluding last 3 bars (those are the sweep candles)
    sl_mask = find_swing_lows(df.iloc[:-3], lookback)
    swing_lows = df.iloc[:-3].loc[sl_mask, "low"]
    if swing_lows.empty:
        return EntrySignal(EntryStrategy.NONE, 0.0, 0.0, 0.0)

    last_swing_low = float(swing_lows.iloc[-1])
    current_atr = atr(df, atr_period)

    # Check last 3 bars: did any wick pierce below the swing low?
    recent = df.iloc[-3:]
    sweep_detected = any(
        float(row["low"]) < last_swing_low * (1 - sweep_pct)
        for _, row in recent.iterrows()
    )
    if not sweep_detected:
        return EntrySignal(EntryStrategy.NONE, 0.0, 0.0, 0.0)

    # Recovery: last bar closes above the sweep level
    last = df.iloc[-1]
    last_close = float(last["close"])
    last_open = float(last["open"])
    last_low = float(last["low"])

    recovered = last_close > last_swing_low
    if not recovered:
        return EntrySignal(EntryStrategy.NONE, 0.0, 0.0, 0.0)

    # Strong bullish candle: close > open, significant body
    body_size = abs(last_close - last_open)
    candle_range = float(last["high"]) - last_low
    body_ratio = body_size / candle_range if candle_range > 0 else 0.0
    is_strong_bull = last_close > last_open and body_ratio > 0.5

    # Volume spike
    vol_r = volume_ratio(df, ind_cfg.get("volume_lookback", 20))
    has_volume_spike = vol_r >= min_vol_ratio * 1.2  # need stronger volume for sweep

    confidence = 0.0
    if is_strong_bull:
        confidence += 0.5
    if has_volume_spike:
        confidence += 0.35
    # Bonus: sweep was clean (wick went far below, recovered quickly)
    sweep_depth = last_swing_low - last_low
    if sweep_depth > current_atr * 0.5:
        confidence += 0.15

    if confidence < 0.5:
        return EntrySignal(EntryStrategy.NONE, 0.0, 0.0, 0.0)

    # SL goes just below the sweep wick — very tight stop
    stop_loss = last_low - current_atr * 0.5
    return EntrySignal(
        strategy=EntryStrategy.LIQUIDITY_SWEEP,
        confidence=min(1.0, confidence),
        entry_price=last_close,
        stop_loss=stop_loss,
        notes=f"swing_low={last_swing_low:.4f}  wick_low={last_low:.4f}  vol_ratio={vol_r:.2f}",
    )
