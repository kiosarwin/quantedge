"""
Short Entry Strategy Classifier (post-distribution short edges).

Two short setups ported from `kiosarwin/Futures` where they are the bot's
primary statistical edge (cohort: `IMMINENT_DUMP + PHASE_D` and
`IMMINENT_DUMP + LIQ_SWEEP`):

1. PHASE_D — Wyckoff Sign of Weakness
   ────────────────────────────────────
   - A distribution range exists (price ranged for ~30-40 bars near the highs).
   - Price closes below the lower quartile of recent lows (broken support).
   - Confirmed by either a volume spike or an EMA 9/21 bearish cross.
   - Bearish stack: close < open AND close < ema_21 < ema_50.
   - This is the post-distribution breakdown trigger — continuation,
     not a reversal-into-distribution. Entry is the breakdown candle.

2. LIQ_SWEEP — Bearish liquidity sweep above equal highs
   ─────────────────────────────────────────────────────────
   - Two recent equal highs (within 0.3%) form a buy-side liquidity pool.
   - Current bar's wick pierces above that pool (stop hunt).
   - Current bar's close drops back below the pool (failed breakout).
   - Confirmed by long upper wick + bearish close + volume spike.
   - This is the classic stop-hunt fade at the top of a distribution.

Both detectors return a strongly-typed :class:`ShortEntrySignal`. The public
entry point :func:`detect_short_entry` evaluates the enabled detectors and
returns the highest-confidence signal, mirroring the long-side
:mod:`src.analysis.entry_strategies` module.

Reference: see `src/engines/dump_detector.py` in `kiosarwin/Futures` —
specifically `_sow()` (Phase D) and `_liquidity_sweep()` (LIQ_SWEEP_HIGH).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from src.analysis.indicators import atr, ema, volume_ratio

log = logging.getLogger(__name__)


class ShortStrategy(str, Enum):
    PHASE_D = "phase_d"
    LIQ_SWEEP = "liq_sweep"
    NONE = "none"


@dataclass
class ShortEntrySignal:
    strategy: ShortStrategy
    confidence: float           # 0.0 – 1.0
    entry_price: float          # close of the trigger bar
    stop_loss: float            # SL above structure
    notes: str = ""

    @property
    def is_valid(self) -> bool:
        return self.strategy != ShortStrategy.NONE and self.confidence > 0.5

    @property
    def label(self) -> str:
        return self.strategy.value


_NONE = ShortEntrySignal(ShortStrategy.NONE, 0.0, 0.0, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def detect_short_entry(df: pd.DataFrame, cfg: dict) -> ShortEntrySignal:
    """Run all enabled short detectors and return the highest-confidence one.

    Args:
        df:  OHLCV DataFrame on the primary timeframe (≥ 60 bars recommended).
        cfg: Full config dict.

    Returns:
        Best :class:`ShortEntrySignal` (always — check ``is_valid``).
    """
    if df is None or len(df) < 50:
        return _NONE

    strategy_cfg = cfg.get("strategy", {})
    if not bool(strategy_cfg.get("enable_short_setups", True)):
        return _NONE

    min_conf = float(strategy_cfg.get("short_setup_min_confidence", 0.65))

    candidates: list[ShortEntrySignal] = []

    if bool(strategy_cfg.get("short_setup_phase_d_enabled", True)):
        sig = _detect_phase_d(df, cfg)
        if sig.is_valid and sig.confidence >= min_conf:
            candidates.append(sig)

    if bool(strategy_cfg.get("short_setup_liq_sweep_enabled", True)):
        sig = _detect_liq_sweep(df, cfg)
        if sig.is_valid and sig.confidence >= min_conf:
            candidates.append(sig)

    if not candidates:
        return _NONE

    best = max(candidates, key=lambda s: s.confidence)
    log.debug(
        "Short setup: %s confidence=%.2f entry=%.6f sl=%.6f notes=%s",
        best.strategy, best.confidence, best.entry_price, best.stop_loss, best.notes,
    )
    return best


# Convenience aliases used by tests/external callers.
def detect_phase_d_short(df: pd.DataFrame, cfg: dict) -> ShortEntrySignal:
    return _detect_phase_d(df, cfg)


def detect_liq_sweep_short(df: pd.DataFrame, cfg: dict) -> ShortEntrySignal:
    return _detect_liq_sweep(df, cfg)


# ──────────────────────────────────────────────────────────────────────────────
#  Strategy 1: PHASE_D (Wyckoff Sign of Weakness)
# ──────────────────────────────────────────────────────────────────────────────

def _detect_phase_d(df: pd.DataFrame, cfg: dict) -> ShortEntrySignal:
    """Port of `_sow()` (post-distribution support break) from Futures.

    Required:
        broke_support AND (vol_spike OR ema_cross_bear)

    Confidence stack:
        +0.45  broke_support
        +0.20  vol_spike (volume_ratio >= 1.5x)
        +0.15  ema_cross_bear (ema9 cross below ema21 in last 3 bars)
        +0.10  bearish stack (close < ema21 < ema50)
        +0.05  bear_close (close < open)
        +0.05  prior distribution range (last 40-bar range pct < 12%)
    """
    ind_cfg = cfg.get("indicators", {})
    ssg = cfg.get("strategy", {}).get("phase_d", {})

    lookback_support = int(ssg.get("support_lookback", 30))
    range_lookback = int(ssg.get("distribution_range_lookback", 40))
    range_pct_max = float(ssg.get("distribution_range_max_pct", 0.12))
    vol_spike_mult = float(ssg.get("vol_spike_min_ratio", 1.5))
    atr_period = int(ind_cfg.get("atr_period", 14))
    vol_lookback = int(ind_cfg.get("volume_lookback", 20))
    sl_atr_mult = float(ssg.get("sl_atr_multiplier", 1.0))

    if len(df) < max(50, range_lookback + 5):
        return _NONE

    last = df.iloc[-1]
    last_close = float(last["close"])
    last_open = float(last["open"])
    last_high = float(last["high"])

    # 1) Distribution range precondition (advisory; small confidence bonus).
    recent_range = df.tail(range_lookback)
    low_min = float(recent_range["low"].min())
    high_max = float(recent_range["high"].max())
    range_pct = (high_max - low_min) / low_min if low_min > 0 else 1.0
    in_distribution_range = range_pct < range_pct_max

    # 2) Broken support: close below 25th percentile of last N lows.
    support_window = df.tail(lookback_support)
    support = float(support_window["low"].quantile(0.25))
    broke_support = last_close < support * 1.003   # 0.3% wiggle as in Futures
    if not broke_support:
        return _NONE

    # 3) Volume spike confirmation.
    try:
        vol_r = volume_ratio(df, vol_lookback)
    except Exception:
        vol_r = 1.0
    vol_spike = vol_r >= vol_spike_mult

    # 4) EMA cross bearish (ema9 crossed below ema21 within last 3 bars).
    ema9_series = ema(df["close"], 9)
    ema21_series = ema(df["close"], 21)
    ema50_series = ema(df["close"], 50)
    ema_cross_bear = False
    if len(df) >= 5:
        diff = (ema9_series - ema21_series).tail(4).tolist()
        # find a sign flip from non-negative to negative within the last 3 bars
        for i in range(1, len(diff)):
            if diff[i - 1] >= 0 and diff[i] < 0:
                ema_cross_bear = True
                break

    # Hard rule from Futures: detected = broke_support AND (vol_spike OR ema_cross_bear).
    if not (vol_spike or ema_cross_bear):
        return _NONE

    # 5) Bearish stack and bearish close (confidence boosters).
    ema21_now = float(ema21_series.iloc[-1])
    ema50_now = float(ema50_series.iloc[-1])
    bearish_stack = last_close < ema21_now < ema50_now
    bear_close = last_close < last_open

    # ── Confidence ─────────────────────────────────────────────────────
    confidence = 0.45  # base for required broke_support
    reasons: list[str] = ["BREAK_SUPPORT"]

    if vol_spike:
        confidence += 0.20
        reasons.append(f"VOL_SPIKE({vol_r:.2f}x)")
    if ema_cross_bear:
        confidence += 0.15
        reasons.append("EMA9_21_CROSS_BEAR")
    if bearish_stack:
        confidence += 0.10
        reasons.append("BEAR_EMA_STACK")
    if bear_close:
        confidence += 0.05
        reasons.append("BEAR_CLOSE")
    if in_distribution_range:
        confidence += 0.05
        reasons.append(f"RANGE({range_pct * 100:.1f}%)")

    confidence = min(1.0, confidence)

    # Stop-loss: above the high of the trigger bar plus a small ATR buffer.
    try:
        atr_value = atr(df, atr_period)
    except Exception:
        atr_value = max(last_high - float(last["low"]), 1e-9)

    # Use the highest of the last 5 bars as structural SL anchor — protects
    # against late entry on a sharp recovery wick.
    structural_high = float(df["high"].tail(5).max())
    stop_loss = structural_high + atr_value * sl_atr_mult

    return ShortEntrySignal(
        strategy=ShortStrategy.PHASE_D,
        confidence=confidence,
        entry_price=last_close,
        stop_loss=stop_loss,
        notes="|".join(reasons),
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Strategy 2: LIQ_SWEEP (bearish stop hunt above equal highs)
# ──────────────────────────────────────────────────────────────────────────────

def _detect_liq_sweep(df: pd.DataFrame, cfg: dict) -> ShortEntrySignal:
    """Port of `_liquidity_sweep()` LIQ_SWEEP_HIGH from Futures.

    Pattern:
      - Two recent equal highs within ``eq_tolerance_pct`` form buy-side liquidity.
      - Current bar's high pierces above that pool by at least ``pierce_pct``.
      - Current close drops back below the pool (sweep + reject).

    Required:
        equal_highs_present AND wick_pierce AND close_back_inside

    Confidence stack:
        +0.50  required pattern (sweep + reject)
        +0.20  long upper wick (upper_wick_ratio >= 0.50 of range)
        +0.15  bearish close (close < open)
        +0.10  volume spike (volume_ratio >= 1.3x)
        +0.05  pierce depth >= 0.5x ATR (clean sweep)
    """
    ind_cfg = cfg.get("indicators", {})
    ssg = cfg.get("strategy", {}).get("liq_sweep", {})

    lookback = int(ssg.get("lookback_bars", 20))
    eq_tol_pct = float(ssg.get("eq_tolerance_pct", 0.003))   # 0.3 %
    pierce_pct = float(ssg.get("pierce_min_pct", 0.002))     # 0.2 %
    vol_spike_mult = float(ssg.get("vol_spike_min_ratio", 1.3))
    sl_atr_mult = float(ssg.get("sl_atr_multiplier", 0.5))
    atr_period = int(ind_cfg.get("atr_period", 14))
    vol_lookback = int(ind_cfg.get("volume_lookback", 20))

    if len(df) < lookback + 2:
        return _NONE

    last = df.iloc[-1]
    last_open = float(last["open"])
    last_close = float(last["close"])
    last_high = float(last["high"])
    last_low = float(last["low"])

    # Look at the prior `lookback` bars (excluding current) for equal-high pairs.
    prior = df.iloc[-(lookback + 1):-1]
    if len(prior) < 6:
        return _NONE

    highs = prior["high"].to_numpy()

    pool_top = None
    found_pair = False
    # Compare each pair {i, j}; require some separation so adjacent bars don't
    # spuriously match. Mirrors the Futures implementation.
    for i in range(len(highs) - 4):
        for j in range(i + 2, len(highs)):
            hi, hj = float(highs[i]), float(highs[j])
            if hj <= 0:
                continue
            if abs(hi - hj) / hj <= eq_tol_pct:
                top = max(hi, hj)
                if pool_top is None or top > pool_top:
                    pool_top = top
                    found_pair = True

    if not found_pair or pool_top is None:
        return _NONE

    # Wick pierces above pool, close drops back inside.
    wick_pierce = last_high > pool_top * (1.0 + pierce_pct)
    close_back_inside = last_close < pool_top
    if not (wick_pierce and close_back_inside):
        return _NONE

    # ── Confidence stack ───────────────────────────────────────────────
    candle_range = max(last_high - last_low, 1e-9)
    body = abs(last_close - last_open)
    body_top = max(last_open, last_close)
    upper_wick = (last_high - body_top) / candle_range
    long_upper_wick = upper_wick >= 0.50
    bear_close = last_close < last_open

    try:
        vol_r = volume_ratio(df, vol_lookback)
    except Exception:
        vol_r = 1.0
    vol_spike = vol_r >= vol_spike_mult

    try:
        atr_value = atr(df, atr_period)
    except Exception:
        atr_value = candle_range
    pierce_depth = max(last_high - pool_top, 0.0)
    deep_sweep = pierce_depth >= atr_value * 0.5

    confidence = 0.50  # base for the required sweep+reject pattern
    reasons = [f"SWEEP_HIGH({pool_top:.6f})"]

    if long_upper_wick:
        confidence += 0.20
        reasons.append(f"UPPER_WICK({upper_wick:.2f})")
    if bear_close:
        confidence += 0.15
        reasons.append("BEAR_CLOSE")
    if vol_spike:
        confidence += 0.10
        reasons.append(f"VOL_SPIKE({vol_r:.2f}x)")
    if deep_sweep:
        confidence += 0.05
        reasons.append("DEEP_SWEEP")
    # Body-dominant bearish candle (engulfing-style) is an extra check —
    # gives weight to candles where the rejection is strong on close.
    if bear_close and body / candle_range >= 0.4:
        confidence += 0.05
        reasons.append("BODY_DOMINANT")

    confidence = min(1.0, confidence)

    # Stop-loss tight above the sweep wick.
    stop_loss = last_high + atr_value * sl_atr_mult

    return ShortEntrySignal(
        strategy=ShortStrategy.LIQ_SWEEP,
        confidence=confidence,
        entry_price=last_close,
        stop_loss=stop_loss,
        notes="|".join(reasons),
    )
