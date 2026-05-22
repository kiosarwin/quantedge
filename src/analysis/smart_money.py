"""
Smart Money Detection Engine.

Detects institutional accumulation, distribution, and liquidity events by
analyzing OI dynamics, funding extremes, volume patterns, and price structure.

Phases:
  ACCUMULATION   — OI rising + price flat/up + volume building → bullish bias.
  DISTRIBUTION   — OI rising + price stagnating + extreme funding → bearish bias.
  LIQUIDITY_SWEEP — Stop hunt detected; reversal setup in progress.
  TRENDING       — OI + price both confirming directional move.
  NEUTRAL        — No clear institutional signal.
  CHAOS          — Conflicting signals; skip.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from src.analysis.structure import detect_liquidity_sweep, detect_bos
from src.analysis.indicators import atr, volume_ratio

log = logging.getLogger(__name__)


class SmartMoneyPhase(str, Enum):
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    TRENDING = "trending"
    NEUTRAL = "neutral"
    CHAOS = "chaos"

    @property
    def has_directional_bias(self) -> bool:
        return self in (
            SmartMoneyPhase.ACCUMULATION,
            SmartMoneyPhase.DISTRIBUTION,
            SmartMoneyPhase.LIQUIDITY_SWEEP,
            SmartMoneyPhase.TRENDING,
        )


@dataclass
class SmartMoneySignal:
    phase: SmartMoneyPhase
    score: float            # 0–100 (conviction level)
    direction_bias: str     # 'long' | 'short' | 'neutral'
    oi_narrative: str       # human-readable OI interpretation
    funding_narrative: str
    volume_narrative: str
    sweep_narrative: str
    reasoning: str          # concise summary for trade card

    def aligns_with(self, direction: str) -> bool:
        """True if smart money bias agrees with proposed trade direction."""
        return self.direction_bias == direction or self.direction_bias == "neutral"


def detect_smart_money(
    df: pd.DataFrame,
    oi_change_pct: float,
    funding_rate: float,
    ls_ratio: float,
    taker_buy_ratio: float,
    cfg: dict,
) -> SmartMoneySignal:
    """
    Analyze market data for institutional footprints.

    Args:
        df:               OHLCV DataFrame (1H preferred, ≥ 50 bars).
        oi_change_pct:    OI change % since last snapshot.
        funding_rate:     Current funding rate (fraction).
        ls_ratio:         Global long/short account ratio.
        taker_buy_ratio:  Taker buy volume / total taker volume (0–1).
        cfg:              Full config dict.
    """
    sm_cfg = cfg.get("smart_money", {})
    ind_cfg = cfg.get("indicators", {})
    struct_cfg = cfg.get("structure", {})

    # FIXED: Use config values with REALISTIC defaults for perpetual futures.
    # OI change per 1h snapshot on Binance is typically 0.05-0.5%.
    # A 0.15% rise per hour IS accumulation. A 2% threshold would never fire.
    oi_acc_thresh = float(sm_cfg.get("oi_accumulation_threshold_pct", 0.15))
    oi_dist_thresh = float(sm_cfg.get("oi_distribution_threshold_pct", -0.10))
    price_stag_pct = sm_cfg.get("price_stagnation_pct", 0.5) / 100
    funding_extreme = sm_cfg.get("funding_extreme_threshold", 0.05)
    vol_lookback = ind_cfg.get("volume_lookback", 20)
    sweep_pct = struct_cfg.get("liquidity_sweep_pct", 0.3) / 100

    if len(df) < 30:
        return _neutral("Insufficient data")

    price = float(df["close"].iloc[-1])
    price_5ago = float(df["close"].iloc[-6]) if len(df) >= 6 else price
    price_change_pct = abs(price - price_5ago) / price_5ago if price_5ago else 0.0
    price_is_flat = price_change_pct < price_stag_pct
    price_rising = float(df["close"].iloc[-1]) > float(df["close"].iloc[-10]) if len(df) >= 10 else False

    # ── Volume narrative ──────────────────────────────────────────────
    vol_r = volume_ratio(df, vol_lookback)
    if vol_r > 2.5:
        vol_narrative = f"Volume spike {vol_r:.1f}x avg (institutional activity)"
    elif vol_r > 1.5:
        vol_narrative = f"Elevated volume {vol_r:.1f}x avg"
    else:
        vol_narrative = f"Normal volume {vol_r:.1f}x avg"

    # ── OI narrative ──────────────────────────────────────────────────
    if oi_change_pct >= oi_acc_thresh:
        oi_narrative = f"OI rising +{oi_change_pct:.2f}% (new contracts entering)"
    elif oi_change_pct <= oi_dist_thresh:
        oi_narrative = f"OI falling {oi_change_pct:.2f}% (positions closing)"
    else:
        oi_narrative = f"OI stable {oi_change_pct:+.2f}%"

    # ── Funding narrative ─────────────────────────────────────────────
    if funding_rate > funding_extreme:
        funding_narrative = f"Extreme positive funding {funding_rate:.4f} (crowded longs → short bias)"
    elif funding_rate < -funding_extreme:
        funding_narrative = f"Extreme negative funding {funding_rate:.4f} (crowded shorts → long bias)"
    elif funding_rate > 0.01:
        funding_narrative = f"Moderately positive funding {funding_rate:.4f}"
    elif funding_rate < -0.005:
        funding_narrative = f"Slightly negative funding {funding_rate:.4f}"
    else:
        funding_narrative = f"Neutral funding {funding_rate:.4f}"

    # ── Liquidity sweep detection ─────────────────────────────────────
    sweep = detect_liquidity_sweep(df, struct_cfg.get("swing_lookback", 10), sweep_pct)
    bos = detect_bos(df, struct_cfg.get("swing_lookback", 10), struct_cfg.get("bos_confirmation_candles", 2))

    if sweep == "swept_lows":
        sweep_narrative = "Stop hunt below swing lows detected — bullish reversal setup"
    elif sweep == "swept_highs":
        sweep_narrative = "Stop hunt above swing highs detected — bearish reversal setup"
    else:
        sweep_narrative = "No liquidity sweep detected"

    # ── L/S ratio signal ─────────────────────────────────────────────
    ls_long_extreme = sm_cfg.get("ls_long_extreme", 0.70)
    ls_short_extreme = sm_cfg.get("ls_short_extreme", 0.30)
    ls_crowded_long = ls_ratio > ls_long_extreme    # contrarian bearish
    ls_crowded_short = ls_ratio < ls_short_extreme  # contrarian bullish

    # ── Phase classification ──────────────────────────────────────────
    score = 50.0
    direction_bias = "neutral"

    # CHAOS: conflicting extreme signals
    chaos_signals = sum([
        funding_rate > funding_extreme and price_rising,
        ls_crowded_long and oi_change_pct >= oi_acc_thresh and not price_rising,
        vol_r > 4.0,
    ])
    if chaos_signals >= 2:
        return SmartMoneySignal(
            phase=SmartMoneyPhase.CHAOS,
            score=20.0,
            direction_bias="neutral",
            oi_narrative=oi_narrative,
            funding_narrative=funding_narrative,
            volume_narrative=vol_narrative,
            sweep_narrative=sweep_narrative,
            reasoning="Conflicting institutional signals — avoid",
        )

    # LIQUIDITY_SWEEP: highest priority — fresh reversal opportunity
    if sweep != "none":
        if sweep == "swept_lows":
            direction_bias = "long"
            score = 80.0 + (15.0 if bos == "bullish_bos" else 0.0)
        else:  # swept_highs
            direction_bias = "short"
            score = 80.0 + (15.0 if bos == "bearish_bos" else 0.0)
        return SmartMoneySignal(
            phase=SmartMoneyPhase.LIQUIDITY_SWEEP,
            score=min(100.0, score),
            direction_bias=direction_bias,
            oi_narrative=oi_narrative,
            funding_narrative=funding_narrative,
            volume_narrative=vol_narrative,
            sweep_narrative=sweep_narrative,
            reasoning=f"Liquidity sweep {sweep} + {'BOS confirms' if bos != 'none' else 'watching for BOS'}",
        )

    # DISTRIBUTION: OI building + price stagnant/down + extreme positive funding
    # UPGRADED: Also detect distribution when OI is FALLING (positions closing = unwinding)
    # or when funding is extreme even without OI threshold hit.
    is_distribution = (
        (oi_change_pct >= oi_acc_thresh and price_is_flat and
         (funding_rate > funding_extreme or ls_crowded_long))
        or (funding_rate > funding_extreme * 1.5 and ls_crowded_long)  # NEW: extreme crowding alone
        or (oi_change_pct <= oi_dist_thresh and funding_rate > funding_extreme)  # NEW: OI drop + high funding
    )
    if is_distribution:
        score = 68.0 + min(22.0, abs(funding_rate) / funding_extreme * 12)
        if taker_buy_ratio < 0.45:
            score += 10.0  # aggressive sellers dominating
        if ls_crowded_long and funding_rate > funding_extreme:
            score += 5.0   # double confirmation
        return SmartMoneySignal(
            phase=SmartMoneyPhase.DISTRIBUTION,
            score=min(100.0, score),
            direction_bias="short",
            oi_narrative=oi_narrative,
            funding_narrative=funding_narrative,
            volume_narrative=vol_narrative,
            sweep_narrative=sweep_narrative,
            reasoning=f"Distribution: {oi_narrative}; {funding_narrative}",
        )

    # ACCUMULATION: OI rising + price flat/slightly rising + neutral/low funding
    # UPGRADED: Also detect when taker buy ratio is very high (buyers absorbing)
    is_accumulation = (
        (oi_change_pct >= oi_acc_thresh and price_is_flat and
         funding_rate < funding_extreme and not ls_crowded_long)
        or (taker_buy_ratio > 0.60 and oi_change_pct > 0 and price_is_flat
            and funding_rate < funding_extreme * 0.5)  # NEW: strong buying absorption
    )
    if is_accumulation:
        score = 63.0 + min(25.0, oi_change_pct / max(oi_acc_thresh, 0.01) * 12)
        if taker_buy_ratio > 0.55:
            score += 10.0  # aggressive buyers absorbing supply
        if vol_r > 1.5:
            score += 5.0   # elevated volume confirms accumulation
        return SmartMoneySignal(
            phase=SmartMoneyPhase.ACCUMULATION,
            score=min(100.0, score),
            direction_bias="long",
            oi_narrative=oi_narrative,
            funding_narrative=funding_narrative,
            volume_narrative=vol_narrative,
            sweep_narrative=sweep_narrative,
            reasoning=f"Accumulation: {oi_narrative}; low funding, compression",
        )

    # TRENDING: OI + price both rising/falling (trend continuation)
    oi_rising_strongly = oi_change_pct >= oi_acc_thresh
    if oi_rising_strongly and not price_is_flat:
        score = 72.0 + min(18.0, vol_r * 6)  # Upgraded: higher base + vol bonus
        direction_bias = "long" if price_rising else "short"
        return SmartMoneySignal(
            phase=SmartMoneyPhase.TRENDING,
            score=min(100.0, score),
            direction_bias=direction_bias,
            oi_narrative=oi_narrative,
            funding_narrative=funding_narrative,
            volume_narrative=vol_narrative,
            sweep_narrative=sweep_narrative,
            reasoning=f"Trending: OI confirms {'uptrend' if price_rising else 'downtrend'}; {vol_narrative}",
        )

    # NEUTRAL fallback
    score = 40.0 + (10.0 if ls_crowded_short else 0.0) + (10.0 if taker_buy_ratio > 0.6 else 0.0)
    if ls_crowded_short:
        direction_bias = "long"
    elif ls_crowded_long:
        direction_bias = "short"

    return SmartMoneySignal(
        phase=SmartMoneyPhase.NEUTRAL,
        score=score,
        direction_bias=direction_bias,
        oi_narrative=oi_narrative,
        funding_narrative=funding_narrative,
        volume_narrative=vol_narrative,
        sweep_narrative=sweep_narrative,
        reasoning="No dominant smart money signal detected",
    )


def _neutral(reason: str) -> SmartMoneySignal:
    return SmartMoneySignal(
        phase=SmartMoneyPhase.NEUTRAL,
        score=0.0,
        direction_bias="neutral",
        oi_narrative="N/A",
        funding_narrative="N/A",
        volume_narrative="N/A",
        sweep_narrative="N/A",
        reasoning=reason,
    )
