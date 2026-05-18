"""
Feature Engineering Layer.

Builds a unified feature vector F(t) used by the EV model and scoring system.

Features:
  volatility_regime      — Current vol vs. historical; 0=calm, 100=extreme.
  momentum_strength      — −100 (strong bear) to +100 (strong bull).
  oi_change_rate         — OI change % (raw, signed).
  funding_deviation      — Funding - rolling mean, normalized.
  volume_anomaly_score   — 0=flat, 100=3× spike.
  order_flow_imbalance   — −100 (ask-heavy) to +100 (bid-heavy).
  liquidation_pressure   — 0=none, 100=extreme proximity to key liq zones.
  market_structure       — 'bullish_bos' | 'bearish_bos' | 'choch_bull' | 'choch_bear' | 'none'.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analysis.indicators import atr, rsi, ema, volume_ratio, buy_volume_ratio
from src.analysis.structure import detect_bos, detect_liquidity_sweep, get_recent_swing_levels
from src.analysis.order_book import bid_ask_imbalance

log = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    symbol: str
    volatility_regime: float        # 0–100 (higher = more volatile)
    momentum_strength: float        # −100 to +100
    oi_change_rate: float           # % (raw signed)
    funding_deviation: float        # normalized −1 to +1
    volume_anomaly_score: float     # 0–100
    order_flow_imbalance: float     # −100 to +100
    liquidation_pressure: float     # 0–100
    market_structure: str           # 'bullish_bos' | 'bearish_bos' | 'none'

    def directional_alignment(self, direction: str) -> float:
        """
        Returns a 0–1 alignment score: how much this feature vector supports
        the proposed trade direction.
        """
        if direction == "long":
            m = (self.momentum_strength + 100) / 200       # 0–1 (higher = more bullish)
            ob = (self.order_flow_imbalance + 100) / 200
            oi = min(1.0, max(0.0, self.oi_change_rate / 5 + 0.5))
            struct = 1.0 if self.market_structure in ("bullish_bos",) else (
                0.5 if self.market_structure == "none" else 0.2
            )
        else:  # short
            m = (100 - self.momentum_strength) / 200
            ob = (100 - self.order_flow_imbalance) / 200
            oi = min(1.0, max(0.0, -self.oi_change_rate / 5 + 0.5))
            struct = 1.0 if self.market_structure in ("bearish_bos",) else (
                0.5 if self.market_structure == "none" else 0.2
            )
        return round((m * 0.3 + ob * 0.2 + oi * 0.3 + struct * 0.2), 4)


def build_feature_vector(
    symbol: str,
    df_primary: pd.DataFrame,
    df_higher: pd.DataFrame,
    order_book: dict,
    oi_change_pct: float,
    funding_rate: float,
    taker_buy_ratio: float,
    cfg: dict,
) -> FeatureVector:
    """
    Compute all features for a single symbol at the current tick.

    Args:
        symbol:          Trading pair.
        df_primary:      Primary timeframe OHLCV (1H).
        df_higher:       Higher timeframe OHLCV (4H).
        order_book:      Current order book dict.
        oi_change_pct:   OI change % vs. previous snapshot.
        funding_rate:    Current funding rate (fraction).
        taker_buy_ratio: Taker buy vol / total taker vol.
        cfg:             Full config dict.
    """
    ind = cfg.get("indicators", {})
    struct_cfg = cfg.get("structure", {})
    atr_period = ind.get("atr_period", 14)
    vol_lookback = ind.get("volume_lookback", 20)
    ema_fast = ind.get("ema_fast", 21)
    ema_slow = ind.get("ema_slow", 55)

    # ── Volatility regime ─────────────────────────────────────────────
    if len(df_primary) >= atr_period * 3:
        current_atr = atr(df_primary, atr_period)
        avg_atr = float(df_primary["high"].sub(df_primary["low"]).rolling(atr_period * 3).mean().iloc[-1])
        vol_regime = min(100.0, (current_atr / avg_atr - 0.5) * 100) if avg_atr > 0 else 50.0
    else:
        vol_regime = 50.0

    # ── Momentum strength ─────────────────────────────────────────────
    if len(df_primary) >= max(ema_fast, ema_slow, 14):
        rsi_v = rsi(df_primary, ind.get("rsi_period", 14))
        ema_f = float(ema(df_primary["close"], ema_fast).iloc[-1])
        ema_s = float(ema(df_primary["close"], ema_slow).iloc[-1])
        price = float(df_primary["close"].iloc[-1])
        # RSI component: −50 to +50
        rsi_component = (rsi_v - 50) * 1.0
        # EMA component: −50 to +50 based on fast vs slow alignment
        ema_component = 50.0 * (1 if ema_f > ema_s else -1) * min(1.0, abs(ema_f - ema_s) / price * 50)
        momentum = float(np.clip(rsi_component + ema_component, -100, 100))
    else:
        momentum = 0.0

    # ── Volume anomaly ────────────────────────────────────────────────
    if len(df_primary) >= vol_lookback + 1:
        vol_r = volume_ratio(df_primary, vol_lookback)
        vol_anomaly = min(100.0, (vol_r - 1.0) * 40)  # 1.0x=0, 3.5x=100
    else:
        vol_anomaly = 0.0

    # ── Order flow imbalance ──────────────────────────────────────────
    ob_imbalance = bid_ask_imbalance(order_book)
    # bid/ask ratio: 1.0 = neutral → map to −100..+100
    # ratio 2.0 → +50, ratio 0.5 → −50
    if ob_imbalance > 0:
        ob_score = min(100.0, (ob_imbalance - 1.0) * 100)
    else:
        ob_score = 0.0
    # Blend with taker buy ratio
    taker_component = (taker_buy_ratio - 0.5) * 200  # −100..+100
    order_flow_imbalance = float(np.clip(ob_score * 0.5 + taker_component * 0.5, -100, 100))

    # ── Liquidation pressure ─────────────────────────────────────────
    # Approximation: distance to nearest significant swing level as % of ATR
    liq_pressure = _liquidation_pressure_index(df_primary, cfg)

    # ── Market structure ─────────────────────────────────────────────
    lookback = struct_cfg.get("swing_lookback", 10)
    confirm = struct_cfg.get("bos_confirmation_candles", 2)
    sweep_pct = struct_cfg.get("liquidity_sweep_pct", 0.3) / 100
    bos = detect_bos(df_primary, lookback, confirm)
    sweep = detect_liquidity_sweep(df_primary, lookback, sweep_pct)

    if bos == "bullish_bos":
        market_structure = "bullish_bos"
    elif bos == "bearish_bos":
        market_structure = "bearish_bos"
    elif sweep == "swept_lows":
        market_structure = "choch_bull"  # potential CHoCH bullish
    elif sweep == "swept_highs":
        market_structure = "choch_bear"
    else:
        market_structure = "none"

    # ── Funding deviation ─────────────────────────────────────────────
    # Normalize against extreme threshold so it's dimensionless
    extreme = cfg.get("smart_money", {}).get("funding_extreme_threshold", 0.05)
    funding_dev = float(np.clip(funding_rate / extreme, -1.0, 1.0))

    return FeatureVector(
        symbol=symbol,
        volatility_regime=round(vol_regime, 2),
        momentum_strength=round(momentum, 2),
        oi_change_rate=round(oi_change_pct, 4),
        funding_deviation=round(funding_dev, 4),
        volume_anomaly_score=round(vol_anomaly, 2),
        order_flow_imbalance=round(order_flow_imbalance, 2),
        liquidation_pressure=round(liq_pressure, 2),
        market_structure=market_structure,
    )


def _liquidation_pressure_index(df: pd.DataFrame, cfg: dict) -> float:
    """
    Estimates proximity of price to potential forced-liquidation clusters.
    High pressure = price is near where leveraged positions would be wiped.
    Uses swing levels + ATR to compute normalized proximity score.
    """
    if len(df) < 30:
        return 0.0

    ind_cfg = cfg.get("indicators", {})
    struct_cfg = cfg.get("structure", {})
    atr_period = ind_cfg.get("atr_period", 14)
    lookback = struct_cfg.get("swing_lookback", 10)

    current_atr = atr(df, atr_period)
    if current_atr == 0:
        return 0.0

    price = float(df["close"].iloc[-1])
    levels = get_recent_swing_levels(df, lookback, count=5)

    all_levels = levels["swing_highs"] + levels["swing_lows"]
    if not all_levels:
        return 0.0

    # Pressure increases as price approaches swing levels within 2 ATR
    distances = [abs(price - lvl) / current_atr for lvl in all_levels]
    min_dist = min(distances)

    # Min dist < 0.5 ATR = very high pressure (100), > 3 ATR = low (0)
    pressure = max(0.0, 100.0 - min_dist / 3.0 * 100.0)
    return float(min(100.0, pressure))
