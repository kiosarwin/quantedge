"""
Spot Scorer — multi-factor scoring engine for Binance Spot swing trading.

Score breakdown (total = 100):
┌──────────────────────────┬────────┐
│ Component                │ Weight │
├──────────────────────────┼────────┤
│ Trend Strength           │  25    │
│ Volume Expansion         │  20    │
│ Structure Quality        │  20    │
│ Momentum Alignment       │  15    │
│ BTC Correlation          │  10    │
│ Volatility Condition     │  10    │
└──────────────────────────┴────────┘

Only trade if total score ≥ 75.
Regime acts as a gate (SIDEWAYS → 0 score → no trade).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analysis.indicators import (
    ema,
    atr,
    rsi,
    volume_ratio,
    buy_volume_ratio,
    trend_strength_score,
    volatility_score,
)
from src.analysis.structure import structure_quality_score, trade_direction_from_structure
from src.analysis.regime import classify_regime, Regime, regime_score_modifier
from src.data.spot_market_data import SpotSnapshot

log = logging.getLogger(__name__)


@dataclass
class SpotSignalBreakdown:
    symbol: str
    direction: str              # 'long' | 'none' (spot = long only)
    total_score: float
    regime: Regime
    # Component scores (each 0–100)
    trend_strength: float
    volume_expansion: float
    structure_quality: float
    momentum_alignment: float
    btc_correlation: float
    volatility_condition: float
    weights_used: dict
    # Entry metadata
    entry_strategy: str = "none"
    entry_confidence: float = 0.0

    @property
    def is_tradeable(self) -> bool:
        return (
            self.direction == "long"
            and self.total_score >= 75
            and self.regime == Regime.TRENDING
        )


class SpotScorer:
    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._weights: dict[str, float] = dict(cfg["scoring"]["weights"])

    def update_weights(self, weights: dict[str, float]) -> None:
        self._weights = {k: float(v) for k, v in weights.items()}
        log.debug("SpotScorer weights updated: %s", self._weights)

    def score(
        self,
        snapshot: SpotSnapshot,
        btc_df: pd.DataFrame | None = None,
    ) -> SpotSignalBreakdown | None:
        """
        Score a single symbol snapshot.

        Args:
            snapshot: SpotSnapshot with OHLCV data
            btc_df:   BTC 4H candles for correlation scoring (optional)

        Returns:
            SpotSignalBreakdown or None if insufficient data
        """
        tf_cfg = self._cfg["timeframes"]
        primary_tf = tf_cfg["primary"]        # 4H
        secondary_tf = tf_cfg.get("secondary", primary_tf)  # 1H

        df_primary = snapshot.candles_for(primary_tf)
        df_secondary = snapshot.candles_for(secondary_tf)

        if df_primary.empty or len(df_primary) < 50:
            log.debug("Not enough 4H candles for %s", snapshot.symbol)
            return None

        # ── Regime gate ───────────────────────────────────────────────
        regime = classify_regime(df_primary, self._cfg)
        regime_mult = regime_score_modifier(regime)
        if regime_mult == 0.0:
            log.debug("[%s] Regime=%s → skip", snapshot.symbol, regime)
            return SpotSignalBreakdown(
                symbol=snapshot.symbol,
                direction="none",
                total_score=0.0,
                regime=regime,
                trend_strength=0.0,
                volume_expansion=0.0,
                structure_quality=0.0,
                momentum_alignment=0.0,
                btc_correlation=0.0,
                volatility_condition=0.0,
                weights_used=dict(self._weights),
            )

        # ── Direction (Spot = LONG only) ──────────────────────────────
        direction = trade_direction_from_structure(df_primary, self._cfg)
        if direction == "none" and not df_secondary.empty and len(df_secondary) >= 50:
            direction = trade_direction_from_structure(df_secondary, self._cfg)

        # For spot we only score LONG setups
        if direction != "long":
            direction = self._check_ema_trend(df_primary)
        if direction != "long":
            return None

        # ── Individual signal scores ──────────────────────────────────
        ts = self._trend_strength(df_primary)
        ve = self._volume_expansion(df_primary)
        sq = self._structure_quality(df_primary)
        ma = self._momentum_alignment(df_primary)
        bc = self._btc_correlation(df_primary, btc_df)
        vc = self._volatility_condition(df_primary)

        # ── Weighted total ────────────────────────────────────────────
        w = self._weights
        total_w = sum(w.values()) or 100.0
        raw_score = (
            ts * w.get("trend_strength", 25) +
            ve * w.get("volume_expansion", 20) +
            sq * w.get("structure_quality", 20) +
            ma * w.get("momentum_alignment", 15) +
            bc * w.get("btc_correlation", 10) +
            vc * w.get("volatility_condition", 10)
        ) / total_w

        # Apply regime multiplier
        final_score = round(raw_score * regime_mult, 2)

        return SpotSignalBreakdown(
            symbol=snapshot.symbol,
            direction=direction,
            total_score=final_score,
            regime=regime,
            trend_strength=ts,
            volume_expansion=ve,
            structure_quality=sq,
            momentum_alignment=ma,
            btc_correlation=bc,
            volatility_condition=vc,
            weights_used=dict(w),
        )

    def score_many(
        self,
        snapshots: dict[str, SpotSnapshot],
        btc_df: pd.DataFrame | None = None,
    ) -> list[SpotSignalBreakdown]:
        results = []
        for snap in snapshots.values():
            bd = self.score(snap, btc_df)
            if bd is not None:
                results.append(bd)
        results.sort(key=lambda x: x.total_score, reverse=True)
        return results

    # ------------------------------------------------------------------ #
    #  Component scorers (each returns 0.0 – 100.0)                      #
    # ------------------------------------------------------------------ #

    def _trend_strength(self, df: pd.DataFrame) -> float:
        """
        EMA alignment + RSI momentum → 0-100.
        Full score when: price > EMA200, EMA50 > EMA200, RSI 55-70.
        """
        try:
            return trend_strength_score(df, self._cfg)
        except Exception:
            return 0.0

    def _volume_expansion(self, df: pd.DataFrame) -> float:
        """
        Volume spike + buy-side pressure → 0-100.
        Score rises when current volume is 2× average AND buying dominates.
        """
        try:
            ind = self._cfg["indicators"]
            lookback = ind.get("volume_lookback", 20)
            spike_mult = ind.get("volume_spike_multiplier", 2.0)

            vol_r = volume_ratio(df, lookback)
            buy_r = buy_volume_ratio(df, lookback)

            # Volume component: how much above average
            vol_score = min(100.0, (vol_r / spike_mult) * 60)
            # Directional component: buy pressure
            buy_score = buy_r * 100
            return round(vol_score * 0.5 + buy_score * 0.5, 2)
        except Exception:
            return 0.0

    def _structure_quality(self, df: pd.DataFrame) -> float:
        """BOS + liquidity sweep alignment → 0-100."""
        try:
            return structure_quality_score(df, self._cfg)
        except Exception:
            return 0.0

    def _momentum_alignment(self, df: pd.DataFrame) -> float:
        """
        RSI position + ADX strength → 0-100.
        Ideal: RSI 50-65 (momentum without overbought), ADX > 25.
        """
        try:
            from src.analysis.regime import adx
            ind = self._cfg["indicators"]
            rsi_val = rsi(df, ind.get("rsi_period", 14))
            adx_val = adx(df, ind.get("adx_period", 14))

            # RSI score: 50-65 is ideal long zone
            if 50 <= rsi_val <= 65:
                rsi_score = 100.0
            elif 40 <= rsi_val < 50:
                rsi_score = (rsi_val - 40) / 10 * 70   # partial
            elif 65 < rsi_val <= 75:
                rsi_score = 100 - (rsi_val - 65) * 7   # getting overbought
            elif rsi_val > 75:
                rsi_score = max(0.0, 30 - (rsi_val - 75) * 3)  # overbought
            else:
                rsi_score = 0.0

            # ADX score: >25 is trending
            adx_score = min(100.0, max(0.0, (adx_val - 15) / 35 * 100))

            return round(rsi_score * 0.5 + adx_score * 0.5, 2)
        except Exception:
            return 50.0

    def _btc_correlation(
        self, df: pd.DataFrame, btc_df: pd.DataFrame | None
    ) -> float:
        """
        BTC trend alignment → 0-100.
        - BTC bullish (price > EMA50) and positively correlated → high score
        - BTC bearish → low score
        - If BTC data unavailable → neutral (50)
        """
        if btc_df is None or btc_df.empty or len(btc_df) < 20:
            return 50.0
        try:
            ind = self._cfg["indicators"]
            ema_period = self._cfg.get("btc", {}).get("ema_period", 50)
            lookback = ind.get("btc_correlation_lookback", 48)

            btc_close = btc_df["close"].iloc[-lookback:]
            btc_ema = float(ema(btc_df["close"], ema_period).iloc[-1])
            btc_last = float(btc_df["close"].iloc[-1])

            # Is BTC in an uptrend?
            btc_bullish = btc_last > btc_ema

            # Compute return correlation between alt and BTC
            alt_returns = df["close"].pct_change().iloc[-lookback:].values
            btc_returns = btc_close.pct_change().iloc[-lookback:].values

            n = min(len(alt_returns), len(btc_returns))
            if n < 10:
                return 50.0

            corr = float(np.corrcoef(alt_returns[-n:], btc_returns[-n:])[0, 1])
            if np.isnan(corr):
                corr = 0.0

            # Score: BTC bullish + positive correlation = best
            btc_trend_score = 80.0 if btc_bullish else 20.0
            corr_score = max(0.0, corr * 100)   # -100 to +100 → 0 to 100

            return round(btc_trend_score * 0.6 + corr_score * 0.4, 2)
        except Exception:
            return 50.0

    def _volatility_condition(self, df: pd.DataFrame) -> float:
        """
        ATR % of price in ideal range (0.5%–3%) → 0-100.
        Too low = dead market. Too high = extreme volatility.
        """
        try:
            return volatility_score(df, self._cfg)
        except Exception:
            return 50.0

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _check_ema_trend(self, df: pd.DataFrame) -> str:
        """Simple EMA-based trend check as fallback direction filter."""
        try:
            ind = self._cfg["indicators"]
            ema_fast = ind.get("ema_fast", 50)
            ema_slow = ind.get("ema_slow", 200)
            if len(df) < ema_slow:
                return "none"
            ema_f = float(ema(df["close"], ema_fast).iloc[-1])
            ema_s = float(ema(df["close"], ema_slow).iloc[-1])
            price = float(df["close"].iloc[-1])
            if price > ema_s and ema_f > ema_s:
                return "long"
        except Exception:
            pass
        return "none"
