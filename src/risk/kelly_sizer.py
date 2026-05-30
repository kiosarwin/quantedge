"""
Fractional Kelly Criterion position sizer.

Kelly fraction = (p × b − q) / b
  where p = P(win), q = P(loss), b = avg_win / avg_loss

Fractional Kelly = Kelly × fraction_factor (default 0.25 = quarter Kelly)

Volatility adjustment: reduce position size when current ATR is elevated
relative to target ATR, and increase when market is calmer than target.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.analysis.indicators import atr
from src.models.ev_model import EVResult

log = logging.getLogger(__name__)


@dataclass
class KellyResult:
    kelly_raw: float            # raw Kelly fraction (uncapped)
    fractional_kelly: float     # quarter-Kelly (or configured fraction)
    vol_scalar: float           # volatility adjustment multiplier (0.5–1.5)
    final_risk_pct: float       # final risk % of equity (hard-capped)
    size_usd: float             # position size in USD notional
    r_distance_pct: float       # stop distance as % of price
    rationale: str

    @property
    def is_valid(self) -> bool:
        return self.size_usd > 5.0


class KellySizer:
    def __init__(self, cfg: dict):
        kelly_cfg = cfg.get("kelly", {})
        self._fraction = kelly_cfg.get("fraction", 0.25)  # Quarter-Kelly (audit H1: reduced from 0.30)
        self._max_risk_pct = kelly_cfg.get("max_kelly_pct", 5.0) / 100  # UPGRADED: 5% cap
        self._min_risk_pct = kelly_cfg.get("min_kelly_pct", 0.7) / 100  # UPGRADED: 0.7% floor
        self._vol_scale = kelly_cfg.get("volatility_scale", True)
        self._target_atr_pct = kelly_cfg.get("vol_target_atr_pct", 1.5) / 100
        # Aggressive confidence scaling: high-confidence setups get extra size
        self._confidence_scaling = kelly_cfg.get("confidence_scaling", True)

        # Fallback if Kelly not used: use config risk %
        self._default_risk_pct = cfg.get("risk", {}).get("risk_per_trade_pct", 1.5) / 100
        self._max_hard_cap = cfg.get("risk", {}).get("max_risk_per_trade_pct", 2.5) / 100

    def size(
        self,
        ev: EVResult,
        equity: float,
        entry_price: float,
        stop_loss: float,
        df: pd.DataFrame,
        cfg: dict,
    ) -> KellyResult:
        """
        Compute risk-adjusted position size.

        Args:
            ev:           EVResult from ev_model.
            equity:       Current portfolio equity in USD.
            entry_price:  Expected fill price.
            stop_loss:    Stop loss price.
            df:           OHLCV DataFrame for volatility check.
            cfg:          Full config dict.
        """
        if equity <= 0 or entry_price <= 0 or stop_loss <= 0:
            return _zero_result("Invalid inputs")

        r_distance = abs(entry_price - stop_loss)
        r_distance_pct = r_distance / entry_price

        if r_distance_pct == 0:
            return _zero_result("Zero stop distance")

        # ── Kelly fraction ────────────────────────────────────────────
        # Prefer ev.kelly_raw when EVModel publishes it (decoupling from the
        # previous hard-coded `× 0.25` convention).  Fall back to the legacy
        # path so older state files still work.
        kelly_raw = float(getattr(ev, "kelly_raw", 0.0) or 0.0)
        if kelly_raw <= 0 and ev.kelly_fraction > 0:
            kelly_raw = ev.kelly_fraction / 0.25
        fractional_kelly = kelly_raw * self._fraction

        # ── Volatility scalar ─────────────────────────────────────────
        vol_scalar = 1.0
        if self._vol_scale and len(df) >= 28:
            ind_cfg = cfg.get("indicators", {})
            current_atr = atr(df, ind_cfg.get("atr_period", 14))
            current_atr_pct = current_atr / entry_price
            if current_atr_pct > 0:
                # Scale inversely: high vol → smaller; low vol → larger, bounded [0.5, 1.5]
                vol_scalar = float(min(1.5, max(0.5, self._target_atr_pct / current_atr_pct)))

        # ── Confidence-based aggressive scaling ───────────────────────
        # Ref: Kelly Criterion optimal sizing — when edge is proven (high
        # confidence + high p_win), size more aggressively up to the
        # theoretical Kelly optimum. This is the "aggressive on proven edge" mode.
        confidence_mult = 1.0
        if self._confidence_scaling and ev.confidence >= 0.7:
            if ev.p_win >= 0.58 and ev.ev_net_pct >= 0.15:
                # High-conviction: scale up to 1.4x the base Kelly
                confidence_mult = 1.0 + (ev.confidence - 0.7) * 1.33  # max ~1.4x at conf=1.0
            elif ev.p_win >= 0.52 and ev.ev_net_pct >= 0.08:
                confidence_mult = 1.0 + (ev.confidence - 0.7) * 0.67  # max ~1.2x

        # ── Final risk % ──────────────────────────────────────────────
        if fractional_kelly <= 0 or not ev.is_tradeable:
            # When EV is available but not statistically tradeable yet, do not
            # give it the same risk as a validated edge. Keep exploration alive,
            # but make the fallback proportional to posterior quality.
            edge_scalar = self._fallback_edge_scalar(ev)
            risk_pct = self._default_risk_pct * edge_scalar
            rationale = (
                f"Fallback {self._default_risk_pct:.1%} ×{edge_scalar:.2f}posterior "
                f"(Kelly={fractional_kelly:.3%})"
            )
        else:
            # H1 audit: cap effective Kelly at 0.30 to prevent over-aggressive sizing
            max_effective_kelly = 0.30
            if fractional_kelly * vol_scalar * confidence_mult > max_effective_kelly:
                confidence_mult = max_effective_kelly / (fractional_kelly * vol_scalar) if (fractional_kelly * vol_scalar) > 0 else 1.0
            risk_pct = fractional_kelly * vol_scalar * confidence_mult
            risk_pct = min(risk_pct, 0.30)  # Defensive hard clamp: effective Kelly never exceeds 30%
            rationale = (
                f"Kelly={kelly_raw:.3%}  ×{self._fraction}frac  ×{vol_scalar:.2f}vol"
                f"  ×{confidence_mult:.2f}conf"
            )

        # Hard caps
        risk_pct = float(min(self._max_hard_cap, max(self._min_risk_pct, risk_pct)))

        # ── USD position size ─────────────────────────────────────────
        risk_usd = equity * risk_pct
        size_usd = (risk_usd / r_distance) * entry_price

        log.debug(
            "Kelly: raw=%.4f frac=%.4f vol_scalar=%.2f risk=%.2f%% size=$%.0f  %s",
            kelly_raw, fractional_kelly, vol_scalar, risk_pct * 100, size_usd, rationale,
        )

        return KellyResult(
            kelly_raw=round(kelly_raw, 6),
            fractional_kelly=round(fractional_kelly, 6),
            vol_scalar=round(vol_scalar, 4),
            final_risk_pct=round(risk_pct * 100, 4),
            size_usd=round(size_usd, 2),
            r_distance_pct=round(r_distance_pct * 100, 4),
            rationale=rationale,
        )

    @staticmethod
    def _fallback_edge_scalar(ev: EVResult) -> float:
        p_win = min(0.85, max(0.15, float(getattr(ev, "p_win", 0.5) or 0.5)))
        conservative_p = float(getattr(ev, "conservative_p_win", 0.0) or 0.0)
        if conservative_p > 0.0:
            p_win = 0.65 * p_win + 0.35 * min(0.85, max(0.15, conservative_p))

        ev_net = float(getattr(ev, "ev_net_pct", 0.0) or 0.0)
        conservative_ev = float(getattr(ev, "conservative_ev_net_pct", 0.0) or 0.0)
        confidence = min(1.0, max(0.0, float(getattr(ev, "confidence", 0.0) or 0.0)))

        probability_term = (p_win - 0.50) * 2.0
        ev_term = max(-1.0, min(1.0, ev_net / 1.0))
        conservative_term = max(-1.0, min(1.0, conservative_ev / 1.0))
        evidence = 0.65 * probability_term + 0.25 * ev_term + 0.10 * conservative_term
        scalar = 0.75 + 0.35 * evidence

        if ev_net < 0.0 and p_win < 0.45:
            scalar -= min(0.25, abs(ev_net) * 0.12 + (0.45 - p_win) * 0.50)
        if confidence < 0.30 and ev_net <= 0.0:
            scalar -= (0.30 - confidence) * 0.25
        if ev_net > 0.0 and p_win >= 0.53:
            scalar += min(0.12, ev_net * 0.04 + (p_win - 0.53) * 0.40)

        return float(min(1.10, max(0.35, scalar)))


def _zero_result(reason: str) -> KellyResult:
    return KellyResult(
        kelly_raw=0.0,
        fractional_kelly=0.0,
        vol_scalar=1.0,
        final_risk_pct=0.0,
        size_usd=0.0,
        r_distance_pct=0.0,
        rationale=reason,
    )
