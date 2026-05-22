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
        self._fraction = kelly_cfg.get("fraction", 0.25)
        self._max_risk_pct = kelly_cfg.get("max_kelly_pct", 3.0) / 100
        self._min_risk_pct = kelly_cfg.get("min_kelly_pct", 0.5) / 100
        self._vol_scale = kelly_cfg.get("volatility_scale", True)
        self._target_atr_pct = kelly_cfg.get("vol_target_atr_pct", 1.5) / 100

        # Fallback if Kelly not used: use config risk %
        self._default_risk_pct = cfg.get("risk", {}).get("risk_per_trade_pct", 1.5) / 100
        self._max_hard_cap = cfg.get("risk", {}).get("max_risk_per_trade_pct", 2.0) / 100

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

        # ── Final risk % ──────────────────────────────────────────────
        if fractional_kelly <= 0 or not ev.is_tradeable:
            # Fall back to default when Kelly is flat or uninformative
            risk_pct = self._default_risk_pct
            rationale = f"Fallback to default {risk_pct:.1%} (Kelly={fractional_kelly:.3%})"
        else:
            risk_pct = fractional_kelly * vol_scalar
            rationale = (
                f"Kelly={kelly_raw:.3%}  ×{self._fraction}frac  ×{vol_scalar:.2f}vol"
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
