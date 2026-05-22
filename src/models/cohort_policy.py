"""
Cohort policy — sleeve / regime / smart-money allowlist + attribution-aware
threshold relief.

Inputs:
    breakdown:        SignalBreakdown from Scorer
    trade_log:        list of TradeRecord (or duck-typed objects)
    attribution:      report from src.reporting.attribution.build_attribution_report()

Decision:
    CohortDecision(allowed, reason, ...)

The class also offers `recommend_symbols(trade_log)` which the scanner uses to
priority-load tickers that have produced winning cohort samples recently.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

log = logging.getLogger(__name__)

__all__ = ["CohortPolicy", "CohortDecision"]


# ---------------------------------------------------------------- decision

@dataclass
class CohortDecision:
    allowed: bool
    reason: str
    cohort_key: str = ""
    threshold_relief: float = 0.0          # negative = relief (lowers thresh)
    ranking_bonus: float = 0.0
    size_mult: float = 1.0                 # multiplier applied on top of FM/Kelly
    attribution_match: dict | None = None  # the row that matched, if any


# ---------------------------------------------------------------- policy

class CohortPolicy:
    """
    Lightweight gate that complements StrategyRouter.  The router decides
    *which sleeve* the bot is trading; this class decides whether the
    sleeve+regime+side combo is allowed *for this exact pair* given recent
    history and overall attribution.
    """

    DEFAULT_ALLOWED_REGIMES = ("trending_expansion", "distribution")
    DEFAULT_ALLOWED_SLEEVES = ("trend_following", "reversal", "compression_breakout")
    DEFAULT_ALLOWED_SM_PHASES = (
        "trending",
        "accumulation",
        "liquidity_sweep",
        "distribution",
        "neutral",
    )

    def __init__(self, cfg: dict):
        self._cfg = cfg or {}
        edge_cfg = self._cfg.get("edge_policy", {}) or {}
        self._enabled = bool(edge_cfg.get("enabled", True))
        self._allowed_regimes = set(
            edge_cfg.get("allowed_regimes", self.DEFAULT_ALLOWED_REGIMES) or self.DEFAULT_ALLOWED_REGIMES
        )
        self._allowed_sleeves = set(
            edge_cfg.get("allowed_strategy_sleeves", self.DEFAULT_ALLOWED_SLEEVES)
            or self.DEFAULT_ALLOWED_SLEEVES
        )
        self._allowed_sm_phases = set(
            edge_cfg.get("allowed_sm_phases", self.DEFAULT_ALLOWED_SM_PHASES)
            or self.DEFAULT_ALLOWED_SM_PHASES
        )
        self._long_only = bool(edge_cfg.get("long_only", False))

        # Local-history (per-pair) gate: block a pair that's bled three+ in a row.
        self._max_recent_pair_losses = int(edge_cfg.get("max_recent_pair_losses", 3) or 3)
        self._pair_history_window = int(edge_cfg.get("pair_history_window", 6) or 6)

        # Attribution gate: only fire when the cohort has reached `min_trades`
        # AND has unhealthy stats.
        attr = edge_cfg.get("attribution", {}) or {}
        self._attr_min_trades = int(attr.get("min_trades", 6) or 6)
        self._attr_block_pf = float(attr.get("block_below_pf", 0.85) or 0.85)
        self._attr_block_wr = float(attr.get("block_below_wr", 0.35) or 0.35)
        self._attr_relief_pf = float(attr.get("relief_above_pf", 1.30) or 1.30)
        self._attr_relief_wr = float(attr.get("relief_above_wr", 0.55) or 0.55)
        self._max_relief = float(attr.get("max_threshold_relief", 5.0) or 5.0)
        self._max_ranking_bonus = float(attr.get("max_ranking_bonus", 4.0) or 4.0)

    # ============================================================== public

    def evaluate(
        self,
        breakdown,
        trade_log: Iterable,
        attribution_report: dict | None = None,
    ) -> CohortDecision:
        cohort_key = self._cohort_key(breakdown)
        if not self._enabled:
            return CohortDecision(True, "cohort policy disabled", cohort_key=cohort_key)

        sleeve = str(getattr(breakdown, "strategy_sleeve", "") or "neutral")
        direction = str(getattr(breakdown, "direction", "") or "")
        regime_obj = getattr(breakdown, "regime", None)
        regime = getattr(regime_obj, "value", "") if regime_obj else ""
        sm_obj = getattr(breakdown, "smart_money", None)
        sm_phase = getattr(getattr(sm_obj, "phase", None), "value", "") if sm_obj else ""

        if sleeve not in self._allowed_sleeves:
            return CohortDecision(
                False,
                f"strategy sleeve filter blocked sleeve={sleeve}",
                cohort_key=cohort_key,
            )

        if regime and regime not in self._allowed_regimes:
            return CohortDecision(
                False,
                f"regime filter blocked regime={regime}",
                cohort_key=cohort_key,
            )

        if sm_phase and sm_phase not in self._allowed_sm_phases:
            return CohortDecision(
                False,
                f"sm phase filter blocked sm_phase={sm_phase}",
                cohort_key=cohort_key,
            )

        if self._long_only and direction != "long":
            return CohortDecision(
                False,
                f"long-only policy blocked direction={direction}",
                cohort_key=cohort_key,
            )

        # Per-pair recent-history gate.
        pair_block = self._pair_history_block(breakdown, trade_log)
        if pair_block:
            return CohortDecision(False, pair_block, cohort_key=cohort_key)

        # Attribution-driven hard block.
        attr_match = self._best_attribution_match(breakdown, attribution_report)
        if attr_match and self._attribution_blocking(attr_match):
            return CohortDecision(
                False,
                self._attribution_reason(attr_match, kind="block"),
                cohort_key=cohort_key,
                attribution_match=attr_match,
            )

        relief = self._attribution_relief(breakdown, attribution_report)
        bonus = self._attribution_bonus(breakdown, attribution_report)
        size_mult = self._attribution_size_mult(breakdown, attribution_report)
        return CohortDecision(
            True,
            "ok",
            cohort_key=cohort_key,
            threshold_relief=relief,
            ranking_bonus=bonus,
            size_mult=size_mult,
            attribution_match=attr_match,
        )

    def threshold_relief_with_attribution(
        self,
        breakdown,
        trade_log: Iterable,
        attribution_report: dict | None,
    ) -> float:
        if not self._enabled:
            return 0.0
        return self._attribution_relief(breakdown, attribution_report)

    def ranking_bonus_with_attribution(
        self,
        breakdown,
        trade_log: Iterable,
        attribution_report: dict | None,
    ) -> float:
        if not self._enabled:
            return 0.0
        return self._attribution_bonus(breakdown, attribution_report)

    def recommend_symbols(self, trade_log: Iterable) -> list[str]:
        # Boost symbols that have produced positive expectancy in recent history.
        scores: dict[str, float] = {}
        for t in (trade_log or []):
            sym = getattr(t, "symbol", None)
            if not sym:
                continue
            scores[sym] = scores.get(sym, 0.0) + float(getattr(t, "pnl_usd", 0.0) or 0.0)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # Keep up to 8 priority symbols, only positive-pnl pairs.
        return [s for s, score in ranked if score > 0][:8]

    # ============================================================ internals

    @staticmethod
    def _cohort_key(breakdown) -> str:
        sleeve = getattr(breakdown, "strategy_sleeve", "") or "neutral"
        regime_obj = getattr(breakdown, "regime", None)
        regime = getattr(regime_obj, "value", "") if regime_obj else "unknown"
        side = getattr(breakdown, "direction", "") or "unknown"
        return f"{sleeve}|{regime}|{side}"

    def _pair_history_block(self, breakdown, trade_log: Iterable) -> str:
        sym = getattr(breakdown, "symbol", "") or ""
        if not sym:
            return ""
        recent = [
            t for t in (trade_log or [])
            if getattr(t, "symbol", "") == sym
        ]
        if not recent:
            return ""
        recent = recent[-self._pair_history_window :]
        losses = sum(1 for t in recent if float(getattr(t, "pnl_usd", 0.0) or 0.0) <= 0)
        if losses >= self._max_recent_pair_losses:
            return f"negative cohort: {losses}/{len(recent)} recent losses on {sym}"
        return ""

    def _best_attribution_match(self, breakdown, report) -> dict | None:
        if not report:
            return None
        target_key = self._cohort_key(breakdown)
        for row in report.get("by_sleeve_regime_side", []) or []:
            if row.get("key") == target_key:
                return row
        # Fall back to coarser groupings.
        sleeve = getattr(breakdown, "strategy_sleeve", "") or "neutral"
        regime_obj = getattr(breakdown, "regime", None)
        regime = getattr(regime_obj, "value", "") if regime_obj else ""
        side = getattr(breakdown, "direction", "") or ""
        for row in report.get("by_regime_side", []) or []:
            if row.get("key") == f"{regime}|{side}":
                return row
        for row in report.get("by_sleeve", []) or []:
            if row.get("key") == sleeve:
                return row
        return None

    def _attribution_blocking(self, row: dict) -> bool:
        if int(row.get("trades", 0)) < self._attr_min_trades:
            return False
        if float(row.get("profit_factor", 0.0) or 0.0) <= self._attr_block_pf:
            return True
        if float(row.get("win_rate", 0.0) or 0.0) <= self._attr_block_wr:
            return True
        if float(row.get("expectancy_usd", 0.0) or 0.0) < 0.0:
            return True
        return False

    @staticmethod
    def _attribution_reason(row: dict, kind: str) -> str:
        return (
            f"attribution {kind}: key={row.get('key', '?')} "
            f"trades={row.get('trades', 0)} "
            f"wr={float(row.get('win_rate', 0.0)):.2f} "
            f"pf={float(row.get('profit_factor', 0.0)):.2f} "
            f"exp=${float(row.get('expectancy_usd', 0.0)):+.2f}"
        )

    def _attribution_relief(self, breakdown, report) -> float:
        row = self._best_attribution_match(breakdown, report)
        if not row or int(row.get("trades", 0)) < self._attr_min_trades:
            return 0.0
        pf = float(row.get("profit_factor", 0.0) or 0.0)
        wr = float(row.get("win_rate", 0.0) or 0.0)
        if pf >= self._attr_relief_pf and wr >= self._attr_relief_wr:
            # Negative relief = lowers threshold.  Scale by how strong the cohort is.
            scale = min(1.0, max(0.0, (pf - self._attr_relief_pf) / 1.0))
            return -round(self._max_relief * (0.5 + 0.5 * scale), 2)
        return 0.0

    def _attribution_bonus(self, breakdown, report) -> float:
        row = self._best_attribution_match(breakdown, report)
        if not row or int(row.get("trades", 0)) < self._attr_min_trades:
            return 0.0
        pf = float(row.get("profit_factor", 0.0) or 0.0)
        wr = float(row.get("win_rate", 0.0) or 0.0)
        exp = float(row.get("expectancy_usd", 0.0) or 0.0)
        if pf >= self._attr_relief_pf and wr >= self._attr_relief_wr and exp > 0:
            scale = min(1.0, max(0.0, (pf - self._attr_relief_pf) / 1.5))
            return round(self._max_ranking_bonus * (0.5 + 0.5 * scale), 2)
        return 0.0

    def _attribution_size_mult(self, breakdown, report) -> float:
        """Modest size scaling driven by realised cohort performance.

        Range is bounded [0.6, 1.25] so it can never overpower hard risk
        caps; the floor still triggers on healthy-but-weakening cohorts.
        """
        row = self._best_attribution_match(breakdown, report)
        if not row or int(row.get("trades", 0)) < self._attr_min_trades:
            return 1.0
        pf = float(row.get("profit_factor", 0.0) or 0.0)
        wr = float(row.get("win_rate", 0.0) or 0.0)
        exp = float(row.get("expectancy_usd", 0.0) or 0.0)
        # Strong cohort -> +up to 25%
        if pf >= self._attr_relief_pf and wr >= self._attr_relief_wr and exp > 0:
            scale = min(1.0, max(0.0, (pf - self._attr_relief_pf) / 1.5))
            return round(1.0 + 0.25 * scale, 3)
        # Weak cohort -> -up to 40%
        if pf <= self._attr_block_pf or wr <= self._attr_block_wr or exp < 0:
            shortfall = max(
                self._attr_block_pf - pf,
                (self._attr_block_wr - wr) * 2,
                0.0,
            )
            scale = min(1.0, shortfall / 0.4)
            return round(max(0.6, 1.0 - 0.4 * scale), 3)
        return 1.0
