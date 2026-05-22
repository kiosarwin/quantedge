"""
Strategy lifecycle manager.

Each closed trade belongs to a *cohort*: a tuple of
(strategy_sleeve, market_regime, sm_phase, session, direction).  The lifecycle
manager tracks each cohort's progression through four stages:

    RESEARCH         — too few trades, no live exposure
    PAPER_VALIDATION — gathered enough samples, checking PF/WR
    SMALL_LIVE       — paper-validated, ramping into small live size
    ACTIVE           — fully validated, full size

The recommendation flag (`PAPER_ONLY` / `TRADE`) is what main.py reads to
decide whether to open *any* trade in live mode.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

log = logging.getLogger(__name__)

__all__ = ["StrategyLifecycleManager", "LifecycleDecision"]


_STATUSES = ("RESEARCH", "PAPER_VALIDATION", "SMALL_LIVE", "ACTIVE", "DISABLED")


# ----------------------------------------------------------------- decision

@dataclass
class LifecycleDecision:
    allowed: bool
    reason: str
    cohort_key: str
    status: str = "RESEARCH"
    recommendation: str = "PAPER_ONLY"


# ----------------------------------------------------------------- manager

class StrategyLifecycleManager:
    DEFAULTS = {
        "default_paper_only": True,
        "research_min_trades": 3,
        "paper_validation_min_trades": 10,
        "small_live_min_trades": 25,
        "active_min_trades": 50,
        "min_profit_factor": 1.10,
        "min_expectancy_usd": 0.0,
        "min_win_rate": 0.45,
        "max_outlier_share": 0.85,
        "max_stop_hit_rate": 0.7,
        "disable_profit_factor_below": 0.85,
        "disable_expectancy_usd_below": -1.0,
        "disable_max_drawdown_usd": 30.0,
    }

    def __init__(self, cfg: dict):
        self._cfg = cfg or {}
        merged = {**self.DEFAULTS, **(self._cfg.get("lifecycle", {}) or {})}
        self._enabled = bool(merged.get("enabled", True))
        self._params = merged

    # ------------------------------------------------------------- API

    def build_report(self, trade_log: Iterable) -> dict:
        if not self._enabled:
            return {
                "cohorts": [],
                "counts": {},
                "recommendation": "TRADE",
                "enabled": False,
            }

        groups: dict[str, list] = {}
        for t in trade_log or []:
            key = self._cohort_key_from_trade(t)
            groups.setdefault(key, []).append(t)

        cohorts: list[dict] = []
        counts = {s: 0 for s in _STATUSES}

        for key, trades in groups.items():
            cohort = self._evaluate_cohort(key, trades)
            cohorts.append(cohort)
            counts[cohort["status"]] = counts.get(cohort["status"], 0) + 1

        # Recommendation: TRADE only if at least one cohort is SMALL_LIVE or ACTIVE.
        any_live = any(c["status"] in {"SMALL_LIVE", "ACTIVE"} for c in cohorts)
        recommendation = "TRADE" if any_live else "PAPER_ONLY"

        # Sort cohorts by trade count desc for nicer display.
        cohorts.sort(key=lambda c: c["trades"], reverse=True)

        return {
            "cohorts": cohorts,
            "counts": counts,
            "recommendation": recommendation,
            "enabled": True,
        }

    def assess_breakdown(
        self,
        breakdown,
        report: dict,
        mode: str = "paper",
        hour_utc: int = 12,
    ) -> LifecycleDecision:
        cohort_key = self._cohort_key_from_breakdown(breakdown)
        if not self._enabled:
            return LifecycleDecision(
                allowed=True,
                reason="lifecycle filter disabled",
                cohort_key=cohort_key,
                status="ACTIVE",
                recommendation="TRADE",
            )

        cohort = next(
            (c for c in report.get("cohorts", []) if c.get("key") == cohort_key),
            None,
        )
        status = (cohort or {}).get("status", "RESEARCH")
        recommendation = "TRADE" if status in {"SMALL_LIVE", "ACTIVE"} else "PAPER_ONLY"

        if mode == "paper":
            return LifecycleDecision(
                allowed=True,
                reason=f"paper mode — accepting {status}",
                cohort_key=cohort_key,
                status=status,
                recommendation=recommendation,
            )

        # Live mode: only ACTIVE / SMALL_LIVE cohorts may proceed.
        if status in {"SMALL_LIVE", "ACTIVE"}:
            return LifecycleDecision(
                allowed=True,
                reason=f"cohort status {status}",
                cohort_key=cohort_key,
                status=status,
                recommendation=recommendation,
            )
        return LifecycleDecision(
            allowed=False,
            reason=f"cohort status {status} — paper-only",
            cohort_key=cohort_key,
            status=status,
            recommendation="PAPER_ONLY",
        )

    # ------------------------------------------------------------- internals

    def _cohort_key_from_trade(self, trade) -> str:
        sleeve = getattr(trade, "strategy_sleeve", "") or "unknown"
        regime = getattr(trade, "regime", "") or "unknown"
        scores = getattr(trade, "scores", {}) or {}
        sm_phase = scores.get("sm_phase", "neutral") or "neutral"
        session = scores.get("session", "unknown") or "unknown"
        direction = getattr(trade, "direction", "") or "unknown"
        return f"{sleeve}|{regime}|{sm_phase}|{session}|{direction}"

    def _cohort_key_from_breakdown(self, breakdown) -> str:
        sleeve = getattr(breakdown, "strategy_sleeve", "") or "unknown"
        regime_obj = getattr(breakdown, "regime", None)
        regime = getattr(regime_obj, "value", "") if regime_obj else "unknown"
        sm_obj = getattr(breakdown, "smart_money", None)
        sm_phase = getattr(getattr(sm_obj, "phase", None), "value", "") if sm_obj else "neutral"
        direction = getattr(breakdown, "direction", "") or "unknown"
        # Session is unknown at scoring time — main.py annotates trades when they close.
        session = "unknown"
        return f"{sleeve}|{regime}|{sm_phase}|{session}|{direction}"

    def _evaluate_cohort(self, key: str, trades: list) -> dict:
        n = len(trades)
        wins = [t for t in trades if float(getattr(t, "pnl_usd", 0.0) or 0.0) > 0]
        losses = [t for t in trades if float(getattr(t, "pnl_usd", 0.0) or 0.0) <= 0]
        gross_profit = sum(float(getattr(t, "pnl_usd", 0.0) or 0.0) for t in wins)
        gross_loss = abs(sum(float(getattr(t, "pnl_usd", 0.0) or 0.0) for t in losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if wins else 0.0)
        wr = len(wins) / n if n else 0.0
        expectancy_usd = sum(float(getattr(t, "pnl_usd", 0.0) or 0.0) for t in trades) / n if n else 0.0
        stop_hits = sum(1 for t in trades if str(getattr(t, "reason", "") or "") == "stop_loss")
        stop_rate = stop_hits / n if n else 0.0
        pnl_seq = [float(getattr(t, "pnl_usd", 0.0) or 0.0) for t in trades]
        max_dd = _max_drawdown_usd(pnl_seq)
        outlier_share = _outlier_share(pnl_seq)

        # Health gate
        unhealthy = (
            n >= self._params["paper_validation_min_trades"]
            and (
                pf < self._params["disable_profit_factor_below"]
                or expectancy_usd < self._params["disable_expectancy_usd_below"]
                or max_dd > self._params["disable_max_drawdown_usd"]
            )
        )
        healthy = (
            pf >= self._params["min_profit_factor"]
            and expectancy_usd >= self._params["min_expectancy_usd"]
            and wr >= self._params["min_win_rate"]
            and stop_rate <= self._params["max_stop_hit_rate"]
            and outlier_share <= self._params["max_outlier_share"]
        )

        if unhealthy:
            status = "DISABLED"
        elif n >= self._params["active_min_trades"] and healthy:
            status = "ACTIVE"
        elif n >= self._params["small_live_min_trades"] and healthy:
            status = "SMALL_LIVE"
        elif n >= self._params["paper_validation_min_trades"]:
            status = "PAPER_VALIDATION"
        elif n >= self._params["research_min_trades"]:
            status = "RESEARCH"
        else:
            status = "RESEARCH"

        return {
            "key": key,
            "trades": n,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": wr,
            "profit_factor": pf,
            "expectancy_usd": expectancy_usd,
            "stop_hit_rate": stop_rate,
            "max_drawdown_usd": max_dd,
            "outlier_share": outlier_share,
            "status": status,
        }


# ----------------------------------------------------------------- helpers

def _max_drawdown_usd(pnl_seq: list[float]) -> float:
    eq = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnl_seq:
        eq += p
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)
    return max_dd


def _outlier_share(pnl_seq: list[float]) -> float:
    if not pnl_seq:
        return 0.0
    largest = max((abs(p) for p in pnl_seq), default=0.0)
    total = sum(abs(p) for p in pnl_seq)
    return (largest / total) if total > 0 else 0.0
