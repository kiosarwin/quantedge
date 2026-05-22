"""
ML hooks — pass-through implementation.

The futures pipeline historically referenced an ML stack (xgboost, sklearn,
walk-forward CV).  We keep the *interface* — `MLPredictor` + `LiveReadiness`
— so the rest of the code stays untouched, but every method here is either a
deterministic no-op or an additive observer.

Nothing in this module *blocks* trading by default.  The pipeline still has
the EV gate, smart-money filter, regime classifier, cohort policy, lifecycle
manager, fund manager, and risk manager.  Re-enabling true ML should slot in
behind the same surface.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

log = logging.getLogger(__name__)

__all__ = ["MLPredictor", "LiveReadiness", "ReadinessReport"]


# ============================================================================
# MLPredictor
# ============================================================================

class MLPredictor:
    """No-op ML predictor with a stable surface for the rest of the bot.

    It records regime PnL outcomes for telemetry, but never trains, never
    predicts a directional signal, and never vetoes a trade.
    """

    def __init__(self) -> None:
        self.is_ready: bool = False
        self.cv_accuracy: float = 0.0
        self.feature_importance: dict[str, float] = {}
        self._regime_pnls: dict[str, list[float]] = defaultdict(list)
        self._last_train_count: int = 0
        self._dynamic_p_win_threshold: float = 0.50  # never blocks at 0.50

    # -------- config-like properties (read-only) ---------------------------

    @property
    def dynamic_p_win_threshold(self) -> float:
        return self._dynamic_p_win_threshold

    @property
    def accuracy_trend(self) -> str:
        return "n/a"

    # -------- predictions --------------------------------------------------

    def predict(self, scores_dict: dict, total_score: float) -> tuple[float, bool]:
        """Return (p_win, is_ready).  We pass through with p_win=0.50 so the
        ML hard-gate in main.py never trips while ML is disabled.
        """
        return 0.50, False

    def confidence_scale(self, p_win: float) -> float:
        return 1.0

    def kelly_adjustment(self, trade_log: Iterable) -> float:
        return 1.0

    def regime_gate(self, regime_key: str) -> tuple[bool, float, str]:
        return True, 1.0, "ml regime gate disabled"

    # -------- telemetry / training ----------------------------------------

    def record_regime_outcome(self, regime_key: str, pnl: float) -> None:
        self._regime_pnls[regime_key].append(float(pnl))

    def maybe_train(self, trade_log: Iterable, extra_records: Iterable | None = None) -> bool:
        # We don't actually train.  We just bookkeep.
        n = sum(1 for _ in trade_log)
        if extra_records is not None:
            n += sum(1 for _ in extra_records)
        if n != self._last_train_count:
            self._last_train_count = n
        return False

    def status_report(self, trade_log: Iterable) -> dict:
        return {
            "ready": self.is_ready,
            "trained_on": self._last_train_count,
            "accuracy": self.cv_accuracy,
            "threshold": self.dynamic_p_win_threshold,
        }


# ============================================================================
# LiveReadiness
# ============================================================================

@dataclass
class ReadinessReport:
    passed: bool = False
    trade_count: int = 0
    criteria: dict[str, dict] = field(default_factory=dict)
    notes: str = ""


class LiveReadiness:
    """Tracks objective live-readiness criteria from realised paper trades.

    The decision to flip to live is still owned by the user — this object
    exists so the bot can show a clear pre-flight checklist.
    """

    DEFAULTS = {
        "min_trades": 30,
        "min_win_rate": 0.45,
        "min_profit_factor": 1.20,
        "min_expectancy_pct": 0.20,
        "max_drawdown_pct": 12.0,
        "min_sharpe_like": 0.20,
    }

    def __init__(self, cfg: dict | None = None, telegram_notifier=None):
        cfg = cfg or {}
        readiness_cfg = cfg.get("readiness", {}) or {}
        self._params = {**self.DEFAULTS, **{k: v for k, v in readiness_cfg.items() if v is not None}}
        self._telegram = telegram_notifier
        self._last_report: ReadinessReport = ReadinessReport()

    # ----------------------------------------------------------- public

    def check(self, trade_log: Iterable, ml: MLPredictor | None, equity_curve: Iterable[float]) -> ReadinessReport:
        trades = list(trade_log or [])
        equity_curve = list(equity_curve or [])
        n = len(trades)

        if n == 0:
            self._last_report = ReadinessReport(
                passed=False,
                trade_count=0,
                criteria={"sample_size": {"passed": False, "value": 0, "needed": self._params["min_trades"]}},
                notes="no trades yet",
            )
            return self._last_report

        wins = [t for t in trades if _pnl_usd(t) > 0]
        losses = [t for t in trades if _pnl_usd(t) <= 0]
        win_rate = len(wins) / n
        gross_profit = sum(_pnl_usd(t) for t in wins)
        gross_loss = abs(sum(_pnl_usd(t) for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if wins else 0.0)
        expectancy_pct = sum(float(getattr(t, "pnl_pct", 0.0) or 0.0) for t in trades) / n
        max_dd_pct = _max_drawdown_pct(equity_curve)
        sharpe = _sharpe_like([_pnl_usd(t) for t in trades])

        crit = {
            "sample_size": _criterion(n, self._params["min_trades"]),
            "win_rate": _criterion(win_rate, self._params["min_win_rate"]),
            "profit_factor": _criterion(profit_factor, self._params["min_profit_factor"]),
            "expectancy_pct": _criterion(expectancy_pct, self._params["min_expectancy_pct"]),
            "drawdown_pct": _criterion(max_dd_pct, self._params["max_drawdown_pct"], inverse=True),
            "sharpe_like": _criterion(sharpe, self._params["min_sharpe_like"]),
        }
        passed = all(c["passed"] for c in crit.values())
        self._last_report = ReadinessReport(
            passed=passed,
            trade_count=n,
            criteria=crit,
        )
        return self._last_report


# --------------------------------------------------------------- helpers

def _pnl_usd(trade) -> float:
    try:
        return float(getattr(trade, "pnl_usd", 0.0) or 0.0)
    except Exception:
        return 0.0


def _max_drawdown_pct(equity_curve: list[float]) -> float:
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        if peak > 0:
            dd = (peak - eq) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
    return max_dd


def _sharpe_like(pnls: list[float]) -> float:
    if len(pnls) < 2:
        return 0.0
    mean = sum(pnls) / len(pnls)
    var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
    std = var ** 0.5
    return 0.0 if std == 0 else mean / std


def _criterion(value: float, target: float, inverse: bool = False) -> dict:
    if inverse:
        passed = value <= target
    else:
        passed = value >= target
    return {"passed": bool(passed), "value": float(value), "target": float(target)}
