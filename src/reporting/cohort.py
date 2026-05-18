"""
Cohort analytics for mini-quant monitoring.

Groups closed trades by strategy/regime/session/volatility/trend/signal context
and computes robust expectancy-oriented metrics.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass
class CohortMetrics:
    key: str
    strategy_name: str
    market_regime: str
    session: str
    volatility_bucket: str
    trend_bucket: str
    signal_type: str
    entry_reason: str
    exit_reason: str
    asset: str
    timeframe: str
    direction: str
    trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    expectancy_usd: float
    expectancy_pct: float
    average_win_usd: float
    average_loss_usd: float
    average_win_pct: float
    average_loss_pct: float
    payoff_ratio: float
    max_drawdown_usd: float
    sharpe_like: float
    sortino_like: float
    tp1_hit_rate: float
    stop_hit_rate: float
    avg_time_in_trade_s: float
    avg_slippage_fee_pct: float
    sample_outlier_share: float


def _safe_score(scores: dict, key: str, default: float = 0.0) -> float:
    try:
        val = scores.get(key, default)
        return float(default if val is None else val)
    except Exception:
        return float(default)


def _safe_text(scores: dict, key: str, default: str = "") -> str:
    val = scores.get(key, default)
    return str(default if val is None else val)


def _regime_from_trade(trade) -> str:
    return getattr(trade, "regime", "") or _safe_text(getattr(trade, "scores", {}) or {}, "regime", "unknown") or "unknown"


def _session_from_trade(trade) -> str:
    explicit = getattr(trade, "session", "") or _safe_text(getattr(trade, "scores", {}) or {}, "session", "")
    if explicit:
        return explicit
    hour = int((getattr(trade, "opened_at", 0.0) // 3600) % 24)
    if hour < 8:
        return "asia"
    if hour < 13:
        return "london"
    if hour < 17:
        return "overlap_london_ny"
    return "ny"


def _bucket_volatility(trade) -> str:
    explicit = getattr(trade, "volatility_bucket", "") or _safe_text(getattr(trade, "scores", {}) or {}, "volatility_bucket", "")
    if explicit:
        return explicit
    val = _safe_score(getattr(trade, "scores", {}) or {}, "volatility", 50.0)
    if val <= 20:
        return "low"
    if val <= 40:
        return "medium"
    if val <= 70:
        return "high"
    return "extreme"


def _bucket_trend(trade) -> str:
    explicit = getattr(trade, "trend_bucket", "") or _safe_text(getattr(trade, "scores", {}) or {}, "trend_bucket", "")
    if explicit:
        return explicit
    val = _safe_score(getattr(trade, "scores", {}) or {}, "trend_strength", 50.0)
    if val < 40:
        return "weak"
    if val < 65:
        return "moderate"
    if val < 85:
        return "strong"
    return "very_strong"


def _asset_from_trade(trade) -> str:
    explicit = getattr(trade, "asset", "") or _safe_text(getattr(trade, "scores", {}) or {}, "asset", "")
    if explicit:
        return explicit
    symbol = getattr(trade, "symbol", "")
    return symbol.split("/")[0] if symbol else "unknown"


def _timeframe_from_trade(trade) -> str:
    return getattr(trade, "timeframe", "") or _safe_text(getattr(trade, "scores", {}) or {}, "timeframe", "unknown") or "unknown"


def _strategy_from_trade(trade) -> str:
    return getattr(trade, "strategy_sleeve", "") or _safe_text(getattr(trade, "scores", {}) or {}, "strategy_sleeve", "unknown") or "unknown"


def _signal_type_from_trade(trade) -> str:
    explicit = getattr(trade, "signal_type", "") or _safe_text(getattr(trade, "scores", {}) or {}, "signal_type", "")
    if explicit:
        return explicit
    regime = _regime_from_trade(trade)
    sm_phase = _safe_text(getattr(trade, "scores", {}) or {}, "sm_phase", "neutral") or "neutral"
    return f"{regime}|{sm_phase}"


def _entry_reason_from_trade(trade) -> str:
    return getattr(trade, "entry_reason", "") or _safe_text(getattr(trade, "scores", {}) or {}, "entry_reason", "unknown") or "unknown"


def cohort_dimensions(trade) -> dict[str, str]:
    return {
        "strategy_name": _strategy_from_trade(trade),
        "market_regime": _regime_from_trade(trade),
        "session": _session_from_trade(trade),
        "volatility_bucket": _bucket_volatility(trade),
        "trend_bucket": _bucket_trend(trade),
        "signal_type": _signal_type_from_trade(trade),
        "entry_reason": _entry_reason_from_trade(trade),
        "exit_reason": getattr(trade, "reason", "") or "unknown",
        "asset": _asset_from_trade(trade),
        "timeframe": _timeframe_from_trade(trade),
        "direction": getattr(trade, "direction", "") or "unknown",
    }


def default_cohort_key(trade) -> str:
    dims = cohort_dimensions(trade)
    return "|".join(
        dims[k]
        for k in (
            "strategy_name",
            "market_regime",
            "session",
            "volatility_bucket",
            "trend_bucket",
            "signal_type",
            "asset",
            "timeframe",
            "direction",
        )
    )


def _max_drawdown(pnls: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return max_dd


def _sharpe_like(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean_r = sum(returns) / len(returns)
    var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    return 0.0 if std == 0 else mean_r / std


def _sortino_like(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean_r = sum(returns) / len(returns)
    downside = [min(0.0, r) for r in returns]
    downside_var = sum(r * r for r in downside) / max(1, len(downside))
    downside_std = math.sqrt(downside_var)
    return 0.0 if downside_std == 0 else mean_r / downside_std


def summarize_cohorts(trade_log: list, key_fn=default_cohort_key, min_trades: int = 1) -> list[dict]:
    groups: dict[str, list] = {}
    for trade in trade_log:
        key = key_fn(trade) or "unknown"
        groups.setdefault(key, []).append(trade)

    rows: list[dict] = []
    for key, trades in groups.items():
        if len(trades) < min_trades:
            continue
        trades = sorted(trades, key=lambda t: getattr(t, "closed_at", getattr(t, "opened_at", 0.0)))
        dims = cohort_dimensions(trades[0])
        wins = [t for t in trades if getattr(t, "pnl_usd", 0.0) > 0]
        losses = [t for t in trades if getattr(t, "pnl_usd", 0.0) <= 0]
        gross_profit = sum(getattr(t, "pnl_usd", 0.0) for t in wins)
        gross_loss = abs(sum(getattr(t, "pnl_usd", 0.0) for t in losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if wins else 0.0)
        avg_win_usd = gross_profit / len(wins) if wins else 0.0
        avg_loss_usd = gross_loss / len(losses) if losses else 0.0
        pnl_usd = [float(getattr(t, "pnl_usd", 0.0)) for t in trades]
        pnl_pct = [float(getattr(t, "pnl_pct", 0.0)) for t in trades]
        tp1_hits = [1.0 for t in trades if bool(getattr(t, "tp1_hit", False))]
        hold_s = [float(getattr(t, "hold_duration_s", 0.0) or 0.0) for t in trades if getattr(t, "hold_duration_s", None) is not None]
        fee_slip_pct = [
            float(getattr(t, "fees_slippage_pct", 0.0) or 0.0)
            for t in trades
            if getattr(t, "fees_slippage_pct", None) is not None
        ]
        largest_abs = max((abs(x) for x in pnl_usd), default=0.0)
        outlier_share = 0.0
        if largest_abs > 0:
            outlier_share = largest_abs / max(1e-9, sum(abs(x) for x in pnl_usd))
        row = CohortMetrics(
            key=key,
            strategy_name=dims["strategy_name"],
            market_regime=dims["market_regime"],
            session=dims["session"],
            volatility_bucket=dims["volatility_bucket"],
            trend_bucket=dims["trend_bucket"],
            signal_type=dims["signal_type"],
            entry_reason=dims["entry_reason"],
            exit_reason=dims["exit_reason"],
            asset=dims["asset"],
            timeframe=dims["timeframe"],
            direction=dims["direction"],
            trades=len(trades),
            wins=len(wins),
            losses=len(losses),
            win_rate=len(wins) / len(trades),
            profit_factor=pf,
            expectancy_usd=sum(pnl_usd) / len(trades),
            expectancy_pct=sum(pnl_pct) / len(trades),
            average_win_usd=avg_win_usd,
            average_loss_usd=avg_loss_usd,
            average_win_pct=(sum(getattr(t, "pnl_pct", 0.0) for t in wins) / len(wins)) if wins else 0.0,
            average_loss_pct=(sum(abs(getattr(t, "pnl_pct", 0.0)) for t in losses) / len(losses)) if losses else 0.0,
            payoff_ratio=(avg_win_usd / avg_loss_usd) if avg_loss_usd > 0 else (float("inf") if avg_win_usd > 0 else 0.0),
            max_drawdown_usd=_max_drawdown(pnl_usd),
            sharpe_like=_sharpe_like(pnl_pct),
            sortino_like=_sortino_like(pnl_pct),
            tp1_hit_rate=(sum(tp1_hits) / len(trades)) if trades else 0.0,
            stop_hit_rate=(sum(1 for t in trades if getattr(t, "reason", "") == "stop_loss") / len(trades)) if trades else 0.0,
            avg_time_in_trade_s=(sum(hold_s) / len(hold_s)) if hold_s else 0.0,
            avg_slippage_fee_pct=(sum(fee_slip_pct) / len(fee_slip_pct)) if fee_slip_pct else 0.0,
            sample_outlier_share=outlier_share,
        )
        rows.append(asdict(row))

    rows.sort(key=lambda r: (r["expectancy_usd"], r["profit_factor"], r["trades"]), reverse=True)
    return rows
