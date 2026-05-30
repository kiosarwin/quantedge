"""
Trade attribution helpers for strategy sleeves, regimes, and directions.

Professional-grade additions:
  - Risk-adjusted performance metrics (Sharpe, Sortino, Calmar)
  - VaR / CVaR from trade return distribution
  - Kelly criterion calculation
  - Streak analysis and recovery metrics
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, asdict
from typing import Callable
from src.models.strategy_passport import sector_for_asset, asset_from_symbol


@dataclass
class AttributionRow:
    key: str
    trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    net_pnl_usd: float
    avg_pnl_usd: float
    avg_pnl_pct: float
    avg_dispersion_value: float = 0.0


def _safe_sleeve(trade) -> str:
    explicit = getattr(trade, "strategy_sleeve", "") or str(getattr(trade, "scores", {}).get("strategy_sleeve", "") or "")
    if explicit:
        return explicit

    scores = getattr(trade, "scores", {}) or {}
    regime = getattr(trade, "regime", "") or _infer_regime(scores)
    sm_phase = str(scores.get("sm_phase", "neutral") or "neutral")
    trend = float(scores.get("trend_strength", 0.0) or 0.0)
    structure = float(scores.get("structure_quality", 0.0) or 0.0)
    volatility = float(scores.get("volatility", 50.0) or 50.0)

    if sm_phase == "liquidity_sweep" or (regime == "distribution" and sm_phase in {"distribution", "liquidity_sweep"} and volatility <= 75.0):
        return "reversal"
    if regime == "trending_expansion" and trend >= 62.0 and structure >= 58.0:
        return "trend_following"
    if regime == "accumulation_compression" and structure >= 60.0:
        return "compression_breakout"
    return "unknown"




def _safe_setup_type(trade) -> str:
    explicit = getattr(trade, "setup_type", "") or str(getattr(trade, "scores", {}).get("setup_type", "") or "")
    return explicit or "unknown"

def _safe_sector(trade) -> str:
    explicit = getattr(trade, "sector", "") or str(getattr(trade, "scores", {}).get("sector", "") or "")
    if explicit:
        return explicit
    asset = getattr(trade, "asset", "") or asset_from_symbol(getattr(trade, "symbol", ""))
    return sector_for_asset(asset)


def _safe_exit_profile(trade) -> str:
    explicit = getattr(trade, "exit_profile", "") or str(getattr(trade, "scores", {}).get("exit_profile", "") or "")
    if explicit:
        return explicit
    sleeve = _safe_sleeve(trade)
    if sleeve in {"trend_following", "reversal", "compression_breakout"}:
        return sleeve
    return "default"


def _safe_dispersion_state(trade) -> str:
    explicit = getattr(trade, "dispersion_state", "") or str(getattr(trade, "scores", {}).get("dispersion_state", "") or "")
    return explicit or "normal"


def _safe_dispersion_value(trade) -> float:
    explicit = getattr(trade, "dispersion_value", None)
    if explicit is not None:
        try:
            return float(explicit)
        except Exception:
            return 0.0
    scores = getattr(trade, "scores", {}) or {}
    try:
        return float(scores.get("dispersion_value", 0.0) or 0.0)
    except Exception:
        return 0.0


def _safe_score_text(trade, key: str, default: str = "unknown") -> str:
    explicit = getattr(trade, key, "") or ""
    if explicit:
        return str(explicit)
    scores = getattr(trade, "scores", {}) or {}
    value = scores.get(key, default)
    return str(value or default)


def _safe_market_risk_state(trade) -> str:
    return _safe_score_text(trade, "market_risk_on_state")


def _safe_market_rotation_state(trade) -> str:
    return _safe_score_text(trade, "market_rotation_state")


def _safe_market_btc_trend(trade) -> str:
    return _safe_score_text(trade, "market_btc_trend")


def _safe_market_eth_btc_trend(trade) -> str:
    return _safe_score_text(trade, "market_eth_btc_trend")


def _infer_regime(scores: dict) -> str:
    code = int(scores.get("regime_code", -1) or -1)
    return {
        3: "trending_expansion",
        2: "accumulation_compression",
        1: "distribution",
        0: "chaos",
    }.get(code, "unknown")


def summarize_grouped(trade_log: list, key_fn: Callable[[object], str], min_trades: int = 1) -> list[AttributionRow]:
    groups: dict[str, list] = {}
    for t in trade_log:
        key = key_fn(t) or "unknown"
        groups.setdefault(key, []).append(t)

    rows: list[AttributionRow] = []
    for key, trades in groups.items():
        if len(trades) < min_trades:
            continue
        wins = [t for t in trades if t.pnl_usd > 0]
        losses = [t for t in trades if t.pnl_usd <= 0]
        gross_profit = sum(t.pnl_usd for t in wins)
        gross_loss = abs(sum(t.pnl_usd for t in losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if wins else 0.0)
        net = sum(t.pnl_usd for t in trades)
        avg_dispersion = sum(_safe_dispersion_value(t) for t in trades) / len(trades) if trades else 0.0
        rows.append(
            AttributionRow(
                key=key,
                trades=len(trades),
                wins=len(wins),
                losses=len(losses),
                win_rate=(len(wins) / len(trades)) if trades else 0.0,
                profit_factor=pf,
                net_pnl_usd=net,
                avg_pnl_usd=(net / len(trades)) if trades else 0.0,
                avg_pnl_pct=(sum(t.pnl_pct for t in trades) / len(trades)) if trades else 0.0,
                avg_dispersion_value=avg_dispersion,
            )
        )

    rows.sort(key=lambda r: (r.net_pnl_usd, r.profit_factor, r.trades), reverse=True)
    return rows


@dataclass
class RiskAdjustedMetrics:
    """Professional-grade risk-adjusted performance metrics.

    Equivalent to what an institutional fund reports to investors.
    """
    sharpe_ratio: float = 0.0         # annualized (risk-free = 0)
    sortino_ratio: float = 0.0        # annualized, downside deviation only
    calmar_ratio: float = 0.0         # annualized return / max drawdown
    var_95_pct: float = 0.0           # 95% VaR as % of equity
    var_99_pct: float = 0.0           # 99% VaR
    cvar_95_pct: float = 0.0          # 95% Conditional VaR (expected shortfall)
    cvar_99_pct: float = 0.0          # 99% Conditional VaR
    max_drawdown_pct: float = 0.0     # worst peak-to-trough
    kelly_criterion: float = 0.0      # optimal bet fraction
    payoff_ratio: float = 0.0         # avg_win / abs(avg_loss)
    tail_risk_score: float = 0.0      # 0-100, higher = fatter tails
    volatility_annualized: float = 0.0
    downside_deviation: float = 0.0
    recovery_factor: float = 0.0      # net profit / max drawdown
    max_consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    avg_trade_duration_s: float = 0.0
    sample_size: int = 0


def compute_risk_adjusted_metrics(trade_log: list) -> RiskAdjustedMetrics:
    """Compute institutional-grade risk-adjusted metrics from trade log.

    This is what a professional quant fund reports alongside raw P&L.
    """
    m = RiskAdjustedMetrics()
    if len(trade_log) < 3:
        m.sample_size = len(trade_log)
        return m

    # Extract returns (% of equity per trade)
    returns = []
    durations = []
    for t in trade_log:
        pnl_pct = float(getattr(t, "pnl_pct", 0.0) or 0.0)
        returns.append(pnl_pct)
        opened = float(getattr(t, "opened_at", 0.0) or 0.0)
        closed = float(getattr(t, "closed_at", 0.0) or 0.0)
        if opened > 0 and closed > opened:
            durations.append(closed - opened)

    m.sample_size = len(returns)
    m.avg_trade_duration_s = statistics.mean(durations) if durations else 0.0

    # --- Basic stats ---
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    avg_win = statistics.mean(wins) if wins else 0.0
    avg_loss = statistics.mean(losses) if losses else 0.0
    m.payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    # --- Win/loss streaks ---
    max_win_streak = max_loss_streak = 0
    current_win = current_loss = 0
    for r in returns:
        if r > 0:
            current_win += 1
            current_loss = 0
            max_win_streak = max(max_win_streak, current_win)
        else:
            current_loss += 1
            current_win = 0
            max_loss_streak = max(max_loss_streak, current_loss)
    m.max_consecutive_wins = max_win_streak
    m.max_consecutive_losses = max_loss_streak

    # --- Kelly criterion ---
    win_rate = len(wins) / len(returns) if returns else 0.0
    if win_rate > 0 and m.payoff_ratio > 0:
        m.kelly_criterion = max(0.0, (win_rate * m.payoff_ratio - (1 - win_rate)) / m.payoff_ratio)

    # --- Volatility ---
    if len(returns) >= 2:
        m.volatility_annualized = statistics.stdev(returns) * math.sqrt(252)
        downside = [r for r in returns if r < 0]
        m.downside_deviation = statistics.stdev(downside) * math.sqrt(252) if len(downside) >= 2 else 0.0

    # --- Risk-adjusted returns ---
    if m.volatility_annualized > 0:
        mean_annual = statistics.mean(returns) * 252
        m.sharpe_ratio = mean_annual / m.volatility_annualized
    if m.downside_deviation > 0:
        mean_annual = statistics.mean(returns) * 252
        m.sortino_ratio = mean_annual / m.downside_deviation

    # --- VaR / CVaR (historical simulation) ---
    sorted_returns = sorted(returns)
    n = len(sorted_returns)
    idx_95 = max(0, int(n * 0.05))
    idx_99 = max(0, int(n * 0.01))
    m.var_95_pct = abs(sorted_returns[idx_95])
    m.var_99_pct = abs(sorted_returns[idx_99])
    tail_95 = sorted_returns[:idx_95 + 1]
    tail_99 = sorted_returns[:idx_99 + 1]
    m.cvar_95_pct = abs(statistics.mean(tail_95)) if tail_95 else m.var_95_pct
    m.cvar_99_pct = abs(statistics.mean(tail_99)) if tail_99 else m.var_99_pct

    # --- Tail risk score ---
    if m.var_95_pct > 0:
        tail_ratio = m.cvar_95_pct / m.var_95_pct
        m.tail_risk_score = min(100.0, max(0.0, (tail_ratio - 1.0) * 200.0))

    # --- Max drawdown from equity curve ---
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in returns:
        cumulative += r
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        max_dd = max(max_dd, dd)
    m.max_drawdown_pct = max_dd

    # --- Calmar ratio ---
    if m.max_drawdown_pct > 0:
        mean_annual = statistics.mean(returns) * 252
        m.calmar_ratio = mean_annual / (m.max_drawdown_pct / 100.0)

    # --- Recovery factor ---
    if m.max_drawdown_pct > 0:
        m.recovery_factor = sum(returns) / (m.max_drawdown_pct / 100.0)

    return m


def build_attribution_report(trade_log: list, min_trades: int = 2) -> dict:
    by_sleeve = summarize_grouped(trade_log, lambda t: _safe_sleeve(t), min_trades=min_trades)
    by_setup_type = summarize_grouped(trade_log, lambda t: _safe_setup_type(t), min_trades=min_trades)
    by_sector = summarize_grouped(trade_log, lambda t: _safe_sector(t), min_trades=min_trades)
    by_exit_profile = summarize_grouped(trade_log, lambda t: _safe_exit_profile(t), min_trades=min_trades)
    by_dispersion_state = summarize_grouped(trade_log, lambda t: _safe_dispersion_state(t), min_trades=min_trades)
    by_market_risk_state = summarize_grouped(trade_log, lambda t: _safe_market_risk_state(t), min_trades=min_trades)
    by_market_rotation_state = summarize_grouped(trade_log, lambda t: _safe_market_rotation_state(t), min_trades=min_trades)
    by_market_context = summarize_grouped(
        trade_log,
        lambda t: f"{_safe_market_risk_state(t)}|{_safe_market_rotation_state(t)}",
        min_trades=min_trades,
    )
    by_market_btc_ethbtc = summarize_grouped(
        trade_log,
        lambda t: f"{_safe_market_btc_trend(t)}|{_safe_market_eth_btc_trend(t)}",
        min_trades=min_trades,
    )
    by_regime = summarize_grouped(trade_log, lambda t: getattr(t, "regime", "") or "unknown", min_trades=min_trades)
    by_side = summarize_grouped(trade_log, lambda t: getattr(t, "direction", "") or "unknown", min_trades=min_trades)
    by_regime_side = summarize_grouped(
        trade_log,
        lambda t: f"{getattr(t, 'regime', '') or 'unknown'}|{getattr(t, 'direction', '') or 'unknown'}",
        min_trades=min_trades,
    )
    by_sleeve_exit_profile = summarize_grouped(
        trade_log,
        lambda t: f"{_safe_sleeve(t)}|{_safe_exit_profile(t)}",
        min_trades=min_trades,
    )
    by_sleeve_dispersion_state = summarize_grouped(
        trade_log,
        lambda t: f"{_safe_sleeve(t)}|{_safe_dispersion_state(t)}",
        min_trades=min_trades,
    )
    by_regime_dispersion_state = summarize_grouped(
        trade_log,
        lambda t: f"{getattr(t, 'regime', '') or 'unknown'}|{_safe_dispersion_state(t)}",
        min_trades=min_trades,
    )
    by_exit_profile_regime_side = summarize_grouped(
        trade_log,
        lambda t: f"{_safe_exit_profile(t)}|{getattr(t, 'regime', '') or 'unknown'}|{getattr(t, 'direction', '') or 'unknown'}",
        min_trades=min_trades,
    )
    by_sleeve_regime_side = summarize_grouped(
        trade_log,
        lambda t: f"{_safe_sleeve(t)}|{getattr(t, 'regime', '') or 'unknown'}|{getattr(t, 'direction', '') or 'unknown'}",
        min_trades=min_trades,
    )
    by_setup_type_regime_side = summarize_grouped(
        trade_log,
        lambda t: f"{_safe_setup_type(t)}|{getattr(t, 'regime', '') or 'unknown'}|{getattr(t, 'direction', '') or 'unknown'}",
        min_trades=min_trades,
    )

    # Professional-grade risk-adjusted metrics
    risk_metrics = compute_risk_adjusted_metrics(trade_log)

    return {
        "by_sleeve": [asdict(r) for r in by_sleeve],
        "by_setup_type": [asdict(r) for r in by_setup_type],
        "by_sector": [asdict(r) for r in by_sector],
        "by_exit_profile": [asdict(r) for r in by_exit_profile],
        "by_dispersion_state": [asdict(r) for r in by_dispersion_state],
        "by_market_risk_state": [asdict(r) for r in by_market_risk_state],
        "by_market_rotation_state": [asdict(r) for r in by_market_rotation_state],
        "by_market_context": [asdict(r) for r in by_market_context],
        "by_market_btc_ethbtc": [asdict(r) for r in by_market_btc_ethbtc],
        "by_regime": [asdict(r) for r in by_regime],
        "by_side": [asdict(r) for r in by_side],
        "by_regime_side": [asdict(r) for r in by_regime_side],
        "by_sleeve_exit_profile": [asdict(r) for r in by_sleeve_exit_profile],
        "by_sleeve_dispersion_state": [asdict(r) for r in by_sleeve_dispersion_state],
        "by_regime_dispersion_state": [asdict(r) for r in by_regime_dispersion_state],
        "by_exit_profile_regime_side": [asdict(r) for r in by_exit_profile_regime_side],
        "by_sleeve_regime_side": [asdict(r) for r in by_sleeve_regime_side],
        "by_setup_type_regime_side": [asdict(r) for r in by_setup_type_regime_side],
        "risk_adjusted_metrics": asdict(risk_metrics),
        "headline": {
            "best_sleeve": by_sleeve[0].key if by_sleeve else None,
            "worst_sleeve": by_sleeve[-1].key if by_sleeve else None,
            "best_setup_type": by_setup_type[0].key if by_setup_type else None,
            "worst_setup_type": by_setup_type[-1].key if by_setup_type else None,
            "best_exit_profile": by_exit_profile[0].key if by_exit_profile else None,
            "worst_exit_profile": by_exit_profile[-1].key if by_exit_profile else None,
            "best_dispersion_state": by_dispersion_state[0].key if by_dispersion_state else None,
            "worst_dispersion_state": by_dispersion_state[-1].key if by_dispersion_state else None,
            "best_market_context": by_market_context[0].key if by_market_context else None,
            "worst_market_context": by_market_context[-1].key if by_market_context else None,
        },
    }


def compact_lines(report: dict, top_n: int = 3) -> list[str]:
    lines: list[str] = []
    sleeve_rows = report.get("by_sleeve", [])[:top_n]
    if sleeve_rows:
        parts = []
        for row in sleeve_rows:
            pf = row["profit_factor"]
            pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
            parts.append(f'{row["key"]}:{row["win_rate"]:.0%}/{pf_str}/{row["net_pnl_usd"]:+.1f}')
        lines.append("Sleeves  " + "  ".join(parts))
    setup_rows = report.get("by_setup_type", [])[:top_n]
    if setup_rows:
        parts = []
        for row in setup_rows:
            pf = row["profit_factor"]
            pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
            parts.append(f'{row["key"]}:{row["trades"]}t/{pf_str}')
        lines.append("Setups   " + "  ".join(parts))
    regime_side_rows = report.get("by_regime_side", [])[:top_n]
    if regime_side_rows:
        parts = []
        for row in regime_side_rows:
            pf = row["profit_factor"]
            pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
            parts.append(f'{row["key"]}:{row["trades"]}t/{pf_str}')
        lines.append("RegSide  " + "  ".join(parts))
    market_rows = report.get("by_market_context", [])[:top_n]
    if market_rows:
        parts = []
        for row in market_rows:
            pf = row["profit_factor"]
            pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
            parts.append(f'{row["key"]}:{row["trades"]}t/{pf_str}')
        lines.append("Market   " + "  ".join(parts))
    exit_rows = report.get("by_exit_profile", [])[:top_n]
    if exit_rows:
        parts = []
        for row in exit_rows:
            pf = row["profit_factor"]
            pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
            parts.append(f'{row["key"]}:{row["win_rate"]:.0%}/{pf_str}/{row["net_pnl_usd"]:+.1f}')
        lines.append("Exits    " + "  ".join(parts))
    return lines
