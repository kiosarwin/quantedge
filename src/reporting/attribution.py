"""
Trade attribution helpers for strategy sleeves, regimes, and directions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable


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


def _safe_exit_profile(trade) -> str:
    explicit = getattr(trade, "exit_profile", "") or str(getattr(trade, "scores", {}).get("exit_profile", "") or "")
    if explicit:
        return explicit
    sleeve = _safe_sleeve(trade)
    if sleeve in {"trend_following", "reversal", "compression_breakout"}:
        return sleeve
    return "default"


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
            )
        )

    rows.sort(key=lambda r: (r.net_pnl_usd, r.profit_factor, r.trades), reverse=True)
    return rows


def build_attribution_report(trade_log: list, min_trades: int = 2) -> dict:
    by_sleeve = summarize_grouped(trade_log, lambda t: _safe_sleeve(t), min_trades=min_trades)
    by_exit_profile = summarize_grouped(trade_log, lambda t: _safe_exit_profile(t), min_trades=min_trades)
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

    return {
        "by_sleeve": [asdict(r) for r in by_sleeve],
        "by_exit_profile": [asdict(r) for r in by_exit_profile],
        "by_regime": [asdict(r) for r in by_regime],
        "by_side": [asdict(r) for r in by_side],
        "by_regime_side": [asdict(r) for r in by_regime_side],
        "by_sleeve_exit_profile": [asdict(r) for r in by_sleeve_exit_profile],
        "by_exit_profile_regime_side": [asdict(r) for r in by_exit_profile_regime_side],
        "by_sleeve_regime_side": [asdict(r) for r in by_sleeve_regime_side],
        "headline": {
            "best_sleeve": by_sleeve[0].key if by_sleeve else None,
            "worst_sleeve": by_sleeve[-1].key if by_sleeve else None,
            "best_exit_profile": by_exit_profile[0].key if by_exit_profile else None,
            "worst_exit_profile": by_exit_profile[-1].key if by_exit_profile else None,
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
    regime_side_rows = report.get("by_regime_side", [])[:top_n]
    if regime_side_rows:
        parts = []
        for row in regime_side_rows:
            pf = row["profit_factor"]
            pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
            parts.append(f'{row["key"]}:{row["trades"]}t/{pf_str}')
        lines.append("RegSide  " + "  ".join(parts))
    exit_rows = report.get("by_exit_profile", [])[:top_n]
    if exit_rows:
        parts = []
        for row in exit_rows:
            pf = row["profit_factor"]
            pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
            parts.append(f'{row["key"]}:{row["win_rate"]:.0%}/{pf_str}/{row["net_pnl_usd"]:+.1f}')
        lines.append("Exits    " + "  ".join(parts))
    return lines
