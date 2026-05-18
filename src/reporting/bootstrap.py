"""
Bootstrap audit reporting helpers.

These helpers turn the closed-trade journal and edge memory into a compact
status snapshot for the bootstrap phase: direction mix, session matrix,
cohort health, and edge memory state.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from math import inf

from src.reporting.attribution import build_attribution_report
from src.reporting.cohort import summarize_cohorts
from src.session_clock import WITA, active_market_session_label, active_market_session_key


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _trade_sort_ts(trade) -> float:
    return _safe_float(getattr(trade, "closed_at", 0.0) or getattr(trade, "opened_at", 0.0))


def _session_from_ts(ts: float) -> str:
    if ts <= 0:
        return "unknown"
    return active_market_session_key(datetime.fromtimestamp(ts, tz=WITA))


def _metrics(trades: list) -> dict[str, float]:
    if not trades:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_pnl_usd": 0.0,
            "expectancy_usd": 0.0,
            "expectancy_pct": 0.0,
            "max_drawdown_usd": 0.0,
            "tp1_hit_rate": 0.0,
        }

    wins = [t for t in trades if _safe_float(getattr(t, "pnl_usd", 0.0)) > 0]
    losses = [t for t in trades if _safe_float(getattr(t, "pnl_usd", 0.0)) <= 0]
    pnl = [_safe_float(getattr(t, "pnl_usd", 0.0)) for t in trades]
    pnl_pct = [_safe_float(getattr(t, "pnl_pct", 0.0)) for t in trades]
    gross_profit = sum(x for x in pnl if x > 0)
    gross_loss = abs(sum(x for x in pnl if x <= 0))
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for val in pnl:
        equity += val
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    tp1_hits = sum(1 for t in trades if bool(getattr(t, "tp1_hit", False)))
    return {
        "trades": float(len(trades)),
        "wins": float(len(wins)),
        "losses": float(len(losses)),
        "win_rate": len(wins) / len(trades),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (inf if wins else 0.0),
        "net_pnl_usd": sum(pnl),
        "expectancy_usd": sum(pnl) / len(trades),
        "expectancy_pct": sum(pnl_pct) / len(trades),
        "max_drawdown_usd": max_dd,
        "tp1_hit_rate": tp1_hits / len(trades),
    }


def _direction_stats(trades: list, direction: str) -> dict[str, float]:
    subset = [t for t in trades if str(getattr(t, "direction", "") or "").lower() == direction]
    metrics = _metrics(subset)
    metrics["count"] = float(len(subset))
    return metrics


def _session_matrix_rows(trades: list) -> list[str]:
    sessions = ["asia", "london", "ny", "overlap_london_ny"]
    labels = {
        "asia": "Asia",
        "london": "London",
        "ny": "NY",
        "overlap_london_ny": "NY+London",
        "unknown": "Unknown",
    }
    matrix: dict[str, Counter] = {s: Counter() for s in sessions}
    for trade in trades:
        open_session = str(getattr(trade, "session", "") or "unknown")
        if open_session not in matrix:
            matrix[open_session] = Counter()
        close_session = _session_from_ts(_trade_sort_ts(trade))
        matrix[open_session][close_session] += 1

    all_cols = sessions + ["unknown"]
    header = "Open\\Close | " + " ".join(f"{labels[col]:>9}" for col in all_cols)
    rows = [header]
    for row_key in sessions + ["unknown"]:
        counts = matrix.get(row_key, Counter())
        rows.append(
            f"{labels[row_key]:>10} | "
            + " ".join(f"{counts.get(col, 0):>9}" for col in all_cols)
        )
    return rows


def build_bootstrap_audit_report(
    trade_log: list,
    lifecycle_report: dict | None = None,
    edge_memory=None,
    recent_window_sizes: tuple[int, int] = (20, 50),
    edge_rows: int = 3,
) -> dict:
    trades = sorted(list(trade_log or []), key=_trade_sort_ts)
    metrics = _metrics(trades)
    long_stats = _direction_stats(trades, "long")
    short_stats = _direction_stats(trades, "short")
    sample_too_small = len(trades) < 100

    recent_metrics = {
        size: _metrics(trades[-size:]) if len(trades) >= size else _metrics(trades)
        for size in recent_window_sizes
    }

    session_counts = Counter(
        str(getattr(t, "session", "") or "unknown") for t in trades
    )
    session_matrix = _session_matrix_rows(trades)

    cohort_report = lifecycle_report or {}
    cohort_counts = dict(cohort_report.get("counts", {}))
    cohort_recommendation = str(cohort_report.get("recommendation", "PAPER_ONLY"))

    edge_total_samples = int(edge_memory.total_samples()) if edge_memory else 0
    edge_rows_data = []
    if edge_memory and hasattr(edge_memory, "summary_rows"):
        edge_rows_data = edge_memory.summary_rows(limit=edge_rows)

    attribution = build_attribution_report(trades, min_trades=2) if trades else {}
    cohort_rows = summarize_cohorts(trades, min_trades=1) if trades else []

    return {
        "sample_too_small": sample_too_small,
        "trade_count": len(trades),
        "overall": metrics,
        "direction": {
            "long": long_stats,
            "short": short_stats,
        },
        "recent": recent_metrics,
        "session_counts": dict(session_counts),
        "session_matrix": session_matrix,
        "cohort_counts": cohort_counts,
        "cohort_recommendation": cohort_recommendation,
        "cohort_rows": cohort_rows[:5],
        "edge_total_samples": edge_total_samples,
        "edge_rows": edge_rows_data,
        "attribution": attribution,
        "session_label": active_market_session_label(),
    }


def format_bootstrap_audit_lines(report: dict) -> list[str]:
    lines: list[str] = []
    n = int(report.get("trade_count", 0))
    overall = report.get("overall", {})
    direction = report.get("direction", {})
    recent = report.get("recent", {})
    cohort_counts = report.get("cohort_counts", {})
    edge_total_samples = int(report.get("edge_total_samples", 0))
    edge_rows = report.get("edge_rows", [])

    lines.append(f"Phase: `{'BOOTSTRAP' if report.get('sample_too_small') else 'MEASURE'}` | N `{n}`")
    lines.append(
        "Overall: "
        f"WR `{overall.get('win_rate', 0.0):.1%}` | "
        f"PF `{overall.get('profit_factor', 0.0):.2f}` | "
        f"Exp `${overall.get('expectancy_usd', 0.0):+.2f}` | "
        f"TP1 `{overall.get('tp1_hit_rate', 0.0):.1%}` | "
        f"MaxDD `${overall.get('max_drawdown_usd', 0.0):.2f}`"
    )
    lines.append(
        "Mix: "
        f"LONG `{int(direction.get('long', {}).get('count', 0))}` "
        f"WR `{direction.get('long', {}).get('win_rate', 0.0):.1%}` "
        f"PF `{direction.get('long', {}).get('profit_factor', 0.0):.2f}`  "
        f"SHORT `{int(direction.get('short', {}).get('count', 0))}` "
        f"WR `{direction.get('short', {}).get('win_rate', 0.0):.1%}` "
        f"PF `{direction.get('short', {}).get('profit_factor', 0.0):.2f}`"
    )
    for window in sorted(recent):
        data = recent[window]
        lines.append(
            f"Last `{window}`: WR `{data.get('win_rate', 0.0):.1%}` "
            f"PF `{data.get('profit_factor', 0.0):.2f}` "
            f"Exp `${data.get('expectancy_usd', 0.0):+.2f}`"
        )
    lines.append(
        "Cohorts: "
        + " ".join(f"`{k}:{v}`" for k, v in sorted(cohort_counts.items()))
        + f" | Rec `{report.get('cohort_recommendation', 'PAPER_ONLY')}`"
    )
    lines.append("Session matrix open→close:")
    lines.extend(f"  {row}" for row in report.get("session_matrix", [])[:5])
    if edge_total_samples > 0:
        lines.append(f"Edge memory: `{edge_total_samples}` samples")
        for row in edge_rows:
            lines.append(
                "  "
                f"`{row['pair']}` `{row['edge_type']}` "
                f"n `{row['sample_size']}` "
                f"wr `{row['overall_wr']:.0%}` recent `{row['recent_wr']:.0%}` "
                f"cov `{row['regime_coverage']}`"
            )
    else:
        lines.append("Edge memory: `_no samples yet_`")
    return lines
