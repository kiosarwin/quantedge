"""Read-only trading brain journal and edge reports.

This module turns closed trade records into human/LLM-readable artifacts.
It deliberately does not feed back into entry, exit, scoring, or risk logic.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from src.reporting.attribution import build_attribution_report


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(value: str) -> str:
    cleaned = _SAFE_NAME.sub("-", str(value or "unknown")).strip("-")
    return cleaned or "unknown"


def _fmt_float(value, digits: int = 2, default: str = "n/a") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return "inf" if number > 0 else default
    return f"{number:.{digits}f}"


def _fmt_pct(value, digits: int = 2) -> str:
    return f"{_fmt_float(value, digits)}%"


def _dt(ts: float) -> datetime:
    try:
        ts = float(ts or 0.0)
    except (TypeError, ValueError):
        ts = 0.0
    if ts <= 0:
        return datetime.now(timezone.utc)
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _score(trade, key: str, default="unknown"):
    scores = getattr(trade, "scores", {}) or {}
    return scores.get(key, default)


def _bucket_lines(title: str, rows: list[dict], *, limit: int) -> list[str]:
    lines = [f"## {title}"]
    if not rows:
        lines.append("- Not enough sample yet.")
        return lines
    for row in rows[:limit]:
        pf = row.get("profit_factor", 0.0)
        pf_text = "inf" if pf == float("inf") else _fmt_float(pf, 2)
        lines.append(
            "- "
            f"{row.get('key', 'unknown')}: "
            f"{row.get('trades', 0)} trades, "
            f"WR {_fmt_pct(float(row.get('win_rate', 0.0) or 0.0) * 100, 1)}, "
            f"PF {pf_text}, "
            f"net ${_fmt_float(row.get('net_pnl_usd', 0.0), 2)}"
        )
    return lines


def build_trading_brain_report(
    trade_log: Iterable,
    *,
    min_trades: int = 2,
    top_n: int = 5,
) -> dict:
    trades = list(trade_log or [])
    n = len(trades)
    wins = [t for t in trades if float(getattr(t, "pnl_usd", 0.0) or 0.0) > 0]
    losses = [t for t in trades if float(getattr(t, "pnl_usd", 0.0) or 0.0) <= 0]
    gross_profit = sum(float(getattr(t, "pnl_usd", 0.0) or 0.0) for t in wins)
    gross_loss = abs(sum(float(getattr(t, "pnl_usd", 0.0) or 0.0) for t in losses))
    net = gross_profit - gross_loss
    attribution = build_attribution_report(trades, min_trades=min_trades) if trades else {}

    watch_sections = [
        "by_sleeve",
        "by_setup_type",
        "by_sector",
        "by_market_context",
        "by_regime_side",
        "by_exit_profile",
    ]
    strengths: list[dict] = []
    weaknesses: list[dict] = []
    for section in watch_sections:
        for row in attribution.get(section, []) or []:
            enriched = dict(row)
            enriched["section"] = section
            if float(row.get("net_pnl_usd", 0.0) or 0.0) > 0:
                strengths.append(enriched)
            elif int(row.get("trades", 0) or 0) >= min_trades:
                weaknesses.append(enriched)

    strengths.sort(key=lambda r: (r.get("net_pnl_usd", 0.0), r.get("profit_factor", 0.0)), reverse=True)
    weaknesses.sort(key=lambda r: (r.get("net_pnl_usd", 0.0), -r.get("trades", 0)))

    return {
        "schema": "trading_brain_report_v1",
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / n) if n else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_pnl": net,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (float("inf") if wins else 0.0),
        "sample_ready": n >= min_trades,
        "min_trades": min_trades,
        "top_strengths": strengths[:top_n],
        "top_weaknesses": weaknesses[:top_n],
        "attribution": attribution,
    }


def format_edge_report_markdown(report: dict, *, title: str = "Trading Brain Edge Report", top_n: int = 5) -> str:
    pf = report.get("profit_factor", 0.0)
    pf_text = "inf" if pf == float("inf") else _fmt_float(pf, 2)
    lines = [
        f"# {title}",
        "",
        "## Summary",
        f"- Trades: {report.get('trades', 0)}",
        f"- Win rate: {_fmt_pct(float(report.get('win_rate', 0.0) or 0.0) * 100, 1)}",
        f"- Profit factor: {pf_text}",
        f"- Net PnL: ${_fmt_float(report.get('net_pnl', 0.0), 2)}",
        f"- Sample ready: {'yes' if report.get('sample_ready') else 'no'}",
        "",
    ]

    lines.extend(_bucket_lines("Strongest Evidence", report.get("top_strengths", []), limit=top_n))
    lines.append("")
    lines.extend(_bucket_lines("Weakest Evidence", report.get("top_weaknesses", []), limit=top_n))
    lines.append("")

    attribution = report.get("attribution", {}) or {}
    for label, key in (
        ("Sleeves", "by_sleeve"),
        ("Setups", "by_setup_type"),
        ("Sectors", "by_sector"),
        ("Market Context", "by_market_context"),
        ("Regime And Side", "by_regime_side"),
    ):
        lines.extend(_bucket_lines(label, attribution.get(key, []), limit=top_n))
        lines.append("")

    lines.append("## Operating Note")
    lines.append("- This report is diagnostic only. It must not open, block, resize, or close trades.")
    return "\n".join(lines).rstrip() + "\n"


def format_closed_trade_markdown(trade, *, trade_count: int = 0) -> str:
    opened = _dt(getattr(trade, "opened_at", 0.0))
    closed = _dt(getattr(trade, "closed_at", 0.0))
    scores = getattr(trade, "scores", {}) or {}
    passport = scores.get("setup_passport") if isinstance(scores.get("setup_passport"), dict) else {}

    raw_payload = asdict(trade) if hasattr(trade, "__dataclass_fields__") else scores

    lines = [
        f"# {getattr(trade, 'symbol', 'unknown')} {str(getattr(trade, 'direction', 'unknown')).upper()} Closed",
        "",
        "## Outcome",
        f"- Closed trades seen: {trade_count}",
        f"- Opened UTC: {opened.isoformat()}",
        f"- Closed UTC: {closed.isoformat()}",
        f"- Entry: {_fmt_float(getattr(trade, 'entry_price', None), 8)}",
        f"- Exit: {_fmt_float(getattr(trade, 'exit_price', None), 8)}",
        f"- PnL USD: ${_fmt_float(getattr(trade, 'pnl_usd', 0.0), 4)}",
        f"- PnL pct: {_fmt_pct(getattr(trade, 'pnl_pct', 0.0), 4)}",
        f"- Exit reason: {getattr(trade, 'reason', 'unknown')}",
        f"- Hold seconds: {_fmt_float(getattr(trade, 'hold_duration_s', 0.0), 1)}",
        "",
        "## Setup Snapshot",
        f"- Sleeve: {getattr(trade, 'strategy_sleeve', '') or _score(trade, 'strategy_sleeve')}",
        f"- Setup type: {_score(trade, 'setup_type')}",
        f"- Regime: {getattr(trade, 'regime', '') or 'unknown'}",
        f"- Direction: {getattr(trade, 'direction', 'unknown')}",
        f"- Sector: {getattr(trade, 'sector', '') or _score(trade, 'sector')}",
        f"- Session: {getattr(trade, 'session', '') or _score(trade, 'session')}",
        f"- Score: {_fmt_float(_score(trade, 'score', _score(trade, 'total_score', 0.0)), 2)}",
        f"- Cohort: {getattr(trade, 'cohort_key', '') or _score(trade, 'cohort_key', '')}",
        f"- Lifecycle: {getattr(trade, 'lifecycle_status', '') or _score(trade, 'lifecycle_status', '')}",
        "",
        "## Market Context",
        f"- Risk-on state: {getattr(trade, 'market_risk_on_state', '') or _score(trade, 'market_risk_on_state')}",
        f"- Rotation state: {getattr(trade, 'market_rotation_state', '') or _score(trade, 'market_rotation_state')}",
        f"- BTC trend: {getattr(trade, 'market_btc_trend', '') or _score(trade, 'market_btc_trend')}",
        f"- ETH/BTC trend: {getattr(trade, 'market_eth_btc_trend', '') or _score(trade, 'market_eth_btc_trend')}",
        f"- Context confidence: {_fmt_float(getattr(trade, 'market_context_confidence', 0.0), 2)}",
        f"- Sector rotation: {_score(trade, 'sector_rotation_state', passport.get('sector_rotation_state', 'unknown'))}",
        "",
        "## Trade Dynamics",
        f"- MFE R: {_fmt_float(getattr(trade, 'mfe_r', 0.0), 2)}",
        f"- MAE R: {_fmt_float(getattr(trade, 'mae_r', 0.0), 2)}",
        f"- TP1 hit: {bool(getattr(trade, 'tp1_hit', False))}",
        f"- Fees USD: ${_fmt_float(getattr(trade, 'fees_usd', 0.0), 4)}",
        f"- Slippage estimate USD: ${_fmt_float(getattr(trade, 'slippage_estimate_usd', 0.0), 4)}",
        "",
        "## Raw Scores",
        "```json",
        json.dumps(raw_payload, indent=2, sort_keys=True, default=str),
        "```",
    ]
    return "\n".join(lines) + "\n"


class TradingBrainJournal:
    """Writes local diagnostic trading-brain artifacts from closed trades."""

    def __init__(self, cfg: dict):
        tb_cfg = cfg.get("trading_brain", {}) or {}
        self.enabled = bool(tb_cfg.get("enabled", True))
        self.closed_trade_notes_enabled = bool(tb_cfg.get("closed_trade_notes_enabled", True))
        self.edge_report_enabled = bool(tb_cfg.get("edge_report_enabled", True))
        self.min_trades = int(tb_cfg.get("min_trades", 2) or 2)
        self.top_n = int(tb_cfg.get("top_n", 5) or 5)
        self.output_dir = Path(tb_cfg.get("output_dir", "data/trading_brain"))
        self.closed_dir = self.output_dir / "trades" / "closed"
        self.analysis_dir = self.output_dir / "analysis"

    def record_closed_trade(self, trade, trade_log: Iterable) -> dict:
        if not self.enabled:
            return {"enabled": False}

        trades = list(trade_log or [])
        self.closed_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        if self.closed_trade_notes_enabled:
            closed = _dt(getattr(trade, "closed_at", 0.0))
            stamp = closed.strftime("%Y-%m-%d-%H%M%S")
            symbol = _safe_filename(getattr(trade, "symbol", "unknown").replace("/", "-").replace(":", "-"))
            direction = _safe_filename(getattr(trade, "direction", "unknown"))
            path = self.closed_dir / f"{stamp}-{symbol}-{direction}-closed.md"
            path.write_text(format_closed_trade_markdown(trade, trade_count=len(trades)), encoding="utf-8")

        report = build_trading_brain_report(trades, min_trades=self.min_trades, top_n=self.top_n)
        if self.edge_report_enabled:
            text = format_edge_report_markdown(report, top_n=self.top_n)
            (self.analysis_dir / "latest-edge-report.md").write_text(text, encoding="utf-8")
        return report
