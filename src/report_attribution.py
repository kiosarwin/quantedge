"""
Standalone strategy attribution report.

Usage:
    python -m src.report_attribution
"""
from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from src.learning.learner import Learner
from src.reporting.attribution import build_attribution_report
from src.main import load_config

console = Console()


def _render_rows(title: str, rows: list[dict], limit: int = 12) -> None:
    tbl = Table(title=title, box=box.ROUNDED)
    tbl.add_column("Key", style="cyan")
    tbl.add_column("Trades", justify="right")
    tbl.add_column("WR", justify="right")
    tbl.add_column("PF", justify="right")
    tbl.add_column("Net $", justify="right")
    tbl.add_column("Avg $", justify="right")
    tbl.add_column("Avg %", justify="right")

    for row in rows[:limit]:
        pf = row["profit_factor"]
        pf_str = "∞" if pf == float("inf") else f"{pf:.2f}"
        wr = row["win_rate"]
        net = row["net_pnl_usd"]
        avg = row["avg_pnl_usd"]
        avg_pct = row["avg_pnl_pct"]
        tbl.add_row(
            row["key"],
            str(row["trades"]),
            f"[{'green' if wr >= 0.5 else 'red'}]{wr:.0%}[/]",
            f"[{'green' if pf >= 1.2 else 'red'}]{pf_str}[/]",
            f"[{'green' if net >= 0 else 'red'}]{net:+.2f}[/]",
            f"[{'green' if avg >= 0 else 'red'}]{avg:+.2f}[/]",
            f"[{'green' if avg_pct >= 0 else 'red'}]{avg_pct:+.2f}%[/]",
        )
    console.print(tbl)


def main() -> None:
    cfg = load_config("config/config.yaml")
    learner = Learner(cfg)
    trade_log = learner._trade_log
    report = build_attribution_report(trade_log, min_trades=2)

    console.print(Panel(
        "\n".join([
            f"Trades: {len(trade_log)}",
            f"Best sleeve: {report['headline'].get('best_sleeve') or 'n/a'}",
            f"Worst sleeve: {report['headline'].get('worst_sleeve') or 'n/a'}",
            f"Best exit: {report['headline'].get('best_exit_profile') or 'n/a'}",
            f"Worst exit: {report['headline'].get('worst_exit_profile') or 'n/a'}",
        ]),
        title="Strategy Attribution",
        border_style="cyan",
    ))

    _render_rows("By Sleeve", report["by_sleeve"])
    _render_rows("By Exit Profile", report["by_exit_profile"])
    _render_rows("By Regime", report["by_regime"])
    _render_rows("By Side", report["by_side"])
    _render_rows("By Regime x Side", report["by_regime_side"])
    _render_rows("By Sleeve x Exit Profile", report["by_sleeve_exit_profile"], limit=20)
    _render_rows("By Exit Profile x Regime x Side", report["by_exit_profile_regime_side"], limit=20)
    _render_rows("By Sleeve x Regime x Side", report["by_sleeve_regime_side"], limit=20)


if __name__ == "__main__":
    main()
