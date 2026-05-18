"""
Backtest results reporter — rich terminal tables and summary stats.
"""
from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from src.backtest.engine import BacktestResult, BacktestTrade

console = Console()


def _color(v: float, good_positive: bool = True) -> str:
    if good_positive:
        return "green" if v > 0 else "red"
    return "red" if v > 0 else "green"


def print_trade_log(result: BacktestResult, max_rows: int = 30) -> None:
    trades = result.closed_trades[-max_rows:]
    tbl = Table(title=f"{result.symbol} — Trade Log (last {len(trades)})", box=box.SIMPLE)
    tbl.add_column("#", justify="right", style="dim")
    tbl.add_column("Dir", style="bold")
    tbl.add_column("Entry", justify="right")
    tbl.add_column("Exit", justify="right")
    tbl.add_column("SL", justify="right")
    tbl.add_column("TP1", justify="right")
    tbl.add_column("TP2", justify="right")
    tbl.add_column("PnL $", justify="right")
    tbl.add_column("MFE R", justify="right")
    tbl.add_column("MAE R", justify="right")
    tbl.add_column("Reason")

    for i, t in enumerate(trades, 1):
        c = "green" if t.pnl_usd > 0 else "red"
        tbl.add_row(
            str(i),
            f"[{'green' if t.direction == 'long' else 'red'}]{t.direction}[/]",
            f"{t.entry_price:.4f}",
            f"{t.exit_price:.4f}",
            f"{t.stop_loss:.4f}",
            f"{t.tp1:.4f}",
            f"{t.tp2:.4f}",
            f"[{c}]{t.pnl_usd:+.2f}[/]",
            f"[cyan]{t.mfe_r:.2f}R[/]",
            f"[yellow]{t.mae_r:.2f}R[/]",
            t.exit_reason,
        )
    console.print(tbl)


def print_score_calibration(results: list[BacktestResult]) -> None:
    """Show win rate and PF broken down by entry score bucket."""
    all_trades = [t for r in results for t in r.closed_trades]
    if not all_trades:
        return

    buckets = [(50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 100)]
    tbl = Table(title="Score Calibration — Threshold Analysis", box=box.ROUNDED)
    tbl.add_column("Score Range", style="cyan")
    tbl.add_column("Trades", justify="right")
    tbl.add_column("Win%", justify="right")
    tbl.add_column("PF", justify="right")
    tbl.add_column("Avg PnL", justify="right")
    tbl.add_column("Suggested?", justify="center")

    best_threshold = None
    best_pf = 0.0

    for lo, hi in buckets:
        bucket = [t for t in all_trades if lo <= t.entry_score < hi]
        if not bucket:
            tbl.add_row(f"{lo}–{hi}", "0", "—", "—", "—", "")
            continue
        wins = [t for t in bucket if t.pnl_usd > 0]
        losses = [t for t in bucket if t.pnl_usd <= 0]
        wr = len(wins) / len(bucket) * 100
        gw = sum(t.pnl_usd for t in wins)
        gl = abs(sum(t.pnl_usd for t in losses))
        pf = gw / gl if gl > 0 else float("inf")
        avg_pnl = sum(t.pnl_usd for t in bucket) / len(bucket)
        pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
        pf_color = "green" if pf >= 1.5 else "yellow" if pf >= 1.0 else "red"
        wr_color = "green" if wr >= 50 else "red"

        if pf > best_pf and len(bucket) >= 3:
            best_pf = pf
            best_threshold = lo

        tbl.add_row(
            f"{lo}–{hi}",
            str(len(bucket)),
            f"[{wr_color}]{wr:.1f}%[/]",
            f"[{pf_color}]{pf_str}[/]",
            f"[{'green' if avg_pnl > 0 else 'red'}]{avg_pnl:+.2f}[/]",
            "",
        )

    console.print(tbl)
    if best_threshold is not None:
        console.print(
            f"[bold cyan]Suggested threshold: {best_threshold}[/bold cyan]  "
            f"(best PF={best_pf:.2f} in backtest — validate on out-of-sample data)"
        )


def print_summary(results: list[BacktestResult]) -> None:
    # Per-symbol summary
    tbl = Table(title="Backtest Summary — Per Symbol", box=box.ROUNDED)
    tbl.add_column("Symbol", style="cyan")
    tbl.add_column("Trades", justify="right")
    tbl.add_column("Win%", justify="right")
    tbl.add_column("PF", justify="right")
    tbl.add_column("Avg W", justify="right")
    tbl.add_column("Avg L", justify="right")
    tbl.add_column("Total PnL", justify="right")
    tbl.add_column("Return%", justify="right")
    tbl.add_column("Max DD%", justify="right")

    for r in results:
        n = len(r.closed_trades)
        pf = r.profit_factor
        wr = r.win_rate
        ret = r.return_pct
        dd = r.max_drawdown
        tbl.add_row(
            r.symbol,
            str(n),
            f"[{'green' if wr >= 50 else 'red'}]{wr:.1f}%[/]",
            f"[{'green' if pf >= 1.5 else 'yellow' if pf >= 1.0 else 'red'}]{pf:.2f}[/]",
            f"[green]{r.avg_win:+.2f}[/]",
            f"[red]{r.avg_loss:+.2f}[/]",
            f"[{'green' if r.total_pnl > 0 else 'red'}]{r.total_pnl:+.2f}[/]",
            f"[{'green' if ret > 0 else 'red'}]{ret:+.1f}%[/]",
            f"[{'red' if dd > 15 else 'yellow' if dd > 8 else 'green'}]{dd:.1f}%[/]",
        )
    console.print(tbl)

    # Aggregate
    total_trades = sum(len(r.closed_trades) for r in results)
    total_wins = sum(len(r.wins) for r in results)
    total_pnl = sum(r.total_pnl for r in results)
    all_wins = [t for r in results for t in r.wins]
    all_losses = [t for r in results for t in r.losses]
    gross_w = sum(t.pnl_usd for t in all_wins)
    gross_l = abs(sum(t.pnl_usd for t in all_losses))
    pf = gross_w / gross_l if gross_l > 0 else float("inf")

    avg_initial = sum(r.initial_capital for r in results) / len(results) if results else 10000
    avg_final = sum(r.final_equity for r in results) / len(results) if results else avg_initial

    exit_reason_counts: dict[str, int] = {}
    for r in results:
        for t in r.closed_trades:
            exit_reason_counts[t.exit_reason] = exit_reason_counts.get(t.exit_reason, 0) + 1

    lines = [
        f"Total trades : [bold]{total_trades}[/bold]  (wins: [green]{total_wins}[/green]  losses: [red]{total_trades - total_wins}[/red])",
        (f"Global win%  : [bold][{'green' if total_wins/total_trades*100 >= 50 else 'red'}]{total_wins/total_trades*100:.1f}%[/][/bold]" if total_trades else "Global win%  : —"),
        f"Profit factor: [bold]{pf:.2f}[/bold]",
        f"Total PnL    : [{'green' if total_pnl > 0 else 'red'}]{total_pnl:+.2f} USDT[/]",
        f"Avg equity   : {avg_initial:.0f} → {avg_final:.0f} USDT",
        "",
        "Exit breakdown: " + "  ".join(f"{k}={v}" for k, v in sorted(exit_reason_counts.items())),
    ]
    console.print(Panel("\n".join(lines), title="Aggregate Results", border_style="cyan"))

    # MFE / MAE analysis
    all_trades = [t for r in results for t in r.closed_trades]
    if all_trades:
        wins  = [t for t in all_trades if t.pnl_usd > 0]
        losses = [t for t in all_trades if t.pnl_usd <= 0]

        def _avg(lst, attr):
            return sum(getattr(t, attr) for t in lst) / len(lst) if lst else 0.0

        mfe_tbl = Table(title="MFE / MAE Analysis", box=box.ROUNDED)
        mfe_tbl.add_column("Cohort",  style="cyan")
        mfe_tbl.add_column("Trades",  justify="right")
        mfe_tbl.add_column("Avg MFE", justify="right")
        mfe_tbl.add_column("Avg MAE", justify="right")
        mfe_tbl.add_column("MFE:MAE", justify="right")
        mfe_tbl.add_column("Insight", style="dim")

        for label, cohort in [("All trades", all_trades), ("Winners", wins), ("Losers", losses)]:
            if not cohort:
                continue
            avg_mfe = _avg(cohort, "mfe_r")
            avg_mae = _avg(cohort, "mae_r")
            ratio   = avg_mfe / avg_mae if avg_mae > 0 else 0.0
            insight = ""
            if label == "Losers" and avg_mfe > 1.0:
                insight = "MFE>1R on losers → TPs too tight?"
            elif label == "Winners" and avg_mae > 0.5:
                insight = "High MAE on winners → SL too tight?"
            mfe_tbl.add_row(
                label,
                str(len(cohort)),
                f"[cyan]{avg_mfe:.2f}R[/]",
                f"[yellow]{avg_mae:.2f}R[/]",
                f"{ratio:.1f}x",
                insight,
            )
        console.print(mfe_tbl)
