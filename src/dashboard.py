"""
Live dashboard — reads data/state.json written by the bot each tick.

Usage:
    python -m src.dashboard
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

STATE_FILE = Path("data/state.json")
REFRESH_RATE = 2  # seconds


def _age(ts: float) -> str:
    s = int(time.time() - ts)
    if s < 60:
        return f"{s}s ago"
    return f"{s // 60}m {s % 60}s ago"


def _pf_str(pf: float) -> str:
    if pf == 0.0:
        return "N/A"
    return "∞" if pf == float("inf") else f"{pf:.2f}"


def _color_score(score: float, threshold: float) -> str:
    if score >= threshold:
        return "green"
    if score >= threshold * 0.9:
        return "yellow"
    return "white"


def build_header(state: dict) -> Panel:
    mode = state.get("mode", "?").upper()
    equity = state.get("equity", 0)
    peak = state.get("peak_equity", equity)
    daily = state.get("daily_pnl_pct", 0.0)
    weekly = state.get("weekly_pnl_pct", 0.0)
    dd = state.get("drawdown_pct", 0.0)
    streak = state.get("consecutive_losses", 0)
    kill_switch = state.get("kill_switch_reason", "")
    updated = state.get("updated_at", 0)
    open_count = len(state.get("open_trades", []))

    mode_color = "green" if mode == "PAPER" else "bold red"
    daily_color = "green" if daily >= 0 else "red"
    dd_color = "green" if dd < 3 else "yellow" if dd < 7 else "red"

    text = Text()
    text.append(f" MODE: ", style="dim")
    text.append(f"{mode} ", style=mode_color)
    text.append(f" │  Equity: ", style="dim")
    text.append(f"${equity:,.2f}", style="bold")
    text.append(f"  Peak: ${peak:,.2f}", style="dim")
    text.append(f"  │  Daily PnL: ", style="dim")
    text.append(f"{daily:+.2f}%", style=daily_color)
    text.append(f"  │  Weekly PnL: ", style="dim")
    text.append(f"{weekly:+.2f}%", style="green" if weekly >= 0 else "red")
    text.append(f"  │  Drawdown: ", style="dim")
    text.append(f"{dd:.1f}%", style=dd_color)
    text.append(f"  │  Open: ", style="dim")
    text.append(f"{open_count}", style="bold cyan")
    text.append(f"  │  Streak Loss: ", style="dim")
    text.append(f"{streak}", style="red" if streak >= 2 else "white")
    if kill_switch:
        text.append(f"  │  Kill: ", style="dim")
        text.append(kill_switch[:24], style="bold red")
    text.append(f"  │  Updated: ", style="dim")
    text.append(f"{_age(updated)}", style="dim")

    return Panel(text, title="[bold cyan]NINJA TRADER[/bold cyan]", border_style="cyan")


def build_open_trades(state: dict) -> Panel:
    trades = state.get("open_trades", [])
    tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    tbl.add_column("Symbol", style="cyan")
    tbl.add_column("Dir", justify="center")
    tbl.add_column("Entry", justify="right")
    tbl.add_column("SL", justify="right")
    tbl.add_column("TP1", justify="right")
    tbl.add_column("TP2", justify="right")
    tbl.add_column("Size $", justify="right")
    tbl.add_column("TP1 Hit", justify="center")
    tbl.add_column("Open", justify="right")

    if not trades:
        tbl.add_row("[dim]No open trades[/dim]", "", "", "", "", "", "", "", "")
    for t in trades:
        color = "green" if t["direction"] == "long" else "red"
        age = _age(t["opened_at"])
        tbl.add_row(
            t["symbol"],
            f"[{color}]{t['direction'].upper()}[/]",
            f"{t['entry']:.4f}",
            f"[red]{t['sl']:.4f}[/]",
            f"{t['tp1']:.4f}",
            f"{t['tp2']:.4f}",
            f"${t['size_usd']:,.0f}",
            "[green]✓[/]" if t["tp1_hit"] else "—",
            f"[dim]{age}[/]",
        )

    return Panel(tbl, title=f"[bold]Open Trades ({len(trades)})[/bold]", border_style="blue")


def build_signals(state: dict) -> Panel:
    signals = state.get("top_signals", [])
    tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    tbl.add_column("Symbol", style="cyan")
    tbl.add_column("Dir", justify="center")
    tbl.add_column("Score", justify="right")
    tbl.add_column("Threshold", justify="right")
    tbl.add_column("Regime", justify="center")
    tbl.add_column("SM Phase", justify="center")
    tbl.add_column("Gates", justify="center")
    tbl.add_column("OI Score", justify="right")
    tbl.add_column("OI Δ%", justify="right")
    tbl.add_column("P(win)", justify="right")
    tbl.add_column("EV%", justify="right")

    if not signals:
        tbl.add_row("[dim]No signals[/dim]", "", "", "", "", "", "", "", "", "", "")
    for s in signals:
        dir_color = "green" if s["direction"] == "long" else "red"
        threshold = s.get("threshold", 63)
        score_color = _color_score(s["score"], threshold)
        regime_map = {
            "trending_expansion": "[green]Trend↑[/]",
            "accumulation_compression": "[blue]Accum[/]",
            "distribution": "[red]Dist[/]",
            "chaos": "[bold red]CHAOS[/]",
        }
        regime_str = regime_map.get(s["regime"], s["regime"][:6])
        gates = s.get("gates", "---")
        gate_color = "green" if gates == "RSE" else "yellow" if gates[0] == "R" else "red"
        ev = s.get("ev", 0.0)
        ev_color = "green" if ev > 0 else "red"
        oi_score = s.get("oi_score", 50.0)
        oi_chg = s.get("oi_change_pct", 0.0)
        oi_score_color = "green" if oi_score > 60 else "red" if oi_score < 40 else "dim"
        oi_chg_color = "green" if oi_chg > 0.5 else "red" if oi_chg < -0.5 else "dim"
        tbl.add_row(
            s["symbol"],
            f"[{dir_color}]{s['direction']}[/]",
            f"[{score_color}]{s['score']:.1f}[/]",
            f"[dim]{threshold:.0f}[/]",
            regime_str,
            s["sm"][:8],
            f"[{gate_color}]{gates}[/]",
            f"[{oi_score_color}]{oi_score:.0f}[/]",
            f"[{oi_chg_color}]{oi_chg:+.2f}%[/]",
            f"{s['pwin']:.0%}",
            f"[{ev_color}]{ev:+.2f}%[/]",
        )

    return Panel(tbl, title="[bold]Top Signals[/bold]", border_style="magenta")


def build_shadow(state: dict) -> Panel:
    s = state.get("shadow", {})
    if not s or s.get("closed", 0) == 0:
        return Panel("[dim]Shadow engine — no closed trades yet[/dim]",
                     title="[bold]Shadow Engine[/bold]", border_style="yellow")

    tbl = Table(box=box.SIMPLE, show_header=False)
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Value", justify="right")

    wr = s.get("winrate", 0)
    pf = s.get("profit_factor", 0)
    pnl = s.get("net_pnl_pct", 0)
    dd = s.get("max_dd", 0)
    t_pf = s.get("trending_pf", 0)
    c_pf = s.get("compression_pf", 0)

    tbl.add_row("Closed / Open", f"{s.get('closed', 0)} / {s.get('open', 0)}")
    tbl.add_row("Win Rate", f"[{'green' if wr >= 0.5 else 'red'}]{wr:.1%}[/]")
    tbl.add_row("Profit Factor", f"[{'green' if pf >= 1.3 else 'red'}]{_pf_str(pf)}[/]")
    tbl.add_row("Net PnL", f"[{'green' if pnl >= 0 else 'red'}]{pnl:+.1f}%[/]")
    tbl.add_row("Max DD", f"[{'red' if dd > 7 else 'yellow' if dd > 4 else 'green'}]{dd:.1f}%[/]")
    tbl.add_row("Trending PF", _pf_str(t_pf))
    tbl.add_row("Compression PF", _pf_str(c_pf))

    return Panel(tbl, title="[bold]Shadow Engine[/bold]", border_style="yellow")


def build_attribution(state: dict) -> Panel:
    attribution = state.get("attribution", {})
    sleeve_rows = attribution.get("by_sleeve", [])
    exit_rows = attribution.get("by_exit_profile", [])
    if not sleeve_rows and not exit_rows:
        return Panel("[dim]Attribution — need more closed trades[/dim]",
                     title="[bold]Strategy Attribution[/bold]", border_style="green")

    tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    tbl.add_column("Bucket", style="cyan")
    tbl.add_column("Trades", justify="right")
    tbl.add_column("WR", justify="right")
    tbl.add_column("PF", justify="right")
    tbl.add_column("Net $", justify="right")

    for row in sleeve_rows[:5]:
        pf = row.get("profit_factor", 0.0)
        pf_str = "∞" if pf == float("inf") else f"{pf:.2f}"
        net = row.get("net_pnl_usd", 0.0)
        wr = row.get("win_rate", 0.0)
        tbl.add_row(
            row.get("key", "unknown"),
            str(row.get("trades", 0)),
            f"[{'green' if wr >= 0.5 else 'red'}]{wr:.0%}[/]",
            f"[{'green' if pf >= 1.2 else 'red'}]{pf_str}[/]",
            f"[{'green' if net >= 0 else 'red'}]{net:+.2f}[/]",
        )
    for row in exit_rows[:3]:
        pf = row.get("profit_factor", 0.0)
        pf_str = "∞" if pf == float("inf") else f"{pf:.2f}"
        net = row.get("net_pnl_usd", 0.0)
        wr = row.get("win_rate", 0.0)
        tbl.add_row(
            f"exit:{row.get('key', 'unknown')}",
            str(row.get("trades", 0)),
            f"[{'green' if wr >= 0.5 else 'red'}]{wr:.0%}[/]",
            f"[{'green' if pf >= 1.2 else 'red'}]{pf_str}[/]",
            f"[{'green' if net >= 0 else 'red'}]{net:+.2f}[/]",
        )
    return Panel(tbl, title="[bold]Strategy Attribution[/bold]", border_style="green")


def build_lifecycle(state: dict) -> Panel:
    lifecycle = state.get("lifecycle", {})
    if not lifecycle:
        return Panel("[dim]Lifecycle — no cohort health yet[/dim]",
                     title="[bold]Lifecycle[/bold]", border_style="red")

    counts = lifecycle.get("counts", {})
    recommendation = lifecycle.get("recommendation", "PAPER_ONLY")
    reco_color = "green" if recommendation == "TRADE" else "yellow" if recommendation == "PAPER_ONLY" else "red"

    tbl = Table(box=box.SIMPLE, show_header=False)
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Value", justify="right")
    tbl.add_row("Recommendation", f"[{reco_color}]{recommendation}[/]")
    tbl.add_row("ACTIVE", str(counts.get("ACTIVE", 0)))
    tbl.add_row("SMALL_LIVE", str(counts.get("SMALL_LIVE", 0)))
    tbl.add_row("PAPER_VALIDATION", str(counts.get("PAPER_VALIDATION", 0)))
    tbl.add_row("RESEARCH", str(counts.get("RESEARCH", 0)))
    tbl.add_row("DEGRADED", f"[yellow]{counts.get('DEGRADED', 0)}[/]")
    tbl.add_row("DISABLED", f"[red]{counts.get('DISABLED', 0)}[/]")
    return Panel(tbl, title="[bold]Lifecycle[/bold]", border_style="red")


def build_layout(state: dict) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(build_header(state), name="header", size=3),
        Layout(name="body"),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=3),
        Layout(name="right", ratio=1),
    )
    layout["right"].split_column(
        Layout(build_lifecycle(state), name="lifecycle"),
        Layout(build_shadow(state), name="shadow"),
        Layout(build_attribution(state), name="attrib"),
    )
    layout["left"].split_column(
        Layout(build_open_trades(state), name="trades", size=10),
        Layout(build_signals(state), name="signals"),
    )
    return layout


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def main() -> None:
    console = Console()
    if not STATE_FILE.exists():
        console.print("[yellow]Waiting for bot to write state file...[/yellow]")

    with Live(console=console, refresh_per_second=1 / REFRESH_RATE, screen=True) as live:
        while True:
            state = load_state()
            if state:
                live.update(build_layout(state))
            else:
                live.update(Panel(
                    "[yellow]Waiting for bot state... is the bot running?[/yellow]",
                    title="NINJA TRADER DASHBOARD",
                ))
            time.sleep(REFRESH_RATE)


if __name__ == "__main__":
    main()
