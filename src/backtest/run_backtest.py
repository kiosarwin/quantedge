"""
Backtest runner — entry point.
Usage:
    cd ninja_trader
    python -m src.backtest.run_backtest
    python -m src.backtest.run_backtest --symbols BTCUSDT ETHUSDT --start 2023-01-01 --end 2024-01-01
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.logging import RichHandler

console = Console()

_DEFAULT_SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=False)],
    )


async def run(args: argparse.Namespace) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.backtest.data_loader import fetch_all_symbols
    from src.backtest.engine import BacktestEngine
    from src.backtest.reporter import print_trade_log, print_summary, print_score_calibration

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["exchange"]["api_key"] = ""
    cfg["exchange"]["api_secret"] = ""

    # CLI overrides
    bt = cfg["backtest"]
    start = args.start or bt["start_date"]
    end = args.end or bt["end_date"]
    capital = args.capital or bt["initial_capital"]

    raw_symbols = args.symbols or _DEFAULT_SYMBOLS
    # Normalise to ccxt perpetual format
    symbols = []
    for s in raw_symbols:
        if "/" not in s:
            base = s.replace("USDT", "")
            s = f"{base}/USDT:USDT"
        symbols.append(s)

    tf_cfg = cfg["timeframes"]
    timeframes = list({tf_cfg["primary"], tf_cfg["higher"], tf_cfg["lower"], tf_cfg["entry"]})

    console.rule("[bold cyan]Ninja Trader Backtest[/bold cyan]")
    console.print(f"Period  : [bold]{start}[/bold] → [bold]{end}[/bold]")
    console.print(f"Capital : [bold]{capital} USDT[/bold]")
    console.print(f"Symbols : {', '.join(symbols)}")
    console.print(f"TFs     : {', '.join(timeframes)}")
    console.rule()

    # ── Download data ─────────────────────────────────────────────────────
    console.print("\n[yellow]Fetching historical data...[/yellow]")
    all_candles = await fetch_all_symbols(symbols, timeframes, start, end, testnet=False)

    # ── Run backtest per symbol ───────────────────────────────────────────
    engine = BacktestEngine(cfg)
    results = []

    for sym in symbols:
        candles = all_candles.get(sym, {})
        primary_df = candles.get(tf_cfg["primary"])
        if primary_df is None or primary_df.empty:
            console.print(f"[red]No data for {sym} — skipping[/red]")
            continue

        console.print(f"\n[cyan]Backtesting {sym}[/cyan]  {len(primary_df)} {tf_cfg['primary']} bars")
        result = engine.run(sym, candles, initial_capital=float(capital))
        results.append(result)

        if result.closed_trades:
            print_trade_log(result)
        else:
            console.print(f"  [dim]No trades taken for {sym}[/dim]")

    # ── Print summary ─────────────────────────────────────────────────────
    if results:
        console.rule("[bold]Summary[/bold]")
        print_summary(results)
        console.rule("[bold]Score Calibration[/bold]")
        print_score_calibration(results)
    else:
        console.print("[red]No results to show.[/red]")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Ninja Trader backtester")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent.parent.parent / "config" / "config.yaml"),
    )
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=None)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
