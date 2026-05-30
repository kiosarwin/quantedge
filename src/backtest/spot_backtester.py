"""
Spot Backtester — walk-forward simulation of the Ninja Spot strategy.

Simulates:
  - Multi-timeframe scoring on historical OHLCV
  - Regime detection gate
  - Entry strategy detection
  - Partial TP exits (TP1/TP2/TP3)
  - Trailing stop after TP1
  - ATR-based stop-loss
  - Commission + slippage

Usage:
    cd ninja_trader
    python -m src.backtest.run_spot_backtest
    python -m src.backtest.run_spot_backtest --symbols BTCUSDT ETHUSDT SOLUSDT
    python -m src.backtest.run_spot_backtest --start 2024-01-01 --end 2024-12-31
"""
from __future__ import annotations

import asyncio
import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich import box

from src.analysis.indicators import atr, is_extreme_volatility
from src.analysis.structure import trade_direction_from_structure
from src.analysis.regime import classify_regime, Regime
from src.analysis.entry_strategies import detect_entry, EntryStrategy
from src.scoring.spot_scorer import SpotScorer
from src.risk.spot_risk_manager import SpotRiskManager, SpotTradeSetup
from src.data.spot_market_data import SpotSnapshot, _ohlcv_to_df
from src.data.spot_client import BinanceSpotClient

console = Console()
log = logging.getLogger(__name__)

_MIN_BARS = 220  # warmup period for indicators


# ──────────────────────────────────────────────────────────────────────────────
#  Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestPosition:
    symbol: str
    entry_bar: int
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    original_qty: float
    remaining_qty: float
    cost_basis: float
    size_usdt: float
    entry_strategy: str = "none"
    tp1_hit: bool = False
    tp2_hit: bool = False
    trailing_stop: Optional[float] = None
    realized_pnl: float = 0.0
    peak_price: float = 0.0

    # Filled on close
    exit_bar: int = 0
    exit_price: float = 0.0
    exit_reason: str = ""
    total_pnl_usdt: float = 0.0
    total_pnl_pct: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.exit_bar == 0


@dataclass
class BacktestResult:
    symbol: str
    trades: list[BacktestPosition] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    initial_capital: float = 10000.0

    @property
    def closed_trades(self) -> list[BacktestPosition]:
        return [t for t in self.trades if not t.is_open]

    @property
    def wins(self) -> list[BacktestPosition]:
        return [t for t in self.closed_trades if t.total_pnl_usdt > 0]

    @property
    def losses(self) -> list[BacktestPosition]:
        return [t for t in self.closed_trades if t.total_pnl_usdt <= 0]

    @property
    def win_rate(self) -> float:
        n = len(self.closed_trades)
        return len(self.wins) / n * 100 if n else 0.0

    @property
    def total_pnl(self) -> float:
        return sum(t.total_pnl_usdt for t in self.closed_trades)

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.total_pnl_usdt for t in self.wins)
        gross_loss = abs(sum(t.total_pnl_usdt for t in self.losses))
        return gross_win / gross_loss if gross_loss > 0 else float("inf")

    @property
    def avg_win_pct(self) -> float:
        return sum(t.total_pnl_pct for t in self.wins) / len(self.wins) if self.wins else 0.0

    @property
    def avg_loss_pct(self) -> float:
        return sum(t.total_pnl_pct for t in self.losses) / len(self.losses) if self.losses else 0.0

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        peak = self.initial_capital
        max_dd = 0.0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1] if self.equity_curve else self.initial_capital

    @property
    def return_pct(self) -> float:
        return (self.final_equity - self.initial_capital) / self.initial_capital * 100

    @property
    def expectancy_pct(self) -> float:
        """Average PnL per trade as % of trade size."""
        n = len(self.closed_trades)
        if not n:
            return 0.0
        return sum(t.total_pnl_pct for t in self.closed_trades) / n


# ──────────────────────────────────────────────────────────────────────────────
#  Engine
# ──────────────────────────────────────────────────────────────────────────────

class SpotBacktestEngine:
    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._bt = cfg["backtest"]
        self._exit = cfg["exit"]
        self._scorer = SpotScorer(cfg)
        self._risk = SpotRiskManager(cfg)
        self._commission_pct = self._bt.get("commission_pct", 0.10) / 100
        self._slippage_pct = self._bt.get("slippage_pct", 0.05) / 100
        self._threshold = cfg["trading"]["min_score_threshold"]
        self._trail_pct = self._exit.get("trailing_stop_pct", 0.05)
        self._max_hold_bars: int = int(self._exit.get("max_hold_hours", 120) * 4)  # 4H bars

    def run(
        self,
        symbol: str,
        candles: dict[str, pd.DataFrame],
        initial_capital: float | None = None,
    ) -> BacktestResult:
        capital = initial_capital or self._bt["initial_capital"]
        self._risk.update_equity(capital)
        self._risk.state.peak_equity = capital
        self._risk.state.daily_start_equity = capital

        tf_cfg = self._cfg["timeframes"]
        primary_tf = tf_cfg["primary"]    # 4H
        entry_tf = tf_cfg["entry"]        # 15m

        df = candles.get(primary_tf)
        df_entry = candles.get(entry_tf, pd.DataFrame())

        if df is None or df.empty:
            log.warning("[%s] No primary candles for backtest", symbol)
            return BacktestResult(symbol=symbol, initial_capital=capital)

        result = BacktestResult(symbol=symbol, initial_capital=capital)
        result.equity_curve.append(capital)
        equity = capital

        open_pos: BacktestPosition | None = None

        for i in range(_MIN_BARS, len(df)):
            bar = df.iloc[i]
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])

            df_slice = df.iloc[:i]

            # ── Manage open position ──────────────────────────────────
            if open_pos is not None:
                open_pos, equity = self._check_exits(open_pos, bar_high, bar_low, bar_close, i, equity)
                if not open_pos.is_open:
                    commission = abs(open_pos.total_pnl_usdt) * self._commission_pct
                    open_pos.total_pnl_usdt -= commission
                    equity += open_pos.total_pnl_usdt
                    equity = max(0.0, equity)
                    self._risk.on_trade_closed(open_pos.total_pnl_pct)
                    result.trades.append(open_pos)
                    result.equity_curve.append(equity)
                    open_pos = None
                    continue

            # ── Entry scan ────────────────────────────────────────────
            if open_pos is None and self._risk.can_open_trade()[0]:
                snap = self._make_fake_snapshot(symbol, df_slice, candles, i, primary_tf, entry_tf)
                bd = self._scorer.score(snap, btc_df=None)  # no BTC data in backtest

                if bd and bd.is_tradeable and bd.total_score >= self._threshold:
                    # Check extreme volatility
                    safety = self._cfg.get("safety", {})
                    if safety.get("pause_on_extreme_volatility", True):
                        mult = safety.get("extreme_vol_atr_multiplier", 2.5)
                        if is_extreme_volatility(df_slice, self._cfg["indicators"]["atr_period"], mult):
                            result.equity_curve.append(equity)
                            continue

                    # Entry with slippage
                    entry_price = bar_close * (1 + self._slippage_pct)
                    self._risk.update_equity(equity)

                    # Detect entry strategy from entry TF slice
                    entry_signal = None
                    if "entry" in tf_cfg and not df_entry.empty:
                        df_entry_slice = df_entry.iloc[:i] if len(df_entry) > i else df_entry
                        if len(df_entry_slice) >= 50:
                            entry_signal = detect_entry(df_entry_slice, self._cfg)

                    sl_override = entry_signal.stop_loss if (entry_signal and entry_signal.is_valid) else None
                    strategy_name = entry_signal.strategy.value if (entry_signal and entry_signal.is_valid) else "score_only"

                    setup = self._risk.calculate_setup(
                        symbol=symbol,
                        df=df_slice,
                        entry_price=entry_price,
                        strategy_sl=sl_override,
                        entry_strategy=strategy_name,
                    )
                    if setup:
                        commission_entry = setup.size_usdt * self._commission_pct
                        equity -= commission_entry
                        open_pos = BacktestPosition(
                            symbol=symbol,
                            entry_bar=i,
                            entry_price=entry_price,
                            stop_loss=setup.stop_loss,
                            tp1=setup.tp1,
                            tp2=setup.tp2,
                            tp3=setup.tp3,
                            original_qty=setup.base_qty,
                            remaining_qty=setup.base_qty,
                            cost_basis=entry_price,
                            size_usdt=setup.size_usdt,
                            entry_strategy=setup.entry_strategy,
                            peak_price=entry_price,
                        )
                        self._risk.on_trade_opened()

            result.equity_curve.append(equity)

        # Force-close open position at last bar
        if open_pos is not None:
            last_price = float(df.iloc[-1]["close"])
            open_pos.exit_price = last_price
            open_pos.exit_reason = "end_of_data"
            open_pos.exit_bar = len(df) - 1
            close_pnl = (last_price - open_pos.cost_basis) * open_pos.remaining_qty
            total_pnl = open_pos.realized_pnl + close_pnl
            commission = abs(total_pnl) * self._commission_pct
            total_pnl -= commission
            open_pos.total_pnl_usdt = total_pnl
            open_pos.total_pnl_pct = total_pnl / open_pos.size_usdt * 100 if open_pos.size_usdt else 0.0
            equity += total_pnl
            result.trades.append(open_pos)
            result.equity_curve.append(equity)

        return result

    # ------------------------------------------------------------------ #
    #  Bar-level exit simulation                                          #
    # ------------------------------------------------------------------ #

    def _check_exits(
        self,
        pos: BacktestPosition,
        high: float,
        low: float,
        close: float,
        bar_idx: int,
        equity: float,
    ) -> tuple[BacktestPosition, float]:
        """Check exit conditions for a bar. Returns (position, equity)."""

        tp1_pct = self._exit.get("tp1_size_pct", 0.40)
        tp2_pct = self._exit.get("tp2_size_pct", 0.35)

        # Update peak for trailing
        if high > pos.peak_price:
            pos.peak_price = high

        # ── TP1 ──────────────────────────────────────────────────────
        if not pos.tp1_hit and high >= pos.tp1:
            sell_qty = pos.remaining_qty * tp1_pct
            tp1_pnl = (pos.tp1 - pos.cost_basis) * sell_qty
            pos.realized_pnl += tp1_pnl
            equity += tp1_pnl
            pos.remaining_qty -= sell_qty
            pos.remaining_qty = max(0.0, pos.remaining_qty)
            pos.tp1_hit = True
            pos.stop_loss = pos.entry_price    # breakeven
            pos.trailing_stop = pos.tp1 * (1 - self._trail_pct)

        # ── TP2 ──────────────────────────────────────────────────────
        if pos.tp1_hit and not pos.tp2_hit and high >= pos.tp2:
            sell_qty = pos.remaining_qty * tp2_pct
            tp2_pnl = (pos.tp2 - pos.cost_basis) * sell_qty
            pos.realized_pnl += tp2_pnl
            equity += tp2_pnl
            pos.remaining_qty -= sell_qty
            pos.remaining_qty = max(0.0, pos.remaining_qty)
            pos.tp2_hit = True

        # ── Update trailing stop ──────────────────────────────────────
        if pos.tp1_hit and pos.trailing_stop is not None:
            new_trail = close * (1 - self._trail_pct)
            if new_trail > pos.trailing_stop:
                pos.trailing_stop = new_trail

        # ── TP3 (full close) ─────────────────────────────────────────
        if pos.tp2_hit and high >= pos.tp3:
            return self._close_bar(pos, pos.tp3, bar_idx, "tp3", equity)

        # ── SL hit ───────────────────────────────────────────────────
        if low <= pos.stop_loss:
            return self._close_bar(pos, pos.stop_loss, bar_idx, "stop_loss", equity)

        # ── Trailing stop hit ─────────────────────────────────────────
        if pos.trailing_stop is not None and low <= pos.trailing_stop:
            return self._close_bar(pos, pos.trailing_stop, bar_idx, "trailing_stop", equity)

        # ── Timeout ──────────────────────────────────────────────────
        if bar_idx - pos.entry_bar >= self._max_hold_bars:
            return self._close_bar(pos, close, bar_idx, "timeout", equity)

        return pos, equity

    def _close_bar(
        self,
        pos: BacktestPosition,
        exit_price: float,
        bar_idx: int,
        reason: str,
        equity: float,
    ) -> tuple[BacktestPosition, float]:
        close_pnl = (exit_price - pos.cost_basis) * pos.remaining_qty
        total_pnl = pos.realized_pnl + close_pnl
        pos.exit_price = exit_price
        pos.exit_reason = reason
        pos.exit_bar = bar_idx
        pos.total_pnl_usdt = total_pnl
        pos.total_pnl_pct = total_pnl / pos.size_usdt * 100 if pos.size_usdt else 0.0
        pos.remaining_qty = 0.0
        return pos, equity

    # ------------------------------------------------------------------ #
    #  Fake snapshot for scoring                                          #
    # ------------------------------------------------------------------ #

    def _make_fake_snapshot(
        self,
        symbol: str,
        df_slice: pd.DataFrame,
        all_candles: dict[str, pd.DataFrame],
        bar_idx: int,
        primary_tf: str,
        entry_tf: str,
    ) -> SpotSnapshot:
        snap = SpotSnapshot(symbol=symbol)
        snap.candles[primary_tf] = df_slice
        snap.last_price = float(df_slice["close"].iloc[-1])

        for tf, df_all in all_candles.items():
            if tf == primary_tf:
                continue
            if len(df_all) > bar_idx:
                snap.candles[tf] = df_all.iloc[:bar_idx]
            else:
                snap.candles[tf] = df_all
        return snap


# ──────────────────────────────────────────────────────────────────────────────
#  Data fetcher
# ──────────────────────────────────────────────────────────────────────────────

async def fetch_historical_spot(
    symbols: list[str],
    timeframes: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Download historical Spot OHLCV from Binance via ccxt."""
    import ccxt.async_support as ccxt
    from datetime import datetime, timezone

    exchange = ccxt.binance({
        "options": {"defaultType": "spot"},
        "enableRateLimit": True,
    })
    await exchange.load_markets()

    since_ms = int(datetime.strptime(start_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    ).timestamp() * 1000)
    until_ms = int(datetime.strptime(end_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    ).timestamp() * 1000)

    tf_ms = {
        "1m": 60_000, "5m": 300_000, "15m": 900_000,
        "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000,
        "1d": 86_400_000,
    }

    result: dict[str, dict[str, pd.DataFrame]] = {}

    for symbol in symbols:
        result[symbol] = {}
        for tf in timeframes:
            all_ohlcv = []
            fetch_since = since_ms
            bar_ms = tf_ms.get(tf, 3_600_000)
            console.print(f"  Downloading {symbol} {tf}...", end="")

            while fetch_since < until_ms:
                try:
                    ohlcv = await exchange.fetch_ohlcv(
                        symbol, tf, since=fetch_since, limit=1000
                    )
                    if not ohlcv:
                        break
                    all_ohlcv.extend(ohlcv)
                    fetch_since = ohlcv[-1][0] + bar_ms
                    await asyncio.sleep(0.3)
                except Exception as exc:
                    log.warning("OHLCV fetch error %s %s: %s", symbol, tf, exc)
                    break

            if all_ohlcv:
                df = _ohlcv_to_df(all_ohlcv)
                df = df[df.index <= pd.Timestamp(end_date, tz="UTC")]
                result[symbol][tf] = df
                console.print(f" {len(df)} bars")
            else:
                console.print(" [red]no data[/red]")

    await exchange.close()
    return result


# ──────────────────────────────────────────────────────────────────────────────
#  Reporter
# ──────────────────────────────────────────────────────────────────────────────

def print_trade_log(result: BacktestResult, max_rows: int = 30) -> None:
    trades = result.closed_trades[-max_rows:]
    tbl = Table(title=f"{result.symbol} — Trade Log (last {len(trades)})", box=box.SIMPLE)
    tbl.add_column("#", justify="right", style="dim")
    tbl.add_column("Strategy")
    tbl.add_column("Entry", justify="right")
    tbl.add_column("Exit", justify="right")
    tbl.add_column("SL", justify="right")
    tbl.add_column("TP1", justify="right")
    tbl.add_column("PnL%", justify="right")
    tbl.add_column("PnL$", justify="right")
    tbl.add_column("Exit Reason")

    for i, t in enumerate(trades, 1):
        c = "green" if t.total_pnl_usdt > 0 else "red"
        tbl.add_row(
            str(i),
            t.entry_strategy[:12],
            f"{t.entry_price:.4f}",
            f"{t.exit_price:.4f}",
            f"{t.stop_loss:.4f}",
            f"{t.tp1:.4f}",
            f"[{c}]{t.total_pnl_pct:+.1f}%[/]",
            f"[{c}]{t.total_pnl_usdt:+.2f}[/]",
            t.exit_reason,
        )
    console.print(tbl)


def print_summary(results: list[BacktestResult]) -> None:
    tbl = Table(title="Spot Backtest Summary", box=box.ROUNDED)
    tbl.add_column("Symbol", style="cyan")
    tbl.add_column("Trades", justify="right")
    tbl.add_column("Win%", justify="right")
    tbl.add_column("PF", justify="right")
    tbl.add_column("Avg W%", justify="right")
    tbl.add_column("Avg L%", justify="right")
    tbl.add_column("Return%", justify="right")
    tbl.add_column("Max DD%", justify="right")
    tbl.add_column("Expect%", justify="right")

    for r in results:
        n = len(r.closed_trades)
        wr = r.win_rate
        pf = r.profit_factor
        ret = r.return_pct
        dd = r.max_drawdown
        tbl.add_row(
            r.symbol,
            str(n),
            f"[{'green' if wr >= 50 else 'red'}]{wr:.1f}%[/]",
            f"[{'green' if pf >= 1.5 else 'yellow' if pf >= 1.0 else 'red'}]{pf:.2f}[/]",
            f"[green]+{r.avg_win_pct:.1f}%[/]",
            f"[red]{r.avg_loss_pct:.1f}%[/]",
            f"[{'green' if ret > 0 else 'red'}]{ret:+.1f}%[/]",
            f"[{'red' if dd > 15 else 'yellow' if dd > 8 else 'green'}]{dd:.1f}%[/]",
            f"{r.expectancy_pct:+.2f}%",
        )
    console.print(tbl)

    # Aggregate stats
    total_trades = sum(len(r.closed_trades) for r in results)
    total_wins = sum(len(r.wins) for r in results)
    total_pnl = sum(r.total_pnl for r in results)
    all_wins = [t for r in results for t in r.wins]
    all_losses = [t for r in results for t in r.losses]
    gross_w = sum(t.total_pnl_usdt for t in all_wins)
    gross_l = abs(sum(t.total_pnl_usdt for t in all_losses))
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    wr = total_wins / total_trades * 100 if total_trades else 0

    exit_counts: dict[str, int] = {}
    for r in results:
        for t in r.closed_trades:
            exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    strategy_counts: dict[str, int] = {}
    for r in results:
        for t in r.closed_trades:
            strategy_counts[t.entry_strategy] = strategy_counts.get(t.entry_strategy, 0) + 1

    lines = [
        f"Total trades:  [bold]{total_trades}[/bold]  wins={total_wins}  losses={total_trades - total_wins}",
        f"Win rate:      [bold]{'green' if wr >= 50 else 'red'}]{wr:.1f}%[/][/bold]",
        f"Profit factor: [bold]{pf:.2f}[/bold]",
        f"Total PnL:     [{'green' if total_pnl > 0 else 'red'}]{total_pnl:+.2f} USDT[/]",
        "",
        "Exit breakdown: " + "  ".join(f"{k}={v}" for k, v in sorted(exit_counts.items())),
        "Entry strategies: " + "  ".join(f"{k}={v}" for k, v in sorted(strategy_counts.items())),
    ]
    console.print(Panel("\n".join(lines), title="Aggregate Results", border_style="cyan"))


# ──────────────────────────────────────────────────────────────────────────────
#  Runner
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "AVAX/USDT"]


async def run(args: argparse.Namespace) -> None:
    from pathlib import Path
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    tf_cfg = cfg["timeframes"]
    timeframes = list({
        tf_cfg["primary"],
        tf_cfg.get("secondary", tf_cfg["primary"]),
        tf_cfg["entry"],
        tf_cfg.get("higher", tf_cfg["primary"]),
    })

    bt = cfg["backtest"]
    start = args.start or bt["start_date"]
    end = args.end or bt["end_date"]
    capital = args.capital or bt["initial_capital"]

    raw_symbols = args.symbols or _DEFAULT_SYMBOLS
    symbols = []
    for s in raw_symbols:
        if "/" not in s:
            base = s.replace("USDT", "")
            s = f"{base}/USDT"
        symbols.append(s)

    console.rule("[bold cyan]Ninja Spot Trader — Backtest[/bold cyan]")
    console.print(f"Period  : [bold]{start}[/bold] → [bold]{end}[/bold]")
    console.print(f"Capital : [bold]${capital:,.0f} USDT[/bold]")
    console.print(f"Symbols : {', '.join(symbols)}")
    console.print(f"TFs     : {', '.join(timeframes)}")
    console.rule()

    console.print("\n[yellow]Downloading historical Spot data...[/yellow]")
    all_candles = await fetch_historical_spot(symbols, timeframes, start, end)

    engine = SpotBacktestEngine(cfg)
    results = []

    for sym in symbols:
        candles = all_candles.get(sym, {})
        primary_df = candles.get(tf_cfg["primary"])
        if primary_df is None or primary_df.empty:
            console.print(f"[red]No data for {sym} — skipping[/red]")
            continue

        console.print(
            f"\n[cyan]Backtesting {sym}[/cyan]  "
            f"{len(primary_df)} {tf_cfg['primary']} bars"
        )
        result = engine.run(sym, candles, initial_capital=float(capital))
        results.append(result)

        if result.closed_trades:
            print_trade_log(result)
        else:
            console.print(f"  [dim]No trades taken for {sym}[/dim]")

    if results:
        console.rule("[bold]Summary[/bold]")
        print_summary(results)
    else:
        console.print("[red]No results.[/red]")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=False)],
    )
    parser = argparse.ArgumentParser(description="Ninja Spot Backtester")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent.parent.parent / "config" / "config_spot.yaml"),
    )
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--capital", type=float, default=None)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
