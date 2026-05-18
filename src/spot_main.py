"""
Ninja Smart Trader — Binance SPOT Edition
Main entry point.

Usage:
    cd ninja_trader
    python -m src.spot_main                   # uses config/config.yaml
    python -m src.spot_main --mode paper
    python -m src.spot_main --mode live
    python -m src.spot_main --config /path/to/config.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from src.data.spot_client import BinanceSpotClient
from src.data.spot_market_data import SpotMarketDataService
from src.scanner.spot_scanner import SpotScanner
from src.scoring.spot_scorer import SpotScorer, SpotSignalBreakdown
from src.risk.spot_risk_manager import SpotRiskManager
from src.execution.spot_executor import SpotExecutor
from src.execution.spot_trade_manager import SpotTradeManager, SpotTrade
from src.analysis.btc_guard import BTCGuard
from src.analysis.entry_strategies import detect_entry, EntryStrategy
from src.learning.learner import Learner, TradeRecord
from src.analysis.regime import Regime

console = Console()
log = logging.getLogger("ninja_trader.spot")


# ──────────────────────────────────────────────────────────────────────────────
#  Setup helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    load_dotenv()
    cfg["exchange"]["api_key"] = os.getenv("BINANCE_API_KEY", cfg["exchange"].get("api_key", ""))
    cfg["exchange"]["api_secret"] = os.getenv("BINANCE_API_SECRET", cfg["exchange"].get("api_secret", ""))
    return cfg


def setup_logging(cfg: dict) -> None:
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    handlers: list[logging.Handler] = [
        RichHandler(console=console, rich_tracebacks=True, show_time=True),
    ]
    log_file = log_cfg.get("log_file")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        from logging.handlers import RotatingFileHandler
        handlers.append(
            RotatingFileHandler(
                log_file,
                maxBytes=log_cfg.get("max_bytes", 10_485_760),
                backupCount=log_cfg.get("backup_count", 5),
            )
        )
    logging.basicConfig(level=level, format="%(message)s", handlers=handlers)


# ──────────────────────────────────────────────────────────────────────────────
#  Bot
# ──────────────────────────────────────────────────────────────────────────────

class NinjaSpotTrader:
    """
    Main trading loop:
      1. BTC guard check
      2. Scan all USDT Spot pairs → filter by volume
      3. Fetch OHLCV snapshots
      4. Score each pair (regime → structure → momentum → volume → BTC)
      5. Run entry strategy detector on top candidates
      6. Open positions on score ≥ 75 setups
      7. Monitor open positions → trigger partial TPs + trailing stop
      8. Self-learning: adjust scoring weights from trade history
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._trading = cfg["trading"]
        self._safety = cfg.get("safety", {})

        self._client = BinanceSpotClient(cfg)
        self._market_data = SpotMarketDataService(self._client, cfg)
        self._scanner = SpotScanner(self._client, cfg)
        self._scorer = SpotScorer(cfg)
        self._risk = SpotRiskManager(cfg)
        self._executor = SpotExecutor(self._client, cfg)
        self._trade_mgr = SpotTradeManager(
            self._client, self._executor, self._risk, cfg,
            on_close=self._on_trade_closed,
        )
        self._btc_guard = BTCGuard(cfg)
        self._learner = Learner(cfg, scorer=None)  # spot scorer differs from futures

        self._running = False
        self._heartbeat_ts = 0.0
        self._pending_scores: dict[str, dict] = {}

    async def start(self) -> None:
        await self._client.connect()

        # Seed equity
        try:
            usdt_free = await self._client.fetch_usdt_balance()
            if usdt_free > 0:
                self._risk.update_equity(usdt_free)
                log.info("Account equity: $%.2f USDT", usdt_free)
            else:
                capital = self._cfg.get("capital", {}).get("total_usdt", 1000.0)
                self._risk.update_equity(capital)
                log.info("Paper equity seeded: $%.2f USDT", capital)
        except Exception as exc:
            capital = self._cfg.get("capital", {}).get("total_usdt", 1000.0)
            log.warning("Could not fetch balance (%s) — using $%.0f", exc, capital)
            self._risk.update_equity(capital)

        self._running = True
        mode = self._trading["mode"].upper()
        console.rule(f"[bold cyan]Ninja Spot Trader — {mode} MODE[/bold cyan]")
        log.info("Bot started. Min score threshold: %d", self._trading["min_score_threshold"])

        try:
            await self._loop()
        finally:
            await self._shutdown()

    async def _loop(self) -> None:
        scan_interval = self._trading["scan_interval_seconds"]

        while self._running:
            try:
                tick_start = time.time()

                # ── Heartbeat ─────────────────────────────────────────
                if time.time() - self._heartbeat_ts > self._safety.get("heartbeat_interval_seconds", 60):
                    await self._heartbeat()

                # ── BTC Safety Gate ───────────────────────────────────
                btc_df = await self._market_data.fetch_btc_candles(
                    self._cfg["timeframes"]["primary"], limit=100
                )
                btc_status = self._btc_guard.evaluate(btc_df)

                if not btc_status.is_safe:
                    console.print(
                        f"[red bold]BTC GUARD ACTIVE[/red bold] — {btc_status.reason}  "
                        f"[dim]No new entries until BTC recovers[/dim]"
                    )

                # ── Scan pairs ────────────────────────────────────────
                pairs = await self._scanner.scan()
                top_n = self._trading.get("top_pairs_to_trade", 2)
                scan_limit = top_n * 15   # score more than we'll trade
                pairs = pairs[:scan_limit]

                if not pairs:
                    log.warning("No pairs found — sleeping")
                    await asyncio.sleep(scan_interval)
                    continue

                # ── Fetch snapshots ───────────────────────────────────
                log.info("Fetching data for %d pairs...", len(pairs))
                snapshots = await self._market_data.fetch_snapshots(pairs)

                # ── Score pairs ───────────────────────────────────────
                breakdowns = self._scorer.score_many(snapshots, btc_df)
                self._display_scores(breakdowns[:10])

                # ── Monitor open trades ───────────────────────────────
                price_map = {sym: snap.last_price for sym, snap in snapshots.items()}
                await self._trade_mgr.monitor_all(price_map)

                # ── Open new trades ───────────────────────────────────
                if btc_status.is_safe:
                    await self._try_open_trades(breakdowns, snapshots, top_n)

                # ── Sleep ─────────────────────────────────────────────
                elapsed = time.time() - tick_start
                sleep_time = max(5.0, scan_interval - elapsed)
                log.debug("Tick %.1fs — sleeping %.0fs", elapsed, sleep_time)
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.exception("Unhandled error in main loop: %s", exc)
                await asyncio.sleep(15)

    async def _try_open_trades(
        self,
        breakdowns: list[SpotSignalBreakdown],
        snapshots: dict,
        top_n: int,
    ) -> None:
        threshold = self._trading["min_score_threshold"]

        eligible = [
            b for b in breakdowns
            if b.is_tradeable
            and b.symbol not in self._trade_mgr.open_symbols
        ][:top_n]

        for bd in eligible:
            snap = snapshots.get(bd.symbol)
            if snap is None:
                continue

            # Extreme volatility guard
            if self._safety.get("pause_on_extreme_volatility", True):
                from src.analysis.indicators import is_extreme_volatility
                df_primary = snap.candles_for(self._cfg["timeframes"]["primary"])
                if not df_primary.empty:
                    mult = self._safety.get("extreme_vol_atr_multiplier", 2.5)
                    if is_extreme_volatility(df_primary, self._cfg["indicators"]["atr_period"], mult):
                        log.warning("[%s] Extreme volatility — skipping", bd.symbol)
                        continue

            # Detect entry strategy on entry timeframe
            df_entry = snap.candles_for(self._cfg["timeframes"]["entry"])
            if df_entry.empty:
                df_entry = snap.candles_for(self._cfg["timeframes"].get("secondary", self._cfg["timeframes"]["primary"]))

            entry_signal = detect_entry(df_entry, self._cfg)
            if not entry_signal.is_valid:
                log.debug("[%s] No valid entry signal — skipping", bd.symbol)
                continue

            # Calculate trade setup
            df_primary = snap.candles_for(self._cfg["timeframes"]["primary"])
            setup = self._risk.calculate_setup(
                symbol=bd.symbol,
                df=df_primary,
                entry_price=snap.last_price,
                strategy_sl=entry_signal.stop_loss if entry_signal.stop_loss > 0 else None,
                entry_strategy=entry_signal.strategy.value,
            )
            if setup is None:
                continue

            # Store scoring snapshot for learning
            self._pending_scores[bd.symbol] = {
                "trend_strength": bd.trend_strength,
                "volume_expansion": bd.volume_expansion,
                "structure_quality": bd.structure_quality,
                "momentum_alignment": bd.momentum_alignment,
                "btc_correlation": bd.btc_correlation,
                "volatility_condition": bd.volatility_condition,
                "total_score": bd.total_score,
                "regime": bd.regime.value,
                "entry_strategy": entry_signal.strategy.value,
                "entry_confidence": entry_signal.confidence,
            }

            console.print(
                f"[green bold]→ ENTRY[/green bold] {bd.symbol}  "
                f"score={bd.total_score:.0f}  "
                f"strategy={entry_signal.strategy.value}  "
                f"confidence={entry_signal.confidence:.2f}  "
                f"entry=${snap.last_price:.6f}  sl=${setup.stop_loss:.6f}  "
                f"R:R={setup.risk_reward:.2f}"
            )

            await self._trade_mgr.open(setup)

    async def _heartbeat(self) -> None:
        self._heartbeat_ts = time.time()
        s = self._risk.state
        try:
            if self._trading["mode"] == "live":
                usdt = await self._client.fetch_usdt_balance()
                self._risk.update_equity(usdt)
                s = self._risk.state
        except Exception as exc:
            log.warning("Heartbeat equity refresh failed: %s", exc)

        log.info(
            "♥ Heartbeat  equity=$%.2f  daily_pnl=%.1f%%  drawdown=%.1f%%  open=%d",
            s.equity, s.daily_pnl_pct, s.drawdown_pct, s.open_trade_count,
        )

    async def _on_trade_closed(
        self, trade: SpotTrade, pnl_usdt: float, reason: str
    ) -> None:
        scores = self._pending_scores.pop(trade.symbol, {})
        pnl_pct = pnl_usdt / trade.setup.size_usdt * 100 if trade.setup.size_usdt else 0.0

        record = TradeRecord(
            symbol=trade.symbol,
            direction="long",
            entry_price=trade.entry_price,
            exit_price=trade.peak_price,   # best available
            pnl_usd=pnl_usdt,
            pnl_pct=pnl_pct,
            reason=reason,
            scores=scores,
            opened_at=trade.opened_at,
            closed_at=time.time(),
        )
        self._learner.record_trade(record)

    async def _shutdown(self) -> None:
        log.info("Shutting down — closing all open trades...")
        await self._trade_mgr.close_all("shutdown")
        await self._client.close()
        console.rule("[yellow]Ninja Spot Trader stopped[/yellow]")

    def _display_scores(self, breakdowns: list[SpotSignalBreakdown]) -> None:
        if not breakdowns:
            return
        tbl = Table(title="Top Spot Signals", show_lines=False)
        tbl.add_column("Symbol", style="cyan", no_wrap=True)
        tbl.add_column("Score", justify="right")
        tbl.add_column("Regime")
        tbl.add_column("Trend", justify="right")
        tbl.add_column("Volume", justify="right")
        tbl.add_column("Struct", justify="right")
        tbl.add_column("Momentum", justify="right")
        tbl.add_column("BTC", justify="right")
        tbl.add_column("Vol", justify="right")

        for b in breakdowns:
            score_color = (
                "green" if b.total_score >= 75
                else "yellow" if b.total_score >= 60
                else "dim"
            )
            regime_color = "green" if b.regime == Regime.TRENDING else "red"
            tbl.add_row(
                b.symbol,
                f"[{score_color}]{b.total_score:.1f}[/]",
                f"[{regime_color}]{b.regime.value}[/]",
                f"{b.trend_strength:.0f}",
                f"{b.volume_expansion:.0f}",
                f"{b.structure_quality:.0f}",
                f"{b.momentum_alignment:.0f}",
                f"{b.btc_correlation:.0f}",
                f"{b.volatility_condition:.0f}",
            )
        console.print(tbl)


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.mode:
        cfg["trading"]["mode"] = args.mode
    setup_logging(cfg)
    bot = NinjaSpotTrader(cfg)
    await bot.start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ninja Smart Trader — Binance Spot")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent.parent / "config" / "config_spot.yaml"),
        help="Path to config_spot.yaml",
    )
    parser.add_argument("--mode", choices=["paper", "live"], default=None)
    args = parser.parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")


if __name__ == "__main__":
    main()
