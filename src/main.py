"""
Ninja Trader — main entry point.

Usage:
    python -m src          # uses config/config.yaml
    python -m src --config /path/to/config.yaml
    python -m src --mode paper
    python -m src --mode backtest
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from src.data.client import BinanceFuturesClient
from src.data.market_data import MarketDataService
from src.scanner.scanner import PairScanner
from src.scoring.scorer import Scorer, SignalBreakdown
from src.risk.risk_manager import RiskManager
from src.execution.executor import Executor
from src.execution.trade_manager import TradeManager, OpenTrade
from src.learning.learner import Learner, TradeRecord
from src.notifications.telegram import TelegramNotifier
from src.backtest.shadow_engine import ShadowEngine
from src.models.ml_engine import MLPredictor, LiveReadiness
from src.models.fund_manager import FundManager
from src.models.cohort_policy import CohortPolicy
from src.models.strategy_lifecycle import StrategyLifecycleManager
from src.data.dataset_logger import DatasetLogger
from src.analysis.rejection_logger import RejectionLogger
from src.reporting.attribution import build_attribution_report

console = Console()
log = logging.getLogger("ninja_trader")


# ──────────────────────────────────────────────────────────────────────────────
#  Setup
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    load_dotenv()
    cfg["exchange"]["api_key"] = os.getenv(
        "BINANCE_API_KEY", cfg["exchange"].get("api_key", "")
    )
    cfg["exchange"]["api_secret"] = os.getenv(
        "BINANCE_API_SECRET", cfg["exchange"].get("api_secret", "")
    )
    return cfg

def normalize_config(cfg: dict) -> dict:
    mode = cfg["trading"]["mode"]
    paper_validation = cfg.get("paper_validation", {})
    # CRITICAL: enforce testnet/live consistency — never allow live + testnet=True
    if mode == "live":
        if not cfg["exchange"]["api_key"] or not cfg["exchange"]["api_secret"]:
            raise RuntimeError(
                "LIVE mode requires BINANCE_API_KEY and BINANCE_API_SECRET via env or config"
            )
        if cfg["exchange"].get("testnet", False):
            raise RuntimeError("LIVE mode cannot have testnet: true — set testnet: false in config")
    elif mode == "paper":
        # Paper mode always forces testnet=True to avoid any accidental real orders
        cfg["exchange"]["testnet"] = True
        if paper_validation.get("enabled", True):
            max_open_override = paper_validation.get("max_open_trades_override")
            if max_open_override is not None:
                cfg["trading"]["max_open_trades"] = int(max_open_override)
            top_pairs_override = paper_validation.get("top_pairs_to_trade_override")
            if top_pairs_override is not None:
                cfg["trading"]["top_pairs_to_trade"] = int(top_pairs_override)

    return cfg


def setup_logging(cfg: dict) -> None:
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    handlers = [
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
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("hpack").setLevel(logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────────
#  Bot
# ──────────────────────────────────────────────────────────────────────────────

class NinjaTrader:
    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._trading = cfg["trading"]
        self._safety = cfg.get("safety", {})

        # Exploration mode: paper = discover edge freely; live = capital protection on
        _expl = self._trading.get("exploration_mode", "auto")
        self._exploration = (
            (self._trading["mode"] == "paper") if _expl == "auto" else bool(_expl)
        )
        if self._exploration:
            log.info("Exploration mode ENABLED — budget-layer gates disabled")

        self._client = BinanceFuturesClient(cfg)
        self._market_data = MarketDataService(self._client, cfg)
        self._scanner = PairScanner(self._client, cfg)
        self._scorer = Scorer(cfg)
        self._risk = RiskManager(cfg, exploration_mode=self._exploration)
        self._executor = Executor(self._client, cfg)
        self._trade_mgr = TradeManager(
            self._client, self._executor, self._risk, cfg,
            on_close=self._on_trade_closed,
            on_progress=self._on_trade_progress,
        )
        self._learner = Learner(cfg, scorer=self._scorer)
        self._telegram = TelegramNotifier(cfg)
        self._ml = MLPredictor()
        self._readiness = LiveReadiness(telegram_notifier=self._telegram)
        self._fund_mgr = FundManager(cfg, telegram_notifier=self._telegram, exploration_mode=self._exploration)
        self._cohort_policy = CohortPolicy(cfg)
        self._lifecycle = StrategyLifecycleManager(cfg)
        self._paper_validation = cfg.get("paper_validation", {})
        self._equity_curve: list[float] = []
        self._running = False
        self._heartbeat_ts = 0.0
        self._tg_heartbeat_ts = 0.0
        self._tg_cycle_report_ts = 0.0
        self._last_report_trade_count = 0   # for fund manager report interval
        self._live_transition_done = False  # guard against double-transition
        self._last_monthly_reset_ts = time.time()
        self._last_reconcile_ts = 0.0
        self._consecutive_wins = 0          # tracked separately from risk_manager losses
        self._starting_equity: float = 0.0  # set after equity seed, used for total PnL display
        self._runtime_error_count = 0
        self._heartbeat_fail_count = 0

        # Feed learner's trade log into the scorer's EV model each tick
        self._scorer.set_trade_log(self._learner._trade_log)

        # Hydrate EdgeDetector memory from existing trade log (first boot after upgrade).
        try:
            _em = self._scorer.edge_detector.memory
            if self._learner._trade_log and _em.total_samples() < len(self._learner._trade_log):
                _em.hydrate_from_trade_log(self._learner._trade_log)
                log.info("EdgeDetector memory hydrated from %d trades", len(self._learner._trade_log))
        except Exception as _exc:
            log.debug("EdgeDetector hydrate failed: %s", _exc)

        # Pending trade score snapshots for learning (keyed by symbol)
        self._pending_scores: dict[str, dict] = {}

        # Signal confirmation tracker: {symbol: {"count": int, "first_seen": float}}
        self._signal_confirm: dict[str, dict] = {}

        # Shadow mode: simulate trades in parallel without real execution
        shadow_cfg = cfg.get("shadow", {})
        self._shadow: ShadowEngine | None = (
            ShadowEngine(cfg) if shadow_cfg.get("enabled", False) else None
        )
        self._shadow_report_interval = shadow_cfg.get("report_interval_trades", 10)
        self._shadow_last_report_count = 0

        # ML dataset logger — Phase 1 schema
        _src = "paper" if cfg["trading"]["mode"] == "paper" else "live"
        self._dataset_logger = DatasetLogger(cfg, source=_src)
        self._dataset_record_ids: dict[str, str] = {}   # symbol → record_id

        # Rejection forensics — diagnostics sink, zero behavioral impact
        self._rej = RejectionLogger(cfg)

    def _paper_validation_enabled(self) -> bool:
        return self._trading["mode"] == "paper" and bool(
            self._paper_validation.get("enabled", True)
        )

    def _paper_trade_scope_check(self, breakdown: SignalBreakdown) -> tuple[bool, str]:
        if not self._paper_validation_enabled():
            return True, "paper validation disabled"
        allowed_directions = set(
            self._paper_validation.get("allowed_directions", ["long"]) or ["long"]
        )
        allowed_regimes = set(
            self._paper_validation.get("allowed_regimes", ["trending_expansion"])
            or ["trending_expansion"]
        )
        allowed_sleeves = set(
            self._paper_validation.get("allowed_strategy_sleeves", ["trend_following"])
            or ["trend_following"]
        )
        direction = str(getattr(breakdown, "direction", "") or "")
        regime = breakdown.regime.value if breakdown.regime else ""
        sleeve = str(getattr(breakdown, "strategy_sleeve", "") or "neutral")
        if direction not in allowed_directions:
            return False, f"direction={direction} not in {sorted(allowed_directions)}"
        if regime not in allowed_regimes:
            return False, f"regime={regime or 'unknown'} not in {sorted(allowed_regimes)}"
        if sleeve not in allowed_sleeves:
            return False, f"sleeve={sleeve} not in {sorted(allowed_sleeves)}"
        return True, "paper scope ok"

    def _paper_trade_scope_allowed(self, breakdown: SignalBreakdown) -> bool:
        allowed, _reason = self._paper_trade_scope_check(breakdown)
        return allowed

    def _paper_ml_hard_gate_min_trades(self) -> int:
        return int(
            self._paper_validation.get(
                "ml_hard_gate_min_trades",
                self._cfg.get("ml", {}).get("hard_gate_min_trades", 20),
            )
        )

    def _paper_high_conviction_candidate(self, breakdown: SignalBreakdown, threshold: float) -> bool:
        ev = breakdown.ev_result
        if not self._paper_validation_enabled() or ev is None:
            return False
        min_trades = int(
            self._paper_validation.get(
                "ev_hard_gate_min_trades",
                self._cfg.get("ev_model", {}).get("min_trades_for_ev", 20),
            )
        )
        if ev.trade_count >= min_trades or not breakdown.regime_ok or not breakdown.smart_money_ok:
            return False
        high_score = float(
            self._paper_validation.get("high_conviction_score_trigger", 68.0) or 68.0
        )
        base_deficit = float(
            self._paper_validation.get("ev_bootstrap_max_deficit_pct", 0.10) or 0.10
        )
        high_deficit = float(
            self._paper_validation.get(
                "high_conviction_ev_max_deficit_pct",
                base_deficit,
            )
            or base_deficit
        )
        direction = str(getattr(breakdown, "direction", "long") or "long")
        regime = breakdown.regime.value if breakdown.regime else ""
        sm_phase = breakdown.smart_money.phase.value if breakdown.smart_money else "neutral"
        sm_bias = str(getattr(breakdown.smart_money, "direction_bias", "neutral") or "neutral")
        short_regimes = set(
            self._paper_validation.get(
                "high_conviction_short_allowed_regimes",
                ["distribution"],
            )
            or ["distribution"]
        )
        short_phases = set(
            self._paper_validation.get(
                "high_conviction_short_allowed_sm_phases",
                ["distribution", "liquidity_sweep"],
            )
            or ["distribution", "liquidity_sweep"]
        )
        long_regimes = set(
            self._paper_validation.get(
                "high_conviction_allowed_regimes",
                ["trending_expansion"],
            )
            or ["trending_expansion"]
        )
        long_phases = set(
            self._paper_validation.get(
                "high_conviction_allowed_sm_phases",
                ["liquidity_sweep", "trending"],
            )
            or ["liquidity_sweep", "trending"]
        )
        if direction == "short":
            return (
                breakdown.strategy_sleeve == "reversal"
                and regime in short_regimes
                and sm_phase in short_phases
                and sm_bias == "short"
                and breakdown.total_score >= max(high_score, threshold)
                and ev.ev_net_pct >= -high_deficit
            )
        return (
            breakdown.total_score >= max(high_score, threshold)
            and regime in long_regimes
            and sm_phase in long_phases
            and ev.ev_net_pct >= -high_deficit
        )

    def _paper_ev_relax_mode(self, breakdown: SignalBreakdown, threshold: float) -> str | None:
        if not self._paper_validation_enabled():
            return None
        ev = breakdown.ev_result
        if ev is None or breakdown.ev_ok:
            return None
        min_trades = int(
            self._paper_validation.get(
                "ev_hard_gate_min_trades",
                self._cfg.get("ev_model", {}).get("min_trades_for_ev", 20),
            )
        )
        if ev.trade_count >= min_trades or not breakdown.regime_ok or not breakdown.smart_money_ok:
            return None

        floor = float(self._paper_validation.get("min_score_floor", 58.0) or 58.0)
        score_buffer = float(self._paper_validation.get("ev_relax_score_buffer", 2.0) or 2.0)
        standard_deficit = float(
            self._paper_validation.get("ev_bootstrap_max_deficit_pct", 0.10) or 0.10
        )
        standard_score = max(floor, threshold - score_buffer)
        if breakdown.total_score >= standard_score and ev.ev_net_pct >= -standard_deficit:
            return "standard"

        if self._paper_high_conviction_candidate(breakdown, threshold):
            return "high_conviction"
        return None

    async def start(self) -> None:
        await self._client.connect()

        # Ensure One-Way position mode (Binance defaults to Hedge mode on new accounts)
        if self._trading["mode"] == "live":
            await self._client.set_position_mode_one_way()

        # Seed equity — use paper_starting_equity in paper mode, real balance in live
        paper_equity = self._trading.get("paper_starting_equity", 0)
        if self._trading["mode"] == "paper" and paper_equity > 0:
            self._risk.update_equity(float(paper_equity))
            self._starting_equity = float(paper_equity)
            log.info("Paper mode — virtual equity: $%.2f USDT", paper_equity)
        else:
            try:
                balance = await self._client.fetch_balance()
                usdt = float(balance.get("USDT", {}).get("free", 1000))
                self._risk.update_equity(usdt)
                self._starting_equity = usdt
                log.info("Starting equity: $%.2f USDT", usdt)
            except Exception as exc:
                log.warning("Could not fetch balance: %s — using $1000 default", exc)
                self._risk.update_equity(1000.0)
                self._starting_equity = 1000.0

        # Restore persisted account state (equity, curve, streaks) so restart doesn't reset
        try:
            sp = Path("data/state.json")
            if sp.exists():
                persisted = json.loads(sp.read_text())
                if persisted.get("mode") == self._trading["mode"]:
                    self._risk.state.equity = float(persisted.get("equity", self._risk.state.equity))
                    self._risk.state.peak_equity = float(persisted.get("peak_equity", self._risk.state.peak_equity))
                    self._risk.state.consecutive_losses = int(persisted.get("consecutive_losses", 0))
                    self._risk.state.daily_start_equity = self._risk.state.equity
                    self._starting_equity = float(persisted.get("starting_equity", self._starting_equity))
                    self._equity_curve = list(persisted.get("equity_curve", []))
                    self._consecutive_wins = int(persisted.get("consecutive_wins", 0))
                    persisted_kill = str(persisted.get("kill_switch_reason", "") or "")
                    transient_kills = {"runtime_error_burst", "balance_fetch_failure"}
                    if persisted_kill and persisted_kill not in transient_kills:
                        self._risk.state.kill_switch_reason = persisted_kill
                    elif persisted_kill:
                        log.info(
                            "Cleared persisted transient kill switch on startup: %s",
                            persisted_kill,
                        )
                    log.info(
                        "Restored state — equity=$%.2f peak=$%.2f start=$%.2f curve=%d pts losses=%d wins=%d",
                        self._risk.state.equity, self._risk.state.peak_equity,
                        self._starting_equity, len(self._equity_curve),
                        self._risk.state.consecutive_losses, self._consecutive_wins,
                    )
        except Exception as exc:
            log.warning("State restore failed: %s — using seeded defaults", exc)

        self._running = True
        log.info("Ninja Trader started in [bold]%s[/bold] mode", self._trading["mode"])
        await self._telegram.startup(self._trading["mode"], self._risk.state.equity)
        restored_positions = self._telegram_open_positions()
        if restored_positions:
            await self._telegram.restored_positions(restored_positions, self._trading["mode"])
        try:
            await self._loop()
        finally:
            await self._shutdown()

    async def _loop(self) -> None:
        scan_interval = self._trading["scan_interval_seconds"]
        heartbeat_interval = self._safety.get("heartbeat_interval_seconds", 30)
        _cycle = 0

        while self._running:
            try:
                tick_start = time.time()
                _cycle += 1

                # ── Heartbeat ─────────────────────────────────────────────
                if time.time() - self._heartbeat_ts > heartbeat_interval:
                    await self._heartbeat()

                # ── Telegram hourly heartbeat ─────────────────────────────
                tg_interval = self._cfg.get("telegram", {}).get("heartbeat_interval_minutes", 60) * 60
                _send_tg_heartbeat = False
                if tg_interval > 0 and time.time() - self._tg_heartbeat_ts > tg_interval:
                    self._tg_heartbeat_ts = time.time()
                    _send_tg_heartbeat = True
                    top_names: list[str] = []  # populated after scoring below

                # ── Scan pairs ────────────────────────────────────────────
                priority = self._cohort_policy.recommend_symbols(self._learner._trade_log)
                pairs = await self._scanner.scan(priority_symbols=priority)

                if not pairs:
                    log.warning("No pairs found — sleeping")
                    await asyncio.sleep(scan_interval)
                    continue

                # Limit to top N by volume for scoring, but only after priority injection.
                pairs = pairs[: self._trading.get("scan_pool_size", 30)]

                # ── Fetch market data ─────────────────────────────────────
                log.info("Fetching data for %d pairs...", len(pairs))
                try:
                    snapshots = await asyncio.wait_for(
                        self._market_data.fetch_snapshots(pairs), timeout=180.0
                    )
                except asyncio.TimeoutError:
                    log.warning("fetch_snapshots timed out after 180s — skipping cycle")
                    await asyncio.sleep(scan_interval)
                    continue

                # ── Score & pick top candidates ───────────────────────────
                breakdowns = self._scorer.score_many(snapshots)
                if breakdowns:
                    top = breakdowns[0]
                    log.info(
                        "Scored %d symbols | top: %s %s score=%.1f regime=%s sm=%s gates=R%sS%sX%s",
                        len(breakdowns), top.symbol, top.direction, top.total_score,
                        top.regime.value if top.regime else "?",
                        top.smart_money.phase.value if top.smart_money else "N/A",
                        "✓" if top.regime_ok else "✗",
                        "✓" if top.smart_money_ok else "✗",
                        "✓" if top.ev_ok else "✗",
                    )
                else:
                    log.info("Scored 0 symbols — all returned None (no direction or insufficient candles)")
                self._display_scores(breakdowns[:10])

                # ── Send cycle report to Telegram ─────────────────────────
                readiness_report = self._readiness.check(
                    self._learner._trade_log, self._ml, self._equity_curve
                )
                jim_status = self._ml.status_report(self._learner._trade_log)
                _tlog = self._learner._trade_log
                _n_trades = len(_tlog)
                _wins = sum(1 for t in _tlog if t.pnl_usd > 0)
                _win_rate = _wins / _n_trades if _n_trades > 0 else 0.0
                _sharpe = self._fund_mgr.sharpe_ratio(self._equity_curve) if len(self._equity_curve) >= 5 else 0.0
                _price_map = {sym: snap.last_price for sym, snap in snapshots.items()}
                _floating = self._trade_mgr.get_floating_pnl(_price_map)
                _open_positions = self._telegram_open_positions()
                cycle_report_interval = self._cfg.get("telegram", {}).get(
                    "cycle_report_interval_minutes", 0
                ) * 60
                if (
                    cycle_report_interval > 0
                    and time.time() - self._tg_cycle_report_ts > cycle_report_interval
                ):
                    self._tg_cycle_report_ts = time.time()
                    await self._telegram.cycle_report(
                        breakdowns=breakdowns,
                        equity=self._risk.state.equity,
                        drawdown_pct=self._risk.state.drawdown_pct,
                        daily_pnl_pct=self._risk.state.daily_pnl_pct,
                        open_trades=self._risk.state.open_trade_count,
                        cycle_num=_cycle,
                        ml_ready=self._ml.is_ready,
                        ml_accuracy=self._ml.cv_accuracy,
                        paper_trades=_n_trades,
                        readiness_passed=readiness_report.passed,
                        consecutive_wins=self._consecutive_wins,
                        consecutive_losses=self._risk.state.consecutive_losses,
                        fm_scale=min(
                            self._ml.kelly_adjustment(_tlog)
                            * self._fund_mgr._recent_performance_mult(_tlog),
                            2.0
                        ),
                        jim_status=jim_status,
                        floating_positions=_floating,
                        open_positions=_open_positions,
                        closed_positions=_tlog[-5:] if _tlog else [],
                        win_rate=_win_rate,
                        sharpe=_sharpe,
                        total_trades=_n_trades,
                        regime_thresholds=self._trading.get("regime_thresholds", {}),
                        jim_bonus=self._fund_mgr.bonus,
                        starting_equity=self._starting_equity,
                    )

                # Send Telegram hourly heartbeat after first scoring
                if _send_tg_heartbeat:
                    top_names = [
                        f"{b.symbol} {b.direction} {b.total_score:.0f}" for b in breakdowns[:5]
                    ]
                    await self._telegram.heartbeat(
                        equity=self._risk.state.equity,
                        drawdown_pct=self._risk.state.drawdown_pct,
                        daily_pnl_pct=self._risk.state.daily_pnl_pct,
                        open_trades=self._risk.state.open_trade_count,
                        top_signals=top_names,
                        open_positions=self._telegram_open_positions(),
                    )

                # ── Monitor open trades ───────────────────────────────────
                price_map = {sym: snap.last_price for sym, snap in snapshots.items()}

                # Fetch price for open trades not in current scan (symbol fell off list)
                missing = [s for s in self._trade_mgr.open_symbols if s not in price_map]
                for sym in missing:
                    try:
                        ticker = await self._client.fetch_ticker(sym)
                        price_map[sym] = float(ticker.get("last", 0) or 0)
                    except Exception as exc:
                        log.warning("Could not fetch price for open trade %s: %s", sym, exc)

                if self._shadow:
                    self._shadow.on_tick(price_map)
                await self._trade_mgr.monitor_all(price_map)

                # ── Exchange position reconciliation (live only, every 5 min) ──
                if self._trading["mode"] == "live" and time.time() - self._last_reconcile_ts > 300:
                    self._last_reconcile_ts = time.time()
                    await self._reconcile_positions()

                # ── Open new trades ───────────────────────────────────────
                top_n = self._trading["top_pairs_to_trade"]
                confirm_scans = 1 if self._trading["mode"] == "paper" else self._trading.get("signal_confirmation_scans", 2)
                signal_max_age = self._trading.get("signal_max_age_seconds", 300)
                now = time.time()
                attribution_report = build_attribution_report(self._learner._trade_log, min_trades=2)
                lifecycle_report = self._lifecycle.build_report(self._learner._trade_log)

                # Update confirmation counters (regime-aware threshold per symbol)
                for b in breakdowns:
                    sym = b.symbol
                    threshold = max(
                        0.0,
                        self._threshold_for(b.regime)
                        + getattr(b, "strategy_threshold_shift", 0.0)
                        + self._cohort_policy.threshold_relief_with_attribution(
                            b,
                            self._learner._trade_log,
                            attribution_report,
                        ),
                    )
                    if sym in self._trade_mgr.open_symbols:
                        self._signal_confirm.pop(sym, None)
                        continue
                    if b.total_score >= threshold:
                        if sym not in self._signal_confirm:
                            self._signal_confirm[sym] = {"count": 1, "first_seen": now}
                        else:
                            self._signal_confirm[sym]["count"] += 1
                    else:
                        self._signal_confirm.pop(sym, None)

                # Expire stale signals
                self._signal_confirm = {
                    sym: v for sym, v in self._signal_confirm.items()
                    if now - v["first_seen"] <= signal_max_age
                }

                # ── ML: train on new data & filter eligible signals ───────
                _shadow_ml = self._shadow.get_ml_trade_log() if self._shadow else []
                newly_trained = self._ml.maybe_train(self._learner._trade_log, extra_records=_shadow_ml)
                if newly_trained:
                    acc = self._ml.cv_accuracy
                    top_feat = max(self._ml.feature_importance, key=self._ml.feature_importance.get, default="?")
                    log.info("ML retrained — CV accuracy=%.1f%%  top_feature=%s", acc * 100, top_feat)
                    await self._telegram.send_raw(
                        f"🧠 *Boss, I just got smarter.*\n"
                        f"ML retrained on `{len(self._learner._trade_log)}` trades.\n"
                        f"Accuracy: `{acc:.1%}`  |  Threshold: `{self._ml.dynamic_p_win_threshold:.0%}`\n"
                        f"Top signal I'm watching: `{top_feat}`\n"
                        f"_Learning from every trade. Getting better every day._ 📈"
                    )

                eligible = []
                _ml_p_win_map: dict[str, float] = {}
                _regime_scale_map: dict[str, float] = {}
                _cohort_bonus_map: dict[str, float] = {}
                for b in breakdowns:
                    paper_scope_allowed, paper_scope_reason = self._paper_trade_scope_check(b)
                    if not paper_scope_allowed:
                        _snap = snapshots.get(b.symbol)
                        _thresh = self._threshold_for(b.regime)
                        _thresh += getattr(b, "strategy_threshold_shift", 0.0)
                        _fr = float(_snap.funding_rate) if _snap is not None else 0.0
                        self._dataset_logger.log_no_trade(b, "paper_scope_blocked", snap=_snap)
                        self._rej.log(
                            stage="paper_scope",
                            reason=paper_scope_reason,
                            breakdown=b,
                            threshold_required=_thresh,
                            funding_rate=_fr,
                        )
                        continue
                    _snap = snapshots.get(b.symbol)
                    _thresh = self._threshold_for(b.regime)
                    _thresh += getattr(b, "strategy_threshold_shift", 0.0)
                    _fr = float(_snap.funding_rate) if _snap is not None else 0.0
                    _ev_bootstrap = (
                        self._exploration
                        and b.ev_result is not None
                        and b.ev_result.trade_count < self._cfg.get("ev_model", {}).get("min_trades_for_ev", 20)
                        and b.regime_ok
                        and b.smart_money_ok
                        and b.total_score >= max(45.0, _thresh - 5.0)
                    )
                    if _ev_bootstrap and not b.ev_ok:
                        log.info("[%s] EV bootstrap override enabled for paper mode", b.symbol)
                        b.ev_ok = True
                    _paper_ev_relax_mode = self._paper_ev_relax_mode(b, _thresh)
                    if _paper_ev_relax_mode and not b.ev_ok:
                        log.info(
                            "[%s] paper validation EV relax enabled (%s, ev_net=%+.3f%% trade_count=%d)",
                            b.symbol,
                            _paper_ev_relax_mode,
                            b.ev_result.ev_net_pct,
                            b.ev_result.trade_count,
                        )
                        b.ev_ok = True
                    if not b.all_gates_passed:
                        if not b.regime_ok:
                            _gate_reason = "gate_regime_fail"
                            _rej_stage = "gate_regime"
                            _rej_detail = f"regime={b.regime.value if b.regime else '?'}"
                        elif not b.smart_money_ok:
                            _gate_reason = "gate_sm_fail"
                            _rej_stage = "gate_sm"
                            _sm = b.smart_money
                            _rej_detail = f"sm_phase={_sm.phase.value if _sm else 'N/A'} score={_sm.score if _sm else 0:.0f}"
                        else:
                            _gate_reason = "gate_ev_fail"
                            _rej_stage = "gate_ev"
                            _ev = b.ev_result
                            _rej_detail = f"ev_net={_ev.ev_net_pct:+.3f}% p_win={_ev.p_win:.3f}" if _ev else "ev=none"
                        self._dataset_logger.log_no_trade(b, _gate_reason, snap=_snap)
                        self._rej.log(
                            stage=_rej_stage, reason=_rej_detail, breakdown=b,
                            threshold_required=_thresh, funding_rate=_fr,
                        )
                        # Dual-stream: shadow EV-rejected signals for uncensored belief update.
                        if _rej_stage == "gate_ev" and self._shadow and _snap is not None:
                            try:
                                _df = _snap.candles_for(self._cfg["timeframes"]["primary"])
                                if not _df.empty:
                                    _sh_setup = self._risk.calculate_setup(
                                        b.symbol, b.direction, _df, _snap.last_price,
                                        strategy_sleeve=b.strategy_sleeve,
                                        ev_result=b.ev_result, kelly_scale=1.0, backtest=True,
                                    )
                                    if _sh_setup is not None:
                                        self._shadow.on_signal(b, _sh_setup)
                            except Exception as _e:
                                log.debug("shadow counterfactual %s: %s", b.symbol, _e)
                        continue
                    if b.total_score < _thresh:
                        self._dataset_logger.log_no_trade(b, "score_below_threshold", snap=_snap)
                        self._rej.log(
                            stage="score_threshold",
                            reason=f"score={b.total_score:.1f} < thresh={_thresh:.1f}",
                            breakdown=b, threshold_required=_thresh, funding_rate=_fr,
                        )
                        continue
                    if b.symbol in self._trade_mgr.open_symbols:
                        continue
                    if self._signal_confirm.get(b.symbol, {}).get("count", 0) < confirm_scans:
                        continue
                    cohort = self._cohort_policy.evaluate(
                        b,
                        self._learner._trade_log,
                        attribution_report=attribution_report,
                    )
                    if not cohort.allowed:
                        log.info("Cohort policy blocked %s — %s", b.symbol, cohort.reason)
                        self._dataset_logger.log_no_trade(b, "cohort_policy_blocked", snap=_snap)
                        self._rej.log(
                            stage="cohort_policy",
                            reason=cohort.reason,
                            breakdown=b, threshold_required=_thresh,
                            funding_rate=_fr,
                        )
                        if self._shadow:
                            self._shadow.record_rejected(b, stage="cohort_policy")
                        continue
                    edge = b.edge_result
                    edge_detector = self._scorer.edge_detector
                    if edge and edge.action == "BLOCK" and edge_detector.is_blocking:
                        log.info(
                            "Edge hard-block %s — %s (%s)",
                            b.symbol, edge.status, ", ".join(edge.reasons) or edge.edge_type,
                        )
                        self._dataset_logger.log_no_trade(b, "edge_blocked", snap=_snap)
                        self._rej.log(
                            stage="edge_blocked",
                            reason=f"{edge.status}: {', '.join(edge.reasons) or edge.edge_type}",
                            breakdown=b, threshold_required=_thresh,
                            funding_rate=_fr,
                        )
                        if self._shadow:
                            self._shadow.record_rejected(b, stage="edge_blocked")
                        continue
                    _life = self._lifecycle.assess_breakdown(
                        b,
                        lifecycle_report,
                        mode=self._trading["mode"],
                        hour_utc=time.gmtime().tm_hour,
                    )
                    b.lifecycle_status = _life.status
                    b.cohort_key = _life.cohort_key
                    b.recommendation = _life.recommendation
                    if not _life.allowed:
                        self._dataset_logger.log_no_trade(b, "lifecycle_blocked", snap=_snap)
                        self._rej.log(
                            stage="lifecycle_blocked",
                            reason=_life.reason,
                            breakdown=b, threshold_required=_thresh, funding_rate=_fr,
                        )
                        continue
                    # ML gate: if model trained, require p_win >= threshold
                    _rm = {"trending_expansion":3,"accumulation_compression":2,"distribution":1,"chaos":0}
                    _sm_m = {"trending":4,"accumulation":3,"liquidity_sweep":2,"neutral":1,"distribution":0,"chaos":-1}
                    _bsm = b.smart_money
                    _bev = b.ev_result
                    scores_dict = {
                        "trend_strength": b.trend_strength,
                        "volume_confirmation": b.volume_confirmation,
                        "structure_quality": b.structure_quality,
                        "open_interest": b.open_interest,
                        "funding_sentiment": b.funding_sentiment,
                        "order_book": b.order_book,
                        "volatility": b.volatility,
                        "legacy_score": b.legacy_score,
                        "base_score": b.base_score,
                        "total_score": b.total_score,
                        "is_long": 1.0 if b.direction == "long" else 0.0,
                        "regime_code": float(_rm.get(b.regime.value if b.regime else "chaos", 0)),
                        "regime_ok": 1.0 if b.regime_ok else 0.0,
                        "sm_phase_code": float(_sm_m.get(_bsm.phase.value if _bsm else "neutral", 1)),
                        "sm_phase": _bsm.phase.value if _bsm else "neutral",
                        "sm_score": float(_bsm.score if _bsm else 50.0),
                        "sm_aligns": 1.0 if (_bsm and _bsm.aligns_with(b.direction)) else 0.0,
                        "sm_ok": 1.0 if b.smart_money_ok else 0.0,
                        "ev_p_win": float(_bev.p_win if _bev else 0.52),
                        "ev_net_pct": float(_bev.ev_net_pct if _bev else 0.0),
                        "ev_confidence": float(_bev.confidence if _bev else 0.4),
                        "ev_ok": 1.0 if b.ev_ok else 0.0,
                        "gates_passed": float(b.regime_ok) + float(b.smart_money_ok) + float(b.ev_ok),
                        "hour_utc": float(time.gmtime().tm_hour),
                    }
                    ml_p_win, ml_ready = self._ml.predict(scores_dict, b.total_score)
                    ml_threshold = self._ml.dynamic_p_win_threshold
                    # Phase 2 gate: ML p_win only closes entries after 20 real closed trades.
                    # Prevents ML (trained partly on shadow trades) from blocking exploration
                    # during bootstrap when CV accuracy is inflated by small-sample overfit.
                    real_trades = len(self._learner._trade_log)
                    ml_hard_gate_min_trades = (
                        self._paper_ml_hard_gate_min_trades()
                        if self._paper_validation_enabled()
                        else int(self._cfg.get("ml", {}).get("hard_gate_min_trades", 20))
                    )
                    if ml_ready and real_trades >= ml_hard_gate_min_trades and ml_p_win < ml_threshold:
                        log.info(
                            "ML rejected %s — p_win=%.1f%% < threshold=%.1f%%",
                            b.symbol, ml_p_win * 100, ml_threshold * 100,
                        )
                        self._dataset_logger.log_no_trade(b, "ml_rejected", snap=_snap)
                        self._rej.log(
                            stage="ml_rejected",
                            reason=f"ml_p_win={ml_p_win:.3f} < thresh={ml_threshold:.3f}",
                            breakdown=b, threshold_required=_thresh,
                            ml_p_win=ml_p_win, ml_threshold=ml_threshold,
                            funding_rate=_fr,
                        )
                        if self._shadow:
                            self._shadow.record_rejected(b, stage="ml_rejected", features=scores_dict)
                        continue
                    # Progressive regime gate: exploration / soft-penalty / hard-lock
                    regime_key = b.regime.value if b.regime else "chaos"
                    if self._paper_validation_enabled():
                        allow, regime_scale, regime_reason = True, 1.0, "paper strategy-only path"
                    else:
                        allow, regime_scale, regime_reason = self._ml.regime_gate(regime_key)
                        if not allow:
                            log.info("Regime hard-lock %s (%s) — skip %s", regime_key, regime_reason, b.symbol)
                            self._dataset_logger.log_no_trade(b, "ml_regime_gate", snap=_snap)
                            self._rej.log(
                                stage="regime_gate",
                                reason=f"{regime_key}: {regime_reason}",
                                breakdown=b, threshold_required=_thresh,
                                ml_p_win=ml_p_win, ml_threshold=ml_threshold,
                                funding_rate=_fr,
                            )
                            if self._shadow:
                                self._shadow.record_rejected(b, stage="regime_gate", features=scores_dict)
                            continue
                    eligible.append(b)
                    _ml_p_win_map[b.symbol] = ml_p_win
                    _regime_scale_map[b.symbol] = regime_scale
                    _cohort_bonus_map[b.symbol] = self._cohort_policy.ranking_bonus_with_attribution(
                        b,
                        self._learner._trade_log,
                        attribution_report,
                    )
                eligible.sort(
                    key=lambda x: (
                        x.total_score
                        + (_ml_p_win_map.get(x.symbol, 0.5) - 0.5) * 20.0
                        + getattr(x, "strategy_ranking_bonus", 0.0)
                        + _cohort_bonus_map.get(x.symbol, 0.0),
                        _regime_scale_map.get(x.symbol, 1.0),
                        _ml_p_win_map.get(x.symbol, 0.5),
                    ),
                    reverse=True,
                )
                eligible = eligible[:top_n]

                # Keep trade log in sync — union realized + shadow counterfactuals.
                # Uncensored posterior: asymmetric shrinkage (pwin_engine) bounds
                # downside risk of including simulated outcomes.
                self._scorer.set_trade_log(self._learner._trade_log)

                for bd in eligible:
                    _bd_thresh = max(
                        0.0,
                        self._threshold_for(bd.regime)
                        + getattr(bd, "strategy_threshold_shift", 0.0)
                        + self._cohort_policy.threshold_relief_with_attribution(
                            bd,
                            self._learner._trade_log,
                            attribution_report,
                        ),
                    )
                    _bd_fr = float(snapshots[bd.symbol].funding_rate) if bd.symbol in snapshots else 0.0
                    # FIX #8: check risk guard BEFORE expensive setup calculation
                    can_open, reason = self._risk.can_open_trade()
                    if not can_open:
                        log.info("[%s] Risk guard blocked: %s", bd.symbol, reason)
                        self._rej.log(
                            stage="risk_guard", reason=reason, breakdown=bd,
                            threshold_required=_bd_thresh, funding_rate=_bd_fr,
                        )
                        if self._shadow:
                            self._shadow.record_rejected(bd, stage="risk_guard")
                        continue

                    snap = snapshots.get(bd.symbol)
                    if snap is None:
                        continue
                    df = snap.candles_for(self._cfg["timeframes"]["primary"])
                    if df.empty:
                        continue

                    # Safety: skip extreme volatility
                    if self._safety.get("pause_on_extreme_volatility", True):
                        from src.analysis.indicators import is_extreme_volatility
                        if is_extreme_volatility(
                            df,
                            self._cfg["indicators"]["atr_period"],
                            self._safety.get("extreme_vol_atr_multiplier", 3.0),
                        ):
                            log.warning("[%s] Extreme volatility — skipping", bd.symbol)
                            self._rej.log(
                                stage="extreme_vol",
                                reason=f"atr > {self._safety.get('extreme_vol_atr_multiplier', 3.0)}x",
                                breakdown=bd, threshold_required=_bd_thresh, funding_rate=_bd_fr,
                            )
                            continue

                    # ── Institutional gate: all 3 filters must pass ───────
                    if not bd.regime_ok:
                        log.info("[%s] BLOCKED regime=%s", bd.symbol, bd.regime.value)
                        continue
                    if not bd.smart_money_ok:
                        sm_phase = bd.smart_money.phase.value if bd.smart_money else "N/A"
                        log.info("[%s] BLOCKED smart_money phase=%s", bd.symbol, sm_phase)
                        continue
                    if not bd.ev_ok:
                        ev_str = f"{bd.ev_result.ev_net_pct:+.3f}%" if bd.ev_result else "N/A"
                        log.info("[%s] BLOCKED EV=%s", bd.symbol, ev_str)
                        continue

                    # Fund manager: combine ML momentum scale + conviction/streak/regime
                    ml_scale = self._ml.kelly_adjustment(self._learner._trade_log)
                    # Jim's per-trade confidence multiplier: if ML is trained and predicted
                    # high p_win for this specific trade, size up accordingly
                    _p_win = _ml_p_win_map.get(bd.symbol, 0.52)
                    ml_confidence = self._ml.confidence_scale(_p_win)
                    fm_decision = self._fund_mgr.get_size_multiplier(
                        breakdown=bd,
                        equity=self._risk.state.equity,
                        consecutive_wins=self._consecutive_wins,
                        consecutive_losses=self._risk.state.consecutive_losses,
                        trade_log=self._learner._trade_log,
                        open_risk_pct=self._risk.state.open_risk_pct,
                        daily_pnl_pct=self._risk.state.daily_pnl_pct,
                        drawdown_pct=self._risk.state.drawdown_pct,
                        open_trade_count=self._risk.state.open_trade_count,
                    )
                    if fm_decision.vetoed:
                        log.warning("[%s] Jim VETOED: %s", bd.symbol, fm_decision.veto_reason)
                        self._rej.log(
                            stage="fm_veto", reason=fm_decision.veto_reason, breakdown=bd,
                            threshold_required=_bd_thresh,
                            fm_scale=fm_decision.size_multiplier,
                            ml_p_win=_p_win, ml_threshold=self._ml.dynamic_p_win_threshold,
                            funding_rate=_bd_fr,
                        )
                        if self._shadow:
                            self._shadow.record_rejected(bd, stage="fm_veto")
                        continue

                    total_scale = min(ml_scale * ml_confidence * fm_decision.size_multiplier, 2.0)
                    total_scale = max(total_scale, 0.40)
                    total_scale *= getattr(bd, "strategy_size_mult", 1.0)
                    if getattr(bd, "strategy_size_mult", 1.0) != 1.0:
                        log.info(
                            "[%s] strategy_scale=%.2fx applied (%s)",
                            bd.symbol, bd.strategy_size_mult, bd.strategy_sleeve,
                        )
                    cohort = self._cohort_policy.evaluate(bd, self._learner._trade_log)
                    if cohort.size_mult != 1.0:
                        total_scale *= cohort.size_mult
                        log.info("[%s] cohort_scale=%.2fx applied", bd.symbol, cohort.size_mult)
                    edge = bd.edge_result
                    if edge and self._scorer.edge_detector.affects_sizing:
                        total_scale *= edge.size_mult
                        log.info(
                            "[%s] edge_scale=%.2fx applied (%s/%s)",
                            bd.symbol, edge.size_mult, edge.status, edge.action,
                        )
                    # Regime gate scale applied AFTER floor so soft-penalty can shrink below 0.40.
                    # Never amplifies (max 1.0 from regime_gate); only shrinks or leaves unchanged.
                    _rscale = _regime_scale_map.get(bd.symbol, 1.0)
                    if _rscale < 1.0:
                        total_scale *= _rscale
                        log.info("[%s] regime_scale=%.2fx applied", bd.symbol, _rscale)
                    if self._paper_high_conviction_candidate(bd, _bd_thresh):
                        floor = float(
                            self._paper_validation.get("high_conviction_min_total_scale", 0.30)
                            or 0.30
                        )
                        if total_scale < floor:
                            log.info(
                                "[%s] paper high-conviction size floor %.2fx applied (raw %.2fx)",
                                bd.symbol,
                                floor,
                                total_scale,
                            )
                            total_scale = floor
                    if fm_decision.notes or ml_confidence > 1.0:
                        conf_note = f"Jim confidence {ml_confidence:.2f}x (p_win={_p_win:.0%})" if ml_confidence > 1.0 else ""
                        all_notes = ([conf_note] if conf_note else []) + fm_decision.notes
                        log.info("[%s] Jim scale=%.2fx  (%s)", bd.symbol, total_scale, ", ".join(all_notes))

                    _live = self._trading["mode"] == "live"
                    setup = self._risk.calculate_setup(
                        bd.symbol, bd.direction, df, snap.last_price,
                        strategy_sleeve=bd.strategy_sleeve,
                        ev_result=bd.ev_result,
                        kelly_scale=total_scale,
                        min_lot_size=self._client.min_lot_size(bd.symbol) if _live else 0.0,
                        min_notional_usd=self._client.min_notional(bd.symbol) if _live else 0.0,
                    )
                    if setup is None:
                        setup_reason = (
                            self._risk.last_setup_rejection_reason
                            or "calculate_setup returned None"
                        )
                        log.info("[%s] calculate_setup returned None — skipping", bd.symbol)
                        self._rej.log(
                            stage="setup_none",
                            reason=setup_reason,
                            breakdown=bd, threshold_required=_bd_thresh,
                            fm_scale=total_scale, ml_p_win=_p_win,
                            ml_threshold=self._ml.dynamic_p_win_threshold,
                            funding_rate=_bd_fr,
                        )
                        continue

                    self._print_trade_card(bd, setup)

                    # Encode all edge dimensions for ML
                    sm = bd.smart_money
                    ev = bd.ev_result
                    _regime_map = {
                        "trending_expansion": 3, "accumulation_compression": 2,
                        "distribution": 1, "chaos": 0,
                    }
                    _sm_map = {
                        "trending": 4, "accumulation": 3, "liquidity_sweep": 2,
                        "neutral": 1, "distribution": 0, "chaos": -1,
                    }
                    self._pending_scores[bd.symbol] = {
                        # Raw signal scores
                        "trend_strength": bd.trend_strength,
                        "volume_confirmation": bd.volume_confirmation,
                        "structure_quality": bd.structure_quality,
                        "open_interest": bd.open_interest,
                        "funding_sentiment": bd.funding_sentiment,
                        "order_book": bd.order_book,
                        "volatility": bd.volatility,
                        "total_score": bd.total_score,
                        # Edge: direction
                        "is_long": 1.0 if bd.direction == "long" else 0.0,
                        # Edge: regime (encoded)
                        "regime_code": float(_regime_map.get(bd.regime.value if bd.regime else "chaos", 0)),
                        "regime_ok": 1.0 if bd.regime_ok else 0.0,
                        # Edge: smart money (encoded)
                        "sm_phase_code": float(_sm_map.get(sm.phase.value if sm else "neutral", 1)),
                        "sm_phase": sm.phase.value if sm else "neutral",
                        "sm_score": float(sm.score if sm else 50.0),
                        "sm_aligns": 1.0 if (sm and sm.aligns_with(bd.direction)) else 0.0,
                        "sm_ok": 1.0 if bd.smart_money_ok else 0.0,
                        # Edge: EV model
                        "ev_p_win": float(ev.p_win if ev else 0.52),
                        "ev_net_pct": float(ev.ev_net_pct if ev else 0.0),
                        "ev_confidence": float(ev.confidence if ev else 0.4),
                        "ev_ok": 1.0 if bd.ev_ok else 0.0,
                        # Edge: gate combination (3-bit signal quality)
                        "gates_passed": float(bd.regime_ok) + float(bd.smart_money_ok) + float(bd.ev_ok),
                        # Time of day (UTC hour) — session awareness
                        "hour_utc": float(time.gmtime().tm_hour),
                        # Strategy sleeve metadata for later attribution
                        "strategy_sleeve": bd.strategy_sleeve,
                        "strategy_reason": bd.strategy_reason,
                        "exit_profile": setup.exit_profile,
                        "strategy_score_mult": float(bd.strategy_score_mult),
                        "strategy_size_mult": float(bd.strategy_size_mult),
                        "dispersion_value": float(bd.dispersion_value),
                        "signal_type": f"{bd.regime.value if bd.regime else 'unknown'}|{sm.phase.value if sm else 'neutral'}",
                        "entry_reason": bd.strategy_reason,
                        "timeframe": self._cfg["timeframes"]["primary"],
                        "asset": bd.symbol.split("/")[0],
                        "session": ("asia" if time.gmtime().tm_hour < 8 else "london" if time.gmtime().tm_hour < 13 else "overlap_london_ny" if time.gmtime().tm_hour < 17 else "ny"),
                        "volatility_bucket": ("low" if bd.volatility <= 20 else "medium" if bd.volatility <= 40 else "high" if bd.volatility <= 70 else "extreme"),
                        "trend_bucket": ("weak" if bd.trend_strength < 40 else "moderate" if bd.trend_strength < 65 else "strong" if bd.trend_strength < 85 else "very_strong"),
                        "cohort_key": getattr(bd, "cohort_key", ""),
                        "lifecycle_status": getattr(bd, "lifecycle_status", "RESEARCH"),
                    }
                    # CRITICAL: only notify + record if trade actually opened
                    opened = await self._trade_mgr.open(setup)
                    if not opened:
                        self._pending_scores.pop(bd.symbol, None)
                        log.warning("[%s] trade_mgr.open() failed — skipping", bd.symbol)
                        continue
                    self._fund_mgr.record_sizing_decision(bd.symbol, total_scale)
                    _rec_id = self._dataset_logger.open_record(bd, setup, snap=snapshots.get(bd.symbol))
                    self._dataset_record_ids[bd.symbol] = _rec_id
                    await self._telegram.trade_opened(bd, setup, jim_notes=fm_decision.notes)
                    if self._shadow:
                        self._shadow.on_signal(bd, setup, scores_dict=self._pending_scores.get(bd.symbol, {}))

                # ── Shadow report every N closed trades ───────────────────
                if self._shadow:
                    m = self._shadow.metrics()
                    if m.closed >= self._shadow_last_report_count + self._shadow_report_interval:
                        self._shadow_last_report_count = m.closed
                        self._shadow.print_report()
                        await self._telegram.shadow_report(
                            self._shadow.format_telegram_report()
                        )

                # ── Write dashboard state ─────────────────────────────────
                self._write_state(breakdowns[:10], lifecycle_report, attribution_report)

                # ── Flush rejection forensics buffer ──────────────────────
                self._rej.flush()

                # ── Sleep until next scan ─────────────────────────────────
                elapsed = time.time() - tick_start
                sleep_time = max(0.0, scan_interval - elapsed)
                log.debug("Tick took %.1fs — sleeping %.1fs", elapsed, sleep_time)
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.exception("Unhandled error in main loop: %s", exc)
                # Classify: external connectivity vs bot bug
                exc_str = str(exc)
                exc_blob = f"{type(exc).__name__}: {exc_str}"
                is_external = any(kw in exc_blob for kw in (
                    "ExchangeNotAvailable", "NetworkError", "RequestTimeout",
                    "SSLCertVerificationError", "ClientConnectorCertificateError",
                    "ClientConnectorDNSError", "ClientConnectorError",
                    "Cannot connect to host", "DNS server returned general failure",
                    "Timeout while contacting DNS servers",
                    "Hostname mismatch", "ssl", "SSL",
                ))
                if is_external:
                    self._runtime_error_count = 0
                    await self._telegram.send_raw(
                        "⚡ *External connectivity issue detected*\n"
                        f"`{exc_str[:200]}`\n"
                        "_This is likely Binance maintenance or network. Bot will self-recover._"
                    )
                else:
                    self._runtime_error_count += 1
                    await self._telegram.send_raw(
                        "🚨 *Bot Error — needs attention*\n"
                        f"`{type(exc).__name__}: {exc_str[:300]}`\n"
                        "_Check the logs._"
                    )
                if self._runtime_error_count >= int(self._safety.get("runtime_error_kill_switch_count", 5)):
                    self._risk.trigger_kill_switch("runtime_error_burst")
                await asyncio.sleep(10)

    async def _heartbeat(self) -> None:
        self._heartbeat_ts = time.time()
        equity = self._risk.state.equity
        # In paper mode, keep the virtual equity — don't overwrite with real $0 balance
        if self._trading["mode"] != "paper":
            try:
                balance = await self._client.fetch_balance()
                external_equity = float(balance.get("USDT", {}).get("total", equity))
                mismatch_tolerance = float(self._safety.get("equity_mismatch_tolerance_pct", 20.0))
                if equity > 0 and self._risk.state.open_trade_count == 0:
                    mismatch_pct = abs(external_equity - equity) / equity * 100
                    if mismatch_pct > mismatch_tolerance:
                        self._risk.trigger_kill_switch("equity_mismatch")
                equity = external_equity
                self._risk.update_equity(equity)
                self._heartbeat_fail_count = 0
                self._runtime_error_count = 0
            except Exception as exc:
                self._heartbeat_fail_count = getattr(self, "_heartbeat_fail_count", 0) + 1
                log.warning("Heartbeat balance fetch failed (#%d): %s", self._heartbeat_fail_count, exc)
                if self._heartbeat_fail_count == 3:
                    exc_str = str(exc)
                    is_external = any(kw in exc_str for kw in ("ssl", "SSL", "ExchangeNotAvailable", "NetworkError", "Hostname mismatch"))
                    prefix = "⚡ *Binance unreachable (maintenance?)*" if is_external else "🚨 *Balance fetch failing*"
                    await self._telegram.send_raw(
                        f"{prefix}\n`{exc_str[:200]}`\n"
                        f"_Equity display frozen at ${equity:,.2f}. Bot continues scanning._"
                    )
                if self._heartbeat_fail_count >= int(self._safety.get("heartbeat_fail_kill_switch_count", 6)):
                    self._risk.trigger_kill_switch("balance_fetch_failure")
        log.info(
            "Heartbeat  equity=$%.2f  drawdown=%.1f%%  daily_pnl=%.1f%%  open=%d",
            equity,
            self._risk.state.drawdown_pct,
            self._risk.state.daily_pnl_pct,
            self._risk.state.open_trade_count,
        )

    async def _on_trade_closed(self, trade: OpenTrade, pnl: float, reason: str) -> None:
        scores = self._pending_scores.pop(trade.symbol, {})
        entry_p = trade.setup.entry_price
        close_p = trade.setup.entry_price
        if "stop" in reason:
            close_p = trade.setup.stop_loss
        elif reason == "tp1":
            close_p = trade.setup.tp1
        elif reason in ("tp2", "tp2_full"):
            close_p = trade.setup.tp2
        elif reason == "tp3":
            close_p = trade.setup.tp3
        elif reason == "trailing_stop" and trade.trailing_stop is not None:
            close_p = trade.trailing_stop
        pre_close_equity = self._risk.state.equity - pnl
        pnl_pct = (pnl / pre_close_equity * 100) if pre_close_equity > 0 else 0.0

        _regime_val = scores.get("regime_code", -1)
        _regime_str = {3: "trending_expansion", 2: "accumulation_compression",
                       1: "distribution", 0: "chaos"}.get(int(_regime_val), "chaos")
        record = TradeRecord(
            symbol=trade.symbol,
            direction=trade.direction,
            entry_price=entry_p,
            exit_price=close_p,
            pnl_usd=pnl,
            pnl_pct=pnl_pct,
            reason=reason,
            scores=scores,
            opened_at=trade.opened_at,
            closed_at=time.time(),
            regime=_regime_str,
            strategy_sleeve=str(scores.get("strategy_sleeve", "")),
            exit_profile=str(getattr(trade.setup, "exit_profile", "") or str(scores.get("exit_profile", "") or "")),
            mfe_r=trade.mfe_r,
            mae_r=trade.mae_r,
            tp1_hit=trade.tp1_hit,
            hold_duration_s=max(0.0, time.time() - trade.opened_at),
            fees_usd=round(trade.setup.size_usd * float(self._cfg.get("ev_model", {}).get("taker_fee_pct", 0.04)) / 100.0 * 2, 4),
            slippage_estimate_usd=round(trade.setup.size_usd * float(self._cfg.get("ev_model", {}).get("slippage_pct", 0.05)) / 100.0 * 2, 4),
            fees_slippage_pct=round((float(self._cfg.get("ev_model", {}).get("taker_fee_pct", 0.04)) + float(self._cfg.get("ev_model", {}).get("slippage_pct", 0.05))) * 2, 4),
            timeframe=str(scores.get("timeframe", self._cfg["timeframes"]["primary"])),
            session=str(scores.get("session", "unknown")),
            asset=str(scores.get("asset", trade.symbol.split('/')[0])),
            signal_type=str(scores.get("signal_type", "unknown")),
            entry_reason=str(scores.get("entry_reason", "unknown")),
            volatility_bucket=str(scores.get("volatility_bucket", "unknown")),
            trend_bucket=str(scores.get("trend_bucket", "unknown")),
            cohort_key=str(scores.get("cohort_key", "")),
            lifecycle_status=str(scores.get("lifecycle_status", "RESEARCH")),
        )
        self._learner.record_trade(record)
        # Feed closed outcome into EdgeDetector memory (pair × edge_type → per-regime WR + decay curve).
        try:
            _sm_phase = str(scores.get("sm_phase", "neutral"))
            _edge_type = f"{_regime_str}|{_sm_phase}|{trade.direction}"
            self._scorer.edge_detector.memory.record(
                pair=trade.symbol,
                edge_type=_edge_type,
                regime=_regime_str,
                won=(pnl_pct > 0),
            )
        except Exception as _exc:
            log.debug("EdgeMemory record failed: %s", _exc)

        _rec_id = self._dataset_record_ids.pop(trade.symbol, None)
        if _rec_id:
            _closed_now = time.time()
            _mfe_peak_s = (trade.mfe_peak_ts - trade.opened_at) if trade.mfe_peak_ts > 0 else None
            # flush adverse counter if still in drawdown at close
            _dd_s = trade.drawdown_total_s
            if trade._adverse_since > 0.0:
                _dd_s += _closed_now - trade._adverse_since
            self._dataset_logger.close_record(
                record_id=_rec_id,
                closed_at=_closed_now,
                exit_price=close_p,
                exit_type=reason,
                pnl_usd=pnl,
                pnl_pct=pnl_pct,
                mfe_r=trade.mfe_r,
                mae_r=trade.mae_r,
                tp1_hit=trade.tp1_hit,
                time_to_mfe_peak_s=_mfe_peak_s,
                drawdown_duration_s=_dd_s if _dd_s > 0 else None,
            )

        bonus_msg = self._fund_mgr.record_sizing_outcome(trade.symbol, pnl)
        if bonus_msg:
            asyncio.create_task(self._telegram.send_raw(
                f"🏆 *Jim's Bonus Update*\n{bonus_msg}\n_— Keep it up, Jim._ 🥷"
            ))
        self._equity_curve.append(self._risk.state.equity)

        # ML: regime outcome tracking + retrain
        regime_val = scores.get("regime_code", -1)
        regime_key = {3: "trending_expansion", 2: "accumulation_compression",
                      1: "distribution", 0: "chaos"}.get(int(regime_val), "chaos")
        self._ml.record_regime_outcome(regime_key, pnl)
        _shadow_ml = self._shadow.get_ml_trade_log() if self._shadow else []
        newly_trained = self._ml.maybe_train(self._learner._trade_log, extra_records=_shadow_ml)
        if newly_trained:
            acc = self._ml.cv_accuracy
            top_feat = max(self._ml.feature_importance, key=self._ml.feature_importance.get, default="?")
            trend = self._ml.accuracy_trend
            log.info("ML retrained — acc=%.1f%%  trend=%s  threshold=%.2f", acc * 100, trend, self._ml.dynamic_p_win_threshold)
            trend_msg = {"improving": "getting sharper 📈", "declining": "hit a rough patch 📉", "stable": "holding steady ➡️"}.get(trend, "")
            await self._telegram.send_raw(
                f"🧠 *Boss, ML just leveled up.*\n"
                f"Retrained on `{len(self._learner._trade_log)}` closed trades.\n"
                f"Accuracy: `{acc:.1%}` — {trend_msg}\n"
                f"Entry gate: `{self._ml.dynamic_p_win_threshold:.0%}` win prob required\n"
                f"Sizing multiplier: `{self._ml.kelly_adjustment(self._learner._trade_log):.2f}x`\n"
                f"Top pattern I see: `{top_feat}`"
            )

        # Live readiness check after every trade
        readiness = self._readiness.check(
            self._learner._trade_log, self._ml, self._equity_curve
        )
        if not readiness.passed and readiness.trade_count >= 5:
            needed = [n for n, c in readiness.criteria.items() if not c.get("passed", True)]
            log.info("Readiness: %d trades — failing: %s", readiness.trade_count, ", ".join(needed) or "none")

        # Track consecutive wins for fund manager sizing
        if pnl > 0:
            self._consecutive_wins += 1
        else:
            self._consecutive_wins = 0

        # Auto-live: switch when all criteria pass (if enabled + not already live)
        if (readiness.passed
                and not self._live_transition_done
                and self._trading["mode"] == "paper"
                and self._safety.get("auto_live_on_readiness", False)):
            self._live_transition_done = True
            await self._transition_to_live()

        # Equity tier unlock check
        tier_changes = self._fund_mgr.update_tiers(self._risk.state.equity)
        for change in tier_changes:
            log.info("Fund manager tier unlock: %s", change)
            await self._telegram.send_raw(f"📊 *Fund Manager: Tier Unlocked*\n`{change}`\nEquity: ${self._risk.state.equity:,.2f}")

        # Milestone check
        new_milestones = self._fund_mgr.check_milestones(self._risk.state.equity)
        for m, gift_emoji, gift_desc in new_milestones:
            await self._telegram.milestone_alert(
                self._risk.state.equity, m, self._risk.state.peak_equity,
                gift_emoji=gift_emoji, gift_desc=gift_desc,
            )

        # Monthly report (reset every 30 days)
        if time.time() - self._last_monthly_reset_ts >= 86400 * 30:
            n_trades = len(self._learner._trade_log)
            fm_report = self._fund_mgr.build_report(
                self._learner._trade_log,
                self._risk.state.equity,
                self._risk.state.peak_equity,
                self._equity_curve,
                self._trading["mode"],
            )
            await self._telegram.monthly_report(fm_report)
            self._fund_mgr.reset_monthly(self._risk.state.equity, n_trades)
            self._last_monthly_reset_ts = time.time()

        # Performance report every N trades
        report_interval = self._safety.get("performance_report_interval_trades", 10)
        n_trades = len(self._learner._trade_log)
        if n_trades > 0 and n_trades >= self._last_report_trade_count + report_interval:
            self._last_report_trade_count = n_trades
            fm_report = self._fund_mgr.build_report(
                self._learner._trade_log,
                self._risk.state.equity,
                self._risk.state.peak_equity,
                self._equity_curve,
                self._trading["mode"],
            )
            await self._telegram.performance_report(
                fm_report,
                self._risk.state.equity,
                self._trading["mode"],
                open_positions=self._telegram_open_positions(),
                recent_closed=[
                    {
                        "symbol": t.symbol,
                        "direction": t.direction,
                        "pnl_usd": t.pnl_usd,
                        "reason": t.reason,
                    }
                    for t in self._learner._trade_log[-3:]
                ],
            )

        emoji = "✅" if pnl > 0 else "❌"
        console.print(
            f"{emoji} [bold]{trade.symbol}[/bold] {trade.direction.upper()} closed "
            f"pnl=[{'green' if pnl > 0 else 'red'}]${pnl:.2f}[/] ({pnl_pct:.1f}%)  reason={reason}"
        )
        await self._telegram.trade_closed(
            symbol=trade.symbol,
            direction=trade.direction,
            pnl_usd=pnl,
            pnl_pct=pnl_pct,
            reason=reason,
            entry=entry_p,
            exit_price=close_p,
        )

    async def _on_trade_progress(self, trade: OpenTrade, event: str, price: float) -> None:
        await self._telegram.trade_progress(
            symbol=trade.symbol,
            direction=trade.direction,
            event=event,
            price=price,
            remaining_contracts=trade.remaining_contracts,
            trailing_stop=trade.trailing_stop,
        )

    async def _transition_to_live(self) -> None:
        """Auto-switch from paper to live mode after all readiness criteria pass."""
        log.info("AUTO-LIVE: readiness criteria passed — preparing live transition")

        # Validate live API keys are present
        api_key = self._cfg["exchange"].get("api_key", "")
        api_secret = self._cfg["exchange"].get("api_secret", "")
        if not api_key or not api_secret:
            log.error("AUTO-LIVE ABORTED: BINANCE_API_KEY/SECRET not set in .env")
            await self._telegram.send_raw(
                "⚠️ *AUTO-LIVE ABORTED*\n"
                "All readiness criteria passed ✅ but live API keys are missing.\n\n"
                "Set `BINANCE_API_KEY` and `BINANCE_API_SECRET` in `.env`, then restart.\n"
                "Bot continues paper trading."
            )
            self._live_transition_done = False  # allow retry after keys are set
            return

        await self._telegram.send_raw(
            "🔄 *Switching to LIVE mode...*\n"
            "Closing all paper positions first."
        )

        # Close all paper positions cleanly
        await self._trade_mgr.close_all("auto_live_transition")

        # Switch config to live
        self._cfg["trading"]["mode"] = "live"
        self._cfg["exchange"]["testnet"] = False
        self._trading = self._cfg["trading"]

        # Update executor paper flag
        self._executor._paper = False

        # Test live credentials before committing the switch
        from src.data.client import BinanceFuturesClient
        test_cfg = {**self._cfg, "exchange": {**self._cfg["exchange"], "testnet": False}}
        test_client = BinanceFuturesClient(test_cfg)
        try:
            await test_client.connect()
            balance = await test_client.fetch_balance()
            usdt = float(balance.get("USDT", {}).get("free", 0))
            await test_client.close()
        except Exception as exc:
            await test_client.close()
            log.error("AUTO-LIVE ABORTED: live authentication failed — %s", exc)
            await self._telegram.send_raw(
                "⚠️ *AUTO-LIVE ABORTED*\n"
                "Readiness criteria passed ✅ but live Binance authentication failed.\n\n"
                f"`{exc}`\n\n"
                "Ensure `BINANCE_API_KEY` and `BINANCE_API_SECRET` are *mainnet* (not testnet) keys.\n"
                "Bot continues paper trading."
            )
            self._live_transition_done = False
            return

        # Credentials verified — now commit the switch
        await self._client.close()
        self._client = BinanceFuturesClient(test_cfg)
        self._trade_mgr._client = self._client
        self._executor._client = self._client
        self._market_data._client = self._client
        self._scanner._client = self._client
        await self._client.connect()
        await self._client.set_position_mode_one_way()

        self._risk.update_equity(usdt)
        self._risk.state.reset_day(usdt)

        log.info("AUTO-LIVE: now LIVE with $%.2f USDT", usdt)
        await self._telegram.live_transition(usdt)

    async def _reconcile_positions(self) -> None:
        """Sync local trade state with actual exchange positions every 5 min (live mode only)."""
        try:
            exchange_positions = await self._client.fetch_positions()
        except Exception as exc:
            log.warning("Reconciliation: could not fetch positions: %s", exc)
            return

        live_symbols: set[str] = set()
        for p in exchange_positions:
            if abs(float(p.get("contracts", 0) or 0)) > 0:
                live_symbols.add(p.get("symbol", ""))
                live_symbols.add(p.get("info", {}).get("symbol", ""))

        for symbol in list(self._trade_mgr.open_symbols):
            if symbol not in live_symbols:
                trade = self._trade_mgr._trades.get(symbol)
                if trade is None:
                    continue
                log.warning("Reconcile: %s not on exchange — forcing local close", symbol)
                try:
                    ticker = await self._client.fetch_ticker(symbol)
                    last_price = float(ticker.get("last", trade.setup.entry_price) or trade.setup.entry_price)
                except Exception:
                    last_price = trade.setup.entry_price
                await self._trade_mgr._close_trade(trade, last_price, "exchange_closed")

    async def _shutdown(self) -> None:
        close_on_shutdown = self._cfg.get("safety", {}).get(
            "close_positions_on_shutdown",
            self._trading["mode"] != "live",
        )
        if close_on_shutdown:
            log.info("Shutting down — closing all open trades...")
            await self._trade_mgr.close_all("shutdown")
        else:
            log.warning(
                "Shutting down — preserving open trades because close_positions_on_shutdown=false"
            )
        await self._client.close()
        log.info("Ninja Trader stopped.")

    def _write_state(
        self,
        breakdowns: list[SignalBreakdown],
        lifecycle_report: dict | None = None,
        attribution_report: dict | None = None,
    ) -> None:
        import json
        try:
            open_trades = self._telegram_open_positions()
            signals = []
            for b in breakdowns:
                fv = b.feature_vector
                signals.append({
                    "symbol": b.symbol,
                    "direction": b.direction,
                    "score": b.total_score,
                    "base_score": b.base_score,
                    "legacy_score": b.legacy_score,
                    "regime": b.regime.value if b.regime else "",
                    "sm": b.smart_money.phase.value if b.smart_money else "N/A",
                    "strategy": b.strategy_sleeve,
                    "strategy_reason": b.strategy_reason,
                    "strategy_score_mult": b.strategy_score_mult,
                    "strategy_size_mult": b.strategy_size_mult,
                    "dispersion_value": b.dispersion_value,
                    "dispersion_state": b.dispersion_state,
                    "cohort_key": b.cohort_key,
                    "lifecycle_status": b.lifecycle_status,
                    "recommendation": b.recommendation,
                    "gates": f"{'R' if b.regime_ok else '-'}{'S' if b.smart_money_ok else '-'}{'E' if b.ev_ok else '-'}",
                    "threshold": self._threshold_for(b.regime) + getattr(b, "strategy_threshold_shift", 0.0),
                    "ev": b.ev_result.ev_net_pct if b.ev_result else 0.0,
                    "pwin": b.ev_result.p_win if b.ev_result else 0.0,
                    "oi_score": b.open_interest,
                    "oi_change_pct": fv.oi_change_rate if fv else 0.0,
                })
            shadow = {}
            if self._shadow:
                m = self._shadow.metrics()
                shadow = {
                    "closed": m.closed, "open": m.open,
                    "winrate": m.winrate, "profit_factor": m.profit_factor,
                    "net_pnl_pct": m.net_pnl_pct, "max_dd": m.max_dd,
                    "trending_pf": m.trending_pf, "compression_pf": m.compression_pf,
                }
            attribution = attribution_report or build_attribution_report(self._learner._trade_log, min_trades=2)
            state = {
                "updated_at": time.time(),
                "mode": self._trading["mode"],
                "equity": self._risk.state.equity,
                "peak_equity": self._risk.state.peak_equity,
                "daily_pnl_pct": self._risk.state.daily_pnl_pct,
                "weekly_pnl_pct": self._risk.state.weekly_pnl_pct,
                "drawdown_pct": self._risk.state.drawdown_pct,
                "consecutive_losses": self._risk.state.consecutive_losses,
                "consecutive_wins": self._consecutive_wins,
                "kill_switch_reason": self._risk.state.kill_switch_reason,
                "paused_until_ts": self._risk.state.paused_until_ts,
                "starting_equity": self._starting_equity,
                "equity_curve": self._equity_curve[-500:],
                "open_trades": open_trades,
                "top_signals": signals,
                "shadow": shadow,
                "attribution": attribution,
                "lifecycle": lifecycle_report or self._lifecycle.build_report(self._learner._trade_log),
            }
            Path("data/state.json").write_text(json.dumps(state))
        except Exception:
            pass

    def _telegram_open_positions(self) -> list[dict]:
        open_trades = []
        for sym, trade in self._trade_mgr._trades.items():
            open_trades.append({
                "symbol": sym,
                "direction": trade.direction,
                "entry": trade.setup.entry_price,
                "sl": trade.setup.stop_loss,
                "tp1": trade.setup.tp1,
                "tp2": trade.setup.tp2,
                "size_usd": trade.setup.size_usd,
                "tp1_hit": trade.tp1_hit,
                "opened_at": trade.opened_at,
                "strategy_sleeve": getattr(trade.setup, "strategy_sleeve", "neutral"),
                "exit_profile": getattr(trade.setup, "exit_profile", "default"),
            })
        return open_trades

    def _threshold_for(self, regime) -> float:
        thresholds = self._trading.get("regime_thresholds", {})
        key = regime.value if regime else ""
        base = thresholds.get(key, self._trading["min_score_threshold"])
        if self._trading["mode"] == "paper" and self._exploration:
            return max(50.0, float(base) - 15.0)
        if self._paper_validation_enabled():
            relax = float(self._paper_validation.get("threshold_relaxation", 0.0) or 0.0)
            floor = float(self._paper_validation.get("min_score_floor", 58.0) or 58.0)
            return max(floor, float(base) - relax)
        return float(base)

    def _print_trade_card(self, bd: SignalBreakdown, setup) -> None:
        sm = bd.smart_money
        ev = bd.ev_result
        color = "green" if bd.direction == "long" else "red"
        tp1_pct = abs(setup.tp1 - setup.entry_price) / setup.entry_price * 100
        tp2_pct = abs(setup.tp2 - setup.entry_price) / setup.entry_price * 100
        tp3_pct = abs(setup.tp3 - setup.entry_price) / setup.entry_price * 100
        sl_pct  = abs(setup.stop_loss - setup.entry_price) / setup.entry_price * 100

        console.rule(f"[bold {color}] TRADE SIGNAL: {bd.symbol} {bd.direction.upper()} [/]")
        console.print(f"  [cyan]Symbol:[/]          {bd.symbol}")
        console.print(f"  [cyan]Direction:[/]       [{color}]{bd.direction.upper()}[/]")
        console.print(f"  [cyan]Entry Zone:[/]      ${setup.entry_price:,.4f}")
        console.print(f"  [cyan]Stop Loss:[/]       ${setup.stop_loss:,.4f}  (-{sl_pct:.2f}%)")
        console.print(f"  [cyan]Take Profit 1:[/]   ${setup.tp1:,.4f}  (+{tp1_pct:.2f}%)  50%")
        console.print(f"  [cyan]Take Profit 2:[/]   ${setup.tp2:,.4f}  (+{tp2_pct:.2f}%)  30%")
        console.print(f"  [cyan]Take Profit 3:[/]   ${setup.tp3:,.4f}  (+{tp3_pct:.2f}%)  20% trail")
        if ev:
            pw_color = "green" if ev.p_win > 0.55 else "yellow"
            ev_color = "green" if ev.ev_net_pct > 0 else "red"
            console.print(f"  [cyan]P(win):[/]          [{pw_color}]{ev.p_win:.1%}[/]  (confidence={ev.confidence:.1%})")
            console.print(f"  [cyan]EV (net):[/]        [{ev_color}]{ev.ev_net_pct:+.3f}%[/]  (gross={ev.ev_gross_pct:+.3f}%  cost={ev.cost_pct:.3f}%)")
        console.print(f"  [cyan]Risk %:[/]          {setup.risk_pct:.2f}%  (${setup.size_usd:,.2f} notional)")
        console.print(f"  [cyan]Market Regime:[/]   {bd.regime.label}")
        console.print(
            f"  [cyan]Strategy Sleeve:[/] {bd.strategy_sleeve}  "
            f"(base={bd.base_score:.1f}  mult={bd.strategy_score_mult:.2f}x  disp={bd.dispersion_value:.1f}/{bd.dispersion_state})"
        )
        console.print(
            f"  [cyan]Exit Profile:[/]    {setup.exit_profile}  "
            f"(TP1={setup.tp1_size_pct:.0%}  TP2={setup.tp2_size_pct:.0%}  trail={setup.trail_size_pct:.0%})"
        )
        if bd.strategy_reason:
            console.print(f"  [cyan]Strategy Note:[/]   {bd.strategy_reason}")
        if sm:
            console.print(f"  [cyan]Smart Money:[/]     [{color}]{sm.phase.value}[/]  score={sm.score:.0f}")
            console.print(f"  [cyan]SM Reasoning:[/]    {sm.reasoning}")
            console.print(f"  [cyan]OI Signal:[/]       {sm.oi_narrative}")
            console.print(f"  [cyan]Funding Signal:[/]  {sm.funding_narrative}")
        console.print(f"  [cyan]Execution Plan:[/]  Limit @ ${setup.entry_price:,.4f}  SL @ ${setup.stop_loss:,.4f}  TP1/TP2/Trail")
        console.print(f"  [cyan]Score:[/]           {bd.total_score:.1f} / 100")
        console.rule()

    def _display_scores(self, breakdowns: list[SignalBreakdown]) -> None:
        if not breakdowns:
            return
        tbl = Table(title="Top Signals", show_lines=False)
        tbl.add_column("Symbol", style="cyan", no_wrap=True)
        tbl.add_column("Dir", style="bold")
        tbl.add_column("Score", justify="right")
        tbl.add_column("Regime", no_wrap=True)
        tbl.add_column("SM Phase", no_wrap=True)
        tbl.add_column("P(win)", justify="right")
        tbl.add_column("EV%", justify="right")
        tbl.add_column("Gates")
        tbl.add_column("Sleeve", no_wrap=True)
        tbl.add_column("Trend", justify="right")
        tbl.add_column("OI", justify="right")
        tbl.add_column("Fund", justify="right")

        for b in breakdowns:
            color = "green" if b.direction == "long" else "red"
            score_color = "green" if b.total_score >= 75 else "yellow" if b.total_score >= 60 else "white"
            regime_short = {
                "trending_expansion": "[green]Trend↑[/]",
                "accumulation_compression": "[blue]Accum[/]",
                "distribution": "[red]Dist[/]",
                "chaos": "[bold red]CHAOS[/]",
            }.get(b.regime.value if b.regime else "", "?")
            sm_phase = b.smart_money.phase.value[:6] if b.smart_money else "N/A"
            ev_str = f"{b.ev_result.ev_net_pct:+.2f}%" if b.ev_result else "N/A"
            pwin_str = f"{b.ev_result.p_win:.0%}" if b.ev_result else "N/A"
            gates = (
                f"{'R' if b.regime_ok else '-'}"
                f"{'S' if b.smart_money_ok else '-'}"
                f"{'E' if b.ev_ok else '-'}"
            )
            gate_color = "green" if b.all_gates_passed else "yellow" if b.regime_ok else "red"
            tbl.add_row(
                b.symbol,
                f"[{color}]{b.direction}[/]",
                f"[{score_color}]{b.total_score:.1f}[/]",
                regime_short,
                sm_phase,
                pwin_str,
                ev_str,
                f"[{gate_color}]{gates}[/]",
                b.strategy_sleeve[:8],
                f"{b.trend_strength:.0f}",
                f"{b.open_interest:.0f}",
                f"{b.funding_sentiment:.0f}",
            )
        console.print(tbl)


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.mode:
        cfg["trading"]["mode"] = args.mode
    if args.testnet is not None:
        cfg["exchange"]["testnet"] = args.testnet
    cfg = normalize_config(cfg)

    setup_logging(cfg)
    bot = NinjaTrader(cfg)
    try:
        await bot.start()
    except Exception:
        try:
            await bot._client.close()
        except Exception:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Ninja Trader — Binance Futures bot")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent.parent / "config" / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument("--mode", choices=["paper", "live", "backtest"], default=None)
    parser.add_argument("--testnet", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")


if __name__ == "__main__":
    main()
