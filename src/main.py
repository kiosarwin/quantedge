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
import math
import os
import pathlib
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from src.data.client import BinanceFuturesClient
from src.data.market_data import MarketDataService, MarketSnapshot
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
from src.models.sector_rotation import attach_sector_rotation, compute_sector_rotation
from src.models.strategy_passport import asset_from_symbol, sector_for_asset
from src.models.adaptive_brain import AdaptiveBrain
from src.risk.correlation_filter import CorrelationFilter
from src.risk.session_modulator import SessionModulator
from src.data.dataset_logger import DatasetLogger
from src.data.crypto_news import build_market_hot_narratives, fetch_crypto_news_highlights
from src.analysis.rejection_logger import RejectionLogger
from src.reporting.attribution import build_attribution_report
from src.reporting.bootstrap import build_bootstrap_audit_report, format_bootstrap_audit_lines
from src.reporting.trading_brain import TradingBrainJournal
from src.session_clock import active_market_session_key
from src.config import validate_config, ConfigValidationError

console = Console()
log = logging.getLogger("ninja_trader")


def setup_passport_with_market_context(breakdown) -> dict:
    """Attach entry broad-market context to the persisted setup passport."""
    passport_obj = getattr(breakdown, "setup_passport", None)
    passport = dict(passport_obj.as_dict() if passport_obj is not None else {})
    market_ctx = getattr(breakdown, "market_context", None)
    sector_rotation = getattr(breakdown, "sector_rotation", None)
    if sector_rotation is not None:
        rotation_context = {
            "sector": str(getattr(sector_rotation, "sector", "unknown") or "unknown"),
            "state": str(getattr(sector_rotation, "state", "unknown") or "unknown"),
            "rank": int(getattr(sector_rotation, "rank", 0) or 0),
            "relative_btc_pct": float(getattr(sector_rotation, "relative_btc_pct", 0.0) or 0.0),
            "breadth": float(getattr(sector_rotation, "breadth", 0.0) or 0.0),
            "confidence": float(getattr(sector_rotation, "confidence", 0.0) or 0.0),
        }
        passport.setdefault("sector_rotation", dict(rotation_context))
        passport.setdefault("sector_rotation_state", rotation_context["state"])
        passport.setdefault("sector_rotation_rank", rotation_context["rank"])
        passport.setdefault("sector_rotation_relative_btc_pct", rotation_context["relative_btc_pct"])
        passport.setdefault("sector_rotation_confidence", rotation_context["confidence"])
    if market_ctx is None:
        return passport

    context = {
        "risk_on_state": str(getattr(market_ctx, "risk_on_state", "unknown") or "unknown"),
        "rotation_state": str(getattr(market_ctx, "rotation_state", "unknown") or "unknown"),
        "btc_trend": str(getattr(market_ctx, "btc_trend", "unknown") or "unknown"),
        "eth_btc_trend": str(getattr(market_ctx, "eth_btc_trend", "unknown") or "unknown"),
        "btc_d_trend": str(getattr(market_ctx, "btc_d_trend", "unknown") or "unknown"),
        "total_trend": str(getattr(market_ctx, "total_trend", "unknown") or "unknown"),
        "confidence": float(getattr(market_ctx, "confidence", 0.0) or 0.0),
    }
    passport.setdefault("market_context", dict(context))
    passport.setdefault("market_risk_on_state", context["risk_on_state"])
    passport.setdefault("market_rotation_state", context["rotation_state"])
    passport.setdefault("market_btc_trend", context["btc_trend"])
    passport.setdefault("market_eth_btc_trend", context["eth_btc_trend"])
    passport.setdefault("market_btc_d_trend", context["btc_d_trend"])
    passport.setdefault("market_total_trend", context["total_trend"])
    passport.setdefault("market_context_confidence", context["confidence"])
    return passport


def enrich_closed_trade_scores(scores: dict | None, trade, cfg: dict) -> dict:
    """Backfill closed-trade metadata when a restored open trade lacks pending scores."""
    enriched = dict(scores or {})
    setup = getattr(trade, "setup", None)
    passport = getattr(setup, "setup_passport", {}) or {}
    opened_at = float(getattr(trade, "opened_at", 0.0) or time.time())
    opened_dt = datetime.fromtimestamp(opened_at, tz=timezone.utc)
    opened_tm = time.gmtime(opened_at)
    symbol = str(getattr(trade, "symbol", "") or getattr(setup, "symbol", ""))
    asset = str(passport.get("asset") or asset_from_symbol(symbol))
    sector = str(passport.get("sector") or sector_for_asset(asset))
    regime = str(passport.get("regime") or "unknown")
    sm_phase = str(passport.get("sm_phase") or "neutral")
    regime_code = {
        "trending_expansion": 3.0,
        "accumulation_compression": 2.0,
        "distribution": 1.0,
        "chaos": 0.0,
    }.get(regime, enriched.get("regime_code", -1.0))

    enriched.setdefault("timeframe", cfg.get("timeframes", {}).get("primary", "unknown"))
    enriched.setdefault("session", active_market_session_key(opened_dt))
    enriched.setdefault("hour_utc", float(opened_tm.tm_hour))
    enriched.setdefault("hour_of_day", int(opened_tm.tm_hour))
    enriched.setdefault("day_of_week", int(opened_tm.tm_wday))
    enriched.setdefault("asset", asset)
    enriched.setdefault("sector", sector)
    passport_market_ctx = passport.get("market_context") if isinstance(passport.get("market_context"), dict) else {}
    enriched.setdefault("market_risk_on_state", passport.get("market_risk_on_state") or passport_market_ctx.get("risk_on_state", "unknown"))
    enriched.setdefault("market_rotation_state", passport.get("market_rotation_state") or passport_market_ctx.get("rotation_state", "unknown"))
    enriched.setdefault("market_btc_trend", passport.get("market_btc_trend") or passport_market_ctx.get("btc_trend", "unknown"))
    enriched.setdefault("market_eth_btc_trend", passport.get("market_eth_btc_trend") or passport_market_ctx.get("eth_btc_trend", "unknown"))
    enriched.setdefault("market_btc_d_trend", passport.get("market_btc_d_trend") or passport_market_ctx.get("btc_d_trend", "unknown"))
    enriched.setdefault("market_total_trend", passport.get("market_total_trend") or passport_market_ctx.get("total_trend", "unknown"))
    enriched.setdefault("market_context_confidence", float(passport.get("market_context_confidence") or passport_market_ctx.get("confidence", 0.0) or 0.0))
    enriched.setdefault("setup_passport", dict(passport))
    enriched.setdefault("strategy_sleeve", str(getattr(setup, "strategy_sleeve", "") or passport.get("sleeve", "")))
    enriched.setdefault("exit_profile", str(getattr(setup, "exit_profile", "") or ""))
    enriched.setdefault("setup_type", str(passport.get("setup_type", "unknown")))
    enriched.setdefault("setup_quality_score", float(passport.get("quality_score", 0.0) or 0.0))
    enriched.setdefault("signal_type", f"{regime}|{sm_phase}")
    enriched.setdefault("entry_reason", str(passport.get("reason", "unknown") or "unknown"))
    enriched.setdefault("regime_code", float(regime_code))
    enriched.setdefault("sm_phase", sm_phase)
    return enriched


def cleanup_workspace_artifacts(root: Path | None = None) -> dict[str, int]:
    """Remove non-essential local artifacts that should not accumulate across runs."""
    workspace = pathlib.Path(root).resolve() if root is not None else pathlib.Path.cwd().resolve()
    removed_files = 0
    removed_dirs = 0

    def _is_runtime_safe(path: Path) -> bool:
        parts = path.parts
        return ".git" not in parts and "venv" not in parts and ".venv" not in parts

    for cache_dir in workspace.rglob("__pycache__"):
        if cache_dir.is_dir() and _is_runtime_safe(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
            removed_dirs += 1

    pytest_cache = workspace / ".pytest_cache"
    if pytest_cache.exists() and _is_runtime_safe(pytest_cache):
        shutil.rmtree(pytest_cache, ignore_errors=True)
        removed_dirs += 1

    for pattern in ("*.pyc", "config/*.bak*", "logs/*test*.log"):
        for path in workspace.glob(pattern):
            if path.is_file() and _is_runtime_safe(path):
                path.unlink(missing_ok=True)
                removed_files += 1

    return {"files": removed_files, "dirs": removed_dirs}


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
        # Paper mode never places real orders; use the configured public market-data endpoint.
        # Binance futures testnet tickers can go stale, which freezes floating PnL/equity.
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
        # Adaptive AI Brain — Thompson bandit + online ML + regime HMM + alpha decay + vol target
        self._brain = AdaptiveBrain(cfg)
        self._session_mod = SessionModulator(cfg)
        self._corr_filter = CorrelationFilter(cfg)
        self._paper_validation = cfg.get("paper_validation", {})
        self._equity_curve: list[float] = []
        self._equity_graph_points: list[dict[str, float]] = []
        self._running = False
        self._heartbeat_ts = 0.0
        self._tg_heartbeat_ts = 0.0
        self._tg_cycle_report_ts = 0.0
        self._tg_equity_graph_ts = 0.0
        self._startup_cycle_report_pending = True
        self._last_fm_report_ts = 0.0       # for fund manager report interval
        self._last_bootstrap_audit_ts = time.time()  # separate bootstrap audit cadence
        self._tg_crypto_news_ts = 0.0      # independent crypto news cadence
        self._live_transition_done = False  # guard against double-transition
        self._last_monthly_reset_ts = time.time()
        self._last_reconcile_ts = 0.0
        self._consecutive_wins = 0          # tracked separately from risk_manager losses
        self._starting_equity: float = 0.0  # set after equity seed, used for total PnL display
        self._runtime_error_count = 0
        self._heartbeat_fail_count = 0
        risk_state = getattr(self._risk, "state", None)
        self._ml_gate_state = {
            "day_start_ts": float(getattr(risk_state, "day_start_ts", time.time()) or time.time()),
            "candidate_count": 0,
            "veto_count": 0,
        }
        self._rotation_state = {
            "day_start_ts": float(getattr(risk_state, "day_start_ts", time.time()) or time.time()),
            "count": 0,
        }

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

        # Trading brain journal -- read-only local diagnostics, no decision impact
        self._trading_brain = TradingBrainJournal(cfg)

        # Rejection forensics — diagnostics sink, zero behavioral impact
        self._rej = RejectionLogger(cfg)


    def _rotation_config(self) -> dict:
        return self._cfg.get("position_rotation", {}) or {}

    def _sync_rotation_budget(self) -> None:
        risk_state = getattr(self._risk, "state", None)
        current_day_start = float(getattr(risk_state, "day_start_ts", 0.0) or 0.0)
        if self._rotation_state.get("day_start_ts") != current_day_start:
            self._rotation_state = {"day_start_ts": current_day_start, "count": 0}

    def _rotation_budget_allows(self, rotations_this_cycle: int) -> tuple[bool, str]:
        cfg = self._rotation_config()
        if not bool(cfg.get("enabled", False)):
            return False, "position rotation disabled"
        max_cycle = int(cfg.get("max_rotations_per_cycle", 1) or 0)
        if max_cycle <= 0 or rotations_this_cycle >= max_cycle:
            return False, "rotation cycle budget exhausted"
        self._sync_rotation_budget()
        max_day = int(cfg.get("max_rotations_per_day", 3) or 0)
        if max_day <= 0 or int(self._rotation_state.get("count", 0)) >= max_day:
            return False, "rotation daily budget exhausted"
        return True, "rotation budget ok"

    @staticmethod
    def _trade_unrealized_r(trade: OpenTrade, price: float) -> float:
        r = float(getattr(trade.setup, "r_distance", 0.0) or 0.0)
        if r <= 0.0 or price <= 0.0:
            return 0.0
        if trade.direction == "long":
            return (price - trade.setup.entry_price) / r
        return (trade.setup.entry_price - price) / r

    @staticmethod
    def _trade_unrealized_pct(trade: OpenTrade, price: float) -> float:
        entry = float(getattr(trade.setup, "entry_price", 0.0) or 0.0)
        if entry <= 0.0 or price <= 0.0:
            return 0.0
        if trade.direction == "long":
            return (price - entry) / entry * 100.0
        return (entry - price) / entry * 100.0

    def _candidate_rotation_score(self, breakdown: SignalBreakdown) -> float:
        ev = getattr(breakdown, "ev_result", None)
        ev_bonus = max(-6.0, min(6.0, float(getattr(ev, "ev_net_pct", 0.0) or 0.0) * 4.0))
        quality_bonus = max(0.0, (float(getattr(breakdown, "setup_quality_score", 0.0) or 0.0) - 50.0) * 0.15)
        return self._positive_ev_probe_score_basis(breakdown) + ev_bonus + quality_bonus

    def _open_trade_rotation_strength(self, trade: OpenTrade, price: float) -> float:
        unrealized_r = self._trade_unrealized_r(trade, price)
        has_passport = bool(getattr(trade.setup, "setup_passport", None))
        passport_bonus = 4.0 if has_passport else -4.0
        ev_net = float(getattr(trade.setup, "ev_net_pct", 0.0) or 0.0)
        ev_score = max(-8.0, min(8.0, ev_net * 8.0))
        tp_bonus = 10.0 if trade.tp1_hit else 0.0
        mfe_bonus = min(12.0, max(0.0, float(getattr(trade, "mfe_r", 0.0) or 0.0)) * 8.0)
        pnl_score = max(-18.0, min(18.0, unrealized_r * 14.0))
        return 50.0 + pnl_score + passport_bonus + ev_score + tp_bonus + mfe_bonus

    def _rotation_positive_ev_probe_mode(self, breakdown: SignalBreakdown) -> str | None:
        try:
            threshold = self._threshold_for(getattr(breakdown, "regime", None))
            return self._paper_positive_ev_probe_mode(breakdown, threshold)
        except Exception:
            return None

    def _select_rotation_victim(
        self,
        breakdown: SignalBreakdown,
        snapshots: dict[str, MarketSnapshot],
        rotations_this_cycle: int,
    ) -> tuple[OpenTrade | None, float, str]:
        allowed, budget_reason = self._rotation_budget_allows(rotations_this_cycle)
        if not allowed:
            return None, 0.0, budget_reason

        cfg = self._rotation_config()
        candidate_score = self._candidate_rotation_score(breakdown)
        positive_ev_probe = self._rotation_positive_ev_probe_mode(breakdown)
        min_score = float(cfg.get("min_candidate_score", 72.0) or 72.0)
        if positive_ev_probe:
            min_score = float(
                cfg.get("positive_ev_probe_min_candidate_score", min_score) or min_score
            )
        if candidate_score < min_score:
            return None, 0.0, f"candidate_score={candidate_score:.1f} < {min_score:.1f}"

        quality = float(getattr(breakdown, "setup_quality_score", 0.0) or 0.0)
        min_quality = float(cfg.get("min_candidate_quality", 58.0) or 58.0)
        if positive_ev_probe:
            min_quality = float(
                cfg.get("positive_ev_probe_min_candidate_quality", min_quality) or 0.0
            )
        if quality < min_quality:
            return None, 0.0, f"setup_quality={quality:.1f} < {min_quality:.1f}"

        ev = getattr(breakdown, "ev_result", None)
        p_win = float(getattr(ev, "p_win", 0.0) or 0.0)
        min_p_win = float(cfg.get("min_candidate_p_win", 0.38) or 0.38)
        if p_win < min_p_win:
            return None, 0.0, f"p_win={p_win:.3f} < {min_p_win:.3f}"
        ev_net = float(getattr(ev, "ev_net_pct", 0.0) or 0.0)
        min_ev = float(cfg.get("min_candidate_ev_net_pct", -0.75) or -0.75)
        if ev_net < min_ev:
            return None, 0.0, f"ev_net={ev_net:+.3f}% < {min_ev:+.3f}%"

        now = time.time()
        min_age_s = float(cfg.get("min_victim_age_s", 900.0) or 900.0)
        protect_after_tp1 = bool(cfg.get("protect_after_tp1", True))
        protect_mfe_r = float(cfg.get("protect_mfe_r", 0.80) or 0.80)
        protect_gain_r = float(cfg.get("protect_unrealized_gain_r", 0.30) or 0.30)
        protect_gain_pct = float(cfg.get("protect_unrealized_gain_pct", 3.0) or 3.0)
        protect_winner_min_age_s = float(cfg.get("protect_winner_min_age_s", 21600.0) or 21600.0)
        max_adverse_r = float(cfg.get("max_victim_adverse_r", 0.90) or 0.90)
        best: tuple[float, OpenTrade, float, float] | None = None
        for trade in self._trade_mgr.open_trades:
            snap = snapshots.get(trade.symbol)
            price = float(getattr(snap, "last_price", 0.0) or 0.0) if snap is not None else 0.0
            if price <= 0.0:
                continue
            age = now - float(getattr(trade, "opened_at", now) or now)
            if age < min_age_s:
                continue
            if protect_after_tp1 and trade.tp1_hit:
                continue
            unrealized_r = self._trade_unrealized_r(trade, price)
            unrealized_pct = self._trade_unrealized_pct(trade, price)
            if unrealized_r >= protect_gain_r:
                continue
            if unrealized_pct >= protect_gain_pct:
                continue
            if age < protect_winner_min_age_s and unrealized_pct > 0.0:
                continue
            if float(getattr(trade, "mfe_r", 0.0) or 0.0) >= protect_mfe_r:
                continue
            if unrealized_r <= -max_adverse_r:
                continue
            strength = self._open_trade_rotation_strength(trade, price)
            if best is None or strength < best[0]:
                best = (strength, trade, price, unrealized_r)

        if best is None:
            return None, 0.0, "no eligible weak open trade"
        victim_strength, victim, victim_price, victim_r = best
        min_advantage = float(cfg.get("min_score_advantage", 14.0) or 14.0)
        if positive_ev_probe:
            min_advantage = float(
                cfg.get("positive_ev_probe_min_score_advantage", min_advantage) or 0.0
            )
        advantage = candidate_score - victim_strength
        if advantage < min_advantage:
            return None, 0.0, (
                f"candidate advantage {advantage:.1f} < {min_advantage:.1f} "
                f"over {victim.symbol}"
            )
        return victim, victim_price, (
            f"candidate_score={candidate_score:.1f} victim={victim.symbol} "
            f"victim_strength={victim_strength:.1f} victim_r={victim_r:+.2f} advantage={advantage:.1f}"
        )

    def _record_position_rotation(self) -> None:
        self._sync_rotation_budget()
        self._rotation_state["count"] = int(self._rotation_state.get("count", 0)) + 1

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
        global_allowed_sleeves = set(
            self._paper_validation.get("allowed_strategy_sleeves", ["trend_following"])
            or ["trend_following"]
        )
        allowed_long_sleeves = set(
            self._paper_validation.get(
                "allowed_long_strategy_sleeves",
                sorted(global_allowed_sleeves),
            )
            or sorted(global_allowed_sleeves)
        )
        allowed_short_sleeves = set(
            self._paper_validation.get(
                "allowed_short_strategy_sleeves",
                sorted(global_allowed_sleeves),
            )
            or sorted(global_allowed_sleeves)
        )
        direction = str(getattr(breakdown, "direction", "") or "")
        regime = breakdown.regime.value if breakdown.regime else ""
        sleeve = str(getattr(breakdown, "strategy_sleeve", "") or "neutral")
        if direction not in allowed_directions:
            return False, f"direction={direction} not in {sorted(allowed_directions)}"
        if regime not in allowed_regimes:
            return False, f"regime={regime or 'unknown'} not in {sorted(allowed_regimes)}"
        if direction == "long":
            allowed_sleeves = allowed_long_sleeves
        elif direction == "short":
            allowed_sleeves = allowed_short_sleeves
        else:
            allowed_sleeves = global_allowed_sleeves
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
                self._cfg.get("ml", {}).get("hard_gate_min_trades", 60),
            )
        )

    def _ml_gate_config(self) -> dict:
        return self._cfg.get("ml", {})

    def _sync_ml_gate_budget(self) -> None:
        risk_state = getattr(self._risk, "state", None)
        current_day_start = float(getattr(risk_state, "day_start_ts", 0.0) or 0.0)
        if self._ml_gate_state.get("day_start_ts") != current_day_start:
            self._ml_gate_state = {
                "day_start_ts": current_day_start,
                "candidate_count": 0,
                "veto_count": 0,
            }

    def _ml_veto_budget_allows(self) -> tuple[bool, str]:
        self._sync_ml_gate_budget()
        cfg = self._ml_gate_config()
        max_daily_vetoes = int(cfg.get("max_daily_vetoes", 2) or 0)
        if max_daily_vetoes <= 0:
            self._ml_gate_state["candidate_count"] += 1
            return True, "veto budget disabled"

        self._ml_gate_state["candidate_count"] += 1
        veto_count = int(self._ml_gate_state["veto_count"])
        if veto_count >= max_daily_vetoes:
            return False, f"daily ML veto budget exhausted ({veto_count}/{max_daily_vetoes})"

        min_candidates = int(cfg.get("veto_budget_min_candidates", 12) or 12)
        candidate_count = int(self._ml_gate_state["candidate_count"])
        max_veto_rate = float(cfg.get("max_veto_rate", 0.20) or 0.20)
        if candidate_count >= min_candidates:
            veto_rate = veto_count / max(1, candidate_count)
            if veto_rate > max_veto_rate:
                return False, f"ML veto rate {veto_rate:.0%} > {max_veto_rate:.0%}"

        return True, "ok"

    def _record_ml_veto(self) -> None:
        self._sync_ml_gate_budget()
        self._ml_gate_state["veto_count"] = int(self._ml_gate_state["veto_count"]) + 1

    def _ev_gate_enabled(self) -> bool:
        return bool(self._cfg.get("ev_model", {}).get("gate_enabled", True))


    def _paper_experimental_setup_stress_check(self, breakdown: SignalBreakdown) -> tuple[bool, str]:
        if not self._paper_validation_enabled():
            return True, "paper validation disabled"
        if not bool(self._paper_validation.get("experimental_setup_stress_block_enabled", True)):
            return True, "experimental setup stress block disabled"

        setup_type = str(getattr(breakdown, "setup_type", "") or "")
        blocked_setups = set(
            self._paper_validation.get(
                "experimental_setup_stress_block_setup_types",
                ["mtf_price_action_continuation", "vwap_pullback_continuation", "liquidity_sweep_reversal"],
            )
            or []
        )
        if setup_type not in blocked_setups:
            return True, "not an experimental stress-block setup"

        risk_state = getattr(self._risk, "state", None)
        daily_pnl = float(getattr(risk_state, "daily_pnl_pct", 0.0) or 0.0)
        consecutive_losses = int(getattr(risk_state, "consecutive_losses", 0) or 0)
        daily_trigger = float(
            self._paper_validation.get(
                "experimental_setup_stress_block_daily_loss_pct",
                self._paper_validation.get("stress_tighten_daily_loss_pct", -3.0),
            )
            or -3.0
        )
        streak_trigger = int(
            self._paper_validation.get(
                "experimental_setup_stress_block_consecutive_losses",
                self._paper_validation.get("stress_tighten_consecutive_losses", 3),
            )
            or 3
        )
        daily_active = daily_trigger < 0.0 and daily_pnl <= daily_trigger
        streak_active = streak_trigger > 0 and consecutive_losses >= streak_trigger
        if daily_active or streak_active:
            return (
                False,
                f"experimental setup stress block: setup={setup_type} "
                f"daily_pnl={daily_pnl:.2f}% trigger={daily_trigger:.2f}% "
                f"losses={consecutive_losses}/{streak_trigger}",
            )
        return True, "ok"

    def _paper_loss_streak_guard_active(self) -> bool:
        if not self._paper_validation_enabled():
            return False
        if not bool(self._paper_validation.get("loss_streak_guard_enabled", True)):
            return False
        risk_state = getattr(self._risk, "state", None)
        daily_pnl = float(getattr(risk_state, "daily_pnl_pct", 0.0) or 0.0)
        consecutive_losses = int(getattr(risk_state, "consecutive_losses", 0) or 0)
        daily_trigger = float(
            self._paper_validation.get("loss_streak_guard_daily_loss_pct", -2.0) or -2.0
        )
        streak_trigger = int(
            self._paper_validation.get("loss_streak_guard_consecutive_losses", 2) or 2
        )
        daily_active = daily_trigger < 0.0 and daily_pnl <= daily_trigger
        streak_active = streak_trigger > 0 and consecutive_losses >= streak_trigger
        return daily_active or streak_active

    def _paper_loss_streak_guard_check(
        self,
        breakdown: SignalBreakdown,
        threshold: float,
    ) -> tuple[bool, str]:
        if not self._paper_loss_streak_guard_active():
            return True, "ok"

        score = float(getattr(breakdown, "total_score", 0.0) or 0.0)
        quality = float(getattr(breakdown, "setup_quality_score", 0.0) or 0.0)
        min_score = threshold + float(
            self._paper_validation.get("loss_streak_guard_min_score_buffer", 8.0) or 8.0
        )
        min_quality = float(
            self._paper_validation.get("loss_streak_guard_min_setup_quality", 68.0) or 68.0
        )
        if score < min_score:
            return False, f"loss-streak guard: score={score:.1f} < {min_score:.1f}"
        if quality < min_quality:
            return False, f"loss-streak guard: setup_quality={quality:.1f} < {min_quality:.1f}"

        ev = getattr(breakdown, "ev_result", None)
        require_ev = bool(self._paper_validation.get("loss_streak_guard_require_ev", True))
        if ev is None:
            if require_ev:
                return False, "loss-streak guard: ev=none"
            return True, "ok"

        min_pwin = float(
            self._paper_validation.get("loss_streak_guard_min_p_win", 0.42) or 0.42
        )
        min_ev = float(
            self._paper_validation.get("loss_streak_guard_min_ev_net_pct", 0.0) or 0.0
        )
        p_win = float(getattr(ev, "p_win", 0.0) or 0.0)
        ev_net = float(getattr(ev, "ev_net_pct", 0.0) or 0.0)
        if p_win < min_pwin:
            return False, f"loss-streak guard: p_win={p_win:.3f} < {min_pwin:.3f}"
        if ev_net < min_ev:
            return False, f"loss-streak guard: ev_net={ev_net:+.3f}% < {min_ev:+.3f}%"
        return True, "ok"

    def _paper_loss_streak_size_cap(self) -> float | None:
        if not self._paper_loss_streak_guard_active():
            return None
        cap = self._paper_validation.get("loss_streak_guard_max_total_scale", 0.35)
        try:
            return max(0.05, float(cap))
        except (TypeError, ValueError):
            return 0.35

    def _paper_ev_stress_tightening_active(self) -> bool:
        if not self._paper_validation_enabled() or not self._ev_gate_enabled():
            return False
        risk_state = getattr(self._risk, "state", None)
        daily_pnl = float(getattr(risk_state, "daily_pnl_pct", 0.0) or 0.0)
        consecutive_losses = int(getattr(risk_state, "consecutive_losses", 0) or 0)
        daily_trigger = float(
            self._paper_validation.get("stress_tighten_daily_loss_pct", -3.0) or -3.0
        )
        streak_trigger = int(
            self._paper_validation.get("stress_tighten_consecutive_losses", 3) or 3
        )
        daily_active = daily_trigger < 0.0 and daily_pnl <= daily_trigger
        streak_active = streak_trigger > 0 and consecutive_losses >= streak_trigger
        return daily_active or streak_active

    def _paper_stress_allows_ev(self, breakdown: SignalBreakdown) -> bool:
        if not self._paper_ev_stress_tightening_active():
            return True
        ev = getattr(breakdown, "ev_result", None)
        if ev is None:
            return True
        min_ev = float(self._paper_validation.get("stress_min_ev_net_pct", 0.0) or 0.0)
        return float(getattr(ev, "ev_net_pct", 0.0) or 0.0) >= min_ev

    def _paper_ev_observation_floor_allows(self, breakdown: SignalBreakdown) -> tuple[bool, str]:
        if not self._paper_validation_enabled():
            return True, "paper validation disabled"
        if self._ev_gate_enabled():
            return True, "EV gate enabled"
        if not bool(self._paper_validation.get("ev_observation_floor_enabled", False)):
            return True, "observation floor disabled"

        ev = getattr(breakdown, "ev_result", None)
        if ev is None:
            return True, "ev=none"

        p_win = float(getattr(ev, "p_win", 0.0) or 0.0)
        ev_net = float(getattr(ev, "ev_net_pct", 0.0) or 0.0)
        min_p_win = float(self._paper_validation.get("ev_observation_min_p_win", 0.30) or 0.30)
        max_deficit = float(
            self._paper_validation.get("ev_observation_max_deficit_pct", 0.75) or 0.75
        )
        if p_win < min_p_win and ev_net < -max_deficit:
            return (
                False,
                f"p_win={p_win:.3f} < {min_p_win:.3f} "
                f"and ev_net={ev_net:+.3f}% < -{max_deficit:.3f}%",
            )
        return True, "ok"

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

    def _paper_ev_stage(self, ev_trade_count: int) -> str:
        min_trades = int(
            self._cfg.get("ev_model", {}).get("min_trades_for_ev", 20)
        )
        hard_gate_min_trades = int(
            self._paper_validation.get(
                "ev_hard_gate_min_trades",
                self._cfg.get("ev_model", {}).get("min_trades_for_ev", 20),
            )
        )
        if ev_trade_count < min_trades:
            return "bootstrap"
        if ev_trade_count < hard_gate_min_trades:
            return "probation"
        return "hard_reject"

    def _paper_ev_relax_mode(self, breakdown: SignalBreakdown, threshold: float) -> str | None:
        if not self._paper_validation_enabled():
            return None
        ev = breakdown.ev_result
        if ev is None or breakdown.ev_ok:
            return None
        stage = self._paper_ev_stage(ev.trade_count)
        if stage == "hard_reject" or not breakdown.regime_ok or not breakdown.smart_money_ok:
            return None

        floor = float(self._paper_validation.get("min_score_floor", 58.0) or 58.0)
        score_buffer = float(self._paper_validation.get("ev_relax_score_buffer", 2.0) or 2.0)
        bootstrap_deficit = float(
            self._paper_validation.get("ev_bootstrap_max_deficit_pct", 0.15) or 0.15
        )
        probation_deficit = float(
            self._paper_validation.get("ev_probation_max_deficit_pct", 0.10) or 0.10
        )
        bootstrap_score = max(floor, threshold - score_buffer)
        probation_score = max(floor + 3.0, threshold)

        if stage == "bootstrap":
            if breakdown.total_score >= bootstrap_score and ev.ev_net_pct >= 0.0:
                return "bootstrap"
            if self._paper_high_conviction_candidate(breakdown, threshold):
                return "bootstrap"
            return None

        if breakdown.total_score >= probation_score and ev.ev_net_pct >= 0.0:
            return "probation"
        if self._paper_high_conviction_candidate(breakdown, threshold):
            return "probation"
        return None

    def _paper_positive_ev_probe_mode(self, breakdown: SignalBreakdown, threshold: float) -> str | None:
        if not self._paper_validation_enabled():
            return None
        if not bool(self._paper_validation.get("positive_ev_probe_enabled", False)):
            return None
        ev = getattr(breakdown, "ev_result", None)
        if ev is None or not getattr(breakdown, "ev_ok", False):
            return None
        if not getattr(breakdown, "smart_money_ok", False):
            return None
        if not self._paper_stress_allows_ev(breakdown):
            return None

        stage = self._paper_ev_stage(int(getattr(ev, "trade_count", 0) or 0))
        if stage == "hard_reject":
            return None

        min_ev = float(self._paper_validation.get("positive_ev_probe_min_ev_net_pct", 0.50) or 0.50)
        min_pwin = float(self._paper_validation.get("positive_ev_probe_min_p_win", 0.53) or 0.53)
        if float(getattr(ev, "ev_net_pct", 0.0) or 0.0) < min_ev:
            return None
        if float(getattr(ev, "p_win", 0.0) or 0.0) < min_pwin:
            return None

        direction = str(getattr(breakdown, "direction", "") or "")
        regime = breakdown.regime.value if getattr(breakdown, "regime", None) else ""
        sm_signal = getattr(breakdown, "smart_money", None)
        sm_phase = sm_signal.phase.value if sm_signal else "neutral"
        sm_score = float(getattr(sm_signal, "score", 0.0) or 0.0)
        sleeve = str(getattr(breakdown, "strategy_sleeve", "") or "neutral")

        allowed_dirs = set(self._paper_validation.get("positive_ev_probe_allowed_directions", ["short"]) or ["short"])
        allowed_regimes = set(self._paper_validation.get("positive_ev_probe_allowed_regimes", ["trending_expansion"]) or ["trending_expansion"])
        allowed_phases = set(self._paper_validation.get("positive_ev_probe_allowed_sm_phases", ["liquidity_sweep"]) or ["liquidity_sweep"])
        allowed_sleeves = set(self._paper_validation.get("positive_ev_probe_allowed_sleeves", ["neutral", "reversal"]) or ["neutral", "reversal"])
        if direction not in allowed_dirs or regime not in allowed_regimes or sm_phase not in allowed_phases:
            return None
        if sleeve not in allowed_sleeves:
            return None

        min_sm_score = float(self._paper_validation.get("positive_ev_probe_min_sm_score", 75.0) or 75.0)
        if sm_score < min_sm_score:
            return None

        score_floor = float(self._paper_validation.get("positive_ev_probe_min_score", 30.0) or 30.0)
        max_deficit = float(self._paper_validation.get("positive_ev_probe_max_score_deficit", 12.0) or 12.0)
        required_score = max(score_floor, float(threshold) - max_deficit)
        if self._positive_ev_probe_score_basis(breakdown) < required_score:
            return None
        return stage

    @staticmethod
    def _positive_ev_probe_score_basis(breakdown: SignalBreakdown) -> float:
        return max(
            float(getattr(breakdown, "total_score", 0.0) or 0.0),
            float(getattr(breakdown, "pre_gate_score", 0.0) or 0.0),
        )

    def _signal_confirmation_allowed(self, breakdown: SignalBreakdown, threshold: float) -> bool:
        if float(getattr(breakdown, "total_score", 0.0) or 0.0) >= float(threshold):
            return True
        return self._paper_positive_ev_probe_mode(breakdown, threshold) is not None

    @staticmethod
    def _bounded(value: float, lo: float, hi: float) -> float:
        return min(hi, max(lo, float(value or 0.0)))

    @staticmethod
    def _clip_probability(value: float, *, default: float = 0.5) -> float:
        try:
            p = float(value)
        except (TypeError, ValueError):
            p = default
        if not math.isfinite(p):
            p = default
        return min(0.99, max(0.01, p))

    @classmethod
    def _logit(cls, probability: float) -> float:
        p = cls._clip_probability(probability)
        return math.log(p / (1.0 - p))

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    def _probabilistic_candidate_rank(
        self,
        breakdown: SignalBreakdown,
        *,
        ml_p_win: float,
        regime_scale: float,
        cohort_bonus: float,
    ) -> tuple[float, float, float, float, float]:
        """Rank admitted candidates by posterior win probability and expected utility."""
        ev = getattr(breakdown, "ev_result", None)
        total_score = float(getattr(breakdown, "total_score", 0.0) or 0.0)
        setup_quality = self._bounded(
            float(getattr(breakdown, "setup_quality_score", 50.0) or 50.0),
            0.0,
            100.0,
        )

        ev_p_win = self._clip_probability(getattr(ev, "p_win", 0.5) if ev else 0.5)
        ml_probability = self._clip_probability(ml_p_win)
        quality_probability = self._clip_probability(0.35 + 0.30 * (setup_quality / 100.0))
        ev_conf = self._bounded(float(getattr(ev, "confidence", 0.0) or 0.0) if ev else 0.0, 0.0, 1.0)

        # Bayesian-style log-odds blend: EV posterior is primary, ML is a
        # second model, and setup quality is a weak structural prior.
        ev_weight = 0.60 + 0.25 * ev_conf
        ml_weight = 0.25
        quality_weight = 0.15
        posterior_log_odds = (
            ev_weight * self._logit(ev_p_win)
            + ml_weight * self._logit(ml_probability)
            + quality_weight * self._logit(quality_probability)
        ) / (ev_weight + ml_weight + quality_weight)
        posterior_p = self._sigmoid(posterior_log_odds)

        conservative_p = float(getattr(ev, "conservative_p_win", 0.0) or 0.0) if ev else 0.0
        if conservative_p > 0.0:
            conservative_p = self._clip_probability(conservative_p)
            posterior_p = posterior_p * (1.0 - 0.35 * ev_conf) + conservative_p * (0.35 * ev_conf)

        ev_net = self._bounded(float(getattr(ev, "ev_net_pct", 0.0) or 0.0) if ev else 0.0, -2.0, 2.0)
        conservative_ev = self._bounded(
            float(getattr(ev, "conservative_ev_net_pct", ev_net) or 0.0) if ev else 0.0,
            -2.0,
            2.0,
        )
        avg_win_pct = max(0.0, float(getattr(ev, "avg_win_pct", 0.0) or 0.0) if ev else 0.0)
        avg_loss_pct = max(0.0, float(getattr(ev, "avg_loss_pct", 0.0) or 0.0) if ev else 0.0)
        cost_pct = max(0.0, float(getattr(ev, "cost_pct", 0.0) or 0.0) if ev else 0.0)
        funding_pct = float(getattr(ev, "funding_cost_pct", 0.0) or 0.0) if ev else 0.0

        if avg_win_pct > 0.0 and avg_loss_pct > 0.0:
            win_return = max(-0.95, (avg_win_pct - cost_pct - funding_pct) / 100.0)
            loss_return = min(0.95, (avg_loss_pct + cost_pct + abs(funding_pct)) / 100.0)
            expected_log_growth_pct = (
                posterior_p * math.log1p(win_return)
                + (1.0 - posterior_p) * math.log1p(-loss_return)
            ) * 100.0
        else:
            expected_log_growth_pct = ev_net

        rank_score = (
            total_score
            + (posterior_p - 0.50) * 40.0
            + expected_log_growth_pct * 12.0
            + ev_net * 2.0
            + conservative_ev * ev_conf * 2.0
            + (setup_quality - 50.0) / 12.0
            + float(getattr(breakdown, "strategy_ranking_bonus", 0.0) or 0.0)
            + float(cohort_bonus or 0.0)
        )
        return (
            rank_score,
            float(regime_scale or 1.0),
            posterior_p,
            expected_log_growth_pct,
            total_score,
        )

    async def start(self) -> None:
        cleanup_stats = cleanup_workspace_artifacts()
        if cleanup_stats["files"] or cleanup_stats["dirs"]:
            log.info(
                "Workspace cleanup removed %d files and %d directories",
                cleanup_stats["files"],
                cleanup_stats["dirs"],
            )

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
                    paper_reset_token = str(self._trading.get("paper_state_reset_token", "") or "")
                    persisted_reset_token = str(persisted.get("paper_state_reset_token", "") or "")
                    reset_paper_state = False
                    # Paper-mode safety: when config explicitly sets a paper starting equity,
                    # treat that as a reset anchor if it differs from the persisted state.
                    # This prevents a stale drawdown/peak from permanently freezing paper runs.
                    if (
                        self._trading["mode"] == "paper"
                        and paper_equity
                        and (
                            float(persisted.get("starting_equity", 0.0) or 0.0) != float(paper_equity)
                            or (
                                paper_reset_token
                                and persisted_reset_token != paper_reset_token
                            )
                        )
                    ):
                        log.info(
                            "Paper state mismatch: persisted starting_equity=$%.2f token=%s vs config paper_starting_equity=$%.2f token=%s",
                            float(persisted.get("starting_equity", 0.0) or 0.0),
                            persisted_reset_token or "-",
                            float(paper_equity),
                            paper_reset_token or "-",
                        )
                        self._risk.update_equity(float(paper_equity))
                        self._risk.state.peak_equity = float(paper_equity)
                        self._risk.state.consecutive_losses = 0
                        self._risk.state.reset_day(float(paper_equity))
                        self._risk.state.reset_week(float(paper_equity))
                        self._starting_equity = float(paper_equity)
                        self._equity_curve = [float(paper_equity)]
                        self._equity_graph_points = []
                        self._consecutive_wins = 0
                        self._risk.state.kill_switch_reason = ""
                        self._risk.state.paused_until_ts = 0.0
                        self._risk.state.open_trade_count = 0
                        self._risk.state.open_risk_pct = 0.0
                        self._risk.state.symbol_risk_pct = {}
                        self._risk.state.direction_risk_pct = {}
                        trade_mgr = getattr(self, "_trade_mgr", None)
                        open_book = getattr(trade_mgr, "_trades", None)
                        if isinstance(open_book, dict) and open_book:
                            dropped = len(open_book)
                            open_book.clear()
                            save_state = getattr(trade_mgr, "_save_state", None)
                            if callable(save_state):
                                save_state()
                            log.info(
                                "Paper reset: cleared %d restored open paper trades",
                                dropped,
                            )
                        log.info(
                            "Paper reset: ignoring persisted state and re-seeding equity to $%.2f",
                            float(paper_equity),
                        )
                        reset_paper_state = True
                    if not reset_paper_state:
                        self._risk.state.equity = float(persisted.get("equity", self._risk.state.equity))
                        self._risk.state.peak_equity = float(persisted.get("peak_equity", self._risk.state.peak_equity))
                        self._risk.state.consecutive_losses = int(persisted.get("consecutive_losses", 0))
                        self._risk.state.daily_start_equity = float(
                            persisted.get("daily_start_equity", self._risk.state.equity)
                        )
                        self._risk.state.day_start_ts = float(
                            persisted.get("day_start_ts", getattr(self._risk.state, "day_start_ts", time.time()))
                        )
                        self._risk.state.weekly_start_equity = float(
                            persisted.get(
                                "weekly_start_equity",
                                getattr(self._risk.state, "weekly_start_equity", self._risk.state.equity),
                            )
                        )
                        self._risk.state.week_start_ts = float(
                            persisted.get("week_start_ts", getattr(self._risk.state, "week_start_ts", time.time()))
                        )
                        self._starting_equity = float(persisted.get("starting_equity", self._starting_equity))
                        self._equity_curve = list(persisted.get("equity_curve", []))
                        self._equity_graph_points = list(persisted.get("equity_graph_points", []))
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

        if not self._equity_graph_points:
            now_ts = time.time()
            self._equity_graph_points = [
                {
                    "ts": now_ts,
                    "balance": float(self._risk.state.equity),
                    "equity": float(self._risk.state.equity),
                }
            ]

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

                await self._maybe_send_crypto_news_report()

                # ── Scan pairs ────────────────────────────────────────────
                lifecycle_report = self._lifecycle.build_report(self._learner._trade_log)
                priority = self._cohort_policy.recommend_symbols(
                    self._learner._trade_log,
                    primary_edge=lifecycle_report.get("primary_edge"),
                )
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

                sector_rotations = compute_sector_rotation(snapshots)
                attach_sector_rotation(snapshots, sector_rotations)
                if sector_rotations:
                    top_rotation = min(sector_rotations.values(), key=lambda item: item.rank or 999)
                    log.info(
                        "Sector rotation top: %s %s rel_btc=%.2f%% breadth=%.2f conf=%.2f",
                        top_rotation.sector, top_rotation.state, top_rotation.relative_btc_pct,
                        top_rotation.breadth, top_rotation.confidence,
                    )

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
                _open_positions = self._telegram_open_positions()
                _price_map = await self._price_map_with_open_trades(snapshots)
                _floating = self._trade_mgr.get_floating_pnl(_price_map)
                _effective_equity, _effective_daily_pnl_pct, _effective_drawdown_pct = (
                    self._effective_account_metrics(_floating)
                )
                self._equity_graph_points.append({
                    "ts": time.time(),
                    "balance": float(self._risk.state.equity),
                    "equity": float(_effective_equity),
                })
                self._equity_graph_points = self._equity_graph_points[-720:]
                await self._maybe_send_fund_manager_report(
                    breakdowns=breakdowns,
                    readiness_report=readiness_report,
                    lifecycle_report=lifecycle_report,
                    jim_status=jim_status,
                    floating_positions=_floating,
                    open_positions=_open_positions,
                    total_trades=_n_trades,
                    win_rate=_win_rate,
                    sharpe=_sharpe,
                    cycle_num=_cycle,
                )
                await self._maybe_send_bootstrap_audit_report(
                    lifecycle_report=lifecycle_report,
                    cycle_num=_cycle,
                )
                await self._maybe_send_equity_graph_report(cycle_num=_cycle)

                # Send Telegram hourly heartbeat after first scoring
                if _send_tg_heartbeat:
                    top_names = [
                        f"{b.symbol} {b.direction} {b.total_score:.0f}" for b in breakdowns[:5]
                    ]
                    await self._telegram.heartbeat(
                        equity=self._risk.state.equity,
                        drawdown_pct=_effective_drawdown_pct,
                        daily_pnl_pct=_effective_daily_pnl_pct,
                        open_trades=self._risk.state.open_trade_count,
                        top_signals=top_names,
                        floating_positions=_floating,
                        open_positions=self._telegram_open_positions(),
                        risk_budget=self._build_risk_budget(),
                        short_setup_summary=self._short_setup_summary(breakdowns),
                    )
                # ── Monitor open trades ───────────────────────────────────
                if self._shadow:
                    self._shadow.on_tick(_price_map)
                await self._trade_mgr.monitor_all(_price_map)

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
                    if self._signal_confirmation_allowed(b, threshold):
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
                    if b.ev_ok and not self._paper_stress_allows_ev(b):
                        log.info(
                            "[%s] paper validation stress tightening blocked EV relax (ev_net=%+.3f%%)",
                            b.symbol,
                            b.ev_result.ev_net_pct if b.ev_result else 0.0,
                        )
                        b.ev_ok = False
                    _paper_positive_ev_probe = self._paper_positive_ev_probe_mode(b, _thresh)
                    if _paper_positive_ev_probe and not b.regime_ok and b.smart_money_ok and b.ev_ok:
                        log.info(
                            "[%s] paper validation positive-EV probe unblocked regime gate "
                            "(%s, regime=%s, score_basis=%.1f, ev_net=%+.3f%% p_win=%.3f)",
                            b.symbol,
                            _paper_positive_ev_probe,
                            b.regime.value if b.regime else "?",
                            self._positive_ev_probe_score_basis(b),
                            b.ev_result.ev_net_pct if b.ev_result else 0.0,
                            b.ev_result.p_win if b.ev_result else 0.0,
                        )
                        b.regime_ok = True
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
                                        setup_passport=setup_passport_with_market_context(b),
                                    )
                                    if _sh_setup is not None:
                                        self._shadow.on_signal(b, _sh_setup)
                            except Exception as _e:
                                log.debug("shadow counterfactual %s: %s", b.symbol, _e)
                        continue
                    _ev_floor_allowed, _ev_floor_reason = self._paper_ev_observation_floor_allows(b)
                    if not _ev_floor_allowed:
                        self._dataset_logger.log_no_trade(b, "paper_ev_observation_floor_blocked", snap=_snap)
                        self._rej.log(
                            stage="gate_ev",
                            reason=f"paper observation floor: {_ev_floor_reason}",
                            breakdown=b, threshold_required=_thresh, funding_rate=_fr,
                        )
                        continue
                    _paper_positive_ev_probe = self._paper_positive_ev_probe_mode(b, _thresh)
                    if b.total_score < _thresh:
                        if _paper_positive_ev_probe:
                            log.info(
                                "[%s] paper validation positive-EV probe enabled (%s, score=%.1f<thresh=%.1f ev_net=%+.3f%% p_win=%.3f)",
                                b.symbol,
                                _paper_positive_ev_probe,
                                b.total_score,
                                _thresh,
                                b.ev_result.ev_net_pct if b.ev_result else 0.0,
                                b.ev_result.p_win if b.ev_result else 0.0,
                            )
                        else:
                            self._dataset_logger.log_no_trade(b, "score_below_threshold", snap=_snap)
                            self._rej.log(
                                stage="score_threshold",
                                reason=f"score={b.total_score:.1f} < thresh={_thresh:.1f}",
                                breakdown=b, threshold_required=_thresh, funding_rate=_fr,
                            )
                            continue
                    paper_scope_allowed, paper_scope_reason = self._paper_trade_scope_check(b)
                    if not paper_scope_allowed:
                        if _paper_positive_ev_probe:
                            log.info(
                                "[%s] paper validation positive-EV probe bypassed scope (%s)",
                                b.symbol,
                                paper_scope_reason,
                            )
                        else:
                            self._dataset_logger.log_no_trade(b, "paper_scope_blocked", snap=_snap)
                            self._rej.log(
                                stage="paper_scope",
                                reason=paper_scope_reason,
                                breakdown=b,
                                threshold_required=_thresh,
                                funding_rate=_fr,
                            )
                            continue
                    exp_setup_ok, exp_setup_reason = self._paper_experimental_setup_stress_check(b)
                    if not exp_setup_ok:
                        self._dataset_logger.log_no_trade(b, "experimental_setup_stress_blocked", snap=_snap)
                        self._rej.log(
                            stage="experimental_setup_stress",
                            reason=exp_setup_reason,
                            breakdown=b,
                            threshold_required=_thresh,
                            funding_rate=_fr,
                        )
                        if self._shadow:
                            self._shadow.record_rejected(b, stage="experimental_setup_stress")
                        continue
                    loss_streak_ok, loss_streak_reason = self._paper_loss_streak_guard_check(b, _thresh)
                    if not loss_streak_ok:
                        self._dataset_logger.log_no_trade(b, "loss_streak_guard_blocked", snap=_snap)
                        self._rej.log(
                            stage="loss_streak_guard",
                            reason=loss_streak_reason,
                            breakdown=b,
                            threshold_required=_thresh,
                            funding_rate=_fr,
                        )
                        if self._shadow:
                            self._shadow.record_rejected(b, stage="loss_streak_guard")
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
                    # Phase 2 gate: ML p_win only closes entries after the live hard-gate minimum.
                    # Prevents ML (trained partly on shadow trades) from blocking exploration
                    # during bootstrap when CV accuracy is inflated by small-sample overfit.
                    real_trades = len(self._learner._trade_log)
                    ml_cfg = self._ml_gate_config()
                    ml_hard_gate_min_trades = (
                        self._paper_ml_hard_gate_min_trades()
                        if self._paper_validation_enabled()
                        else int(ml_cfg.get("hard_gate_min_trades", 60))
                    )
                    if ml_ready and real_trades >= ml_hard_gate_min_trades and ml_p_win < ml_threshold:
                        ml_veto_allowed, ml_budget_reason = self._ml_veto_budget_allows()
                        if ml_veto_allowed:
                            self._record_ml_veto()
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
                        log.info(
                            "ML soft-pass %s — %s; continuing in size-only mode",
                            b.symbol, ml_budget_reason,
                        )
                    # Progressive regime gate: exploration / soft-penalty / hard-lock
                    regime_key = b.regime.value if b.regime else "chaos"
                    if self._paper_validation_enabled():
                        allow, regime_scale, regime_reason = True, 1.0, "paper strategy-only path"
                    else:
                        allow, regime_scale, regime_reason = self._ml.regime_gate(regime_key)
                        regime_hard_lock_min_trades = int(ml_cfg.get("regime_hard_lock_min_trades", 30) or 30)
                        if not allow and real_trades < regime_hard_lock_min_trades:
                            log.info(
                                "ML regime hard-lock deferred %s (%s) — continuing in size-only mode",
                                regime_key, regime_reason,
                            )
                            allow, regime_scale, regime_reason = True, 0.5, f"deferred {regime_reason}"
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
                    key=lambda x: self._probabilistic_candidate_rank(
                        x,
                        ml_p_win=_ml_p_win_map.get(x.symbol, 0.5),
                        regime_scale=_regime_scale_map.get(x.symbol, 1.0),
                        cohort_bonus=_cohort_bonus_map.get(x.symbol, 0.0),
                    ),
                    reverse=True,
                )
                eligible = eligible[:top_n]

                # Keep trade log in sync — union realized + shadow counterfactuals.
                # Uncensored posterior: asymmetric shrinkage (pwin_engine) bounds
                # downside risk of including simulated outcomes.
                self._scorer.set_trade_log(self._learner._trade_log)

                rotations_this_cycle = 0
                for bd in eligible:
                    rotation_victim: OpenTrade | None = None
                    rotation_price = 0.0
                    rotation_reason = ""
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
                        if str(reason).startswith("max_open_trades"):
                            rotation_victim, rotation_price, rotation_reason = self._select_rotation_victim(
                                bd, snapshots, rotations_this_cycle
                            )
                        if rotation_victim is None:
                            log.info("[%s] Risk guard blocked: %s", bd.symbol, reason)
                            if str(reason).startswith("max_open_trades") and rotation_reason:
                                log.info("[%s] rotation skipped: %s", bd.symbol, rotation_reason)
                            self._rej.log(
                                stage="risk_guard", reason=reason, breakdown=bd,
                                threshold_required=_bd_thresh, funding_rate=_bd_fr,
                            )
                            if self._shadow:
                                self._shadow.record_rejected(bd, stage="risk_guard")
                            continue
                        log.info(
                            "[%s] position rotation candidate selected; %s pending downstream gates before any close (%s)",
                            bd.symbol, rotation_victim.symbol, rotation_reason,
                        )

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

                    # ── Adaptive Brain layer (Thompson + ML + regime HMM + alpha decay + vol target) ──
                    brain_decision = self._brain.decide(bd)
                    if brain_decision.should_block:
                        log.warning(
                            "[%s] BRAIN BLOCKED — %s",
                            bd.symbol, brain_decision.telegram_summary(),
                        )
                        self._rej.log(
                            stage="brain_block",
                            reason=f"alpha decayed: {brain_decision.telegram_summary()}",
                            breakdown=bd, threshold_required=_bd_thresh,
                            fm_scale=total_scale, ml_p_win=_p_win,
                            funding_rate=_bd_fr,
                        )
                        if self._shadow:
                            self._shadow.record_rejected(bd, stage="brain_block")
                        continue
                    total_scale *= brain_decision.overall_size_mult
                    log.info(
                        "[%s] 🧠 Brain: %s",
                        bd.symbol, brain_decision.telegram_summary(),
                    )
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
                    # Session-aware sizing — Asia/London/NY have very different
                    # vol profiles; scale notional accordingly so a 1R bet
                    # represents comparable real-world risk across sessions.
                    if self._session_mod.is_enabled():
                        ss = self._session_mod.current()
                        if ss.multiplier != 1.0:
                            total_scale *= ss.multiplier
                            log.info(
                                "[%s] session_scale=%.2fx applied (%s)",
                                bd.symbol, ss.multiplier, ss.session,
                            )
                    # Regime gate scale applied AFTER floor so soft-penalty can shrink below 0.40.
                    # Never amplifies (max 1.0 from regime_gate); only shrinks or leaves unchanged.
                    _rscale = _regime_scale_map.get(bd.symbol, 1.0)
                    if _rscale < 1.0:
                        total_scale *= _rscale
                        log.info("[%s] regime_scale=%.2fx applied", bd.symbol, _rscale)
                    if (
                        self._paper_high_conviction_candidate(bd, _bd_thresh)
                        and not self._paper_loss_streak_guard_active()
                    ):
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
                    loss_cap = self._paper_loss_streak_size_cap()
                    if loss_cap is not None and total_scale > loss_cap:
                        log.info(
                            "[%s] paper loss-streak size cap %.2fx applied (raw %.2fx)",
                            bd.symbol,
                            loss_cap,
                            total_scale,
                        )
                        total_scale = loss_cap
                    if fm_decision.notes or ml_confidence > 1.0:
                        conf_note = f"Jim confidence {ml_confidence:.2f}x (p_win={_p_win:.0%})" if ml_confidence > 1.0 else ""
                        all_notes = ([conf_note] if conf_note else []) + fm_decision.notes
                        log.info("[%s] Jim scale=%.2fx  (%s)", bd.symbol, total_scale, ", ".join(all_notes))

                    _live = self._trading["mode"] == "live"
                    _short_setup = getattr(bd, "short_setup", None)
                    _short_setup_label = (
                        getattr(_short_setup, "label", "")
                        if _short_setup is not None and getattr(_short_setup, "is_valid", False)
                        else ""
                    )
                    _short_setup_confidence = (
                        float(getattr(_short_setup, "confidence", 0.0))
                        if _short_setup is not None and getattr(_short_setup, "is_valid", False)
                        else 0.0
                    )
                    setup = self._risk.calculate_setup(
                        bd.symbol, bd.direction, df, snap.last_price,
                        strategy_sleeve=bd.strategy_sleeve,
                        ev_result=bd.ev_result,
                        kelly_scale=total_scale,
                        min_lot_size=self._client.min_lot_size(bd.symbol) if _live else 0.0,
                        min_notional_usd=self._client.min_notional(bd.symbol) if _live else 0.0,
                        short_setup_label=_short_setup_label,
                        short_setup_confidence=_short_setup_confidence,
                        setup_passport=setup_passport_with_market_context(bd),
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

                    # ── Correlation cap — block compounding bets ────────────
                    # Concurrent longs on highly correlated assets are
                    # effectively one beta-1 trade, not three.  When the
                    # cluster goes offside, all stops fire together.
                    if self._corr_filter.is_enabled():
                        cand_returns = self._symbol_returns(snap)
                        open_book = self._open_trades_returns(snapshots)
                        cd = self._corr_filter.evaluate(
                            candidate_symbol=bd.symbol,
                            candidate_direction=bd.direction,
                            candidate_returns=cand_returns,
                            open_trades=open_book,
                        )
                        if not cd.allowed:
                            log.info("[%s] correlation cap blocked — %s", bd.symbol, cd.reason)
                            self._rej.log(
                                stage="correlation_cap",
                                reason=cd.reason,
                                breakdown=bd,
                                threshold_required=_bd_thresh,
                                fm_scale=total_scale,
                                ml_p_win=_p_win,
                                ml_threshold=self._ml.dynamic_p_win_threshold,
                                funding_rate=_bd_fr,
                            )
                            if self._shadow:
                                self._shadow.record_rejected(bd, stage="correlation_cap")
                            continue

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
                    market_ctx = getattr(bd, "market_context", None)
                    sector_rotation = getattr(bd, "sector_rotation", None)
                    ts_diag = getattr(bd, "time_series", None)
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
                        # Time of day (UTC) — session/context awareness
                        "hour_utc": float(time.gmtime().tm_hour),
                        "hour_of_day": int(time.gmtime().tm_hour),
                        "day_of_week": int(time.gmtime().tm_wday),
                        # Broad-market context at entry, used for later attribution.
                        "market_risk_on_state": str(getattr(market_ctx, "risk_on_state", "unknown") or "unknown"),
                        "market_rotation_state": str(getattr(market_ctx, "rotation_state", "unknown") or "unknown"),
                        "market_btc_trend": str(getattr(market_ctx, "btc_trend", "unknown") or "unknown"),
                        "market_eth_btc_trend": str(getattr(market_ctx, "eth_btc_trend", "unknown") or "unknown"),
                        "market_btc_d_trend": str(getattr(market_ctx, "btc_d_trend", "unknown") or "unknown"),
                        "market_total_trend": str(getattr(market_ctx, "total_trend", "unknown") or "unknown"),
                        "market_context_confidence": float(getattr(market_ctx, "confidence", 0.0) or 0.0),
                        "sector_rotation_state": str(getattr(sector_rotation, "state", "unknown") or "unknown"),
                        "sector_rotation_rank": int(getattr(sector_rotation, "rank", 0) or 0),
                        "sector_rotation_relative_btc_pct": float(getattr(sector_rotation, "relative_btc_pct", 0.0) or 0.0),
                        "sector_rotation_breadth": float(getattr(sector_rotation, "breadth", 0.0) or 0.0),
                        "sector_rotation_confidence": float(getattr(sector_rotation, "confidence", 0.0) or 0.0),
                        # Causal time-series diagnostics at entry.
                        "time_series_state": str(getattr(ts_diag, "state", "unknown") if ts_diag else "unknown"),
                        "time_series_ewma_vol_pct": float(getattr(ts_diag, "ewma_vol_pct", 0.0) if ts_diag else 0.0),
                        "time_series_realized_vol_pct": float(getattr(ts_diag, "realized_vol_pct", 0.0) if ts_diag else 0.0),
                        "time_series_volatility_ratio": float(getattr(ts_diag, "volatility_ratio", 1.0) if ts_diag else 1.0),
                        "time_series_latest_abs_return_z": float(getattr(ts_diag, "latest_abs_return_z", 0.0) if ts_diag else 0.0),
                        "time_series_drift_t_stat": float(getattr(ts_diag, "drift_t_stat", 0.0) if ts_diag else 0.0),
                        "time_series_stability_score": float(getattr(ts_diag, "stability_score", 0.0) if ts_diag else 0.0),
                        "time_series_markov_state": str(getattr(ts_diag, "markov_state", "unknown") if ts_diag else "unknown"),
                        "time_series_markov_bull_prob": float(getattr(ts_diag, "markov_bull_prob", 0.0) if ts_diag else 0.0),
                        "time_series_markov_bear_prob": float(getattr(ts_diag, "markov_bear_prob", 0.0) if ts_diag else 0.0),
                        "time_series_markov_sideways_prob": float(getattr(ts_diag, "markov_sideways_prob", 0.0) if ts_diag else 0.0),
                        "time_series_markov_edge": float(getattr(ts_diag, "markov_edge", 0.0) if ts_diag else 0.0),
                        "time_series_score_mult": float(getattr(bd, "time_series_score_mult", 1.0) or 1.0),
                        "time_series_size_mult": float(getattr(bd, "time_series_size_mult", 1.0) or 1.0),
                        "time_series_reason": str(getattr(bd, "time_series_reason", "") or ""),
                        # Strategy sleeve + setup passport metadata for later attribution
                        "strategy_sleeve": bd.strategy_sleeve,
                        "strategy_reason": bd.strategy_reason,
                        "exit_profile": setup.exit_profile,
                        "strategy_score_mult": float(bd.strategy_score_mult),
                        "strategy_size_mult": float(bd.strategy_size_mult),
                        "setup_type": getattr(bd, "setup_type", "unknown"),
                        "setup_quality_score": float(getattr(bd, "setup_quality_score", 0.0) or 0.0),
                        "setup_passport": setup_passport_with_market_context(bd),
                        "sector": str(setup_passport_with_market_context(bd).get("sector", "unknown")),
                        "dispersion_value": float(bd.dispersion_value),
                        "signal_type": f"{bd.regime.value if bd.regime else 'unknown'}|{sm.phase.value if sm else 'neutral'}",
                        "entry_reason": bd.strategy_reason,
                        "timeframe": self._cfg["timeframes"]["primary"],
                        "asset": bd.symbol.split("/")[0],
                        "session": active_market_session_key(),
                        "volatility_bucket": ("low" if bd.volatility <= 20 else "medium" if bd.volatility <= 40 else "high" if bd.volatility <= 70 else "extreme"),
                        "trend_bucket": ("weak" if bd.trend_strength < 40 else "moderate" if bd.trend_strength < 65 else "strong" if bd.trend_strength < 85 else "very_strong"),
                        "cohort_key": getattr(bd, "cohort_key", ""),
                        "lifecycle_status": getattr(bd, "lifecycle_status", "RESEARCH"),
                        # Phase D / Liq Sweep dedicated short-setup metadata
                        # (empty when the trade did not fire a dedicated detector).
                        "short_setup_label": _short_setup_label,
                        "short_setup_confidence": _short_setup_confidence,
                    }
                    if rotation_victim is not None:
                        closed = await self._trade_mgr.close_for_rotation(
                            rotation_victim.symbol,
                            rotation_price,
                            reason=f"rotation_upgrade:{bd.symbol}",
                        )
                        if not closed:
                            self._pending_scores.pop(bd.symbol, None)
                            log.warning(
                                "[%s] rotation close failed for %s — skipping open",
                                bd.symbol, rotation_victim.symbol,
                            )
                            continue
                        rotations_this_cycle += 1
                        self._record_position_rotation()
                        log.info(
                            "[%s] rotation freed slot by closing %s @ %.4f",
                            bd.symbol, rotation_victim.symbol, rotation_price,
                        )

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
            float(self._risk.state.drawdown_pct or 0.0),
            float(self._risk.state.daily_pnl_pct or 0.0),
            int(self._risk.state.open_trade_count or 0),
        )
        # Feed daily return into adaptive brain's vol targeter (throttled internally)
        try:
            self._brain.record_daily_return(float(self._risk.state.daily_pnl_pct or 0.0))
        except Exception as _exc:
            log.debug("Brain daily return record failed: %s", _exc)

    async def _on_trade_closed(self, trade: OpenTrade, pnl: float, reason: str) -> None:
        scores = enrich_closed_trade_scores(
            self._pending_scores.pop(trade.symbol, {}), trade, self._cfg
        )
        entry_p = trade.setup.entry_price
        # Prefer the actual market close price stamped by TradeManager
        # (covers every exit reason: stop_loss, trailing_stop, tp1, tp2,
        # tp2_full, tp3, early_adverse_cut, max_hold, manual, end_of_data).
        # Falls back to the closest setup level only when the stamp is
        # missing (older state files) so older OpenTrade payloads keep
        # working after the upgrade.
        close_p = float(getattr(trade, "exit_price", 0.0) or 0.0)
        if close_p <= 0.0:
            if reason == "stop_loss":
                close_p = trade.setup.stop_loss
            elif reason == "trailing_stop" and trade.trailing_stop is not None:
                close_p = trade.trailing_stop
            elif reason == "tp1":
                close_p = trade.setup.tp1
            elif reason in ("tp2", "tp2_full"):
                close_p = trade.setup.tp2
            elif reason == "tp3":
                close_p = trade.setup.tp3
            else:
                # early_adverse_cut / max_hold / manual / end_of_data — best
                # we can do without the stamp is the entry price; pnl_usd is
                # still authoritative so reporting stays correct.
                close_p = entry_p
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
            dispersion_value=float(scores.get("dispersion_value", 0.0) or 0.0),
            dispersion_state=str(scores.get("dispersion_state", "normal") or "normal"),
            mfe_r=trade.mfe_r,
            mae_r=trade.mae_r,
            tp1_hit=trade.tp1_hit,
            hold_duration_s=max(0.0, time.time() - trade.opened_at),
            fees_usd=round(trade.setup.size_usd * float(self._cfg.get("ev_model", {}).get("taker_fee_pct", 0.04)) / 100.0 * 2, 4),
            slippage_estimate_usd=round(trade.setup.size_usd * float(self._cfg.get("ev_model", {}).get("slippage_pct", 0.05)) / 100.0 * 2, 4),
            fees_slippage_pct=round((float(self._cfg.get("ev_model", {}).get("taker_fee_pct", 0.04)) + float(self._cfg.get("ev_model", {}).get("slippage_pct", 0.05))) * 2, 4),
            timeframe=str(scores.get("timeframe", self._cfg["timeframes"]["primary"])),
            session=str(scores.get("session", "unknown")),
            hour_of_day=int(scores.get("hour_of_day", scores.get("hour_utc", -1))),
            day_of_week=int(scores.get("day_of_week", -1)),
            asset=str(scores.get("asset", trade.symbol.split('/')[0])),
            sector=str(scores.get("sector", sector_for_asset(asset_from_symbol(trade.symbol)))),
            market_risk_on_state=str(scores.get("market_risk_on_state", "unknown")),
            market_rotation_state=str(scores.get("market_rotation_state", "unknown")),
            market_btc_trend=str(scores.get("market_btc_trend", "unknown")),
            market_eth_btc_trend=str(scores.get("market_eth_btc_trend", "unknown")),
            market_btc_d_trend=str(scores.get("market_btc_d_trend", "unknown")),
            market_total_trend=str(scores.get("market_total_trend", "unknown")),
            market_context_confidence=float(scores.get("market_context_confidence", 0.0) or 0.0),
            signal_type=str(scores.get("signal_type", "unknown")),
            entry_reason=str(scores.get("entry_reason", "unknown")),
            volatility_bucket=str(scores.get("volatility_bucket", "unknown")),
            trend_bucket=str(scores.get("trend_bucket", "unknown")),
            cohort_key=str(scores.get("cohort_key", "")),
            lifecycle_status=str(scores.get("lifecycle_status", "RESEARCH")),
        )
        self._learner.record_trade(record)
        try:
            self._trading_brain.record_closed_trade(record, self._learner._trade_log)
        except Exception as _exc:
            log.debug("Trading brain journal write failed: %s", _exc)


        # ── Adaptive Brain learning — feeds outcome to Thompson, ML, alpha decay ──
        try:
            # Reconstruct a minimal breakdown-like object from the closed-trade record
            # so the brain can extract features it saw at entry time
            brain_features = type("BrainBreakdown", (), {
                "symbol": trade.symbol,
                "direction": trade.direction,
                "total_score": float(scores.get("score", 50.0)),
                "structure_quality": float(scores.get("structure_quality", 50.0)),
                "trend_strength": float(scores.get("trend_strength", 50.0)),
                "volume_confirmation": float(scores.get("volume_confirmation", 50.0)),
                "funding_sentiment": float(scores.get("funding_sentiment", 50.0)),
                "open_interest": float(scores.get("open_interest", 50.0)),
                "volatility": float(scores.get("volatility", 50.0)),
                "regime": type("R", (), {"value": _regime_str})(),
                "smart_money": None,
                "feature_vector": type("FV", (), {
                    "momentum_strength": float(scores.get("momentum_strength", 0.0)),
                })(),
                "strategy_sleeve": str(scores.get("strategy_sleeve", "neutral")),
            })()
            self._brain.learn(brain_features, won=(pnl > 0), pnl_pct=pnl_pct)
        except Exception as _exc:
            log.debug("Brain learn failed: %s", _exc)
        # Feed closed outcome into EdgeDetector memory (pair × edge_type → per-regime WR + decay curve).
        try:
            _sm_phase = str(scores.get("sm_phase", "neutral"))
            _edge_type = f"{_regime_str}|{_sm_phase}|{trade.direction}"
            self._scorer.edge_detector.memory.record_outcome(
                pair=trade.symbol,
                edge_type=_edge_type,
                regime=_regime_str,
                pnl_pct=pnl_pct,
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
            short_setup_label=str(
                getattr(trade.setup, "short_setup_label", "")
                or scores.get("short_setup_label", "")
                or ""
            ),
            short_setup_confidence=float(
                getattr(trade.setup, "short_setup_confidence", 0.0)
                or scores.get("short_setup_confidence", 0.0)
                or 0.0
            ),
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

    def _effective_account_metrics(self, floating_positions: list[dict] | None = None) -> tuple[float, float, float]:
        floating_total = sum(float(p.get("pnl_usd", 0.0)) for p in (floating_positions or []))
        effective_equity = self._risk.state.equity + floating_total

        daily_base = float(self._risk.state.daily_start_equity or 0.0)
        if daily_base > 0:
            daily_pnl_pct = (effective_equity - daily_base) / daily_base * 100
        else:
            daily_pnl_pct = 0.0

        peak_equity = max(float(self._risk.state.peak_equity or 0.0), effective_equity)
        if peak_equity > 0:
            drawdown_pct = max(0.0, (peak_equity - effective_equity) / peak_equity * 100)
        else:
            drawdown_pct = 0.0

        return effective_equity, daily_pnl_pct, drawdown_pct

    def _bootstrap_audit_lines(self, lifecycle_report: dict | None = None) -> list[str]:
        trade_log = self._learner._trade_log
        edge_memory = None
        if getattr(self, "_scorer", None) is not None:
            edge_memory = self._scorer.edge_detector.memory
        audit = build_bootstrap_audit_report(
            trade_log,
            lifecycle_report=lifecycle_report,
            edge_memory=edge_memory,
        )
        return format_bootstrap_audit_lines(audit)

    def _build_risk_budget(self) -> dict:
        """Snapshot of every active capital-protection budget for Telegram.

        Pure read-only — touches RiskManager state and config caps. Returns a
        dict whose shape matches TelegramNotifier._render_risk_budget_block.

        Halt is "active" whenever the bot will refuse the next trade for
        risk reasons (kill-switch, cooldown, daily/weekly cap, max DD,
        consecutive-loss cap). The reason follows can_open_trade() priority
        so the heartbeat shows the same explanation an operator would see
        in logs.

        Defensive `getattr` reads keep the telemetry pipeline alive even if
        a future RiskManager refactor renames/removes a state field.
        """
        risk_cfg = self._cfg.get("risk", {}) or {}
        trading_cfg = self._cfg.get("trading", {}) or {}
        safety_cfg = self._cfg.get("safety", {}) or {}
        state = self._risk.state

        def _g(name: str, default):
            return getattr(state, name, default)

        # ── Halt status (mirror RiskManager.can_open_trade priority) ──
        now = time.time()
        halt_active = False
        halt_reason = ""
        halt_remaining_s: int | None = None

        kill_reason = _g("kill_switch_reason", "")
        paused_until = float(_g("paused_until_ts", 0.0) or 0.0)
        if kill_reason:
            halt_active = True
            halt_reason = f"kill switch: {kill_reason}"
        elif paused_until and paused_until > now:
            halt_active = True
            halt_remaining_s = int(paused_until - now)
            halt_reason = "cooldown after loss streak"

        # ── Daily loss budget ─────────────────────────────────────
        daily_cap = float(risk_cfg.get("daily_loss_cap_pct", 0.0) or 0.0)
        daily_used = float(_g("daily_pnl_pct", 0.0) or 0.0)
        # Utilization tracks distance into the loss cap; positive PnL = 0%.
        if daily_cap > 0 and daily_used < 0:
            daily_util = min(1.0, abs(daily_used) / daily_cap)
        else:
            daily_util = 0.0
        if not halt_active and daily_cap > 0 and abs(daily_used) >= daily_cap and daily_used < 0:
            halt_active = True
            halt_reason = f"daily loss cap hit ({daily_used:.2f}%)"

        # ── Weekly loss budget (optional) ──────────────────────────
        weekly_cap = float(risk_cfg.get("weekly_loss_cap_pct", 0.0) or 0.0)
        weekly_block: dict | None = None
        if weekly_cap > 0:
            weekly_used = float(_g("weekly_pnl_pct", 0.0) or 0.0)
            if weekly_used < 0:
                weekly_util = min(1.0, abs(weekly_used) / weekly_cap)
            else:
                weekly_util = 0.0
            weekly_block = {
                "used_pct": weekly_used,
                "cap_pct": weekly_cap,
                "utilization": weekly_util,
            }
            if not halt_active and weekly_used <= -weekly_cap:
                halt_active = True
                halt_reason = f"weekly loss cap hit ({weekly_used:.2f}%)"

        # ── Drawdown ──────────────────────────────────────────────
        dd_cap = float(risk_cfg.get("max_drawdown_pct", 0.0) or 0.0)
        dd_cur = float(_g("drawdown_pct", 0.0) or 0.0)
        dd_util = min(1.0, dd_cur / dd_cap) if dd_cap > 0 else 0.0
        if not halt_active and dd_cap > 0 and dd_cur >= dd_cap:
            halt_active = True
            halt_reason = f"max drawdown hit ({dd_cur:.2f}%)"

        # ── Consecutive losses ────────────────────────────────────
        cons_cur = int(_g("consecutive_losses", 0) or 0)
        cons_cap = int(safety_cfg.get("max_consecutive_losses", 0) or 0)
        cooldown_min = int(risk_cfg.get("cooldown_minutes", 0) or 0)
        cooldown_after = int(risk_cfg.get("cooldown_after_loss_streak", 0) or 0)
        if not halt_active and cons_cap > 0 and cons_cur >= cons_cap:
            halt_active = True
            halt_reason = f"consecutive losses limit ({cons_cur})"

        # ── Concurrent positions ──────────────────────────────────
        open_now = int(_g("open_trade_count", 0) or 0)
        max_open = int(trading_cfg.get("max_open_trades", 0) or 0)
        open_util = min(1.0, open_now / max_open) if max_open > 0 else 0.0

        # ── Aggregate open risk ───────────────────────────────────
        per_trade_cap = float(risk_cfg.get("max_risk_per_trade_pct", 0.0) or 0.0)
        agg_cap = per_trade_cap * max_open if (per_trade_cap > 0 and max_open > 0) else 0.0
        agg_used = float(_g("open_risk_pct", 0.0) or 0.0)
        agg_util = min(1.0, agg_used / agg_cap) if agg_cap > 0 else 0.0

        # ── Correlation filter ────────────────────────────────────
        corr_cfg = (risk_cfg.get("correlation_filter") or {}) if isinstance(risk_cfg.get("correlation_filter"), dict) else {}
        correlation = {
            "enabled": bool(corr_cfg.get("enabled", True)),
            "cap": float(corr_cfg.get("corr_max", 0.85) or 0.85),
        }

        return {
            "halt": {
                "active": halt_active,
                "reason": halt_reason,
                "remaining_s": halt_remaining_s,
            },
            "daily_loss": {
                "used_pct": daily_used,
                "cap_pct": daily_cap,
                "utilization": daily_util,
            },
            "weekly_loss": weekly_block,
            "drawdown": {
                "current_pct": dd_cur,
                "cap_pct": dd_cap,
                "utilization": dd_util,
            },
            "concurrent": {
                "open": open_now,
                "cap": max_open,
                "utilization": open_util,
            },
            "aggregate_risk": {
                "open_pct": agg_used,
                "cap_pct": agg_cap,
                "utilization": agg_util,
            },
            "consecutive_losses": {
                "current": cons_cur,
                "cap": cons_cap,
                "cooldown_minutes": cooldown_min,
                "cooldown_after": cooldown_after,
            },
            "correlation": correlation,
        }

    def _rejection_summary(self, top_n: int = 3, reset: bool = True) -> dict:
        """Pull the per-cycle rejection tally for Telegram, tolerating an
        absent `_rej` attribute (e.g. paper-validation harness mocks)."""
        rej = getattr(self, "_rej", None)
        if rej is None or not hasattr(rej, "cycle_summary"):
            return {"total": 0, "stages": []}
        try:
            return rej.cycle_summary(top_n=top_n, reset=reset)
        except Exception as exc:  # pragma: no cover — telemetry must not crash trading
            log.warning("rejection summary failed: %s", exc)
            return {"total": 0, "stages": []}

    @staticmethod
    def _short_setup_summary(breakdowns: list) -> dict:
        """Tally Phase D / Liq Sweep detections across the current scan.

        Returns a dict consumed by ``TelegramNotifier.cycle_report`` /
        ``heartbeat`` to surface dedicated short-edge activity. The tally
        only reflects valid (high-confidence) detections — the same gate
        the StrategyRouter uses to admit the trade — so a high count here
        means the bot saw real post-distribution breakdown opportunities
        in the universe, not just noise.

        Shape:
            {
                "total": int,           # total valid short_setup detections
                "phase_d": int,
                "liq_sweep": int,
                "top": [
                    {"symbol": str, "label": str, "confidence": float},
                    ...
                ],
            }
        """
        out = {"total": 0, "phase_d": 0, "liq_sweep": 0, "top": []}
        if not breakdowns:
            return out
        rows: list[tuple[str, str, float]] = []
        for bd in breakdowns:
            setup = getattr(bd, "short_setup", None)
            if setup is None or not getattr(setup, "is_valid", False):
                continue
            label = str(getattr(setup, "label", "") or "")
            if not label or label == "none":
                continue
            confidence = float(getattr(setup, "confidence", 0.0) or 0.0)
            out["total"] += 1
            if label == "phase_d":
                out["phase_d"] += 1
            elif label == "liq_sweep":
                out["liq_sweep"] += 1
            rows.append((str(getattr(bd, "symbol", "?")), label, confidence))
        rows.sort(key=lambda r: r[2], reverse=True)
        out["top"] = [
            {"symbol": sym, "label": lab, "confidence": conf}
            for sym, lab, conf in rows[:3]
        ]
        return out

    async def _maybe_send_fund_manager_report(
        self,
        breakdowns: list[SignalBreakdown],
        readiness_report,
        jim_status: dict,
        floating_positions: list[dict],
        open_positions: list[dict],
        total_trades: int,
        win_rate: float,
        sharpe: float,
        cycle_num: int,
        lifecycle_report: dict | None = None,
    ) -> None:
        cycle_minutes = float(
            self._cfg.get("telegram", {}).get("cycle_report_interval_minutes", 0) or 0
        )
        fallback_minutes = float(
            self._safety.get("performance_report_interval_minutes", 5) or 5
        )
        report_interval = (cycle_minutes if cycle_minutes > 0 else fallback_minutes) * 60
        should_send_now = self._startup_cycle_report_pending
        if report_interval <= 0:
            return
        if not should_send_now and time.time() - self._tg_cycle_report_ts < report_interval:
            return

        self._startup_cycle_report_pending = False
        self._tg_cycle_report_ts = time.time()
        trade_log = self._learner._trade_log
        await self._telegram.cycle_report(
            breakdowns=breakdowns,
            equity=self._risk.state.equity,
            drawdown_pct=self._risk.state.drawdown_pct,
            daily_pnl_pct=self._risk.state.daily_pnl_pct,
            open_trades=self._risk.state.open_trade_count,
            cycle_num=cycle_num,
            ml_ready=self._ml.is_ready,
            ml_accuracy=self._ml.cv_accuracy,
            paper_trades=total_trades,
            readiness_passed=readiness_report.passed,
            consecutive_wins=self._consecutive_wins,
            consecutive_losses=self._risk.state.consecutive_losses,
            fm_scale=min(
                self._ml.kelly_adjustment(trade_log)
                * self._fund_mgr._recent_performance_mult(trade_log),
                2.0
            ),
            jim_status=jim_status,
            floating_positions=floating_positions,
            open_positions=open_positions,
            closed_positions=trade_log[-5:] if trade_log else [],
            win_rate=win_rate,
            sharpe=sharpe,
            total_trades=total_trades,
            trade_log=trade_log,
            equity_curve=self._equity_curve,
            starting_equity=self._starting_equity,
            regime_thresholds=self._trading.get("regime_thresholds", {}),
            jim_bonus=self._fund_mgr.bonus,
            risk_budget=self._build_risk_budget(),
            rejection_summary=self._rejection_summary(top_n=3, reset=True),
            short_setup_summary=self._short_setup_summary(breakdowns),
        )

    async def _maybe_send_equity_graph_report(self, cycle_num: int = 0) -> None:
        graph_minutes = float(
            self._cfg.get("telegram", {}).get("equity_graph_interval_minutes", 60) or 0
        )
        if graph_minutes <= 0 or len(self._equity_graph_points) < 2:
            return

        report_interval = graph_minutes * 60
        if time.time() - self._tg_equity_graph_ts < report_interval:
            return

        self._tg_equity_graph_ts = time.time()
        await self._telegram.equity_graph_report(
            self._equity_graph_points,
            cycle_num=cycle_num,
            interval_minutes=graph_minutes,
            mode=self._trading["mode"],
        )

    async def _maybe_send_bootstrap_audit_report(
        self,
        lifecycle_report: dict | None = None,
        cycle_num: int = 0,
    ) -> None:
        audit_hours = float(
            self._cfg.get("telegram", {}).get("bootstrap_audit_interval_hours", 12) or 0
        )
        if audit_hours <= 0:
            return
        report_interval = audit_hours * 3600
        if time.time() - self._last_bootstrap_audit_ts < report_interval:
            return
        lines = self._bootstrap_audit_lines(lifecycle_report)
        if not lines:
            return
        self._last_bootstrap_audit_ts = time.time()
        await self._telegram.bootstrap_audit_report(
            lines,
            cycle_num=cycle_num,
            interval_hours=audit_hours,
        )

    async def _maybe_send_crypto_news_report(self) -> None:
        tg_cfg = self._cfg.get("telegram", {}) or {}
        interval_minutes = float(tg_cfg.get("crypto_news_interval_minutes", 60) or 0)
        if interval_minutes <= 0:
            return
        report_interval = interval_minutes * 60
        if time.time() - self._tg_crypto_news_ts < report_interval:
            return
        self._tg_crypto_news_ts = time.time()
        limit = int(tg_cfg.get("crypto_news_limit", 5) or 5)
        timeout_s = float(tg_cfg.get("crypto_news_timeout_seconds", 8.0) or 8.0)
        try:
            items = await fetch_crypto_news_highlights(
                self._cfg,
                limit=max(1, min(10, limit)),
                timeout_s=max(1.0, timeout_s),
            )
        except Exception as exc:
            log.debug("Crypto news report failed: %s", exc)
            return
        if not items:
            return
        narratives = await build_market_hot_narratives(
            self._cfg,
            items,
            limit=max(1, min(10, limit)),
        )
        if not narratives:
            return
        await self._telegram.crypto_news_report(
            narratives,
            interval_minutes=interval_minutes,
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
        open_trades = len(self._trade_mgr._trades)
        if close_on_shutdown:
            log.info("Shutting down — closing all open trades...")
            await self._trade_mgr.close_all("shutdown")
        else:
            log.warning(
                "Shutting down — preserving open trades because close_positions_on_shutdown=false"
            )
        await self._telegram.shutdown(
            mode=self._trading["mode"],
            equity=self._risk.state.equity,
            open_trades=open_trades,
            close_positions=close_on_shutdown,
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
                "paper_state_reset_token": (
                    str(self._trading.get("paper_state_reset_token", "") or "")
                    if self._trading["mode"] == "paper"
                    else ""
                ),
                "equity": self._risk.state.equity,
                "peak_equity": self._risk.state.peak_equity,
                "daily_start_equity": self._risk.state.daily_start_equity,
                "day_start_ts": self._risk.state.day_start_ts,
                "weekly_start_equity": self._risk.state.weekly_start_equity,
                "week_start_ts": self._risk.state.week_start_ts,
                "daily_pnl_pct": self._risk.state.daily_pnl_pct,
                "weekly_pnl_pct": self._risk.state.weekly_pnl_pct,
                "drawdown_pct": self._risk.state.drawdown_pct,
                "consecutive_losses": self._risk.state.consecutive_losses,
                "consecutive_wins": self._consecutive_wins,
                "kill_switch_reason": self._risk.state.kill_switch_reason,
                "paused_until_ts": self._risk.state.paused_until_ts,
                "starting_equity": self._starting_equity,
                "equity_curve": self._equity_curve[-500:],
                "equity_graph_points": self._equity_graph_points[-720:],
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
                # Phase D / Liq Sweep dedicated edge metadata — empty when
                # the trade did not fire a dedicated short setup.
                "short_setup_label": str(getattr(trade.setup, "short_setup_label", "") or ""),
                "short_setup_confidence": float(getattr(trade.setup, "short_setup_confidence", 0.0) or 0.0),
            })
        return open_trades

    async def _price_map_with_open_trades(self, snapshots) -> dict[str, float]:
        price_map = {sym: snap.last_price for sym, snap in snapshots.items()}
        for symbol in self._trade_mgr.open_symbols:
            try:
                ticker = await self._client.fetch_ticker(symbol)
                live_price = float(ticker.get("last", 0) or 0)
                if live_price > 0:
                    price_map[symbol] = live_price
            except Exception as exc:
                log.warning("Could not fetch price for open trade %s: %s", symbol, exc)
        return price_map

    # ------------------------------------------------------------------ #
    #  Correlation-filter helpers                                         #
    # ------------------------------------------------------------------ #

    def _symbol_returns(self, snap):
        """Return log-returns of the primary timeframe for a snapshot."""
        if snap is None:
            return None
        try:
            df = snap.candles_for(self._cfg["timeframes"]["primary"])
            if df is None or df.empty or len(df) < 24:
                return None
            import numpy as np
            close = df["close"].astype(float)
            return np.log(close / close.shift(1)).dropna()
        except Exception as exc:
            log.debug("symbol_returns failed: %s", exc)
            return None

    def _open_trades_returns(self, snapshots) -> dict:
        """Build {symbol: {direction, returns}} for currently open trades."""
        out: dict[str, dict] = {}
        for sym in self._trade_mgr.open_symbols:
            trade = self._trade_mgr._trades.get(sym)
            if not trade:
                continue
            snap = snapshots.get(sym)
            returns = self._symbol_returns(snap) if snap is not None else None
            if returns is None:
                continue
            out[sym] = {
                "direction": trade.setup.direction,
                "returns": returns,
            }
        return out

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
    # Fail-fast schema validation AFTER env injection and CLI overrides so
    # `--mode live` against a config still set to testnet=true is caught at
    # boot, not at first trade. Identity-preserving — returns the same dict.
    # Catches risk-critical typos (kelly.max_kelly_pct: 4.0 -> 40, exit
    # sizing not summing to 1.0, etc.) instead of silently trading with the
    # wrong knob until the first oversized loss surfaces it.
    validate_config(cfg)
    cfg = normalize_config(cfg)

    setup_logging(cfg)
    if getattr(args, "bootstrap_audit_now", False):
        bot = NinjaTrader(cfg)
        try:
            lifecycle_report = bot._lifecycle.build_report(bot._learner._trade_log)
            lines = bot._bootstrap_audit_lines(lifecycle_report)
            await bot._telegram.bootstrap_audit_report(
                lines,
                interval_hours=float(
                    cfg.get("telegram", {}).get("bootstrap_audit_interval_hours", 12) or 12
                ),
            )
        finally:
            try:
                await bot._client.close()
            except Exception:
                pass
        return

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
    parser.add_argument(
        "--bootstrap-audit-now",
        action="store_true",
        help="Send one bootstrap audit report to Telegram and exit",
    )
    args = parser.parse_args()

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
    except ConfigValidationError as exc:
        # Print the structured multi-issue summary directly to stderr so
        # operators see the full issue list, not a Python traceback.
        console.print(f"[red]{exc}[/red]")
        import sys
        sys.exit(2)

if __name__ == "__main__":
    main()
