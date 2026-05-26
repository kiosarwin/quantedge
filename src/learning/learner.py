"""
Self-learning weight adjuster.
Tracks closed trade outcomes and adjusts scoring weights toward signals
that correlate with winning trades.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def _trade_record_from_dict(data: dict) -> "TradeRecord":
    allowed = {field.name for field in fields(TradeRecord)}
    record = TradeRecord(**{key: value for key, value in data.items() if key in allowed})
    _backfill_trade_context(record)
    return record


def _backfill_trade_context(record: "TradeRecord") -> bool:
    """Fill context fields for legacy/restored records without pending scores."""
    from datetime import datetime, timezone

    from src.models.strategy_passport import asset_from_symbol, sector_for_asset
    from src.session_clock import active_market_session_key

    changed = False
    scores = record.scores if isinstance(record.scores, dict) else {}
    if not isinstance(record.scores, dict):
        record.scores = scores
        changed = True

    opened_at = float(record.opened_at or 0.0)
    if opened_at > 0:
        tm = time.gmtime(opened_at)
        dt = datetime.fromtimestamp(opened_at, tz=timezone.utc)
        if Learner._safe_int(record.hour_of_day, -1) < 0:
            record.hour_of_day = int(tm.tm_hour)
            changed = True
        if Learner._safe_int(record.day_of_week, -1) < 0:
            record.day_of_week = int(tm.tm_wday)
            changed = True
        if not record.session or str(record.session).lower() == "unknown":
            record.session = active_market_session_key(dt)
            changed = True

    if not record.asset or str(record.asset).lower() == "unknown":
        record.asset = asset_from_symbol(record.symbol)
        changed = True
    if not record.sector or str(record.sector).lower() == "unknown":
        record.sector = sector_for_asset(record.asset)
        changed = True

    market_defaults = {
        "market_risk_on_state": "unknown",
        "market_rotation_state": "unknown",
        "market_btc_trend": "unknown",
        "market_eth_btc_trend": "unknown",
        "market_btc_d_trend": "unknown",
        "market_total_trend": "unknown",
    }
    for key, fallback in market_defaults.items():
        current = getattr(record, key, fallback)
        score_value = scores.get(key)
        if (not current or str(current).lower() == "unknown") and score_value:
            setattr(record, key, str(score_value))
            changed = True
    if float(getattr(record, "market_context_confidence", 0.0) or 0.0) <= 0.0:
        try:
            score_conf = float(scores.get("market_context_confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            score_conf = 0.0
        if score_conf > 0.0:
            record.market_context_confidence = score_conf
            changed = True

    defaults = {
        "session": record.session,
        "hour_of_day": record.hour_of_day,
        "day_of_week": record.day_of_week,
        "asset": record.asset,
        "sector": record.sector,
        "market_risk_on_state": record.market_risk_on_state,
        "market_rotation_state": record.market_rotation_state,
        "market_btc_trend": record.market_btc_trend,
        "market_eth_btc_trend": record.market_eth_btc_trend,
        "market_btc_d_trend": record.market_btc_d_trend,
        "market_total_trend": record.market_total_trend,
        "market_context_confidence": record.market_context_confidence,
    }
    for key, value in defaults.items():
        current = scores.get(key)
        if current is None or str(current).lower() == "unknown" or current == -1:
            scores[key] = value
            changed = True
    return changed


@dataclass
class TradeRecord:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    pnl_usd: float
    pnl_pct: float
    reason: str
    scores: dict        # signal breakdown at entry
    opened_at: float
    closed_at: float = 0.0
    regime: str = ""    # regime at entry — used for per-regime edge tracking
    strategy_sleeve: str = ""
    exit_profile: str = ""
    dispersion_value: float = 0.0
    dispersion_state: str = "normal"
    mfe_r: float = 0.0
    mae_r: float = 0.0
    tp1_hit: bool = False
    hold_duration_s: float = 0.0
    fees_usd: float = 0.0
    slippage_estimate_usd: float = 0.0
    fees_slippage_pct: float = 0.0
    timeframe: str = "unknown"
    session: str = "unknown"
    hour_of_day: int = -1
    day_of_week: int = -1
    asset: str = "unknown"
    sector: str = "unknown"
    market_risk_on_state: str = "unknown"
    market_rotation_state: str = "unknown"
    market_btc_trend: str = "unknown"
    market_eth_btc_trend: str = "unknown"
    market_btc_d_trend: str = "unknown"
    market_total_trend: str = "unknown"
    market_context_confidence: float = 0.0
    signal_type: str = "unknown"
    entry_reason: str = "unknown"
    volatility_bucket: str = "unknown"
    trend_bucket: str = "unknown"
    cohort_key: str = ""
    lifecycle_status: str = "RESEARCH"


class Learner:
    def __init__(self, cfg: dict, scorer=None):
        self._cfg = cfg
        self._scorer = scorer       # Scorer instance to push updated weights to
        learn_cfg = cfg["learning"]
        self._min_trades = learn_cfg["min_trades_to_adjust"]
        self._lookback = learn_cfg["lookback_trades"]
        self._adj_rate = cfg["scoring"]["adjustment_rate"]

        self._save_path = Path(learn_cfg["save_path"])
        self._save_path.parent.mkdir(parents=True, exist_ok=True)

        self._weights: dict[str, float] = dict(cfg["scoring"]["weights"])
        self._trade_log: list[TradeRecord] = []
        self._load_state()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def record_trade(self, record: TradeRecord) -> None:
        self._trade_log.append(record)
        if len(self._trade_log) >= self._min_trades:
            self._adjust_weights()
        self._save_state()

    @property
    def current_weights(self) -> dict[str, float]:
        return dict(self._weights)

    # ------------------------------------------------------------------ #
    #  Weight adjustment                                                   #
    # ------------------------------------------------------------------ #

    def _adjust_weights(self) -> None:
        recent = self._trade_log[-self._lookback:]
        if len(recent) < self._min_trades:
            return

        signal_keys = list(self._weights.keys())
        wins = [t for t in recent if t.pnl_usd > 0]
        losses = [t for t in recent if t.pnl_usd <= 0]

        if not wins or not losses:
            return

        # ── Standard signal-level adjustment ──────────────────────────
        win_avg = self._avg_scores(wins, signal_keys)
        loss_avg = self._avg_scores(losses, signal_keys)

        for key in signal_keys:
            win_s = win_avg.get(key, 50.0)
            loss_s = loss_avg.get(key, 50.0)
            delta = win_s - loss_s
            adjustment = delta * self._adj_rate
            old = self._weights[key]
            self._weights[key] = max(1.0, old + adjustment)

        # ── NEW: Regime × Direction weight modulation ─────────────────
        # Learns which signals matter MORE in specific regime-direction combos.
        # E.g., "structure_quality" matters more for short-in-distribution than
        # for long-in-trending. This gives the bot adaptive edge per context.
        self._regime_direction_adjust(recent, signal_keys)
        self._context_adjust(recent, signal_keys)

        # Re-normalise to keep total weight sum unchanged
        total = sum(self._weights.values())
        target = sum(self._cfg["scoring"]["weights"].values())
        if total > 0:
            factor = target / total
            for k in self._weights:
                self._weights[k] = round(self._weights[k] * factor, 3)

        log.info("Learner adjusted weights: %s", self._weights)

        if self._scorer is not None:
            self._scorer.update_weights(self._weights)

    def _regime_direction_adjust(self, recent: list["TradeRecord"], signal_keys: list[str]) -> None:
        """
        Per regime×direction learning: boost weights that predict wins
        within each specific context. This makes the bot smarter over time
        about WHAT matters in each market condition.

        Practical impact: if "open_interest" is the winning signal in
        distribution+short trades but not in trending+long trades, the
        weight adjusts accordingly across the combined pool.
        """
        from collections import defaultdict

        # Group trades by regime × direction
        groups: dict[str, list] = defaultdict(list)
        for t in recent:
            key = f"{getattr(t, 'regime', 'unknown')}|{t.direction}"
            groups[key].append(t)

        # Only learn from groups with enough samples (≥4 trades)
        regime_adj_rate = self._adj_rate * 0.5  # Half the normal rate (conservative)

        for group_key, trades in groups.items():
            if len(trades) < 4:
                continue

            group_wins = [t for t in trades if t.pnl_usd > 0]
            group_losses = [t for t in trades if t.pnl_usd <= 0]

            if not group_wins or not group_losses:
                continue

            # Win rate in this group
            wr = len(group_wins) / len(trades)

            # If this regime×direction combo has HIGH win rate (>60%),
            # boost the signals that are elevated in its winning trades.
            # If LOW win rate (<40%), dampen the signals that led us astray.
            if wr >= 0.60:
                win_avg = self._avg_scores(group_wins, signal_keys)
                for key in signal_keys:
                    boost = (win_avg.get(key, 50.0) - 50.0) / 50.0 * regime_adj_rate
                    self._weights[key] = max(1.0, self._weights[key] + boost)
            elif wr <= 0.40:
                loss_avg = self._avg_scores(group_losses, signal_keys)
                for key in signal_keys:
                    penalty = (loss_avg.get(key, 50.0) - 50.0) / 50.0 * regime_adj_rate
                    self._weights[key] = max(1.0, self._weights[key] - penalty * 0.5)

    def _context_adjust(self, recent: list["TradeRecord"], signal_keys: list[str]) -> None:
        """
        Conservative sector x time learning.

        This is intentionally a small modulation of the existing score weights,
        not a new gate. It lets the learner notice that a signal component works
        better in a specific sector/session/hour/day context while preserving
        the current admission, risk, and scoring contracts.
        """
        from collections import defaultdict

        groups: dict[str, list[TradeRecord]] = defaultdict(list)
        for trade in recent:
            for key in self._context_keys(trade):
                groups[key].append(trade)

        context_adj_rate = self._adj_rate * 0.25
        accumulated = {key: 0.0 for key in signal_keys}
        for trades in groups.values():
            if len(trades) < 4:
                continue

            wins = [t for t in trades if t.pnl_usd > 0]
            losses = [t for t in trades if t.pnl_usd <= 0]
            if not wins or not losses:
                continue

            wr = len(wins) / len(trades)
            if 0.45 < wr < 0.55:
                continue

            win_avg = self._avg_scores(wins, signal_keys)
            loss_avg = self._avg_scores(losses, signal_keys)
            strength = min(1.0, len(trades) / max(8.0, float(self._min_trades)))

            for key in signal_keys:
                delta = (win_avg.get(key, 50.0) - loss_avg.get(key, 50.0)) / 50.0
                adjustment = delta * context_adj_rate * strength
                accumulated[key] += adjustment if wr >= 0.55 else adjustment * 0.5

        cap = context_adj_rate * 2.0
        for key, adjustment in accumulated.items():
            clipped = max(-cap, min(cap, adjustment))
            if clipped:
                self._weights[key] = max(1.0, self._weights[key] + clipped)

    @staticmethod
    def _context_keys(trade: "TradeRecord") -> list[str]:
        scores = trade.scores if isinstance(trade.scores, dict) else {}
        sector = Learner._metadata_value(trade.sector, scores.get("sector"), scores.get("setup_sector"))
        session = Learner._metadata_value(trade.session, scores.get("session"))
        hour = Learner._safe_int(getattr(trade, "hour_of_day", -1), -1)
        if hour < 0:
            hour = Learner._safe_int(scores.get("hour_of_day", scores.get("hour_utc", -1)), -1)
        day = Learner._safe_int(getattr(trade, "day_of_week", -1), -1)
        if day < 0:
            day = Learner._safe_int(scores.get("day_of_week", -1), -1)

        sector = str(sector or "unknown").strip().lower() or "unknown"
        session = str(session or "unknown").strip().lower() or "unknown"
        keys: list[str] = []
        if sector != "unknown" and session != "unknown" and hour >= 0 and day >= 0:
            hour_bucket = f"h{(hour // 4) * 4:02d}"
            keys.extend([
                f"sector_session_hour_day|{sector}|{session}|{hour_bucket}|d{day}",
                f"sector_session_hour|{sector}|{session}|{hour_bucket}",
                f"sector_session_day|{sector}|{session}|d{day}",
                f"sector_session|{sector}|{session}",
            ])

        risk_state = Learner._metadata_value(
            getattr(trade, "market_risk_on_state", "unknown"),
            scores.get("market_risk_on_state"),
        ).lower()
        rotation_state = Learner._metadata_value(
            getattr(trade, "market_rotation_state", "unknown"),
            scores.get("market_rotation_state"),
        ).lower()
        btc_trend = Learner._metadata_value(
            getattr(trade, "market_btc_trend", "unknown"),
            scores.get("market_btc_trend"),
        ).lower()
        eth_btc_trend = Learner._metadata_value(
            getattr(trade, "market_eth_btc_trend", "unknown"),
            scores.get("market_eth_btc_trend"),
        ).lower()

        if risk_state != "unknown":
            keys.append(f"market_risk|{risk_state}")
        if rotation_state != "unknown":
            keys.append(f"market_rotation|{rotation_state}")
        if risk_state != "unknown" and rotation_state != "unknown":
            keys.append(f"market_context|{risk_state}|{rotation_state}")
        if btc_trend != "unknown" and eth_btc_trend != "unknown":
            keys.append(f"market_btc_ethbtc|{btc_trend}|{eth_btc_trend}")
        return keys

    @staticmethod
    def _metadata_value(*values) -> str:
        for value in values:
            text = str(value or "").strip()
            if text and text.lower() != "unknown":
                return text
        return "unknown"

    @staticmethod
    def _safe_int(value, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _avg_scores(trades: list[TradeRecord], keys: list[str]) -> dict[str, float]:
        acc: dict[str, float] = {k: 0.0 for k in keys}
        count = 0
        for t in trades:
            for k in keys:
                acc[k] += t.scores.get(k, 50.0)
            count += 1
        if count:
            return {k: v / count for k, v in acc.items()}
        return acc

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def _save_state(self) -> None:
        state = {
            "weights": self._weights,
            "trade_log": [asdict(t) for t in self._trade_log],
        }
        try:
            self._save_path.write_text(json.dumps(state, indent=2))
        except Exception as exc:
            log.warning("Could not save learner state: %s", exc)

    def _load_state(self) -> None:
        if not self._save_path.exists():
            return
        try:
            state = json.loads(self._save_path.read_text())
            loaded_weights = state.get("weights", {})
            for k in self._weights:
                if k in loaded_weights:
                    self._weights[k] = float(loaded_weights[k])
            backfilled = False
            for t in state.get("trade_log", []):
                record = _trade_record_from_dict(t)
                self._trade_log.append(record)
                backfilled = backfilled or any(
                    t.get(key) != getattr(record, key)
                    for key in (
                        "session",
                        "hour_of_day",
                        "day_of_week",
                        "asset",
                        "sector",
                        "market_risk_on_state",
                        "market_rotation_state",
                        "market_btc_trend",
                        "market_eth_btc_trend",
                        "market_btc_d_trend",
                        "market_total_trend",
                        "market_context_confidence",
                    )
                )
            if backfilled:
                self._save_state()
            log.info("Learner state loaded: %d trades, weights=%s", len(self._trade_log), self._weights)
        except Exception as exc:
            log.warning("Could not load learner state: %s", exc)
