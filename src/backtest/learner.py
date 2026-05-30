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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


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
    mfe_r: float = 0.0
    mae_r: float = 0.0


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

        # Average signal scores for wins vs losses
        win_avg = self._avg_scores(wins, signal_keys)
        loss_avg = self._avg_scores(losses, signal_keys)

        # Nudge weights: signals higher in wins than losses get a boost
        total_adj = 0.0
        for key in signal_keys:
            win_s = win_avg.get(key, 50.0)
            loss_s = loss_avg.get(key, 50.0)
            delta = win_s - loss_s
            adjustment = delta * self._adj_rate
            old = self._weights[key]
            self._weights[key] = max(1.0, old + adjustment)
            total_adj += adjustment

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
            for t in state.get("trade_log", []):
                self._trade_log.append(TradeRecord(**t))
            log.info("Learner state loaded: %d trades, weights=%s", len(self._trade_log), self._weights)
        except Exception as exc:
            log.warning("Could not load learner state: %s", exc)
