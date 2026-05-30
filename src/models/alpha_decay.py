"""
Alpha decay tracker — per-pair rolling edge monitor.

In crypto, edges decay rapidly. A pair that gave 70% WR last week may
have negative edge this week due to:
  - Other algos finding the same pattern (alpha capacity exhaustion)
  - Regime shift specific to that asset
  - Coordinated market making activity neutralizing the inefficiency

This tracker:
  1. Maintains rolling window of last N pnl_pct per pair
  2. Computes Sharpe-like edge score (mean / std)
  3. Marks pair "dead" if edge score crosses negative threshold
  4. Auto-revives dead pairs after observation gap (markets cycle)
  5. Provides edge_score for sizing — hot pairs get press, cold get cut

Reference: Korajczyk & Sadka (2004) "Are Momentum Profits Robust to Trading Costs?"
           Industry practice at Citadel, Two Sigma, Renaissance
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path

from src.utils.atomic_write import atomic_write

log = logging.getLogger(__name__)


class AlphaDecayTracker:
    """
    Per-pair rolling edge tracker with auto-revival.

    Each pair's edge is tracked via Sharpe-like ratio over its last N trades.
    If edge drops below threshold → marked dead, no new positions allowed.
    After revival_after_s elapses → reset and try again (markets change).
    """

    STATE_PATH = Path("models/alpha_decay.json")
    WINDOW = 15

    def __init__(
        self,
        decay_threshold: float = -0.15,
        min_obs_to_kill: int = 8,
        revival_after_s: float = 86400.0,  # 24 hours
    ):
        self._returns: dict[str, list[float]] = defaultdict(list)
        self._dead_pairs: dict[str, float] = {}  # symbol → ts when killed
        self._decay_threshold = decay_threshold
        self._min_obs_to_kill = min_obs_to_kill
        self._revival_after_s = revival_after_s
        self._load()

    def record(self, symbol: str, pnl_pct: float) -> None:
        """Record a closed trade outcome."""
        self._returns[symbol].append(float(pnl_pct))
        self._returns[symbol] = self._returns[symbol][-self.WINDOW:]

        # Re-evaluate alive status
        if len(self._returns[symbol]) >= self._min_obs_to_kill:
            score = self._raw_edge_score(symbol)
            if score < self._decay_threshold and symbol not in self._dead_pairs:
                self._dead_pairs[symbol] = time.time()
                log.warning("AlphaDecay: %s marked DEAD (score=%.3f, n=%d)",
                            symbol, score, len(self._returns[symbol]))
            elif score > 0.05 and symbol in self._dead_pairs:
                # Pair recovered organically (rare but possible)
                del self._dead_pairs[symbol]
                log.info("AlphaDecay: %s REVIVED (score=%.3f)", symbol, score)
        self._save()

    def _raw_edge_score(self, symbol: str) -> float:
        """Sharpe-like ratio: mean / std of recent returns. Range roughly [-2, +2]."""
        rets = self._returns.get(symbol, [])
        if len(rets) < 5:
            return 0.0
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / max(1, len(rets) - 1)
        std = math.sqrt(max(var, 1e-9))
        if std < 1e-6:
            return 0.0 if mean == 0 else (1.0 if mean > 0 else -1.0)
        return mean / std

    def is_alive(self, symbol: str) -> tuple[bool, float]:
        """
        Returns (is_alive, normalized_edge_score ∈ [0, 1]).
        Auto-revives pairs after revival window.
        """
        # Check for auto-revival
        if symbol in self._dead_pairs:
            elapsed = time.time() - self._dead_pairs[symbol]
            if elapsed > self._revival_after_s:
                del self._dead_pairs[symbol]
                self._returns[symbol] = []  # fresh slate
                self._save()
                log.info("AlphaDecay: %s auto-revived after %.0fs", symbol, elapsed)
                return True, 0.5  # neutral starting score
            return False, 0.0

        score = self._raw_edge_score(symbol)
        # Map roughly [-1, 1] → [0, 1] via sigmoid-ish
        normalized = max(0.0, min(1.0, (score + 1.0) / 2.0))
        return True, normalized

    def get_size_mult(self, symbol: str) -> float:
        """
        Returns size multiplier for a pair based on its edge:
          - dead pair: 0.0 (block trade)
          - hot pair (edge > 0.5): up to 1.20x
          - cold pair (edge < 0.5): down to 0.70x
          - new/neutral: 1.00x
        """
        alive, score = self.is_alive(symbol)
        if not alive:
            return 0.0
        if len(self._returns.get(symbol, [])) < 5:
            return 1.0  # not enough data — neutral
        # score ∈ [0, 1] → mult ∈ [0.7, 1.2]
        return 0.7 + score * 0.5

    def summary(self) -> dict:
        """Telemetry summary."""
        alive_pairs = []
        for sym in self._returns:
            if sym in self._dead_pairs:
                continue
            score = self._raw_edge_score(sym)
            n = len(self._returns[sym])
            if n >= 3:
                alive_pairs.append((sym, round(score, 3), n))
        alive_pairs.sort(key=lambda x: x[1], reverse=True)

        return {
            "n_pairs_tracked": len(self._returns),
            "n_dead_pairs": len(self._dead_pairs),
            "dead_pairs": list(self._dead_pairs.keys()),
            "top_alive_pairs": alive_pairs[:5],
            "worst_alive_pairs": alive_pairs[-3:] if len(alive_pairs) >= 3 else [],
        }

    def _save(self) -> None:
        try:
            self.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "returns": dict(self._returns),
                "dead_pairs": self._dead_pairs,
            }
            atomic_write(self.STATE_PATH, json.dumps(data, indent=2))
        except Exception as exc:
            log.warning("AlphaDecay save failed: %s", exc)

    def _load(self) -> None:
        if not self.STATE_PATH.exists():
            return
        try:
            data = json.loads(self.STATE_PATH.read_text())
            for k, v in data.get("returns", {}).items():
                self._returns[k] = list(v)
            self._dead_pairs = data.get("dead_pairs", {})
            log.info("AlphaDecayTracker loaded: %d pairs (%d dead)",
                     len(self._returns), len(self._dead_pairs))
        except Exception as exc:
            log.warning("AlphaDecay load failed: %s", exc)
