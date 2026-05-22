"""
Thompson Sampling Multi-Armed Bandit for adaptive strategy sleeve selection.

This is the same algorithm Renaissance Technologies and quantitative funds use
for portfolio allocation across strategies. Each "arm" is a strategy sleeve
(trend_following, reversal, compression_breakout). The bandit maintains a
Beta(alpha, beta) posterior over win-rate per arm × context, where:
  alpha = wins observed (+ prior)
  beta  = losses observed (+ prior)

On each trade decision, it samples from each posterior and picks the arm
with the highest sample. This automatically balances:
  - EXPLORATION: arms with few samples have wide posteriors → occasional picks
  - EXPLOITATION: arms with strong track records have tight, high posteriors

Context = "regime|direction" — bandit learns which sleeve works in which market.

Reference: Thompson (1933), Russo et al. (2018) "A Tutorial on Thompson Sampling"
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

log = logging.getLogger(__name__)


class ThompsonBandit:
    """
    Context-aware Thompson Sampling for sleeve selection.

    Each (context × arm) combination has its own Beta posterior. This means
    the bandit can simultaneously learn:
      - "trend_following works in trending|long" (high posterior)
      - "trend_following struggles in distribution|long" (low posterior)
      - "reversal works in distribution|short" (high posterior)
    """

    STATE_PATH = Path("models/bandit_state.json")

    def __init__(self, arms: list[str], prior_alpha: float = 2.0, prior_beta: float = 2.0):
        self._arms = list(arms)
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta
        self._counts: dict[str, dict[str, float]] = {}
        self._load()

    @staticmethod
    def _key(arm: str, context: str = "global") -> str:
        return f"{context}|{arm}"

    def _bucket(self, arm: str, context: str = "global") -> dict:
        k = self._key(arm, context)
        if k not in self._counts:
            self._counts[k] = {
                "alpha": self._prior_alpha,
                "beta": self._prior_beta,
                "trades": 0,
                "wins": 0,
            }
        return self._counts[k]

    def sample(self, arm: str, context: str = "global") -> float:
        """Sample win-rate from Beta posterior. The randomness is the explore signal."""
        b = self._bucket(arm, context)
        return random.betavariate(b["alpha"], b["beta"])

    def best_arm(self, context: str = "global") -> tuple[str, float]:
        """Thompson sampling: sample each arm, return arm with highest sample."""
        samples = {arm: self.sample(arm, context) for arm in self._arms}
        best = max(samples.items(), key=lambda x: x[1])
        return best

    def expected_winrate(self, arm: str, context: str = "global") -> float:
        """Posterior mean of win rate (Bayesian estimate)."""
        b = self._bucket(arm, context)
        return b["alpha"] / (b["alpha"] + b["beta"])

    def confidence(self, arm: str, context: str = "global") -> float:
        """0..1 — how much real data we have (saturates at 20 observations)."""
        b = self._bucket(arm, context)
        n = (b["alpha"] - self._prior_alpha) + (b["beta"] - self._prior_beta)
        return min(1.0, max(0.0, n / 20.0))

    def size_mult(self, arm: str, context: str = "global") -> float:
        """
        Returns sizing multiplier in [0.6, 1.4] based on posterior performance.
        Scales with both win rate AND confidence — only sizes up when we
        have BOTH evidence of edge AND enough samples to trust it.
        """
        wr = self.expected_winrate(arm, context)
        conf = self.confidence(arm, context)
        # Scale: at conf=1 + wr=0.7 → 1.4x; at conf=1 + wr=0.3 → 0.6x
        scale = 1.0 + (wr - 0.5) * 2.0 * conf
        return max(0.6, min(1.4, scale))

    def update(self, arm: str, won: bool, context: str = "global") -> None:
        """Bayesian posterior update: alpha += win, beta += loss."""
        b = self._bucket(arm, context)
        b["trades"] += 1
        if won:
            b["alpha"] += 1
            b["wins"] += 1
        else:
            b["beta"] += 1
        self._save()

    def report(self) -> dict:
        """Returns full posterior state for telemetry."""
        out = {}
        for k, v in self._counts.items():
            n = v["trades"]
            wr = v["alpha"] / (v["alpha"] + v["beta"])
            out[k] = {
                "trades": n,
                "wins": int(v["wins"]),
                "post_wr": round(wr, 3),
                "alpha": round(v["alpha"], 2),
                "beta": round(v["beta"], 2),
            }
        return out

    def top_contexts(self, n: int = 5) -> list[tuple[str, float, int]]:
        """Returns top-N (context|arm, win_rate, trade_count) tuples."""
        rows = [(k, v["alpha"] / (v["alpha"] + v["beta"]), v["trades"])
                for k, v in self._counts.items() if v["trades"] >= 3]
        rows.sort(key=lambda r: (r[1], r[2]), reverse=True)
        return rows[:n]

    def _save(self) -> None:
        try:
            self.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.STATE_PATH.write_text(json.dumps(self._counts, indent=2))
        except Exception as exc:
            log.warning("ThompsonBandit save failed: %s", exc)

    def _load(self) -> None:
        if not self.STATE_PATH.exists():
            return
        try:
            self._counts = json.loads(self.STATE_PATH.read_text())
            log.info("ThompsonBandit loaded %d (context|arm) buckets", len(self._counts))
        except Exception as exc:
            log.warning("ThompsonBandit load failed: %s", exc)
            self._counts = {}
