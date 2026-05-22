"""
HMM-style regime transition tracker.

This is the same idea behind Hidden Markov Model regime detectors used by
quant funds for tactical asset allocation (e.g., AQR's regime models).
We track empirical P(regime_t+1 | regime_t) per symbol and use it to:

  1. Predict which regime is likely NEXT given current state
  2. Detect when current regime has lasted longer than typical
     (= regime exhaustion, shift imminent)
  3. Provide stability score for sizing — high stability → press;
     low stability → reduce exposure

Key insight: crypto regimes don't last forever. The transition matrix
captures the natural cycle:
  trending_expansion → distribution → chaos → accumulation → trending...

Reference: Rabiner (1989) "A Tutorial on HMMs"
           Ang & Bekaert (2002) "Regime Switches in Interest Rates"
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)


class RegimeTransitionMatrix:
    """
    Empirical regime transition tracker.

    Maintains:
      - transitions[from_regime][to_regime] = count
      - durations[regime] = list of seconds spent in regime before transitioning
      - per-symbol current state and start timestamp

    Smoothed with Laplace prior (+1) to avoid zero probabilities.
    """

    STATE_PATH = Path("models/regime_transitions.json")
    REGIMES = ["trending_expansion", "accumulation_compression", "distribution", "chaos"]

    def __init__(self):
        self._transitions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._last_regime: dict[str, str] = {}
        self._regime_durations: dict[str, list[float]] = defaultdict(list)
        self._regime_start_ts: dict[str, float] = {}
        self._load()

    def observe(self, symbol: str, current_regime: str) -> None:
        """Record observation. Detects transitions and updates the matrix."""
        if current_regime not in self.REGIMES:
            return
        prev = self._last_regime.get(symbol)
        now = time.time()

        if prev and prev != current_regime:
            # Transition detected!
            self._transitions[prev][current_regime] += 1
            duration = now - self._regime_start_ts.get(symbol, now)
            self._regime_durations[prev].append(duration)
            self._regime_durations[prev] = self._regime_durations[prev][-50:]
            self._regime_start_ts[symbol] = now
            log.debug("Regime transition for %s: %s → %s (lasted %.1fs)",
                      symbol, prev, current_regime, duration)
        elif prev is None:
            self._regime_start_ts[symbol] = now

        self._last_regime[symbol] = current_regime

    def transition_probs(self, from_regime: str) -> dict[str, float]:
        """Smoothed transition probabilities using Laplace prior."""
        counts = self._transitions.get(from_regime, {})
        total_count = sum(counts.values())
        # +1 smoothing per regime + total denominator
        denom = total_count + len(self.REGIMES)
        return {
            r: (counts.get(r, 0) + 1) / denom
            for r in self.REGIMES
        }

    def most_likely_next(self, from_regime: str) -> tuple[str, float]:
        """Returns (next_regime, probability) — most likely transition target."""
        probs = self.transition_probs(from_regime)
        # Exclude self-loop to find next "different" regime
        non_self = {k: v for k, v in probs.items() if k != from_regime}
        if not non_self:
            return from_regime, probs.get(from_regime, 0.5)
        best = max(non_self.items(), key=lambda x: x[1])
        return best

    def expected_duration(self, regime: str) -> float:
        """Median seconds typically spent in this regime before transitioning."""
        durations = self._regime_durations.get(regime, [])
        if len(durations) < 3:
            return 7200.0  # 2hr fallback
        sorted_d = sorted(durations)
        return sorted_d[len(sorted_d) // 2]

    def is_due_for_shift(self, symbol: str, current_regime: str, threshold_mult: float = 1.5) -> bool:
        """
        Returns True if symbol has been in current regime longer than
        threshold_mult × median duration → shift imminent → reduce exposure.
        """
        start = self._regime_start_ts.get(symbol)
        if start is None:
            return False
        elapsed = time.time() - start
        expected = self.expected_duration(current_regime)
        return elapsed > expected * threshold_mult

    def stability_score(self, current_regime: str) -> float:
        """
        Returns P(regime persists) ∈ [0, 1].
        Higher = current regime is "sticky" → trends/edges should continue.
        Lower = regime flips often → reduce conviction.
        """
        probs = self.transition_probs(current_regime)
        return probs.get(current_regime, 0.5)

    def report(self) -> dict:
        """Full state for telemetry."""
        return {
            "transitions": {
                k: dict(v) for k, v in self._transitions.items()
            },
            "median_durations_s": {
                r: round(self.expected_duration(r), 0) for r in self.REGIMES
            },
            "stability": {
                r: round(self.stability_score(r), 3) for r in self.REGIMES
            },
            "n_symbols_tracked": len(self._last_regime),
        }

    def _save(self) -> None:
        try:
            self.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "transitions": {k: dict(v) for k, v in self._transitions.items()},
                "last_regime": self._last_regime,
                "durations": {k: list(v) for k, v in self._regime_durations.items()},
                "start_ts": self._regime_start_ts,
            }
            self.STATE_PATH.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            log.warning("RegimeTransition save failed: %s", exc)

    def _load(self) -> None:
        if not self.STATE_PATH.exists():
            return
        try:
            data = json.loads(self.STATE_PATH.read_text())
            for k, v in data.get("transitions", {}).items():
                self._transitions[k] = defaultdict(int, v)
            self._last_regime = data.get("last_regime", {})
            for k, v in data.get("durations", {}).items():
                self._regime_durations[k] = list(v)
            self._regime_start_ts = data.get("start_ts", {})
            log.info("RegimeTransitionMatrix loaded: %d transitions tracked",
                     sum(sum(v.values()) for v in self._transitions.values()))
        except Exception as exc:
            log.warning("RegimeTransition load failed: %s", exc)
