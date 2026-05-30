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
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

from src.utils.atomic_write import atomic_write
from src.session_clock import active_market_session_key

log = logging.getLogger(__name__)


@dataclass
class _SessionState:
    transitions: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float))
    )
    regime_durations: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    last_decay_ts: float = 0.0


@dataclass
class _SymbolState:
    session: str
    regime: str
    start_ts: float


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

    def __init__(self, decay_half_life_s: float = 14 * 24 * 3600.0):
        self._decay_half_life_s = max(1.0, float(decay_half_life_s))
        self._sessions: dict[str, _SessionState] = {}
        self._symbol_state: dict[str, _SymbolState] = {}
        self._load()

    def observe(self, symbol: str, current_regime: str, session: str | None = None) -> None:
        """Record observation. Detects transitions and updates the matrix."""
        if current_regime not in self.REGIMES:
            return
        session_key = session or active_market_session_key()
        now = time.time()
        session_state = self._session_state(session_key)
        self._apply_decay(session_key, now)

        prev = self._symbol_state.get(symbol)
        if prev is None or prev.session != session_key:
            self._symbol_state[symbol] = _SymbolState(session=session_key, regime=current_regime, start_ts=now)
            return

        prev_regime = prev.regime
        if prev_regime != current_regime:
            # Transition detected!
            session_state.transitions[prev_regime][current_regime] += 1.0
            duration = now - prev.start_ts
            session_state.regime_durations[prev_regime].append(duration)
            session_state.regime_durations[prev_regime] = session_state.regime_durations[prev_regime][-50:]
            prev.regime = current_regime
            prev.start_ts = now
            log.debug("Regime transition for %s: %s → %s (lasted %.1fs)",
                      symbol, prev_regime, current_regime, duration)

    def transition_probs(self, from_regime: str, session: str | None = None) -> dict[str, float]:
        """Smoothed transition probabilities using Laplace prior."""
        session_key = session or active_market_session_key()
        self._apply_decay(session_key)
        counts = self._session_state(session_key).transitions.get(from_regime, {})
        total_count = sum(counts.values())
        # +1 smoothing per regime + total denominator
        denom = total_count + len(self.REGIMES)
        return {
            r: (counts.get(r, 0) + 1) / denom
            for r in self.REGIMES
        }

    def most_likely_next(self, from_regime: str, session: str | None = None) -> tuple[str, float]:
        """Returns (next_regime, probability) — most likely transition target."""
        probs = self.transition_probs(from_regime, session=session)
        # Exclude self-loop to find next "different" regime
        non_self = {k: v for k, v in probs.items() if k != from_regime}
        if not non_self:
            return from_regime, probs.get(from_regime, 0.5)
        best = max(non_self.items(), key=lambda x: x[1])
        return best

    def expected_duration(self, regime: str, session: str | None = None) -> float:
        """Median seconds typically spent in this regime before transitioning."""
        session_key = session or active_market_session_key()
        durations = self._session_state(session_key).regime_durations.get(regime, [])
        if len(durations) < 3:
            return 7200.0  # 2hr fallback
        sorted_d = sorted(durations)
        return sorted_d[len(sorted_d) // 2]

    def is_due_for_shift(
        self,
        symbol: str,
        current_regime: str,
        threshold_mult: float = 1.5,
        session: str | None = None,
    ) -> bool:
        """
        Returns True if symbol has been in current regime longer than
        threshold_mult × median duration → shift imminent → reduce exposure.
        """
        session_key = session or active_market_session_key()
        state = self._symbol_state.get(symbol)
        if state is None or state.session != session_key:
            return False
        elapsed = time.time() - state.start_ts
        expected = self.expected_duration(current_regime, session=session_key)
        return elapsed > expected * threshold_mult

    def stability_score(self, current_regime: str, session: str | None = None) -> float:
        """
        Returns P(regime persists) ∈ [0, 1].
        Higher = current regime is "sticky" → trends/edges should continue.
        Lower = regime flips often → reduce conviction.
        """
        probs = self.transition_probs(current_regime, session=session)
        return probs.get(current_regime, 0.5)

    def report(self) -> dict:
        """Full state for telemetry."""
        current_session = active_market_session_key()
        active_state = self._session_state(current_session)
        return {
            "active_session": current_session,
            "transitions": {
                k: dict(v) for k, v in active_state.transitions.items()
            },
            "median_durations_s": {
                r: round(self.expected_duration(r, session=current_session), 0) for r in self.REGIMES
            },
            "stability": {
                r: round(self.stability_score(r, session=current_session), 3) for r in self.REGIMES
            },
            "n_symbols_tracked": len(self._symbol_state),
            "n_sessions_tracked": len(self._sessions),
        }

    def _save(self) -> None:
        try:
            self.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "decay_half_life_s": self._decay_half_life_s,
                "sessions": {
                    session: {
                        "transitions": {k: dict(v) for k, v in state.transitions.items()},
                        "durations": {k: list(v) for k, v in state.regime_durations.items()},
                        "last_decay_ts": state.last_decay_ts,
                    }
                    for session, state in self._sessions.items()
                },
                "symbols": {
                    symbol: {
                        "session": state.session,
                        "regime": state.regime,
                        "start_ts": state.start_ts,
                    }
                    for symbol, state in self._symbol_state.items()
                },
            }
            atomic_write(self.STATE_PATH, json.dumps(data, indent=2))
        except Exception as exc:
            log.warning("RegimeTransition save failed: %s", exc)

    def _load(self) -> None:
        if not self.STATE_PATH.exists():
            return
        try:
            data = json.loads(self.STATE_PATH.read_text())
            self._decay_half_life_s = float(data.get("decay_half_life_s", self._decay_half_life_s))
            sessions = data.get("sessions")
            if sessions is None:
                sessions = {
                    "legacy": {
                        "transitions": data.get("transitions", {}),
                        "durations": data.get("durations", {}),
                        "last_decay_ts": time.time(),
                    }
                }
                symbols = {}
                for symbol, regime in (data.get("last_regime", {}) or {}).items():
                    symbols[symbol] = {
                        "session": "legacy",
                        "regime": regime,
                        "start_ts": float(data.get("start_ts", {}).get(symbol, time.time())),
                    }
            else:
                symbols = data.get("symbols", {}) or {}

            for session, state in sessions.items():
                sess = _SessionState()
                sess.transitions = defaultdict(
                    lambda: defaultdict(float),
                    {
                        k: defaultdict(float, {kk: float(vv) for kk, vv in row.items()})
                        for k, row in (state.get("transitions", {}) or {}).items()
                    },
                )
                sess.regime_durations = defaultdict(
                    list,
                    {k: list(v) for k, v in (state.get("durations", {}) or {}).items()},
                )
                sess.last_decay_ts = float(state.get("last_decay_ts", time.time()))
                self._sessions[session] = sess

            self._symbol_state = {
                symbol: _SymbolState(
                    session=str(state.get("session", "legacy")),
                    regime=str(state.get("regime", "chaos")),
                    start_ts=float(state.get("start_ts", time.time())),
                )
                for symbol, state in symbols.items()
            }

            total = sum(
                sum(sum(row.values()) for row in state.transitions.values())
                for state in self._sessions.values()
            )
            log.info("RegimeTransitionMatrix loaded: %d transitions tracked across %d sessions",
                     int(total), len(self._sessions))
        except Exception as exc:
            log.warning("RegimeTransition load failed: %s", exc)

    def _session_state(self, session: str) -> _SessionState:
        state = self._sessions.get(session)
        if state is None:
            state = _SessionState(last_decay_ts=time.time())
            self._sessions[session] = state
        return state

    def _apply_decay(self, session: str, now: float | None = None) -> None:
        state = self._sessions.get(session)
        if state is None:
            return
        now_ts = now or time.time()
        elapsed = now_ts - float(state.last_decay_ts or now_ts)
        if elapsed <= 0:
            return
        factor = 0.5 ** (elapsed / self._decay_half_life_s)
        if factor >= 0.999:
            state.last_decay_ts = now_ts
            return
        for from_regime, row in list(state.transitions.items()):
            for to_regime in list(row.keys()):
                row[to_regime] *= factor
                if row[to_regime] < 1e-6:
                    del row[to_regime]
            if not row:
                del state.transitions[from_regime]
        state.last_decay_ts = now_ts
