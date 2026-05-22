"""
Adaptive Mini-Quant Brain — top-level adaptive AI orchestrator.

This module combines five institutional-grade adaptive components into a
single decision engine that approximates how a real quant fund operates:

  1. Thompson Sampling Bandit  — adaptive sleeve allocation per market context
  2. Online Bayesian Logistic  — multi-feature P(win) prediction
  3. Regime Transition Matrix  — forward-looking regime stability
  4. Alpha Decay Tracker        — per-pair edge monitoring
  5. Volatility Targeter        — portfolio-level vol control

Each component learns from every closed trade and feeds back into the
sizing decision. The brain DOES NOT replace the existing FundManager,
StrategyRouter, or RiskManager — it augments them with an adaptive layer.

Architecture:
  Existing pipeline: regime → SM → EV → score → sleeve → size
  Brain layer:       brain.decide(breakdown) → adaptive_size_mult, p_win, ...
  Combined:          final_size = base × brain_size_mult × vol_target_mult

State persistence: each component has its own JSON state under models/.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.models.thompson_bandit import ThompsonBandit
from src.models.online_logistic import OnlineLogistic, FeatureVector as MLFeatures
from src.models.regime_transition import RegimeTransitionMatrix
from src.models.alpha_decay import AlphaDecayTracker
from src.risk.vol_targeting import VolTargeter
from src.session_clock import active_market_session_key

log = logging.getLogger(__name__)


@dataclass
class BrainDecision:
    """Adaptive brain output — feeds into sizing pipeline."""
    p_win: float                    # ML-predicted probability
    p_win_confidence: float         # 0..1 confidence in prediction
    sleeve_choice: str              # bandit's recommended sleeve
    sleeve_score: float             # bandit's posterior expected WR
    sleeve_size_mult: float         # bandit-derived multiplier
    regime_stability: float         # P(current regime persists)
    regime_due_shift: bool          # True if current regime overdue
    pair_edge_score: float          # alpha decay score 0..1
    pair_alive: bool                # is pair still showing edge?
    pair_size_mult: float           # alpha decay multiplier
    vol_target_mult: float          # portfolio vol scaling
    overall_size_mult: float        # FINAL combined multiplier
    notes: list[str] = field(default_factory=list)

    @property
    def should_block(self) -> bool:
        """Brain's verdict to block trade entirely."""
        return not self.pair_alive

    def telegram_summary(self) -> str:
        bits = []
        bits.append(f"P(win)={self.p_win:.0%}±{(1 - self.p_win_confidence):.0%}")
        bits.append(f"sleeve={self.sleeve_choice}({self.sleeve_score:.0%})")
        bits.append(f"regime={'stable' if self.regime_stability > 0.6 else 'flux'}")
        if self.regime_due_shift:
            bits.append("⚠️shift-due")
        if not self.pair_alive:
            bits.append("💀dead-pair")
        elif self.pair_edge_score > 0.6:
            bits.append("🔥hot-pair")
        bits.append(f"size_mult={self.overall_size_mult:.2f}x")
        return " | ".join(bits)


class AdaptiveBrain:
    """
    Mini-quant adaptive intelligence.

    Public API:
      decide(breakdown) → BrainDecision  (called before opening a trade)
      learn(breakdown, won, pnl_pct)     (called when trade closes)
      record_daily_return(pnl_pct)       (called once per day)
      status_report()                    (for Telegram dashboards)
    """

    SLEEVES = ["trend_following", "reversal", "compression_breakout", "neutral"]

    def __init__(self, cfg: dict):
        self._cfg = cfg
        # Pull config or use defaults
        brain_cfg = cfg.get("adaptive_brain", {}) or {}
        self._enabled = bool(brain_cfg.get("enabled", True))
        self._size_blend = float(brain_cfg.get("size_blend_weight", 1.0))
        self._ml_confidence_floor = float(brain_cfg.get("ml_confidence_floor", 0.55))
        self._regime_stability_floor = float(brain_cfg.get("regime_stability_floor", 0.55))

        self._bandit = ThompsonBandit(arms=self.SLEEVES)
        self._logistic = OnlineLogistic(
            learning_rate=float(brain_cfg.get("ml_learning_rate", 0.05)),
            l2=float(brain_cfg.get("ml_l2", 0.01)),
        )
        self._transitions = RegimeTransitionMatrix()
        self._alpha_decay = AlphaDecayTracker(
            decay_threshold=float(brain_cfg.get("alpha_decay_threshold", -0.15)),
            min_obs_to_kill=int(brain_cfg.get("alpha_decay_min_obs", 8)),
            revival_after_s=float(brain_cfg.get("alpha_decay_revival_s", 86400)),
        )
        self._vol_targeter = VolTargeter(
            target_annual_vol=float(brain_cfg.get("vol_target_annual", 0.40)),
            lookback=int(brain_cfg.get("vol_lookback_days", 30)),
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def decide(self, breakdown) -> BrainDecision:
        """
        Run full adaptive inference for a candidate trade.

        Returns a BrainDecision that the caller can multiply into the
        existing FundManager output. The brain never DICTATES — it advises.
        """
        if not self._enabled:
            return self._neutral_decision()

        sm = breakdown.smart_money
        regime = breakdown.regime.value if breakdown.regime else "chaos"
        symbol = breakdown.symbol
        direction = breakdown.direction
        sleeve = getattr(breakdown, "strategy_sleeve", "neutral") or "neutral"
        session = getattr(breakdown, "session", "") or active_market_session_key()

        # ── 1. ML P(win) prediction ──────────────────────────────────────
        ml_feats = MLFeatures(
            score=breakdown.total_score,
            structure_quality=breakdown.structure_quality,
            trend_strength=breakdown.trend_strength,
            volume_confirmation=breakdown.volume_confirmation,
            funding_sentiment=breakdown.funding_sentiment,
            open_interest=breakdown.open_interest,
            volatility=breakdown.volatility,
            momentum_strength=(
                getattr(breakdown.feature_vector, "momentum_strength", 0.0)
                if breakdown.feature_vector else 0.0
            ),
            direction_long=1.0 if direction == "long" else 0.0,
        )
        p_win = self._logistic.predict(ml_feats)
        ml_confidence = self._logistic.confidence()

        # ── 2. Bandit sleeve recommendation per (regime|direction) ───────
        context = f"{regime}|{direction}"
        recommended_sleeve, sleeve_score = self._bandit.best_arm(context)
        sleeve_size_mult = self._bandit.size_mult(sleeve, context)

        # ── 3. Regime transition awareness ───────────────────────────────
        self._transitions.observe(symbol, regime, session=session)
        stability = self._transitions.stability_score(regime, session=session)
        due_shift = self._transitions.is_due_for_shift(symbol, regime, session=session)

        # ── 4. Alpha decay per pair ──────────────────────────────────────
        pair_alive, edge_score = self._alpha_decay.is_alive(symbol)
        pair_size_mult = self._alpha_decay.get_size_mult(symbol)

        # ── 5. Portfolio vol-targeting ───────────────────────────────────
        vol_mult = self._vol_targeter.size_multiplier()

        # ── Combined size multiplier ─────────────────────────────────────
        notes = []
        size_mult = sleeve_size_mult

        # ML-based adjustment (only when we have confidence)
        if ml_confidence >= self._ml_confidence_floor:
            ml_signal = (p_win - 0.5) * 2.0  # -1..+1
            adjustment = 1.0 + ml_signal * 0.30 * ml_confidence
            size_mult *= adjustment
            notes.append(f"ML p_win={p_win:.0%} conf={ml_confidence:.0%}")
        elif p_win > 0.5:
            notes.append(f"ML confidence low ({ml_confidence:.0%})")

        # Regime stability adjustment
        if due_shift:
            size_mult *= 0.85
            notes.append("regime due shift")
        elif stability < self._regime_stability_floor:
            size_mult *= 0.95
            notes.append(f"regime confidence low ({stability:.0%})")
        elif stability >= 0.75:
            size_mult *= 1.05
            notes.append("regime stable")

        # Alpha decay adjustment
        size_mult *= pair_size_mult
        if not pair_alive:
            notes.append(f"💀 {symbol} edge decayed")
        elif edge_score >= 0.65:
            notes.append(f"🔥 {symbol} hot pair")

        # Apply portfolio vol-targeting
        size_mult *= vol_mult
        if vol_mult < 0.95:
            notes.append(f"vol-elevated ({vol_mult:.2f}x)")
        elif vol_mult > 1.05:
            notes.append(f"vol-calm ({vol_mult:.2f}x)")

        # Bandit context bonus: if recommended sleeve == current sleeve, small boost
        if recommended_sleeve == sleeve and sleeve != "neutral":
            size_mult *= 1.05
            notes.append(f"bandit confirms {sleeve}")

        # Final bounds
        size_mult = max(0.40, min(2.50, size_mult))

        return BrainDecision(
            p_win=round(p_win, 4),
            p_win_confidence=round(ml_confidence, 3),
            sleeve_choice=recommended_sleeve,
            sleeve_score=round(sleeve_score, 3),
            sleeve_size_mult=round(sleeve_size_mult, 3),
            regime_stability=round(stability, 3),
            regime_due_shift=due_shift,
            pair_edge_score=round(edge_score, 3),
            pair_alive=pair_alive,
            pair_size_mult=round(pair_size_mult, 3),
            vol_target_mult=round(vol_mult, 3),
            overall_size_mult=round(size_mult, 3),
            notes=notes,
        )

    def learn(self, breakdown, won: bool, pnl_pct: float) -> None:
        """Train all components on closed trade outcome."""
        if not self._enabled:
            return

        symbol = breakdown.symbol
        direction = breakdown.direction
        regime = breakdown.regime.value if breakdown.regime else "chaos"
        sleeve = getattr(breakdown, "strategy_sleeve", "neutral") or "neutral"
        context = f"{regime}|{direction}"

        # 1. Bandit update
        if sleeve in self.SLEEVES:
            self._bandit.update(sleeve, won, context)

        # 2. ML update
        ml_feats = MLFeatures(
            score=breakdown.total_score,
            structure_quality=breakdown.structure_quality,
            trend_strength=breakdown.trend_strength,
            volume_confirmation=breakdown.volume_confirmation,
            funding_sentiment=breakdown.funding_sentiment,
            open_interest=breakdown.open_interest,
            volatility=breakdown.volatility,
            momentum_strength=(
                getattr(breakdown.feature_vector, "momentum_strength", 0.0)
                if breakdown.feature_vector else 0.0
            ),
            direction_long=1.0 if direction == "long" else 0.0,
        )
        loss = self._logistic.update(ml_feats, won)

        # 3. Alpha decay update
        self._alpha_decay.record(symbol, pnl_pct)

        log.info(
            "🧠 Brain learned: %s %s pnl=%.2f%% won=%s | ctx=%s sleeve=%s | ml_loss=%.3f",
            symbol, direction, pnl_pct, won, context, sleeve, loss,
        )

    def record_daily_return(self, daily_pnl_pct: float) -> None:
        """Record portfolio daily return for vol targeter."""
        if self._enabled:
            self._vol_targeter.record_return(daily_pnl_pct)

    def status_report(self) -> dict:
        """Comprehensive brain status for telemetry / Telegram."""
        if not self._enabled:
            return {"enabled": False}

        bandit_top = self._bandit.top_contexts(n=5)
        return {
            "enabled": True,
            "ml": {
                "n_updates": self._logistic._n_updates,
                "recent_loss": round(self._logistic.recent_loss(), 4),
                "confidence": round(self._logistic.confidence(), 3),
                "is_drifting": self._logistic.is_drifting(),
                "feature_directions": self._logistic.feature_directions(),
            },
            "bandit": {
                "top_contexts": [
                    {"context": ctx, "wr": round(wr, 3), "n": n}
                    for ctx, wr, n in bandit_top
                ],
                "n_buckets": len(self._bandit._counts),
            },
            "regime_transitions": self._transitions.report(),
            "alpha_decay": self._alpha_decay.summary(),
            "vol_targeter": self._vol_targeter.report(),
        }

    def _neutral_decision(self) -> BrainDecision:
        return BrainDecision(
            p_win=0.5,
            p_win_confidence=0.0,
            sleeve_choice="neutral",
            sleeve_score=0.5,
            sleeve_size_mult=1.0,
            regime_stability=0.5,
            regime_due_shift=False,
            pair_edge_score=0.5,
            pair_alive=True,
            pair_size_mult=1.0,
            vol_target_mult=1.0,
            overall_size_mult=1.0,
            notes=["brain disabled"],
        )
