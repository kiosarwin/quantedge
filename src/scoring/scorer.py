"""
Institutional scoring engine.

Pipeline per symbol:
  1. Feature vector F(t) built from all available signals.
  2. Four-state regime classified — chaos blocks the trade.
  3. Smart money phase detected — must have directional bias.
  4. EV model gate — EV must be positive after fees.
  5. Weighted signal score (unchanged from original for continuity).
  6. Cross-sectional z-score normalization (professional upgrade).
  7. Adaptive threshold based on regime and market context.

Final score is regime-gated and smart-money-gated.
"""
from __future__ import annotations

import collections
import logging
import math
import statistics
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING

from src.session_clock import active_market_session_key

import pandas as pd

from src.analysis.indicators import (
    trend_strength_score,
    volatility_score,
    volume_ratio,
    buy_volume_ratio,
    ema_value,
)
from src.analysis.structure import (
    structure_quality_score,
    trade_direction_from_structure,
)
from src.analysis.order_book import order_book_score
from src.analysis.sentiment import funding_sentiment_score, open_interest_score
from src.analysis.smart_money import detect_smart_money, SmartMoneySignal, SmartMoneyPhase
from src.analysis.short_strategies import detect_short_entry, ShortEntrySignal, ShortStrategy
from src.analysis.mtf_price_action import (
    detect_mtf_price_action_continuation,
    MTFPriceActionSignal,
)
from src.analysis.vwap_pullback import (
    detect_vwap_pullback_continuation,
    VWAPPullbackSignal,
)
from src.analysis.liquidity_sweep_reversal import (
    detect_liquidity_sweep_reversal,
    LiquiditySweepReversalSignal,
)
from src.analysis.feature_engine import build_feature_vector, FeatureVector
from src.analysis.spot_context import SpotContext, compute_spot_mult
from src.models.ev_model import EVResult
from src.models.pwin_engine import PwinContext
from src.models.edge_detector import EdgeContext, EdgeResult
from src.models.regime_classifier import classify_four_state, RegimeState
from src.models.sector_rotation import SectorRotation, sector_rotation_score_mult
from src.models.strategy_router import StrategyRouter, DispersionState
from src.models.strategy_passport import asset_from_symbol, sector_for_asset
from src.models.market_context import MarketContext, market_context_score_mult
from src.models.time_series import TimeSeriesDiagnostics, time_series_directional_mult
from src.data.market_data import MarketSnapshot

if TYPE_CHECKING:
    from src.learning.learner import TradeRecord

log = logging.getLogger(__name__)


@dataclass
class SignalBreakdown:
    symbol: str
    direction: str              # 'long' | 'short' | 'none'
    total_score: float
    base_score: float

    # Individual signal scores
    trend_strength: float
    volume_confirmation: float
    structure_quality: float
    open_interest: float
    funding_sentiment: float
    order_book: float
    volatility: float
    legacy_score: float = 0.0

    # Institutional-grade overlays
    regime: RegimeState = RegimeState.HIGH_VOLATILITY_CHAOS
    smart_money: SmartMoneySignal | None = None
    ev_result: EVResult | None = None
    edge_result: EdgeResult | None = None
    feature_vector: FeatureVector | None = None
    spot_context: SpotContext | None = None
    market_context: MarketContext | None = None
    sector_rotation: SectorRotation | None = None
    time_series: TimeSeriesDiagnostics | None = None
    short_setup: ShortEntrySignal | None = None
    mtf_price_action: MTFPriceActionSignal | None = None
    vwap_pullback: VWAPPullbackSignal | None = None
    liquidity_sweep_reversal: LiquiditySweepReversalSignal | None = None

    weights_used: dict = field(default_factory=dict)
    strategy_sleeve: str = "neutral"
    strategy_reason: str = ""
    strategy_score_mult: float = 1.0
    strategy_size_mult: float = 1.0
    strategy_threshold_shift: float = 0.0
    strategy_ranking_bonus: float = 0.0
    probability_score_mult: float = 1.0
    time_series_score_mult: float = 1.0
    time_series_size_mult: float = 1.0
    time_series_reason: str = ""
    short_macro_state: str = "neutral"
    short_macro_score: float = 0.0
    short_macro_reason: str = ""
    short_macro_score_mult: float = 1.0
    short_macro_size_mult: float = 1.0
    short_macro_threshold_shift: float = 0.0
    pre_gate_score: float = 0.0
    setup_type: str = "none"
    setup_quality_score: float = 0.0
    setup_passport: object | None = None
    dispersion_value: float = 0.0
    dispersion_state: str = "normal"
    cohort_key: str = ""
    lifecycle_status: str = "RESEARCH"
    recommendation: str = "PAPER_ONLY"

    # Gate flags (all must pass to be tradeable)
    regime_ok: bool = False
    smart_money_ok: bool = False
    ev_ok: bool = False

    @property
    def all_gates_passed(self) -> bool:
        return self.regime_ok and self.smart_money_ok and self.ev_ok and self.total_score > 0

    def trade_card(self) -> str:
        """Returns the full institutional trade card string."""
        sm = self.smart_money
        ev = self.ev_result
        fv = self.feature_vector

        sm_phase = sm.phase.value if sm else "N/A"
        sm_score = f"{sm.score:.1f}" if sm else "N/A"
        sm_reasoning = sm.reasoning if sm else "N/A"

        ev_str = (
            f"P(win)={ev.p_win:.1%}  EV_net={ev.ev_net_pct:+.3f}%  "
            f"confidence={ev.confidence:.1%}"
        ) if ev else "N/A (insufficient history)"

        regime_label = self.regime.label if self.regime else "N/A"

        gates = (
            f"regime={'✓' if self.regime_ok else '✗'}  "
            f"sm={'✓' if self.smart_money_ok else '✗'}  "
            f"expectancy={'✓' if self.ev_ok else '✗'}"
        )

        edge = self.edge_result
        edge_str = (
            f"{edge.status}  score={edge.score:.0f}/100  conf={edge.confidence:.2f}  "
            f"action={edge.action}  size={edge.size_mult:.2f}x  cov={edge.regime_coverage}"
        ) if edge else "N/A"

        lines = [
            f"  Symbol:            {self.symbol}",
            f"  Direction:         {self.direction.upper()}",
            f"  Score:             {self.total_score:.1f} / 100",
            f"  Market Regime:     {regime_label}",
            f"  Smart Money Phase: {sm_phase} (score={sm_score})",
            f"  Smart Money:       {sm_reasoning}",
            f"  Expectancy Gate:   {ev_str}",
            f"  Edge Detector:     {edge_str}",
            f"  Gates:             {gates}",
        ]
        if fv:
            lines += [
                f"  Momentum:          {fv.momentum_strength:+.1f}",
                f"  Vol Regime:        {fv.volatility_regime:.1f}",
                f"  OI Change:         {fv.oi_change_rate:+.2f}%",
                f"  Order Flow:        {fv.order_flow_imbalance:+.1f}",
                f"  Liq Pressure:      {fv.liquidation_pressure:.1f}",
                f"  Structure:         {fv.market_structure}",
            ]
        if self.spot_context:
            sc = self.spot_context
            lines += [
                f"  Spot Basis:        {sc.basis_pct:+.3f}%",
                f"  Spot Vol Ratio:    {sc.spot_volume_ratio:.2f}x",
            ]
            if sc.coinbase_premium_pct != 0.0:
                lines.append(f"  Coinbase Premium:  {sc.coinbase_premium_pct:+.3f}%")
        if self.market_context:
            mc = self.market_context
            lines.append(
                f"  Market Context:    {mc.risk_on_state} / {mc.rotation_state} (conf={mc.confidence:.2f})"
            )
        if self.time_series:
            ts = self.time_series
            lines.append(
                f"  Time Series:       {ts.state} ewma_vol={ts.ewma_vol_pct:.3f}% shock_z={ts.latest_abs_return_z:.2f}"
            )
        return "\n".join(lines)


class Scorer:
    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._weights: dict[str, float] = dict(cfg["scoring"]["weights"])
        self._trade_log: list[TradeRecord] = []
        trading_cfg = cfg.get("trading", {})
        self._paper_mode = trading_cfg.get("mode") == "paper"
        self._ev_bootstrap_enabled = bool(
            self._paper_mode and trading_cfg.get("exploration_mode", False)
        )

        # Lazy-import to avoid circular dependency
        from src.models.ev_model import EVModel
        from src.models.edge_detector import EdgeDetector
        self._ev_model = EVModel(cfg)
        self._edge_detector = EdgeDetector(cfg)
        self._strategy_router = StrategyRouter(cfg)
        score_cfg = cfg.get("scoring", {})
        self._score_threshold = float(
            trading_cfg.get("min_score_threshold", score_cfg.get("min_score_to_trade", 65))
        )

        # Professional-grade: cross-sectional score normalization
        # Tracks recent score distribution for z-score computation
        self._score_history: collections.deque[float] = collections.deque(maxlen=200)
        self._regime_score_history: dict[str, collections.deque[float]] = {
            "trending_expansion": collections.deque(maxlen=100),
            "accumulation_compression": collections.deque(maxlen=100),
            "distribution": collections.deque(maxlen=100),
            "chaos": collections.deque(maxlen=100),
        }

    def _paper_soft_ev_allowed(self, ev_result: EVResult | None) -> bool:
        if ev_result is None or not self._paper_mode:
            return False
        paper_validation = self._cfg.get("paper_validation", {})
        if not bool(paper_validation.get("enabled", True)):
            return False
        hard_gate_min_trades = int(
            paper_validation.get(
                "ev_hard_gate_min_trades",
                self._cfg.get("ev_model", {}).get("min_trades_for_ev", 20),
            )
        )
        if ev_result.trade_count >= hard_gate_min_trades:
            return False
        # The scorer has no strategy/session context. Keep generic paper EV
        # relaxation non-negative here; contextual negative-EV exceptions live
        # in NinjaTrader._paper_ev_relax_mode() high-conviction allowlists.
        return float(ev_result.ev_net_pct or 0.0) >= 0.0

    def _ev_gate_allows(self, ev_result: EVResult | None) -> bool:
        ev_cfg = self._cfg.get("ev_model", {})
        if not bool(ev_cfg.get("gate_enabled", True)):
            return True
        if ev_result is None:
            return False

        min_trades = int(ev_cfg.get("min_trades_for_ev", 20) or 20)
        statistical_gate = bool(ev_cfg.get("statistical_gate_enabled", False))
        if statistical_gate:
            if ev_result.trade_count < min_trades:
                return self._ev_bootstrap_enabled
            return ev_result.is_tradeable

        if ev_result.trade_count < min_trades:
            return (
                self._ev_bootstrap_enabled
                or ev_result.ev_net_pct > 0
                or self._paper_soft_ev_allowed(ev_result)
            )
        return ev_result.is_tradeable or self._paper_soft_ev_allowed(ev_result)

    @staticmethod
    def _bounded(value: float, lo: float, hi: float) -> float:
        return min(hi, max(lo, float(value or 0.0)))

    @staticmethod
    def _clip_probability(value: float, default: float = 0.5) -> float:
        try:
            p = float(value)
        except (TypeError, ValueError):
            p = default
        if not math.isfinite(p):
            p = default
        return min(0.99, max(0.01, p))

    def _probabilistic_score_multiplier(self, ev_result: EVResult | None) -> float:
        if ev_result is None:
            return 1.0
        p_win = self._clip_probability(getattr(ev_result, "p_win", 0.5))
        confidence = self._bounded(float(getattr(ev_result, "confidence", 0.0) or 0.0), 0.0, 1.0)
        conservative_p = float(getattr(ev_result, "conservative_p_win", 0.0) or 0.0)
        posterior_p = p_win
        if conservative_p > 0.0:
            posterior_p = (
                p_win * (1.0 - 0.35 * confidence)
                + self._clip_probability(conservative_p) * (0.35 * confidence)
            )

        ev_net = self._bounded(float(getattr(ev_result, "ev_net_pct", 0.0) or 0.0), -3.0, 3.0)
        conservative_ev = self._bounded(
            float(getattr(ev_result, "conservative_ev_net_pct", ev_net) or 0.0),
            -3.0,
            3.0,
        )
        probability_term = (posterior_p - 0.50) * 1.50   # Increased from 1.20
        ev_term = math.tanh(ev_net / 1.0) * 0.25         # Increased from 0.18
        conservative_term = math.tanh(conservative_ev / 1.0) * 0.15 * confidence # Increased from 0.10
        return round(self._bounded(1.0 + probability_term + ev_term + conservative_term, 0.40, 1.25), 4) # Expanded range

    @property
    def edge_detector(self):
        return self._edge_detector

    def update_weights(self, weights: dict[str, float]) -> None:
        self._weights = {k: float(v) for k, v in weights.items()}
        log.debug("Scorer weights updated: %s", self._weights)

    def set_trade_log(self, trade_log: list) -> None:
        """Feed the learner's trade log into the EV model."""
        self._trade_log = trade_log

    @staticmethod
    def _clamp_score(value: float) -> float:
        return round(min(100.0, max(0.0, value)), 2)

    # ------------------------------------------------------------------ #
    #  Professional: cross-sectional normalization & adaptive thresholds   #
    # ------------------------------------------------------------------ #

    def _record_score(self, score: float, regime: str = "") -> None:
        """Record score for rolling distribution tracking."""
        if not hasattr(self, "_score_history"):
            self._score_history = collections.deque(maxlen=200)
        if not hasattr(self, "_regime_score_history"):
            self._regime_score_history = {}
        self._score_history.append(score)
        if regime in self._regime_score_history:
            self._regime_score_history[regime].append(score)

    def score_z_score(self, raw_score: float) -> float:
        """Compute z-score of raw_score relative to recent score distribution.

        This is how institutional quant desks normalize signals cross-sectionally:
        a score of 65 means nothing in isolation — it only matters relative to
        the current distribution of all scores.

        Returns z-score in range [-3, +3].
        """
        history = getattr(self, "_score_history", None)
        if history is None or len(history) < 10:
            return 0.0  # not enough history
        scores = list(history)
        mean = statistics.mean(scores)
        stdev = statistics.stdev(scores) if len(scores) >= 2 else 1.0
        if stdev < 0.01:
            return 0.0
        z = (raw_score - mean) / stdev
        return max(-3.0, min(3.0, z))

    def adaptive_threshold(self, regime: str = "", market_ctx: object | None = None) -> float:
        """Compute adaptive score threshold based on regime and market conditions.

        Professional behavior:
        - Trending expansion: lower threshold (more opportunities)
        - Distribution/chaos: higher threshold (be selective)
        - High-confidence market context: slight relaxation
        """
        base = self._score_threshold

        # Regime adjustment
        regime_shifts = {
            "trending_expansion": -3.0,       # more lenient in trends
            "accumulation_compression": 0.0,  # standard
            "distribution": 5.0,              # more selective
            "chaos": 10.0,                    # very selective
        }
        regime_shift = regime_shifts.get(regime, 0.0)

        # Market context adjustment
        ctx_shift = 0.0
        if market_ctx is not None:
            confidence = float(getattr(market_ctx, "confidence", 0.0) or 0.0)
            if confidence > 0.7:
                ctx_shift -= 1.5  # high-confidence context relaxes threshold
            elif confidence < 0.3:
                ctx_shift += 2.0  # low-confidence context tightens threshold

        return max(30.0, min(80.0, base + regime_shift + ctx_shift))

    def cross_sectional_rank_bonus(self, z: float) -> float:
        """Convert z-score to a score bonus/penalty.

        Top-decile signals (z > 1.5) get a bonus, bottom-decile (z < -1.0) get penalized.
        This ensures the best relative signals get priority even if absolute scores are similar.
        """
        if z > 2.0:
            return 4.0   # top ~2.5%
        if z > 1.5:
            return 2.5   # top ~7%
        if z > 1.0:
            return 1.0   # top ~16%
        if z < -1.5:
            return -3.0  # bottom ~7%
        if z < -1.0:
            return -1.5  # bottom ~16%
        return 0.0

    def _directional_momentum_score(self, breakdown: SignalBreakdown) -> float:
        fv = breakdown.feature_vector
        if fv is None:
            return 50.0
        raw = float(getattr(fv, "momentum_strength", 0.0) or 0.0)
        signed = raw if breakdown.direction == "long" else -raw
        return self._clamp_score(50.0 + signed * 0.60)

    def _directional_smart_money_score(self, breakdown: SignalBreakdown) -> float:
        sm = breakdown.smart_money
        if sm is None:
            return 50.0
        score = float(sm.score)
        if sm.direction_bias == breakdown.direction:
            return self._clamp_score(55.0 + (score - 50.0) * 0.90)
        if sm.direction_bias == "neutral":
            return self._clamp_score(50.0 + (score - 50.0) * 0.25)
        return self._clamp_score(45.0 - (score - 50.0) * 0.90)

    def _liquidation_pressure_score(self, breakdown: SignalBreakdown) -> float:
        fv = breakdown.feature_vector
        if fv is None:
            return 50.0
        pressure = float(getattr(fv, "liquidation_pressure", 0.0) or 0.0)
        signed = pressure if breakdown.direction == "long" else -pressure
        return self._clamp_score(50.0 + signed * 0.70)

    def _trend_base_score(self, breakdown: SignalBreakdown) -> float:
        """
        Trend-following base score. BOTH long and short are first-class.
        Ref: Jegadeesh & Titman (1993) — momentum persistence applies to both
        directions; crypto exhibits strong short-horizon momentum (Dobrynskaya 2023).
        """
        momentum = self._directional_momentum_score(breakdown)
        sm_score = self._directional_smart_money_score(breakdown)
        base = (
            breakdown.trend_strength * 0.26
            + breakdown.structure_quality * 0.22
            + breakdown.volume_confirmation * 0.14
            + breakdown.open_interest * 0.10
            + breakdown.order_book * 0.10
            + breakdown.volatility * 0.08
            + momentum * 0.06
            + sm_score * 0.04
        )
        if breakdown.regime.value == "trending_expansion":
            base += 6.0
        else:
            base -= 10.0
        sm = breakdown.smart_money
        if sm is not None and sm.phase.value in {"trending", "accumulation"}:
            base += 4.0
        # NO direction penalty — shorts in trending regime are equally valid
        # when structure confirms bearish BOS (per momentum literature)
        return self._clamp_score(base)

    def _reversal_base_score(self, breakdown: SignalBreakdown) -> float:
        sm_score = self._directional_smart_money_score(breakdown)
        liq_score = self._liquidation_pressure_score(breakdown)
        momentum = self._directional_momentum_score(breakdown)
        base = (
            breakdown.structure_quality * 0.24
            + sm_score * 0.22
            + breakdown.volume_confirmation * 0.14
            + breakdown.order_book * 0.12
            + breakdown.volatility * 0.10
            + breakdown.open_interest * 0.08
            + liq_score * 0.06
            + momentum * 0.04
        )
        sm = breakdown.smart_money
        if sm is not None and sm.phase.value == "liquidity_sweep":
            base += 7.0
        if breakdown.regime.value == "distribution" and breakdown.direction == "short":
            base += 6.0
        elif breakdown.regime.value == "trending_expansion" and breakdown.direction == "long":
            base += 3.0
        return self._clamp_score(base)

    def _compression_base_score(self, breakdown: SignalBreakdown) -> float:
        momentum = self._directional_momentum_score(breakdown)
        return self._clamp_score(
            breakdown.structure_quality * 0.30
            + breakdown.volume_confirmation * 0.18
            + breakdown.trend_strength * 0.16
            + breakdown.open_interest * 0.10
            + breakdown.order_book * 0.10
            + breakdown.volatility * 0.08
            + momentum * 0.08
        )

    def _neutral_base_score(self, breakdown: SignalBreakdown) -> float:
        sm_score = self._directional_smart_money_score(breakdown)
        return self._clamp_score(
            breakdown.structure_quality * 0.24
            + breakdown.volume_confirmation * 0.18
            + breakdown.order_book * 0.16
            + breakdown.volatility * 0.12
            + sm_score * 0.10
        )

    def _strategy_base_score(self, breakdown: SignalBreakdown, sleeve: str) -> float:
        if sleeve == "trend_following":
            return self._trend_base_score(breakdown)
        if sleeve == "reversal":
            return self._reversal_base_score(breakdown)
        if sleeve == "compression_breakout":
            return self._compression_base_score(breakdown)
        return self._neutral_base_score(breakdown)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def score(self, snapshot: MarketSnapshot) -> SignalBreakdown | None:
        tf_cfg = self._cfg["timeframes"]
        primary_tf = tf_cfg["primary"]
        higher_tf = tf_cfg["higher"]
        df_primary = snapshot.candles_for(primary_tf)
        df_higher = snapshot.candles_for(higher_tf)

        if df_primary.empty or len(df_primary) < 50:
            log.debug("Not enough candles for %s on %s", snapshot.symbol, primary_tf)
            return None

        short_setup: ShortEntrySignal | None = None
        try:
            setup = detect_short_entry(df_primary, self._cfg)
            if setup.is_valid:
                short_setup = setup
        except Exception as exc:
            log.debug("Short setup detection failed for %s: %s", snapshot.symbol, exc)

        direction = trade_direction_from_structure(df_primary, self._cfg)
        if direction == "none" and not df_higher.empty and len(df_higher) >= 50:
            direction = trade_direction_from_structure(df_higher, self._cfg)

        lower_tf = tf_cfg.get("lower") or tf_cfg.get("secondary") or tf_cfg.get("entry")
        df_lower = snapshot.candles_for(lower_tf) if lower_tf else pd.DataFrame()
        mtf_price_action: MTFPriceActionSignal | None = None
        try:
            mtf_signal = detect_mtf_price_action_continuation(
                df_primary,
                df_higher,
                df_lower,
                self._cfg,
                direction_hint=direction,
            )
            if mtf_signal.is_valid:
                mtf_price_action = mtf_signal
                direction = mtf_signal.direction
        except Exception as exc:
            log.debug("MTF price-action detection failed for %s: %s", snapshot.symbol, exc)

        vwap_pullback: VWAPPullbackSignal | None = None
        try:
            vwap_signal = detect_vwap_pullback_continuation(
                df_primary,
                df_higher,
                self._cfg,
                direction_hint=direction,
            )
            if vwap_signal.is_valid:
                vwap_pullback = vwap_signal
                direction = vwap_signal.direction
        except Exception as exc:
            log.debug("VWAP pullback detection failed for %s: %s", snapshot.symbol, exc)

        liquidity_sweep_reversal: LiquiditySweepReversalSignal | None = None
        try:
            sweep_signal = detect_liquidity_sweep_reversal(
                df_primary,
                self._cfg,
                direction_hint="none",
            )
            if sweep_signal.is_valid:
                liquidity_sweep_reversal = sweep_signal
                direction = sweep_signal.direction
        except Exception as exc:
            log.debug("Liquidity sweep reversal detection failed for %s: %s", snapshot.symbol, exc)

        if short_setup is not None and direction != "short":
            log.debug(
                "%s dedicated short setup %s overrides structure direction=%s",
                snapshot.symbol,
                short_setup.label,
                direction,
            )
            direction = "short"
        if direction == "none":
            return None

        # ── 1. Feature vector ─────────────────────────────────────────
        fv = None
        try:
            fv = build_feature_vector(
                symbol=snapshot.symbol,
                df_primary=df_primary,
                df_higher=df_higher,
                order_book=snapshot.order_book,
                oi_change_pct=snapshot.oi_change_pct,
                funding_rate=snapshot.funding_rate,
                taker_buy_ratio=snapshot.taker_buy_ratio,
                cfg=self._cfg,
            )
        except Exception as exc:
            log.debug("Feature vector failed for %s: %s", snapshot.symbol, exc)

        # ── 2. Four-state regime ──────────────────────────────────────
        regime = RegimeState.HIGH_VOLATILITY_CHAOS
        try:
            regime = classify_four_state(
                df_higher if not df_higher.empty else df_primary,
                snapshot.oi_change_pct,
                snapshot.funding_rate,
                self._cfg,
            )
        except Exception as exc:
            log.debug("Regime classification failed for %s: %s", snapshot.symbol, exc)

        regime_ok = regime.tradeable

        # ── 3. Smart money detection ──────────────────────────────────
        sm_signal = None
        smart_money_ok = False
        try:
            sm_signal = detect_smart_money(
                df=df_primary,
                oi_change_pct=snapshot.oi_change_pct,
                funding_rate=snapshot.funding_rate,
                ls_ratio=snapshot.ls_ratio,
                taker_buy_ratio=snapshot.taker_buy_ratio,
                cfg=self._cfg,
            )
            sm_threshold = self._cfg.get("smart_money", {}).get("min_smart_money_score", 55)
            # NEUTRAL = no institutional signal detected — allow through, don't block.
            # Only block on CHAOS or high-confidence contradiction (SM strongly opposes direction).
            if sm_signal.phase == SmartMoneyPhase.CHAOS:
                smart_money_ok = False
            elif sm_signal.phase == SmartMoneyPhase.NEUTRAL:
                smart_money_ok = True   # no info → don't block
            else:
                smart_money_ok = (
                    sm_signal.score >= sm_threshold
                    and sm_signal.aligns_with(direction)
                )
        except Exception as exc:
            log.debug("Smart money detection failed for %s: %s", snapshot.symbol, exc)

        # ── 4. Pre-compute features needed for setup-specific EV context ─
        # (ts, sq, vs are also re-used by the weighted-signal score below.)
        ts = vol_score = sq = oi = fs = ob = vs = 0.0
        try:
            ts = trend_strength_score(df_primary, self._cfg)
        except Exception:
            pass
        try:
            vol_r = volume_ratio(df_primary, self._cfg["indicators"]["volume_lookback"])
            buy_r = buy_volume_ratio(df_primary, self._cfg["indicators"]["volume_lookback"])
            raw_vol = min(100.0, vol_r / self._cfg["indicators"]["volume_spike_multiplier"] * 50)
            vol_score = (raw_vol * 0.5) + (buy_r * 100 * 0.5 if direction == "long" else (1 - buy_r) * 100 * 0.5)
        except Exception:
            pass
        try:
            sq = structure_quality_score(df_primary, self._cfg)
        except Exception:
            pass
        try:
            oi = open_interest_score(snapshot.oi_change_pct, self._cfg)
        except Exception:
            oi = 50.0
        try:
            fs = funding_sentiment_score(snapshot.funding_rate, self._cfg)
        except Exception:
            fs = 50.0
        try:
            ob = order_book_score(snapshot.order_book, self._cfg, direction)
        except Exception:
            ob = 50.0
        try:
            vs = volatility_score(df_primary, self._cfg)
        except Exception:
            vs = 50.0

        # Distribution is usually non-tradeable, except for configured short
        # reversal sleeves that explicitly require this regime.
        if not regime_ok and regime.value == "distribution":
            reversal_probe = SimpleNamespace(
                direction=direction,
                regime=regime,
                smart_money=sm_signal,
                structure_quality=sq,
                volume_confirmation=vol_score,
                volatility=vs,
            )
            if (
                self._strategy_router.is_short_reversal_candidate(reversal_probe)
                or self._strategy_router.is_long_reversal_candidate(reversal_probe)
            ):
                regime_ok = True

        # ── Phase D / Liq Sweep dedicated short setup detector ─────────
        # These are the proven post-distribution short edges from
        # `quantedge` (cohort: IMMINENT_DUMP + PHASE_D / LIQ_SWEEP).
        # Detection runs only for short candidates and is purely informational
        # here — the StrategyRouter consumes `breakdown.short_setup` to decide
        # whether to admit the trade via the reversal sleeve.
        if short_setup is not None:
            # Phase D / Liq Sweep encode the post-distribution breakdown thesis
            # themselves; if the regime classifier tagged this candle as
            # `distribution` and the dedicated detector fires, the regime gate
            # is no longer the right blocker — the setup-specific structure is.
            if not regime_ok and regime.value == "distribution":
                regime_ok = True

        # Smart-money alignment is normally enforced via OI / funding / sweeps,
        # but Phase D and Liq Sweep have their own price-structure thesis. When
        # a high-confidence dedicated short_setup fires and the SM detector
        # only returned NEUTRAL (no info — not a contradiction), unblock the
        # SM gate so the dedicated edge can express itself.
        if (
            short_setup is not None
            and not smart_money_ok
            and sm_signal is not None
            and sm_signal.phase == SmartMoneyPhase.NEUTRAL
        ):
            smart_money_ok = True

        # ── 5. EV model gate (setup-specific p_win via PwinEngine) ─────
        ev_result = None
        ev_ok = False
        try:
            sm_aligned = bool(sm_signal and sm_signal.aligns_with(direction))
            sm_phase_val = sm_signal.phase.value if sm_signal else "neutral"
            asset = asset_from_symbol(snapshot.symbol)
            sector = sector_for_asset(asset)
            pwin_ctx = PwinContext(
                regime=regime.value,
                sm_phase=sm_phase_val,
                sm_aligned=sm_aligned,
                direction=direction,
                sector=sector,
                structure_quality=float(sq),
                volatility_score=float(vs),
                trend_strength=float(ts),
                btc_state=getattr(snapshot, "btc_state", "unknown") or "unknown",
                btcd_state=getattr(snapshot, "btcd_state", "unknown") or "unknown",
            )
            ev_result = self._ev_model.compute(
                trade_log=self._trade_log,
                funding_rate=snapshot.funding_rate,
                pwin_ctx=pwin_ctx,
            )
            ev_ok = self._ev_gate_allows(ev_result)
        except Exception as exc:
            log.debug("EV model failed for %s: %s", snapshot.symbol, exc)
            ev_ok = True

        # HTF alignment bonus: 1H and 4H trend in the same direction (+up to 8pts)
        htf_bonus = 0.0
        if not df_higher.empty and len(df_higher) >= 50:
            try:
                ts_higher = trend_strength_score(df_higher, self._cfg)
                if (ts > 60 and ts_higher > 60) or (ts < 40 and ts_higher < 40):
                    htf_bonus = min(8.0, (abs(ts - 50) + abs(ts_higher - 50)) / 2 / 50 * 8)
            except Exception:
                pass

        w = self._weights
        total_weight = sum(w.values()) or 1.0
        raw_score = (
            ts       * w.get("trend_strength", 20) +
            vol_score* w.get("volume_confirmation", 15) +
            sq       * w.get("structure_quality", 20) +
            oi       * w.get("open_interest", 15) +
            fs       * w.get("funding_sentiment", 10) +
            ob       * w.get("order_book", 10) +
            vs       * w.get("volatility", 10)
        ) / total_weight

        raw_score += htf_bonus
        pre_gate_score = round(min(100.0, max(0.0, raw_score)), 2)

        # Gate penalties: reduce score if institutional filters fail
        if not regime_ok:
            raw_score *= 0.35   # chaos regime = heavy penalty but not zero (was 0.0)
        elif not smart_money_ok:
            raw_score *= 0.6    # heavy penalty; may still pass threshold
        elif not ev_ok:
            raw_score *= 0.7    # EV negative = penalty
        elif sm_signal and sm_signal.phase == SmartMoneyPhase.NEUTRAL:
            raw_score *= 0.95   # soft penalty: no institutional confirmation

        # Smart money score bonus (up to +10 pts) when directional SM aligns
        if sm_signal and sm_signal.phase.has_directional_bias and sm_signal.aligns_with(direction):
            raw_score += (sm_signal.score - 50) / 50 * 10

        # ── Fix #5: 4H trend direction confirmation ───────────────────
        if not df_higher.empty and len(df_higher) >= 55:
            try:
                ema21_4h = ema_value(df_higher, 21)
                ema55_4h = ema_value(df_higher, 55)
                price_4h = float(df_higher["close"].iloc[-1])
                four_h_bull = price_4h > ema55_4h and ema21_4h > ema55_4h
                four_h_bear = price_4h < ema55_4h and ema21_4h < ema55_4h
                if direction == "long" and four_h_bull:
                    raw_score *= 1.08   # 4H confirms long bias
                elif direction == "short" and four_h_bear:
                    raw_score *= 1.08   # 4H confirms short bias
                elif direction == "long" and four_h_bear:
                    raw_score *= 0.88   # counter-trend long — relaxed from 0.78; audit shows all trades in risk_off, longs need room
                elif direction == "short" and four_h_bull:
                    raw_score *= 0.78   # counter-trend short — strong penalty
            except Exception:
                pass

        # ── 6. Spot market context (basis, volume, Coinbase premium) ─────
        spot_ctx = snapshot.spot_context
        if spot_ctx is not None and raw_score > 0:
            try:
                spot_mult, spot_reasons = compute_spot_mult(spot_ctx, direction)
                if spot_mult != 1.0:
                    raw_score *= spot_mult
                    log.debug(
                        "%s spot mult %.3fx: %s",
                        snapshot.symbol, spot_mult, "; ".join(spot_reasons),
                    )
            except Exception:
                pass

        market_ctx = getattr(snapshot, "market_context", None)
        if market_ctx is not None and raw_score > 0:
            try:
                market_mult, market_reason = market_context_score_mult(market_ctx, direction)
                if market_mult != 1.0:
                    raw_score *= market_mult
                    log.debug(
                        "%s market context mult %.3fx: %s",
                        snapshot.symbol, market_mult, market_reason,
                    )
            except Exception:
                pass

        sector_rotation = getattr(snapshot, "sector_rotation", None)
        if sector_rotation is not None and raw_score > 0:
            try:
                sector_mult, sector_reason = sector_rotation_score_mult(sector_rotation, direction)
                if sector_mult != 1.0:
                    raw_score *= sector_mult
                    log.debug(
                        "%s sector rotation mult %.3fx: %s",
                        snapshot.symbol, sector_mult, sector_reason,
                    )
            except Exception:
                pass

        ts_diag = getattr(snapshot, "time_series", None)
        ts_score_mult = 1.0
        ts_size_mult = 1.0
        ts_reason = ""
        if ts_diag is not None and raw_score > 0:
            try:
                ts_score_mult, ts_size_mult, ts_reason = time_series_directional_mult(ts_diag, direction)
                if ts_score_mult != 1.0:
                    raw_score *= ts_score_mult
                    log.debug(
                        "%s time-series mult %.3fx size %.3fx: %s",
                        snapshot.symbol, ts_score_mult, ts_size_mult, ts_reason,
                    )
            except Exception:
                ts_score_mult, ts_size_mult, ts_reason = 1.0, 1.0, "time_series_failed"

        final_total = round(min(100.0, max(0.0, raw_score)), 2)

        # ── 6b. Cross-sectional normalization (professional upgrade) ───
        # Record score and compute z-score relative to recent distribution.
        # Top-decile signals get a ranking bonus; bottom-decile get penalized.
        self._record_score(final_total, regime.value)
        z = self.score_z_score(final_total)
        cs_bonus = self.cross_sectional_rank_bonus(z)
        if cs_bonus != 0.0:
            final_total = round(min(100.0, max(0.0, final_total + cs_bonus)), 2)
            log.debug(
                "%s cross-sectional: z=%.2f bonus=%+.1f → score=%.1f",
                snapshot.symbol, z, cs_bonus, final_total,
            )

        # ── 6c. Adaptive threshold (professional upgrade) ─────────────
        # The threshold adapts to regime and market context — stricter in
        # chaos/distribution, more lenient in trending expansion.
        adaptive_thresh = self.adaptive_threshold(
            regime=regime.value,
            market_ctx=market_ctx,
        )

        # ── 6. Edge Detector (observer-only by default) ───────────────
        edge_result = None
        try:
            p_win_ctx = ev_result.p_win if ev_result else 0.5
            vol_anom = fv.volume_anomaly_score if fv else 50.0
            vol_zscore = (vol_anom - 50.0) / 15.0   # maps 0–100 → ~ -3..+3
            edge_ctx = EdgeContext(
                pair=snapshot.symbol,
                direction=direction,
                regime=regime.value,
                smart_money_phase=(sm_signal.phase.value if sm_signal else "neutral"),
                structure_quality=float(sq),
                trend_strength=float(ts),
                volume_zscore=float(vol_zscore),
                oi_change_pct=float(snapshot.oi_change_pct or 0.0),
                funding_rate=float(snapshot.funding_rate or 0.0),
                volatility=float(vs),
                p_win=float(p_win_ctx),
                total_score=float(final_total),
                score_threshold=self._score_threshold,
            )
            edge_result = self._edge_detector.evaluate(edge_ctx)
            log.info("[EDGE] %s %s", snapshot.symbol, edge_result.summary())
        except Exception as exc:
            log.debug("Edge detector failed for %s: %s", snapshot.symbol, exc)

        return SignalBreakdown(
            symbol=snapshot.symbol,
            direction=direction,
            total_score=final_total,
            base_score=final_total,
            legacy_score=final_total,
            edge_result=edge_result,
            trend_strength=ts,
            volume_confirmation=vol_score,
            structure_quality=sq,
            open_interest=oi,
            funding_sentiment=fs,
            order_book=ob,
            volatility=vs,
            regime=regime,
            smart_money=sm_signal,
            ev_result=ev_result,
            feature_vector=fv,
            spot_context=spot_ctx,
            market_context=market_ctx,
            sector_rotation=sector_rotation,
            time_series=ts_diag,
            short_setup=short_setup,
            mtf_price_action=mtf_price_action,
            vwap_pullback=vwap_pullback,
            liquidity_sweep_reversal=liquidity_sweep_reversal,
            weights_used=dict(w),
            regime_ok=regime_ok,
            smart_money_ok=smart_money_ok,
            ev_ok=ev_ok,
            time_series_score_mult=ts_score_mult,
            time_series_size_mult=ts_size_mult,
            time_series_reason=ts_reason,
            pre_gate_score=pre_gate_score,
        )

    def _loss_contributor_adjustment(self, bd: SignalBreakdown) -> tuple[float, float, float, str]:
        guard = (getattr(self, "_cfg", {}) or {}).get("loss_contributor_guard", {}) or {}
        if not bool(guard.get("enabled", False)):
            return 1.0, 1.0, 0.0, ""

        score_mult = 1.0
        size_mult = 1.0
        threshold_shift = 0.0
        reasons: list[str] = []

        market_ctx = getattr(bd, "market_context", None)
        market_rotation = str(getattr(market_ctx, "rotation_state", "") or "unknown")
        if market_rotation == "mixed_rotation":
            score_mult *= float(guard.get("mixed_rotation_score_mult", 0.94) or 0.94)
            size_mult *= float(guard.get("mixed_rotation_size_mult", 0.80) or 0.80)
            threshold_shift += float(guard.get("mixed_rotation_threshold_shift", 5.0) or 5.0)
            reasons.append("mixed_rotation")

        btc_trend = str(getattr(market_ctx, "btc_trend", "") or "").lower()
        if market_rotation == "alts_outperforming" and btc_trend == "down":
            score_mult *= float(guard.get("alts_btc_down_score_mult", 0.85) or 0.85)
            size_mult *= float(guard.get("alts_btc_down_size_mult", 0.50) or 0.50)
            threshold_shift += float(guard.get("alts_btc_down_threshold_shift", 10.0) or 10.0)
            reasons.append("alts_btc_down")

        rotation = getattr(bd, "sector_rotation", None)
        rotation_state = str(getattr(rotation, "state", "") or "unknown")
        rotation_conf = float(getattr(rotation, "confidence", 0.0) or 0.0)
        rotation_missing = rotation is None or rotation_state in {"", "unknown"} or rotation_conf <= 0.0
        if rotation_missing:
            score_mult *= float(guard.get("missing_sector_rotation_score_mult", 0.96) or 0.96)
            size_mult *= float(guard.get("missing_sector_rotation_size_mult", 0.80) or 0.80)
            threshold_shift += float(guard.get("missing_sector_rotation_threshold_shift", 3.0) or 3.0)
            reasons.append("sector_rotation_missing")

        passport = getattr(bd, "setup_passport", None)
        sector = str(getattr(passport, "sector", "") or sector_for_asset(asset_from_symbol(bd.symbol)))
        symbol = str(getattr(bd, "symbol", "") or "")
        asset = asset_from_symbol(symbol)
        probation_sectors = {str(x) for x in guard.get("probation_sectors", []) or []}
        probation_symbols = {str(x) for x in guard.get("probation_symbols", []) or []}
        if sector in probation_sectors or symbol in probation_symbols or asset in probation_symbols:
            score_mult *= float(guard.get("probation_score_mult", 0.92) or 0.92)
            size_mult *= float(guard.get("probation_size_mult", 0.65) or 0.65)
            threshold_shift += float(guard.get("probation_threshold_shift", 6.0) or 6.0)
            reasons.append(f"probation={sector or asset}")

        session = active_market_session_key().lower()
        session_penalties = guard.get("session_penalties", {}) or {}
        if session and session in session_penalties:
            sp = session_penalties[session]
            score_mult *= float(sp.get("score_mult", 1.0) or 1.0)
            size_mult *= float(sp.get("size_mult", 1.0) or 1.0)
            threshold_shift += float(sp.get("threshold_shift", 0.0) or 0.0)
            reasons.append(f"session={session}")

        if str(getattr(bd, "direction", "") or "").lower() == "short":
            ev = getattr(bd, "ev_result", None)
            ev_net = float(getattr(ev, "ev_net_pct", 0.0) or 0.0) if ev is not None else 0.0
            if ev_net <= 0.0:
                score_mult *= float(guard.get("short_nonpositive_ev_score_mult", 0.90) or 0.90)
                size_mult *= float(guard.get("short_nonpositive_ev_size_mult", 0.70) or 0.70)
                threshold_shift += float(guard.get("short_nonpositive_ev_threshold_shift", 4.0) or 4.0)
                reasons.append("short_ev<=0")
            if rotation is not None and rotation_conf > 0.0 and rotation_state not in {"rotating_out", "weakening"}:
                score_mult *= float(guard.get("short_unsupported_rotation_score_mult", 0.94) or 0.94)
                size_mult *= float(guard.get("short_unsupported_rotation_size_mult", 0.75) or 0.75)
                threshold_shift += float(guard.get("short_unsupported_rotation_threshold_shift", 4.0) or 4.0)
                reasons.append(f"short_sector_rotation={rotation_state}")

        if not reasons:
            return 1.0, 1.0, 0.0, ""
        return (
            round(max(0.50, min(1.0, score_mult)), 4),
            round(max(0.35, min(1.0, size_mult)), 4),
            round(threshold_shift, 2),
            "loss_guard=" + "+".join(reasons),
        )

    def _short_macro_overlay_adjustment(
        self,
        bd: SignalBreakdown,
    ) -> tuple[float, float, float, str, float, str]:
        cfg = (getattr(self, "_cfg", {}) or {}).get("short_macro_overlay", {}) or {}
        if not bool(cfg.get("enabled", False)):
            return 1.0, 1.0, 0.0, "neutral", 0.0, ""
        if str(getattr(bd, "direction", "") or "").lower() != "short":
            return 1.0, 1.0, 0.0, "neutral", 0.0, ""

        setup_type = str(getattr(bd, "setup_type", "") or "unknown")
        boost_setups = set(
            cfg.get(
                "boost_setup_types",
                ["trend_continuation", "liquidity_sweep_reversal", "compression_breakout"],
            )
            or []
        )
        can_boost = setup_type in boost_setups

        market_ctx = getattr(bd, "market_context", None)
        sector_rotation = getattr(bd, "sector_rotation", None)
        if market_ctx is None:
            return 1.0, 1.0, 0.0, "unknown", 0.0, "short_macro=market_context_missing"

        btc = str(getattr(market_ctx, "btc_trend", "unknown") or "unknown")
        btc_d = str(getattr(market_ctx, "btc_d_trend", "unknown") or "unknown")
        total = str(getattr(market_ctx, "total_trend", "unknown") or "unknown")
        eth_btc = str(getattr(market_ctx, "eth_btc_trend", "unknown") or "unknown")
        risk = str(getattr(market_ctx, "risk_on_state", "unknown") or "unknown")
        rotation = str(getattr(market_ctx, "rotation_state", "unknown") or "unknown")
        market_conf = float(getattr(market_ctx, "confidence", 0.0) or 0.0)

        sector_state = str(getattr(sector_rotation, "state", "unknown") or "unknown")
        sector_conf = float(getattr(sector_rotation, "confidence", 0.0) or 0.0)
        min_sector_conf = float(cfg.get("min_sector_confidence", 0.35) or 0.35)
        require_sector = bool(cfg.get("require_sector_confirmation_for_boost", True))
        sector_support = sector_state in {"rotating_out", "weakening"} and sector_conf >= min_sector_conf
        sector_hostile = sector_state in {"rotating_in", "firming"} and sector_conf >= min_sector_conf

        score = 0.0
        reasons: list[str] = []
        if btc == "down":
            score += 2.0
            reasons.append("btc_down")
        elif btc == "up":
            score -= 2.0
            reasons.append("btc_up")
        if btc_d == "down":
            score += 1.5
            reasons.append("btc_d_down")
        elif btc_d == "up":
            score += 0.5
            reasons.append("btc_d_up_alt_underperformance")
        if total == "down":
            score += 1.5
            reasons.append("total_down")
        elif total == "up":
            score -= 1.0
            reasons.append("total_up")
        if risk == "risk_off":
            score += 1.0
            reasons.append("risk_off")
        elif risk in {"risk_on_alts", "risk_on_btc"}:
            score -= 2.0
            reasons.append(risk)
        if eth_btc in {"down", "flat"}:
            score += 0.5
            reasons.append(f"eth_btc_{eth_btc}")
        elif eth_btc == "up":
            score -= 0.5
            reasons.append("eth_btc_up")
        if sector_support:
            score += 2.0
            reasons.append(f"sector_{sector_state}")
        elif sector_hostile:
            score -= 2.0
            reasons.append(f"sector_{sector_state}")

        broad_unwind = btc == "down" and btc_d == "down" and total in {"down", "flat", "unknown"} and risk in {"risk_off", "mixed"}
        alt_underperformance = btc in {"down", "flat", "unknown"} and btc_d == "up" and sector_support
        hostile = btc == "up" or risk in {"risk_on_alts", "risk_on_btc"} or sector_hostile

        if can_boost and broad_unwind and (sector_support or not require_sector) and score >= 5.0:
            state = "strong_broad_unwind"
            score_mult = float(cfg.get("strong_score_mult", 1.06) or 1.06)
            size_mult = float(cfg.get("strong_size_mult", 1.08) or 1.08)
            threshold_shift = float(cfg.get("strong_threshold_shift", -2.0) or -2.0)
        elif can_boost and (sector_support or alt_underperformance or (not require_sector and score >= 3.5)) and score >= 3.0:
            state = "supportive_short"
            score_mult = float(cfg.get("supportive_score_mult", 1.03) or 1.03)
            size_mult = float(cfg.get("supportive_size_mult", 1.03) or 1.03)
            threshold_shift = float(cfg.get("supportive_threshold_shift", -1.0) or -1.0)
        elif hostile and score <= -1.0:
            state = "short_hostile"
            score_mult = float(cfg.get("hostile_score_mult", 0.92) or 0.92)
            size_mult = float(cfg.get("hostile_size_mult", 0.75) or 0.75)
            threshold_shift = float(cfg.get("hostile_threshold_shift", 4.0) or 4.0)
        else:
            state = "mixed_short" if market_conf > 0.0 else "unknown"
            score_mult = 1.0
            size_mult = 1.0
            threshold_shift = 0.0

        reason = "short_macro=" + state
        if reasons:
            reason += ":" + "+".join(reasons)
        if not can_boost and state in {"strong_broad_unwind", "supportive_short"}:
            reason += f"+setup_not_boosted={setup_type}"
        return (
            round(max(0.80, min(1.12, score_mult)), 4),
            round(max(0.50, min(1.15, size_mult)), 4),
            round(threshold_shift, 2),
            state,
            round(score, 2),
            reason,
        )

    def _binance_alpha_risk_adjustment(self, bd: SignalBreakdown) -> tuple[float, float, str]:
        passport = getattr(bd, "setup_passport", None)
        sector = str(getattr(passport, "sector", "") or sector_for_asset(asset_from_symbol(bd.symbol)))
        if sector != "binance_alpha":
            return 1.0, 1.0, ""

        reasons: list[str] = []
        score_mult = 1.0
        size_mult = 1.0
        setup_quality = float(getattr(bd, "setup_quality_score", 0.0) or 0.0)
        participation = min(float(getattr(bd, "volume_confirmation", 0.0) or 0.0), float(getattr(bd, "open_interest", 0.0) or 0.0))
        volatility = float(getattr(bd, "volatility", 0.0) or 0.0)

        if setup_quality < 65.0:
            score_mult *= 0.92
            size_mult *= 0.85
            reasons.append("quality<65")
        if participation < 55.0:
            score_mult *= 0.92
            size_mult *= 0.85
            reasons.append("participation<55")
        if volatility >= 75.0:
            score_mult *= 0.90
            size_mult *= 0.75
            reasons.append("volatility>=75")

        if not reasons:
            return 1.0, 1.0, ""
        return score_mult, size_mult, "binance_alpha_risk=" + "+".join(reasons)

    def score_many(self, snapshots: dict[str, MarketSnapshot]) -> list[SignalBreakdown]:
        results = []
        for snap in snapshots.values():
            bd = self.score(snap)
            if bd is not None:
                results.append(bd)
        dispersion = self._strategy_router.classify_dispersion(results)
        for bd in results:
            decision = self._strategy_router.evaluate(bd, dispersion)
            strategy_base = self._strategy_base_score(bd, decision.sleeve)
            # Keep alpha sleeves anchored to the institutional score, but let
            # neutral stay conservative so a non-alpha fallback cannot inherit
            # the full legacy conviction.
            if decision.sleeve == "neutral":
                anchored_base = min(float(bd.legacy_score), float(strategy_base))
            else:
                anchored_base = max(float(bd.legacy_score), float(strategy_base))
            bd.strategy_sleeve = decision.sleeve
            bd.strategy_reason = decision.reason
            bd.strategy_score_mult = decision.score_mult
            bd.strategy_size_mult = round(decision.size_mult * getattr(bd, "time_series_size_mult", 1.0), 4)
            bd.strategy_threshold_shift = decision.threshold_shift
            bd.strategy_ranking_bonus = decision.ranking_bonus
            bd.probability_score_mult = self._probabilistic_score_multiplier(getattr(bd, "ev_result", None))
            bd.setup_type = getattr(decision, "setup_type", "unknown")
            bd.setup_quality_score = float(getattr(decision, "quality_score", 0.0) or 0.0)
            bd.setup_passport = getattr(decision, "passport", None)
            alpha_score_mult, alpha_size_mult, alpha_reason = self._binance_alpha_risk_adjustment(bd)
            if alpha_reason:
                bd.strategy_reason = f"{bd.strategy_reason}; {alpha_reason}" if bd.strategy_reason else alpha_reason
                bd.strategy_score_mult = round(bd.strategy_score_mult * alpha_score_mult, 4)
                bd.strategy_size_mult = round(bd.strategy_size_mult * alpha_size_mult, 4)
            (
                macro_score_mult,
                macro_size_mult,
                macro_threshold_shift,
                macro_state,
                macro_score,
                macro_reason,
            ) = self._short_macro_overlay_adjustment(bd)
            bd.short_macro_state = macro_state
            bd.short_macro_score = macro_score
            bd.short_macro_reason = macro_reason
            bd.short_macro_score_mult = macro_score_mult
            bd.short_macro_size_mult = macro_size_mult
            bd.short_macro_threshold_shift = macro_threshold_shift
            if macro_reason:
                bd.strategy_reason = f"{bd.strategy_reason}; {macro_reason}" if bd.strategy_reason else macro_reason
            if macro_score_mult != 1.0 or macro_size_mult != 1.0 or macro_threshold_shift != 0.0:
                bd.strategy_score_mult = round(bd.strategy_score_mult * macro_score_mult, 4)
                bd.strategy_size_mult = round(bd.strategy_size_mult * macro_size_mult, 4)
                bd.strategy_threshold_shift = round(bd.strategy_threshold_shift + macro_threshold_shift, 2)
            loss_score_mult, loss_size_mult, loss_threshold_shift, loss_reason = self._loss_contributor_adjustment(bd)
            if loss_reason:
                bd.strategy_reason = f"{bd.strategy_reason}; {loss_reason}" if bd.strategy_reason else loss_reason
                bd.strategy_score_mult = round(bd.strategy_score_mult * loss_score_mult, 4)
                bd.strategy_size_mult = round(bd.strategy_size_mult * loss_size_mult, 4)
                bd.strategy_threshold_shift = round(bd.strategy_threshold_shift + loss_threshold_shift, 2)
            bd.dispersion_value = dispersion.value
            bd.dispersion_state = dispersion.state
            bd.base_score = anchored_base
            bd.total_score = round(
                min(100.0, max(0.0, anchored_base * bd.strategy_score_mult * bd.probability_score_mult)),
                2,
            )
        results.sort(key=lambda x: x.total_score, reverse=True)
        return results
