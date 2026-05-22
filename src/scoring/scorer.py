"""
Institutional scoring engine.

Pipeline per symbol:
  1. Feature vector F(t) built from all available signals.
  2. Four-state regime classified — chaos blocks the trade.
  3. Smart money phase detected — must have directional bias.
  4. EV model gate — EV must be positive after fees.
  5. Weighted signal score (unchanged from original for continuity).

Final score is regime-gated and smart-money-gated.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING

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
from src.analysis.feature_engine import build_feature_vector, FeatureVector
from src.analysis.spot_context import SpotContext, compute_spot_mult
from src.models.ev_model import EVResult
from src.models.pwin_engine import PwinContext
from src.models.edge_detector import EdgeContext, EdgeResult
from src.models.regime_classifier import classify_four_state, RegimeState
from src.models.strategy_router import StrategyRouter, DispersionState
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
    short_setup: ShortEntrySignal | None = None

    weights_used: dict = field(default_factory=dict)
    strategy_sleeve: str = "neutral"
    strategy_reason: str = ""
    strategy_score_mult: float = 1.0
    strategy_size_mult: float = 1.0
    strategy_threshold_shift: float = 0.0
    strategy_ranking_bonus: float = 0.0
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
        min_trades = int(self._cfg.get("ev_model", {}).get("min_trades_for_ev", 20))
        if ev_result.trade_count >= hard_gate_min_trades:
            return False
        if ev_result.trade_count < min_trades:
            max_deficit = float(
                paper_validation.get("ev_bootstrap_max_deficit_pct", 0.15) or 0.15
            )
        else:
            max_deficit = float(
                paper_validation.get("ev_probation_max_deficit_pct", 0.10) or 0.10
            )
        return ev_result.ev_net_pct >= -max_deficit

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
        if breakdown.direction == "short":
            base -= 8.0
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

        direction = trade_direction_from_structure(df_primary, self._cfg)
        if direction == "none" and not df_higher.empty and len(df_higher) >= 50:
            direction = trade_direction_from_structure(df_higher, self._cfg)
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
        # `kiosarwin/Futures` (cohort: IMMINENT_DUMP + PHASE_D / LIQ_SWEEP).
        # Detection runs only for short candidates and is purely informational
        # here — the StrategyRouter consumes `breakdown.short_setup` to decide
        # whether to admit the trade via the reversal sleeve.
        short_setup: ShortEntrySignal | None = None
        if direction == "short":
            try:
                setup = detect_short_entry(df_primary, self._cfg)
                if setup.is_valid:
                    short_setup = setup
                    # Phase D / Liq Sweep encode the post-distribution
                    # breakdown thesis themselves; if the regime classifier
                    # tagged this candle as `distribution` and the dedicated
                    # detector fires, the regime gate is no longer the right
                    # blocker — the setup-specific structure is.
                    if not regime_ok and regime.value == "distribution":
                        regime_ok = True
            except Exception as exc:
                log.debug("Short setup detection failed for %s: %s", snapshot.symbol, exc)

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
            pwin_ctx = PwinContext(
                regime=regime.value,
                sm_phase=sm_phase_val,
                sm_aligned=sm_aligned,
                direction=direction,
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
            # Bootstrap EV only when paper mode is explicitly in exploration mode.
            min_trades = self._cfg.get("ev_model", {}).get("min_trades_for_ev", 20)
            if ev_result.trade_count < min_trades:
                ev_ok = (
                    self._ev_bootstrap_enabled
                    or ev_result.ev_net_pct > 0
                    or self._paper_soft_ev_allowed(ev_result)
                )
            else:
                ev_ok = ev_result.is_tradeable or self._paper_soft_ev_allowed(ev_result)
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

        # Gate penalties: reduce score if institutional filters fail
        if not regime_ok:
            raw_score *= 0.0    # chaos regime = no trade
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
                    raw_score *= 0.78   # counter-trend long — strong penalty
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

        final_total = round(min(100.0, max(0.0, raw_score)), 2)

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
            short_setup=short_setup,
            weights_used=dict(w),
            regime_ok=regime_ok,
            smart_money_ok=smart_money_ok,
            ev_ok=ev_ok,
        )

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
            bd.strategy_size_mult = decision.size_mult
            bd.strategy_threshold_shift = decision.threshold_shift
            bd.strategy_ranking_bonus = decision.ranking_bonus
            bd.dispersion_value = dispersion.value
            bd.dispersion_state = dispersion.state
            bd.base_score = anchored_base
            bd.total_score = round(
                min(100.0, max(0.0, anchored_base * decision.score_mult)),
                2,
            )
        results.sort(key=lambda x: x.total_score, reverse=True)
        return results
