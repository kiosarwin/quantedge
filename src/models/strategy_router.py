"""
Research-backed strategy router for perp directional trading.

Maps a scored symbol into a concrete sleeve:
  - trend_following
  - compression_breakout
  - reversal
  - neutral

The router also applies cross-sectional dispersion-aware scaling so trend
exposure is reduced when the ranking signal becomes less reliable.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.models.strategy_passport import SetupPassport, asset_from_symbol, classify_rotation_state, sector_for_asset


@dataclass
class DispersionState:
    value: float
    state: str


@dataclass
class StrategyDecision:
    sleeve: str
    score_mult: float
    size_mult: float
    threshold_shift: float
    ranking_bonus: float
    reason: str
    setup_type: str = "none"
    quality_score: float = 0.0
    passport: SetupPassport | None = None


class StrategyRouter:
    def __init__(self, cfg: dict):
        s = cfg.get("strategy", {})
        self._trend_min = float(s.get("trend_min_score", 62.0))
        self._trend_sq_min = float(s.get("trend_min_structure", 58.0))
        self._trend_long_only = False  # UPGRADED: both directions valid per momentum research
        self._trend_require_participation = bool(s.get("trend_require_participation", True))
        self._trend_min_alignment = float(s.get("trend_min_alignment", 0.54))
        self._breakout_min_alignment = float(s.get("breakout_min_alignment", 0.57))
        self._reversal_min_alignment = float(s.get("reversal_min_alignment", 0.45))
        self._enable_compression_breakout = bool(s.get("enable_compression_breakout", False))
        self._compression_sq_min = float(s.get("compression_min_structure", 60.0))
        self._reversal_vol_max = float(s.get("reversal_max_volatility", 75.0))
        self._allow_long_reversal = bool(s.get("allow_long_reversal", True))
        self._allow_short_reversal = bool(s.get("allow_short_reversal", False))
        self._short_reversal_require_distribution = bool(
            s.get("short_reversal_require_distribution", True)
        )
        self._short_reversal_min_sm_score = float(s.get("short_reversal_min_sm_score", 85.0))
        self._short_reversal_min_structure = float(s.get("short_reversal_min_structure", 62.0))
        self._short_reversal_min_volume = float(s.get("short_reversal_min_volume", 45.0))
        # Dedicated post-distribution short setups (Phase D / Liq Sweep) ported
        # from `quantedge`. These admit shorts via a price-structure
        # thesis that does not depend on `regime=='distribution'`.
        self._enable_short_setups = bool(s.get("enable_short_setups", True))
        self._short_setup_min_confidence = float(
            s.get("short_setup_min_confidence", 0.65)
        )
        self._short_setup_min_structure = float(
            s.get("short_setup_min_structure", 50.0)
        )
        self._short_setup_max_volatility = float(
            s.get("short_setup_max_volatility", 85.0)
        )
        mtf_cfg = s.get("mtf_price_action_continuation", {}) or {}
        self._mtf_pa_enabled = bool(mtf_cfg.get("enabled", False))
        self._mtf_pa_min_quality = float(mtf_cfg.get("min_quality_score", 72.0) or 72.0)
        self._mtf_pa_require_participation = bool(mtf_cfg.get("require_participation", True))
        self._mtf_pa_block_high_dispersion = bool(mtf_cfg.get("block_high_dispersion", True))
        vwap_cfg = s.get("vwap_pullback_continuation", {}) or {}
        self._vwap_pb_enabled = bool(vwap_cfg.get("enabled", False))
        self._vwap_pb_min_quality = float(vwap_cfg.get("min_quality_score", 74.0) or 74.0)
        self._vwap_pb_require_participation = bool(vwap_cfg.get("require_participation", True))
        self._vwap_pb_block_high_dispersion = bool(vwap_cfg.get("block_high_dispersion", True))
        sweep_cfg = s.get("liquidity_sweep_reversal", {}) or {}
        self._lsr_enabled = bool(sweep_cfg.get("enabled", False))
        self._lsr_min_quality = float(sweep_cfg.get("min_quality_score", 76.0) or 76.0)
        self._lsr_require_participation = bool(sweep_cfg.get("require_participation", True))
        self._lsr_block_high_dispersion = bool(sweep_cfg.get("block_high_dispersion", True))
        self._lsr_max_volatility = float(sweep_cfg.get("max_volatility", self._reversal_vol_max) or self._reversal_vol_max)
        self._dispersion_warn = float(s.get("dispersion_warn", 28.0))
        self._dispersion_high = float(s.get("dispersion_high", 38.0))

    @staticmethod
    def _signal_value(breakdown, field: str, default: float = 50.0) -> float:
        return float(getattr(breakdown, field, default) or default)

    @staticmethod
    def _feature_vector(breakdown):
        return getattr(breakdown, "feature_vector", None)

    def _directional_alignment(self, breakdown) -> float:
        fv = self._feature_vector(breakdown)
        if fv is None:
            return 0.60
        align = getattr(fv, "directional_alignment", None)
        if callable(align):
            return float(align(breakdown.direction))
        return 0.60

    def _participation_score(self, breakdown) -> float:
        volume = min(100.0, max(0.0, self._signal_value(breakdown, "volume_confirmation")))
        oi = min(100.0, max(0.0, self._signal_value(breakdown, "open_interest")))
        return round((volume * 0.55 + oi * 0.45) / 100.0, 4)

    def _crowding_state(self, breakdown) -> str:
        fs = self._signal_value(breakdown, "funding_sentiment", 50.0)
        direction = str(getattr(breakdown, "direction", "") or "")
        if fs <= 20.0:
            return "crowded_longs" if direction == "short" else "long_crowding_risk"
        if fs >= 80.0:
            return "crowded_shorts" if direction == "long" else "short_crowding_risk"
        return "balanced"

    def _funding_state(self, breakdown) -> str:
        fs = self._signal_value(breakdown, "funding_sentiment", 50.0)
        direction = str(getattr(breakdown, "direction", "") or "")
        if self._funding_extreme_against_direction(breakdown):
            return "stretched_against_direction"
        if direction == "long" and fs >= 60.0:
            return "supportive_long"
        if direction == "short" and fs <= 40.0:
            return "supportive_short"
        return "neutral"

    def _microstructure_state(self, breakdown) -> str:
        if self._microstructure_contradicts(breakdown):
            return "contradicts_direction"
        fv = self._feature_vector(breakdown)
        if fv is None:
            return "unknown"
        alignment = self._directional_alignment(breakdown)
        if alignment >= 0.64:
            return "aligned"
        if alignment >= 0.54:
            return "mixed_but_acceptable"
        return "weak_alignment"

    @staticmethod
    def _invalidation_for(setup_type: str, direction: str) -> str:
        if setup_type == "vwap_pullback_continuation":
            return "loses_vwap_and_pullback_extreme"
        if setup_type == "mtf_price_action_continuation":
            return "breaks_pullback_structure"
        if setup_type == "trend_continuation":
            return "breaks_recent_structure_or_stop"
        if setup_type == "compression_breakout":
            return "failed_breakout_reenters_range"
        if setup_type in {"sweep_reversal", "liquidity_sweep_reversal", "phase_d", "liq_sweep"}:
            return "sweep_extreme_reclaimed_against_trade"
        return "no_valid_setup"

    @staticmethod
    def _expected_path_for(setup_type: str) -> str:
        if setup_type == "vwap_pullback_continuation":
            return "vwap_reclaim_continuation"
        if setup_type == "mtf_price_action_continuation":
            return "mtf_impulse_continuation"
        if setup_type == "trend_continuation":
            return "impulse_continuation"
        if setup_type == "compression_breakout":
            return "range_expansion"
        if setup_type in {"sweep_reversal", "liquidity_sweep_reversal", "phase_d", "liq_sweep"}:
            return "snapback_then_follow_through"
        return "none"

    @staticmethod
    def _hold_profile_for(setup_type: str) -> str:
        if setup_type == "vwap_pullback_continuation":
            return "intraday_trend_pullback"
        if setup_type == "mtf_price_action_continuation":
            return "intraday_to_swing"
        if setup_type == "trend_continuation":
            return "intraday_to_swing"
        if setup_type == "compression_breakout":
            return "intraday_breakout"
        if setup_type in {"sweep_reversal", "liquidity_sweep_reversal", "phase_d", "liq_sweep"}:
            return "fast_reversal"
        return "none"

    def _build_passport(
        self,
        breakdown,
        *,
        sleeve: str,
        setup_type: str,
        quality_score: float,
        decision: str,
        reason: str,
    ) -> SetupPassport:
        asset = asset_from_symbol(getattr(breakdown, "symbol", ""))
        sector = sector_for_asset(asset)
        sm_signal = getattr(breakdown, "smart_money", None)
        sm_phase = sm_signal.phase.value if sm_signal else "neutral"
        regime = breakdown.regime.value if getattr(breakdown, "regime", None) else "unknown"
        direction = str(getattr(breakdown, "direction", "") or "")
        return SetupPassport(
            symbol=str(getattr(breakdown, "symbol", "") or ""),
            asset=asset,
            sector=sector,
            rotation_state=classify_rotation_state(breakdown, sector),
            direction=direction,
            setup_type=setup_type,
            sleeve=sleeve,
            regime=regime,
            sm_phase=sm_phase,
            quality_score=quality_score,
            alignment=round(self._directional_alignment(breakdown), 4),
            participation_score=self._participation_score(breakdown),
            crowding_state=self._crowding_state(breakdown),
            funding_state=self._funding_state(breakdown),
            microstructure_state=self._microstructure_state(breakdown),
            invalidation=self._invalidation_for(setup_type, direction),
            expected_path=self._expected_path_for(setup_type),
            hold_profile=self._hold_profile_for(setup_type),
            decision=decision,
            reason=reason,
        )

    def _setup_quality_score(self, breakdown) -> float:
        fv = self._feature_vector(breakdown)
        alignment = self._directional_alignment(breakdown)
        participation = (
            min(100.0, max(0.0, self._signal_value(breakdown, "volume_confirmation"))) * 0.45
            + min(100.0, max(0.0, self._signal_value(breakdown, "open_interest"))) * 0.35
        ) / 80.0
        score = alignment * 70.0 + min(1.0, participation) * 20.0
        if fv is not None:
            liq = float(getattr(fv, "liquidation_pressure", 0.0) or 0.0)
            score += min(10.0, liq / 10.0)
        return round(min(100.0, max(0.0, score)), 2)

    def _microstructure_contradicts(self, breakdown) -> bool:
        fv = self._feature_vector(breakdown)
        if fv is None:
            return False
        direction = breakdown.direction
        structure = str(getattr(fv, "market_structure", "none") or "none")
        order_flow = float(getattr(fv, "order_flow_imbalance", 0.0) or 0.0)
        momentum = float(getattr(fv, "momentum_strength", 0.0) or 0.0)
        vwap_dist = float(getattr(fv, "vwap_distance", 0.0) or 0.0)
        if direction == "long":
            return (
                structure == "bearish_bos"
                and order_flow <= -20.0
                and (momentum <= -15.0 or vwap_dist <= -0.75)
            )
        if direction == "short":
            return (
                structure == "bullish_bos"
                and order_flow >= 20.0
                and (momentum >= 15.0 or vwap_dist >= 0.75)
            )
        return False

    def _setup_quality_supports(self, breakdown, min_alignment: float) -> bool:
        if self._microstructure_contradicts(breakdown):
            return False
        return self._directional_alignment(breakdown) >= min_alignment

    def _funding_supports_direction(self, breakdown) -> bool:
        funding_sentiment = self._signal_value(breakdown, "funding_sentiment", 50.0)
        if breakdown.direction == "long":
            return funding_sentiment <= 80.0
        if breakdown.direction == "short":
            return funding_sentiment >= 20.0
        return True

    def _funding_extreme_against_direction(self, breakdown) -> bool:
        funding_sentiment = self._signal_value(breakdown, "funding_sentiment", 50.0)
        if breakdown.direction == "long":
            return funding_sentiment > 85.0
        if breakdown.direction == "short":
            return funding_sentiment < 15.0
        return False


    def _mtf_price_action_candidate(self, breakdown) -> bool:
        signal = getattr(breakdown, "mtf_price_action", None)
        if not self._mtf_pa_enabled or signal is None or not bool(getattr(signal, "is_valid", False)):
            return False
        if str(getattr(signal, "direction", "none") or "none") != str(getattr(breakdown, "direction", "") or ""):
            return False
        regime = getattr(getattr(breakdown, "regime", None), "value", "")
        if regime != "trending_expansion":
            return False
        if float(getattr(signal, "quality_score", 0.0) or 0.0) < self._mtf_pa_min_quality:
            return False
        if self._mtf_pa_require_participation and not self._participation_supports_trend(breakdown):
            return False
        return (
            not self._funding_extreme_against_direction(breakdown)
            and self._setup_quality_supports(breakdown, self._trend_min_alignment)
        )


    def _vwap_pullback_candidate(self, breakdown) -> bool:
        signal = getattr(breakdown, "vwap_pullback", None)
        if not self._vwap_pb_enabled or signal is None or not bool(getattr(signal, "is_valid", False)):
            return False
        if str(getattr(signal, "direction", "none") or "none") != str(getattr(breakdown, "direction", "") or ""):
            return False
        regime = getattr(getattr(breakdown, "regime", None), "value", "")
        if regime != "trending_expansion":
            return False
        if float(getattr(signal, "quality_score", 0.0) or 0.0) < self._vwap_pb_min_quality:
            return False
        if self._vwap_pb_require_participation and not self._participation_supports_trend(breakdown):
            return False
        return (
            not self._funding_extreme_against_direction(breakdown)
            and self._setup_quality_supports(breakdown, self._trend_min_alignment)
        )


    def _liquidity_sweep_reversal_candidate(self, breakdown) -> bool:
        signal = getattr(breakdown, "liquidity_sweep_reversal", None)
        if not self._lsr_enabled or signal is None or not bool(getattr(signal, "is_valid", False)):
            return False
        if str(getattr(signal, "direction", "none") or "none") != str(getattr(breakdown, "direction", "") or ""):
            return False
        if float(getattr(signal, "quality_score", 0.0) or 0.0) < self._lsr_min_quality:
            return False
        if float(getattr(breakdown, "volatility", 0.0) or 0.0) > self._lsr_max_volatility:
            return False
        if self._lsr_require_participation and not self._reversal_is_supported(breakdown):
            return False
        direction = str(getattr(breakdown, "direction", "") or "")
        if direction == "long" and not self._allow_long_reversal:
            return False
        if direction == "short" and not self._allow_short_reversal:
            return False
        return self._setup_quality_supports(breakdown, self._reversal_min_alignment)

    def _participation_supports_trend(self, breakdown) -> bool:
        volume_confirmation = self._signal_value(breakdown, "volume_confirmation", 50.0)
        open_interest = self._signal_value(breakdown, "open_interest", 50.0)
        return volume_confirmation >= 45.0 and open_interest >= 45.0

    def _participation_supports_breakout(self, breakdown) -> bool:
        volume_confirmation = self._signal_value(breakdown, "volume_confirmation", 50.0)
        open_interest = self._signal_value(breakdown, "open_interest", 50.0)
        return volume_confirmation >= 55.0 and open_interest >= 50.0

    def _reversal_is_supported(self, breakdown) -> bool:
        volume_confirmation = self._signal_value(breakdown, "volume_confirmation", 50.0)
        open_interest = self._signal_value(breakdown, "open_interest", 50.0)
        return volume_confirmation >= 45.0 or open_interest >= 45.0

    def is_long_reversal_candidate(self, breakdown) -> bool:
        sm_signal = breakdown.smart_money
        sm = sm_signal.phase.value if sm_signal else "neutral"
        sm_bias = sm_signal.direction_bias if sm_signal else "neutral"
        return (
            self._allow_long_reversal
            and breakdown.direction == "long"
            and sm == "liquidity_sweep"
            and sm_bias in {"long", "neutral"}
            and self._reversal_is_supported(breakdown)
        )

    def is_short_reversal_candidate(self, breakdown) -> bool:
        sm_signal = breakdown.smart_money
        sm = sm_signal.phase.value if sm_signal else "neutral"
        sm_bias = sm_signal.direction_bias if sm_signal else "neutral"
        sm_score = float(sm_signal.score) if sm_signal else 0.0
        return (
            self._allow_short_reversal
            and breakdown.direction == "short"
            and sm_bias == "short"
            and sm in {"distribution", "liquidity_sweep"}
            and sm_score >= self._short_reversal_min_sm_score
            and breakdown.structure_quality >= self._short_reversal_min_structure
            and breakdown.volume_confirmation >= self._short_reversal_min_volume
            and breakdown.volatility <= self._reversal_vol_max
            and self._reversal_is_supported(breakdown)
            and (
                not self._short_reversal_require_distribution
                or breakdown.regime.value == "distribution"
            )
        )

    def is_short_setup_candidate(self, breakdown) -> bool:
        """Admit a short via the dedicated Phase D / Liq Sweep detector.

        These are the post-distribution short edges ported from
        ``quantedge``. They encode the breakdown thesis directly
        from price structure, so they admit shorts without requiring
        ``regime=='distribution'`` or smart-money DISTRIBUTION/LIQ_SWEEP
        phases. The standard volatility, OI, and volume floors still apply
        so that low-quality candles cannot smuggle through.
        """
        if not (self._enable_short_setups and self._allow_short_reversal):
            return False
        if breakdown.direction != "short":
            return False
        setup = getattr(breakdown, "short_setup", None)
        if setup is None or not getattr(setup, "is_valid", False):
            return False
        if float(getattr(setup, "confidence", 0.0)) < self._short_setup_min_confidence:
            return False
        if float(getattr(breakdown, "structure_quality", 0.0)) < self._short_setup_min_structure:
            return False
        if float(getattr(breakdown, "volatility", 0.0)) > self._short_setup_max_volatility:
            return False
        if not self._reversal_is_supported(breakdown):
            return False
        return True

    def short_setup_label(self, breakdown) -> str:
        """Return the dedicated short-setup label (e.g. 'phase_d') or '' if absent."""
        setup = getattr(breakdown, "short_setup", None)
        if setup is None or not getattr(setup, "is_valid", False):
            return ""
        return str(getattr(setup, "label", ""))

    def classify_dispersion(self, breakdowns: list) -> DispersionState:
        vals = []
        for b in breakdowns:
            fv = getattr(b, "feature_vector", None)
            if fv is not None:
                vals.append(float(fv.momentum_strength))
        if len(vals) < 3:
            return DispersionState(value=0.0, state="normal")
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
        disp = var ** 0.5
        if disp >= self._dispersion_high:
            state = "high"
        elif disp >= self._dispersion_warn:
            state = "warn"
        else:
            state = "normal"
        return DispersionState(value=round(disp, 2), state=state)

    def evaluate(self, breakdown, dispersion: DispersionState) -> StrategyDecision:
        sm_signal = breakdown.smart_money
        sm = sm_signal.phase.value if sm_signal else "neutral"
        sm_bias = sm_signal.direction_bias if sm_signal else "neutral"
        sm_score = float(sm_signal.score) if sm_signal else 0.0
        direction = breakdown.direction
        is_short = direction == "short"
        is_long = direction == "long"

        has_mtf_price_action_signal = bool(getattr(getattr(breakdown, "mtf_price_action", None), "is_valid", False))
        has_vwap_pullback_signal = bool(getattr(getattr(breakdown, "vwap_pullback", None), "is_valid", False))
        has_liquidity_sweep_reversal_signal = bool(getattr(getattr(breakdown, "liquidity_sweep_reversal", None), "is_valid", False))
        is_mtf_price_action = self._mtf_price_action_candidate(breakdown)
        is_vwap_pullback = self._vwap_pullback_candidate(breakdown)
        is_liquidity_sweep_reversal = self._liquidity_sweep_reversal_candidate(breakdown)
        experimental_blocked_by_dispersion = False
        if dispersion.state == "high":
            if self._mtf_pa_block_high_dispersion and has_mtf_price_action_signal:
                is_mtf_price_action = False
                experimental_blocked_by_dispersion = True
            if self._vwap_pb_block_high_dispersion and has_vwap_pullback_signal:
                is_vwap_pullback = False
                experimental_blocked_by_dispersion = True
            if self._lsr_block_high_dispersion and has_liquidity_sweep_reversal_signal:
                is_liquidity_sweep_reversal = False
        is_trend = (
            breakdown.regime.value == "trending_expansion"
            and breakdown.trend_strength >= self._trend_min
            and breakdown.structure_quality >= self._trend_sq_min
            and sm in {"trending", "accumulation", "neutral"}
            and (is_long or not self._trend_long_only)
            and not self._funding_extreme_against_direction(breakdown)
            and self._setup_quality_supports(breakdown, self._trend_min_alignment)
            and (not self._trend_require_participation or self._participation_supports_trend(breakdown))
            and not experimental_blocked_by_dispersion
        )
        is_compression_breakout = (
            self._enable_compression_breakout
            and
            breakdown.regime.value == "accumulation_compression"
            and breakdown.structure_quality >= self._compression_sq_min
            and sm in {"accumulation", "liquidity_sweep", "neutral"}
            and self._participation_supports_breakout(breakdown)
            and self._funding_supports_direction(breakdown)
            and self._setup_quality_supports(breakdown, self._breakout_min_alignment)
        )

        is_long_reversal = self.is_long_reversal_candidate(breakdown)
        is_short_reversal = self.is_short_reversal_candidate(breakdown)
        is_short_setup = self.is_short_setup_candidate(breakdown)
        is_reversal = (
            is_liquidity_sweep_reversal
            or is_long_reversal
            or is_short_reversal
            or is_short_setup
        ) and self._setup_quality_supports(
            breakdown, self._reversal_min_alignment
        )
        quality_score = self._setup_quality_score(breakdown)

        if is_reversal:
            score_mult = 1.08
            size_mult = 0.75
            threshold_shift = -2.0
            ranking_bonus = 4.0
            if is_liquidity_sweep_reversal:
                sweep_signal = getattr(breakdown, "liquidity_sweep_reversal", None)
                reason = "reversal sleeve via liquidity sweep reclaim"
                setup_type = "liquidity_sweep_reversal"
                quality_score = max(quality_score, float(getattr(sweep_signal, "quality_score", 0.0) or 0.0))
                score_mult = 1.10
                size_mult = 0.72
                threshold_shift = -2.0
                ranking_bonus += 2.0
            elif is_short_setup:
                label = self.short_setup_label(breakdown) or "phase_d"
                reason = f"short setup sleeve via {label}"
                # Phase D / Liq Sweep are the bot's primary statistical edge
                # cohorts in quantedge. Give them a slightly larger
                # ranking bonus so they surface ahead of generic reversal
                # candidates when both fire on the same scan.
                ranking_bonus += 1.5
            elif is_short_reversal:
                reason = f"short reversal sleeve via {sm} sm={sm_score:.0f}"
            else:
                reason = f"reversal sleeve via {sm}"
            if not is_liquidity_sweep_reversal:
                setup_type = "sweep_reversal" if not is_short_setup else self.short_setup_label(breakdown) or "phase_d"
            if dispersion.state == "high":
                score_mult *= 1.04
                size_mult *= 1.08
                ranking_bonus += 2.0
                reason += " + dispersion tail"
            reason += f" q={quality_score:.0f}"
            passport = self._build_passport(
                breakdown, sleeve="reversal", setup_type=setup_type,
                quality_score=quality_score, decision="eligible", reason=reason,
            )
            return StrategyDecision(
                sleeve="reversal",
                score_mult=score_mult,
                size_mult=size_mult,
                threshold_shift=threshold_shift,
                ranking_bonus=ranking_bonus,
                reason=reason,
                setup_type=setup_type,
                quality_score=quality_score,
                passport=passport,
            )

        if is_trend or is_mtf_price_action or is_vwap_pullback:
            score_mult = 1.10
            size_mult = 1.12
            threshold_shift = -3.0
            ranking_bonus = 3.0
            setup_type = "trend_continuation"
            reason = "trend sleeve via trending_expansion"
            if is_mtf_price_action:
                mtf_signal = getattr(breakdown, "mtf_price_action", None)
                score_mult = 1.12
                ranking_bonus = 4.0
                setup_type = "mtf_price_action_continuation"
                reason = "trend sleeve via MTF price-action continuation"
                quality_score = max(quality_score, float(getattr(mtf_signal, "quality_score", 0.0) or 0.0))
            if is_vwap_pullback:
                vwap_signal = getattr(breakdown, "vwap_pullback", None)
                score_mult = 1.13
                ranking_bonus = 4.5
                setup_type = "vwap_pullback_continuation"
                reason = "trend sleeve via VWAP pullback continuation"
                quality_score = max(quality_score, float(getattr(vwap_signal, "quality_score", 0.0) or 0.0))
            if not self._participation_supports_trend(breakdown):
                score_mult *= 0.96
                size_mult *= 0.94
                reason += " (thin participation)"
            if not self._funding_supports_direction(breakdown):
                score_mult *= 0.95
                size_mult *= 0.95
                reason += " (funding stretched)"
            if dispersion.state == "warn":
                score_mult *= 0.94
                size_mult *= 0.88
                reason += " (dispersion warn)"
            elif dispersion.state == "high":
                score_mult *= 0.88
                size_mult *= 0.72
                threshold_shift += 2.0
                ranking_bonus -= 1.5
                reason += " (dispersion high)"
            reason += f" q={quality_score:.0f}"
            passport = self._build_passport(
                breakdown, sleeve="trend_following", setup_type=setup_type,
                quality_score=quality_score, decision="eligible", reason=reason,
            )
            return StrategyDecision(
                sleeve="trend_following",
                score_mult=score_mult,
                size_mult=size_mult,
                threshold_shift=threshold_shift,
                ranking_bonus=ranking_bonus,
                reason=reason,
                setup_type=setup_type,
                quality_score=quality_score,
                passport=passport,
            )

        if is_compression_breakout:
            score_mult = 1.04
            size_mult = 0.90
            threshold_shift = -1.0
            ranking_bonus = 1.5
            reason = "compression breakout sleeve via compression + participation"
            if dispersion.state == "high":
                score_mult *= 0.95
                size_mult *= 0.85
                reason += " (dispersion high)"
            reason += f" q={quality_score:.0f}"
            passport = self._build_passport(
                breakdown, sleeve="compression_breakout", setup_type="compression_breakout",
                quality_score=quality_score, decision="eligible", reason=reason,
            )
            return StrategyDecision(
                sleeve="compression_breakout",
                score_mult=score_mult,
                size_mult=size_mult,
                threshold_shift=threshold_shift,
                ranking_bonus=ranking_bonus,
                reason=reason,
                setup_type="compression_breakout",
                quality_score=quality_score,
                passport=passport,
            )

        score_mult = 0.93
        size_mult = 0.85
        threshold_shift = 0.0
        ranking_bonus = 0.0
        reason = "no validated sleeve"
        if is_short and not self._allow_short_reversal and not self._trend_long_only:
            # Shorts are fully enabled but no specific sleeve qualified —
            # treat same as long neutral (no directional penalty)
            reason = "short neutral — no sleeve matched but direction active"
            score_mult = 0.93
            size_mult = 0.85
        elif is_short and not self._allow_short_reversal:
            reason = "short side parked — reversal not enabled"
            score_mult = 0.88
            size_mult = 0.75
            threshold_shift = 2.0
        elif direction == "none":
            reason = "no directional edge"
            score_mult = 0.85
            size_mult = 0.80
        else:
            reason = "no validated sleeve (regime/flow/funding/OI not aligned)"
        if self._microstructure_contradicts(breakdown):
            reason = "no validated sleeve (microstructure contradicts direction)"
            score_mult *= 0.90
            size_mult *= 0.80
            threshold_shift += 2.0
        elif self._directional_alignment(breakdown) < self._trend_min_alignment:
            reason += " (weak directional alignment)"
            score_mult *= 0.94
            size_mult *= 0.88
        if dispersion.state == "high":
            score_mult *= 0.95
            size_mult *= 0.90
        passport = self._build_passport(
            breakdown, sleeve="neutral", setup_type="no_trade",
            quality_score=quality_score, decision="no_trade", reason=reason,
        )
        return StrategyDecision(
            sleeve="neutral",
            score_mult=score_mult,
            size_mult=size_mult,
            threshold_shift=threshold_shift,
            ranking_bonus=ranking_bonus,
            reason=reason,
            setup_type="no_trade",
            quality_score=quality_score,
            passport=passport,
        )
