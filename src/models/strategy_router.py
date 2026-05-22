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


class StrategyRouter:
    def __init__(self, cfg: dict):
        s = cfg.get("strategy", {})
        self._trend_min = float(s.get("trend_min_score", 62.0))
        self._trend_sq_min = float(s.get("trend_min_structure", 58.0))
        self._trend_long_only = False  # UPGRADED: both directions valid per momentum research
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
        # from `kiosarwin/Futures`. These admit shorts via a price-structure
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
        self._dispersion_warn = float(s.get("dispersion_warn", 28.0))
        self._dispersion_high = float(s.get("dispersion_high", 38.0))

    @staticmethod
    def _signal_value(breakdown, field: str, default: float = 50.0) -> float:
        return float(getattr(breakdown, field, default) or default)

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
        ``kiosarwin/Futures``. They encode the breakdown thesis directly
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

        is_trend = (
            breakdown.regime.value == "trending_expansion"
            and breakdown.trend_strength >= self._trend_min
            and breakdown.structure_quality >= self._trend_sq_min
            and sm in {"trending", "accumulation", "neutral"}
            and (is_long or not self._trend_long_only)
            and not self._funding_extreme_against_direction(breakdown)
        )
        is_compression_breakout = (
            self._enable_compression_breakout
            and
            breakdown.regime.value == "accumulation_compression"
            and breakdown.structure_quality >= self._compression_sq_min
            and sm in {"accumulation", "liquidity_sweep", "neutral"}
            and self._participation_supports_breakout(breakdown)
            and self._funding_supports_direction(breakdown)
        )

        is_long_reversal = self.is_long_reversal_candidate(breakdown)
        is_short_reversal = self.is_short_reversal_candidate(breakdown)
        is_short_setup = self.is_short_setup_candidate(breakdown)
        is_reversal = is_long_reversal or is_short_reversal or is_short_setup

        if is_reversal:
            score_mult = 1.08
            size_mult = 0.75
            threshold_shift = -2.0
            ranking_bonus = 4.0
            if is_short_setup:
                label = self.short_setup_label(breakdown) or "phase_d"
                reason = f"short setup sleeve via {label}"
                # Phase D / Liq Sweep are the bot's primary statistical edge
                # cohorts in kiosarwin/Futures. Give them a slightly larger
                # ranking bonus so they surface ahead of generic reversal
                # candidates when both fire on the same scan.
                ranking_bonus += 1.5
            elif is_short_reversal:
                reason = f"short reversal sleeve via {sm} sm={sm_score:.0f}"
            else:
                reason = f"reversal sleeve via {sm}"
            if dispersion.state == "high":
                score_mult *= 1.04
                size_mult *= 1.08
                ranking_bonus += 2.0
                reason += " + dispersion tail"
            return StrategyDecision(
                sleeve="reversal",
                score_mult=score_mult,
                size_mult=size_mult,
                threshold_shift=threshold_shift,
                ranking_bonus=ranking_bonus,
                reason=reason,
            )

        if is_trend:
            score_mult = 1.10
            size_mult = 1.12
            threshold_shift = -3.0
            ranking_bonus = 3.0
            reason = "trend sleeve via trending_expansion"
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
            return StrategyDecision(
                sleeve="trend_following",
                score_mult=score_mult,
                size_mult=size_mult,
                threshold_shift=threshold_shift,
                ranking_bonus=ranking_bonus,
                reason=reason,
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
            return StrategyDecision(
                sleeve="compression_breakout",
                score_mult=score_mult,
                size_mult=size_mult,
                threshold_shift=threshold_shift,
                ranking_bonus=ranking_bonus,
                reason=reason,
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
        if dispersion.state == "high":
            score_mult *= 0.95
            size_mult *= 0.90
        return StrategyDecision(
            sleeve="neutral",
            score_mult=score_mult,
            size_mult=size_mult,
            threshold_shift=threshold_shift,
            ranking_bonus=ranking_bonus,
            reason=reason,
        )
