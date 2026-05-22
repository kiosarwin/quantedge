"""
Expected-Value gate.

The Scorer asks the EV model: given today's setup and yesterday's outcomes,
do we expect a positive *net* return after fees + slippage + funding?

Sample-based P(win)
-------------------
The model filters the trade log to records that share key context with the
current setup (regime / sm_phase / direction).  When no realised samples
match, we fall back to the analytic prior in `pwin_engine.context_prior`.

Asymmetric Bayesian shrinkage
-----------------------------
Small samples are noisy, so we Bayes-shrink the win-rate towards the prior
with an effective sample size α (default 8).  Crucially, when the realised
P(win) is *below* the prior we shrink less aggressively (we trust losses
faster than wins) — this prevents the bot from re-arming a strategy that
just bled its way into the trade log.

Kelly fraction
--------------
``EVResult.kelly_fraction`` is **already** the quarter-Kelly value the
sizer expects (the sizer multiplies by its own configured fraction on
top, with hard caps).  ``kelly_raw`` keeps the un-discounted reference.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

from src.models.pwin_engine import PwinContext, context_prior

log = logging.getLogger(__name__)

__all__ = ["EVModel", "EVResult"]


# ---- Result dataclass ------------------------------------------------------

@dataclass
class EVResult:
    p_win: float = 0.5
    p_loss: float = 0.5
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    payoff_ratio: float = 0.0          # avg_win_pct / avg_loss_pct
    cost_pct: float = 0.0              # round-trip cost (taker fees + slippage)
    funding_cost_pct: float = 0.0
    ev_gross_pct: float = 0.0          # before cost
    ev_net_pct: float = 0.0            # after cost (REAL value, never mangled)
    confidence: float = 0.4            # 0..1, scales with sample size
    trade_count: int = 0               # number of realised samples used
    kelly_raw: float = 0.0             # un-discounted Kelly fraction
    kelly_fraction: float = 0.0        # already quarter-Kelly (×0.25)
    rationale: str = ""
    # auxiliary: for telemetry / training
    matched_keys: list[str] = field(default_factory=list)
    # Live gate flags — applied without mutating the underlying EV math.
    p_win_floor_ok: bool = True
    ev_floor_ok: bool = True

    @property
    def is_tradeable(self) -> bool:
        return (
            self.ev_net_pct > 0.0
            and self.p_win > 0.0
            and self.p_win_floor_ok
            and self.ev_floor_ok
        )


# ---- Engine ----------------------------------------------------------------

class EVModel:
    """Sample-based EV computation with asymmetric Bayesian shrinkage."""

    def __init__(self, cfg: dict):
        self._cfg = cfg
        ev_cfg = (cfg or {}).get("ev_model", {}) or {}
        risk_cfg = (cfg or {}).get("risk", {}) or {}
        backtest_cfg = (cfg or {}).get("backtest", {}) or {}

        self._min_trades_for_ev = int(ev_cfg.get("min_trades_for_ev", 20) or 20)
        self._min_p_win = float(ev_cfg.get("min_p_win", 0.42) or 0.42)
        self._min_ev_pct = float(ev_cfg.get("min_ev_pct", 0.05) or 0.05)
        self._prior_weight = float(ev_cfg.get("prior_weight_alpha", 8) or 8)
        self._loss_shrink_factor = float(ev_cfg.get("loss_shrink_factor", 0.5) or 0.5)
        self._kelly_quarter = 0.25  # documented contract for KellySizer

        self._taker_fee_pct = float(
            risk_cfg.get("taker_fee_pct", backtest_cfg.get("commission_pct", 0.04) or 0.04)
            or 0.04
        )
        self._slippage_pct = float(
            risk_cfg.get("slippage_pct", backtest_cfg.get("slippage_pct", 0.02) or 0.02)
            or 0.02
        )
        # Round-trip cost in percent (taker + taker + slippage on both legs).
        self._round_trip_cost_pct = (self._taker_fee_pct + self._slippage_pct) * 2

    # --------------------------------------------------------------- public

    def compute(
        self,
        trade_log: Iterable,
        funding_rate: float | None = None,
        pwin_ctx: PwinContext | None = None,
    ) -> EVResult:
        ctx = pwin_ctx or PwinContext()
        prior_p = context_prior(ctx)

        matched, all_samples = self._match_samples(trade_log, ctx)
        sample_size = len(matched)
        # ---- P(win) with Bayesian shrinkage --------------------------
        if sample_size > 0:
            wins_pct = [t for t in matched if _pnl_pct(t) > 0]
            losses_pct = [t for t in matched if _pnl_pct(t) <= 0]
            raw_p = len(wins_pct) / sample_size
            p_win = self._bayes_shrink(raw_p, prior_p, sample_size)
            avg_win = (sum(_pnl_pct(t) for t in wins_pct) / len(wins_pct)) if wins_pct else 0.0
            avg_loss = (
                abs(sum(_pnl_pct(t) for t in losses_pct) / len(losses_pct))
                if losses_pct
                else 0.0
            )
        else:
            # Fall back to the global log to estimate average win/loss size,
            # but keep p_win on the analytic prior.
            wins_all = [t for t in all_samples if _pnl_pct(t) > 0]
            losses_all = [t for t in all_samples if _pnl_pct(t) <= 0]
            p_win = prior_p
            avg_win = (
                (sum(_pnl_pct(t) for t in wins_all) / len(wins_all)) if wins_all else 1.5
            )
            avg_loss = (
                abs(sum(_pnl_pct(t) for t in losses_all) / len(losses_all))
                if losses_all
                else 1.0
            )

        # If avg_loss collapsed to 0 (very few losses), guard the math.
        if avg_loss <= 0:
            avg_loss = max(0.5, avg_win * 0.5) or 1.0
        if avg_win <= 0:
            avg_win = max(0.5, avg_loss * 0.6) or 1.0
        payoff = avg_win / avg_loss

        # ---- Cost model (% per trade) -------------------------------
        cost_pct = self._round_trip_cost_pct
        # Direction-aware funding-rate proxy.  Longs pay positive funding and
        # earn it when negative; shorts are the mirror image.  Treating both
        # sides as paying (the historic abs() default) systematically
        # under-counts EV for the side that actually receives funding.
        # `funding_rate` is fractional per 8h period — we charge it as a
        # % cost (or credit) on the trade leg.  Conservatively only credit
        # *up to* the cost ceiling so a one-off large negative funding
        # never masquerades as additional alpha.
        fr = float(funding_rate or 0.0)
        if ctx.direction == "short":
            funding_cost_pct = -fr * 100.0
        else:
            funding_cost_pct = fr * 100.0

        # ---- EV --------------------------------------------------------
        ev_gross = p_win * avg_win - (1.0 - p_win) * avg_loss
        ev_net = ev_gross - cost_pct - funding_cost_pct

        # ---- Kelly -----------------------------------------------------
        # Kelly = (p × b − q) / b, where b = payoff (avg_win/avg_loss).
        if payoff > 0:
            kelly_raw = max(0.0, (p_win * payoff - (1.0 - p_win)) / payoff)
        else:
            kelly_raw = 0.0
        # Hard cap on raw Kelly to avoid runaway sizing in lucky-streak regimes.
        kelly_raw = min(kelly_raw, 0.25)
        kelly_fraction = kelly_raw * self._kelly_quarter

        # ---- Confidence ------------------------------------------------
        # Confidence saturates around min_trades_for_ev.
        confidence = min(1.0, sample_size / max(1, self._min_trades_for_ev))

        rationale = (
            f"p_win={p_win:.3f} (raw={p_win if sample_size else prior_p:.3f}, "
            f"prior={prior_p:.3f}, n={sample_size}); "
            f"avg_win={avg_win:.3f}% avg_loss={avg_loss:.3f}% payoff={payoff:.2f}; "
            f"ev_gross={ev_gross:+.3f}% cost={cost_pct:.3f}% funding={funding_cost_pct:.3f}% "
            f"ev_net={ev_net:+.3f}%"
        )

        result = EVResult(
            p_win=round(p_win, 4),
            p_loss=round(1.0 - p_win, 4),
            avg_win_pct=round(avg_win, 4),
            avg_loss_pct=round(avg_loss, 4),
            payoff_ratio=round(payoff, 4),
            cost_pct=round(cost_pct, 4),
            funding_cost_pct=round(funding_cost_pct, 4),
            ev_gross_pct=round(ev_gross, 4),
            ev_net_pct=round(ev_net, 4),
            confidence=round(confidence, 3),
            trade_count=sample_size,
            kelly_raw=round(kelly_raw, 6),
            kelly_fraction=round(kelly_fraction, 6),
            rationale=rationale,
            matched_keys=[_match_key(t) for t in matched],
            p_win_floor_ok=p_win >= self._min_p_win,
            ev_floor_ok=ev_net >= self._min_ev_pct,
        )
        return result

    # ------------------------------------------------------------ internals

    @staticmethod
    def _match_samples(trade_log: Iterable, ctx: PwinContext):
        all_samples = list(trade_log or [])
        if not all_samples:
            return [], []

        regime_target = (ctx.regime or "").strip()
        sm_target = (ctx.sm_phase or "").strip()
        direction_target = (ctx.direction or "").strip()

        def _matches(t):
            t_regime = getattr(t, "regime", None) or _from_scores(t, "regime", "")
            t_sm = _from_scores(t, "sm_phase", "")
            t_dir = getattr(t, "direction", None) or ""
            return (
                t_regime == regime_target
                and t_sm == sm_target
                and t_dir == direction_target
            )

        matched = [t for t in all_samples if _matches(t)]
        return matched, all_samples

    def _bayes_shrink(self, raw_p: float, prior_p: float, sample_size: int) -> float:
        # Standard Bayes shrinkage with effective prior weight α.
        # When raw_p < prior_p we *reduce* α (loss_shrink_factor < 1) so we
        # trust realised losses faster.
        alpha = self._prior_weight
        if raw_p < prior_p:
            alpha = alpha * self._loss_shrink_factor
        return (sample_size * raw_p + alpha * prior_p) / (sample_size + alpha)


# ---- Helpers ---------------------------------------------------------------

def _pnl_pct(trade) -> float:
    try:
        return float(getattr(trade, "pnl_pct", 0.0) or 0.0)
    except Exception:
        return 0.0


def _from_scores(trade, key: str, default):
    scores = getattr(trade, "scores", None)
    if not isinstance(scores, dict):
        return default
    val = scores.get(key, default)
    return val if val is not None else default


def _match_key(trade) -> str:
    regime = getattr(trade, "regime", "") or _from_scores(trade, "regime", "")
    sm = _from_scores(trade, "sm_phase", "")
    direction = getattr(trade, "direction", "") or ""
    return f"{regime}|{sm}|{direction}"
