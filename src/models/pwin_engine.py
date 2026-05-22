"""
P(win) priors and posterior adjustments.

This module is intentionally small — most heavy lifting lives in `ev_model.py`,
which owns sample-based P(win) computation.  The dataclass below is the
context object that travels alongside scoring → EV → sizing.

The class also offers a light helper to compute a context-prior P(win) when
the trade log is empty.  These priors are based on widely-published outcome
frequencies for the corresponding setups; they should never be used as a
standalone gate, only as a Bayesian starting point that the EV model blends
with realised outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass

# Public — keep stable for tests + type hints.
__all__ = ["PwinContext", "context_prior"]


@dataclass(frozen=True)
class PwinContext:
    """All inputs the EV model needs to derive a setup-specific P(win)."""

    regime: str = "trending_expansion"
    sm_phase: str = "neutral"
    sm_aligned: bool = False
    direction: str = "long"
    structure_quality: float = 50.0
    volatility_score: float = 50.0
    trend_strength: float = 50.0
    btc_state: str = "unknown"
    btcd_state: str = "unknown"


def _clip(value: float, lo: float = 0.05, hi: float = 0.95) -> float:
    return max(lo, min(hi, value))


def context_prior(ctx: PwinContext) -> float:
    """
    Deterministic prior P(win) used when the trade log is too small.

    The mapping is conservative on purpose: the prior centres at 0.50 and only
    drifts ± 0.10 based on regime + smart-money alignment + structure quality.
    The EV model still applies its own min_p_win / min_ev_pct gates on top.
    """
    p = 0.50

    # Regime nudge
    p += {
        "trending_expansion": 0.04,
        "accumulation_compression": 0.01,
        "distribution": -0.02,
        "chaos": -0.07,
    }.get(ctx.regime, 0.0)

    # Smart money nudge
    if ctx.sm_phase in {"trending", "accumulation"} and ctx.sm_aligned:
        p += 0.05
    elif ctx.sm_phase == "liquidity_sweep" and ctx.sm_aligned:
        p += 0.03
    elif ctx.sm_phase == "distribution" and not ctx.sm_aligned:
        p -= 0.04
    elif ctx.sm_phase == "chaos":
        p -= 0.05

    # Structure / trend / volatility quality (each contributes ± 0.04 max)
    p += (ctx.structure_quality - 50.0) / 50.0 * 0.04
    p += (ctx.trend_strength - 50.0) / 50.0 * 0.03
    # Mid-range volatility is best.  Treat 50 as "ideal", penalise extremes.
    vol_dev = abs(ctx.volatility_score - 50.0) / 50.0
    p -= vol_dev * 0.03

    # BTC macro context — only mild adjustment, the scorer already handles it.
    if ctx.direction == "long" and ctx.btc_state == "bear":
        p -= 0.02
    elif ctx.direction == "short" and ctx.btc_state == "bull":
        p -= 0.02

    return _clip(p)
