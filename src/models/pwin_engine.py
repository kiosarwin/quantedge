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
    sector: str = "unknown"
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

    UPGRADED: Direction-aware regime priors. In perpetual futures, short in
    distribution is a HIGH-PROBABILITY setup (not penalized). The prior now
    reflects empirical edge per (regime × direction) combination.

    The EV model still applies its own min_p_win / min_ev_pct gates on top.
    """
    p = 0.50

    # Direction-aware regime nudge — the key insight is that regime quality
    # depends on WHICH direction you're trading:
    # - Trending + long = strong edge (momentum persistence)
    # - Trending + short = strong edge (bearish momentum equally valid)
    # - Distribution + short = HIGH edge (institutional distribution → price drop)
    # - Distribution + long = negative edge (fighting distribution)
    # - Compression + either = moderate (breakout can go both ways)
    regime_direction_prior = {
        ("trending_expansion", "long"): 0.06,
        ("trending_expansion", "short"): 0.05,  # Was 0.04; short trends are valid
        ("accumulation_compression", "long"): 0.02,
        ("accumulation_compression", "short"): 0.01,
        ("distribution", "short"): 0.05,   # HIGH: distribution favors shorts heavily
        ("distribution", "long"): -0.04,   # Negative: fighting distribution
        ("chaos", "long"): -0.07,
        ("chaos", "short"): -0.06,  # Slightly less bad (shorts profit from panic)
    }
    p += regime_direction_prior.get((ctx.regime, ctx.direction), 0.0)

    # Smart money nudge — direction-aligned SM is very strong
    if ctx.sm_phase in {"trending", "accumulation"} and ctx.sm_aligned:
        p += 0.06  # Upgraded from 0.05
    elif ctx.sm_phase == "liquidity_sweep" and ctx.sm_aligned:
        p += 0.05  # Upgraded from 0.03 — sweeps are high-conviction reversals
    elif ctx.sm_phase == "distribution" and ctx.sm_aligned and ctx.direction == "short":
        p += 0.06  # NEW: distribution + short aligned = very strong
    elif ctx.sm_phase == "distribution" and not ctx.sm_aligned:
        p -= 0.04
    elif ctx.sm_phase == "chaos":
        p -= 0.05

    # Structure / trend / volatility quality (each contributes ± 0.05 max)
    p += (ctx.structure_quality - 50.0) / 50.0 * 0.05  # Upgraded from 0.04
    p += (ctx.trend_strength - 50.0) / 50.0 * 0.04     # Upgraded from 0.03
    # Mid-range volatility is best.  Treat 50 as "ideal", penalise extremes.
    vol_dev = abs(ctx.volatility_score - 50.0) / 50.0
    p -= vol_dev * 0.02  # Reduced penalty from 0.03 — moderate vol is fine

    # BTC macro context — direction-aware
    if ctx.direction == "long" and ctx.btc_state == "bear":
        p -= 0.03  # Stronger penalty for longing against BTC
    elif ctx.direction == "short" and ctx.btc_state == "bull":
        p -= 0.02
    elif ctx.direction == "short" and ctx.btc_state == "bear":
        p += 0.02  # NEW: shorting alts when BTC is bearish = bonus
    elif ctx.direction == "long" and ctx.btc_state == "bull":
        p += 0.01  # Mild bonus for aligned macro

    return _clip(p)
