"""Unit tests for Phase D and Liq Sweep short detectors.

These detectors port the post-distribution short edges from
``kiosarwin/Futures``. The tests build small synthetic OHLCV frames that
satisfy or violate each rule individually so a regression on a specific
gate is immediately visible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.short_strategies import (
    ShortStrategy,
    detect_liq_sweep_short,
    detect_phase_d_short,
    detect_short_entry,
)


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cfg(**overrides) -> dict:
    base = {
        "indicators": {
            "atr_period": 14,
            "volume_lookback": 20,
        },
        "strategy": {
            "enable_short_setups": True,
            "short_setup_min_confidence": 0.55,
            "short_setup_phase_d_enabled": True,
            "short_setup_liq_sweep_enabled": True,
            "phase_d": {
                "support_lookback": 30,
                "distribution_range_lookback": 40,
                "distribution_range_max_pct": 0.12,
                "vol_spike_min_ratio": 1.5,
                "sl_atr_multiplier": 1.0,
            },
            "liq_sweep": {
                "lookback_bars": 20,
                "eq_tolerance_pct": 0.003,
                "pierce_min_pct": 0.002,
                "vol_spike_min_ratio": 1.3,
                "sl_atr_multiplier": 0.5,
            },
        },
    }
    for k, v in overrides.items():
        # shallow override under "strategy" section
        if k == "strategy":
            base["strategy"].update(v)
        else:
            base[k] = v
    return base


def _df_from_bars(bars: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(bars)
    expected = {"open", "high", "low", "close", "volume"}
    assert expected.issubset(df.columns)
    return df


def _ranged_distribution(n_bars: int = 60, base: float = 100.0,
                         range_pct: float = 0.05, seed: int = 7) -> list[dict]:
    """Generate a tight ranging distribution near the highs.

    range_pct is the half-width as a fraction of base.
    """
    rng = np.random.default_rng(seed)
    bars: list[dict] = []
    for _ in range(n_bars):
        mid = base * (1 + rng.uniform(-range_pct, range_pct))
        body = base * 0.002 * rng.uniform(-1, 1)
        o = mid - body / 2
        c = mid + body / 2
        h = max(o, c) + base * 0.003 * abs(rng.normal())
        l = min(o, c) - base * 0.003 * abs(rng.normal())
        bars.append({
            "open": float(o), "high": float(h), "low": float(l),
            "close": float(c), "volume": float(1_000 + rng.uniform(-100, 100)),
        })
    return bars


# ──────────────────────────────────────────────────────────────────────────────
#  Phase D
# ──────────────────────────────────────────────────────────────────────────────

def test_phase_d_fires_on_post_distribution_breakdown():
    bars = _ranged_distribution(n_bars=55)
    # Append a clean Phase D breakdown candle: high vol, close below 25th pctile,
    # bear close, deep into the prior range.
    bars.append({
        "open": 100.0, "high": 100.2, "low": 92.0, "close": 92.5,
        "volume": 4_000.0,   # ~4× average → guaranteed vol spike
    })
    df = _df_from_bars(bars)
    sig = detect_phase_d_short(df, _cfg())
    assert sig.is_valid, sig.notes
    assert sig.strategy is ShortStrategy.PHASE_D
    assert sig.entry_price == pytest.approx(92.5)
    assert sig.stop_loss > sig.entry_price
    assert "BREAK_SUPPORT" in sig.notes
    assert "VOL_SPIKE" in sig.notes


def test_phase_d_skips_when_support_holds():
    bars = _ranged_distribution(n_bars=60)
    df = _df_from_bars(bars)
    sig = detect_phase_d_short(df, _cfg())
    assert not sig.is_valid
    assert sig.strategy is ShortStrategy.NONE


def test_phase_d_skips_breakdown_without_volume_or_ema_cross():
    # Same shape as the breakdown but no vol spike and no EMA-cross trigger.
    bars = _ranged_distribution(n_bars=55)
    bars.append({
        "open": 100.0, "high": 100.2, "low": 95.0, "close": 95.4,
        "volume": 1_000.0,   # ~1× average → no spike
    })
    df = _df_from_bars(bars)
    cfg = _cfg(strategy={"phase_d": {
        "support_lookback": 30,
        "distribution_range_lookback": 40,
        "distribution_range_max_pct": 0.12,
        "vol_spike_min_ratio": 5.0,   # impossibly high → block vol_spike path
        "sl_atr_multiplier": 1.0,
    }})
    sig = detect_phase_d_short(df, cfg)
    # Without vol spike or EMA cross there is no required confirmation.
    assert not sig.is_valid


def test_phase_d_respects_disabled_flag_via_public_api():
    bars = _ranged_distribution(n_bars=55)
    bars.append({
        "open": 100.0, "high": 100.2, "low": 92.0, "close": 92.5,
        "volume": 4_000.0,
    })
    df = _df_from_bars(bars)
    cfg = _cfg(strategy={"short_setup_phase_d_enabled": False,
                          "short_setup_liq_sweep_enabled": False})
    sig = detect_short_entry(df, cfg)
    assert sig.strategy is ShortStrategy.NONE


# ──────────────────────────────────────────────────────────────────────────────
#  Liq Sweep
# ──────────────────────────────────────────────────────────────────────────────

def test_liq_sweep_fires_on_equal_high_stop_hunt():
    rng = np.random.default_rng(11)
    bars: list[dict] = []
    for i in range(25):
        # Two equal highs at indices 8 and 16 — both at exactly 100.0.
        if i in (8, 16):
            o, c, h, l = 99.5, 99.7, 100.0, 99.4
        else:
            o = 99.5 + rng.uniform(-0.3, 0.3)
            c = o + rng.uniform(-0.2, 0.2)
            h = max(o, c) + rng.uniform(0.0, 0.2)
            l = min(o, c) - rng.uniform(0.0, 0.2)
        bars.append({
            "open": float(o), "high": float(h), "low": float(l),
            "close": float(c), "volume": float(1_000),
        })
    # Sweep candle: pierces the 100.0 pool, closes back inside, big upper wick,
    # bearish body, volume spike.
    bars.append({
        "open": 99.9, "high": 100.8, "low": 99.4, "close": 99.4,
        "volume": 3_000.0,
    })
    df = _df_from_bars(bars)
    sig = detect_liq_sweep_short(df, _cfg())
    assert sig.is_valid, sig.notes
    assert sig.strategy is ShortStrategy.LIQ_SWEEP
    assert sig.entry_price == pytest.approx(99.4)
    # SL must sit above the wick high.
    assert sig.stop_loss > 100.8
    assert "SWEEP_HIGH" in sig.notes
    assert "BEAR_CLOSE" in sig.notes


def test_liq_sweep_skips_when_close_holds_above_pool():
    rng = np.random.default_rng(11)
    bars: list[dict] = []
    for i in range(25):
        if i in (8, 16):
            o, c, h, l = 99.5, 99.7, 100.0, 99.4
        else:
            o = 99.5 + rng.uniform(-0.3, 0.3)
            c = o + rng.uniform(-0.2, 0.2)
            h = max(o, c) + rng.uniform(0.0, 0.2)
            l = min(o, c) - rng.uniform(0.0, 0.2)
        bars.append({
            "open": float(o), "high": float(h), "low": float(l),
            "close": float(c), "volume": float(1_000),
        })
    # Genuine breakout — close stays above the pool. Should NOT trigger.
    bars.append({
        "open": 99.9, "high": 101.0, "low": 99.8, "close": 100.6,
        "volume": 2_500.0,
    })
    df = _df_from_bars(bars)
    sig = detect_liq_sweep_short(df, _cfg())
    assert not sig.is_valid


def test_liq_sweep_skips_without_equal_highs():
    bars: list[dict] = []
    # Strictly ascending highs that are well separated — guarantees no
    # eligible equal-high pair within `eq_tolerance_pct=0.3%`.
    for i in range(25):
        base = 90.0 + i * 1.5     # 90, 91.5, 93, ... 126 — each 1.5 apart
        bars.append({
            "open": float(base - 0.2),
            "high": float(base + 0.5),
            "low": float(base - 0.5),
            "close": float(base + 0.1),
            "volume": 1_000.0,
        })
    bars.append({
        "open": 130.0, "high": 135.0, "low": 128.0, "close": 128.5,
        "volume": 4_000.0,
    })
    df = _df_from_bars(bars)
    sig = detect_liq_sweep_short(df, _cfg())
    # No equal-high pool exists, so the sweep pattern cannot be confirmed.
    assert not sig.is_valid


# ──────────────────────────────────────────────────────────────────────────────
#  Public dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def test_detect_short_entry_returns_best_when_both_fire():
    bars = _ranged_distribution(n_bars=55)
    # Append a Phase D breakdown that ALSO sweeps an equal-high pool.
    # The two equal highs lie within the ranged distribution itself.
    bars.append({
        "open": 100.0, "high": 100.5, "low": 92.0, "close": 92.5,
        "volume": 4_000.0,
    })
    df = _df_from_bars(bars)
    sig = detect_short_entry(df, _cfg())
    assert sig.is_valid
    # Either detector winning is acceptable; the dispatcher must pick the
    # higher-confidence one and never return NONE here.
    assert sig.strategy in {ShortStrategy.PHASE_D, ShortStrategy.LIQ_SWEEP}


def test_detect_short_entry_respects_master_disable():
    bars = _ranged_distribution(n_bars=55)
    bars.append({
        "open": 100.0, "high": 100.2, "low": 92.0, "close": 92.5,
        "volume": 4_000.0,
    })
    df = _df_from_bars(bars)
    cfg = _cfg(strategy={"enable_short_setups": False})
    sig = detect_short_entry(df, cfg)
    assert sig.strategy is ShortStrategy.NONE


def test_detect_short_entry_short_dataframe_returns_none():
    df = pd.DataFrame({
        "open": [1.0] * 10, "high": [1.1] * 10, "low": [0.9] * 10,
        "close": [1.0] * 10, "volume": [1.0] * 10,
    })
    sig = detect_short_entry(df, _cfg())
    assert sig.strategy is ShortStrategy.NONE
