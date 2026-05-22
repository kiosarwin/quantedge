"""Tests for the futures-only aggressive-edge upgrade layer.

Covers:
  - SessionModulator returns a multiplier within configured bounds.
  - CorrelationFilter blocks compounding bets and allows hedging bets.
  - EdgeDetector mode mapping (observer / soft_gate / active_gate).
  - CohortPolicy.evaluate populates a sensible size_mult.
  - TradeManager early-adverse-cut fires when MAE high and MFE low,
    and is suppressed when a meaningful favourable excursion has occurred.
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.models.cohort_policy import CohortPolicy
from src.models.edge_detector import EdgeDetector
from src.risk.correlation_filter import CorrelationFilter
from src.risk.session_modulator import SessionModulator


# ---------- SessionModulator -------------------------------------------------

def test_session_modulator_returns_within_bounds():
    cfg = {
        "safety": {
            "session_aware_sizing": True,
            "session_size_multipliers": {
                "asia": 0.7,
                "london": 1.05,
                "ny": 1.05,
                "overlap_london_ny": 1.30,
            },
        }
    }
    mod = SessionModulator(cfg)
    s = mod.current()
    assert 0.5 <= s.multiplier <= 1.5
    assert s.session in {"asia", "london", "ny", "overlap_london_ny"}


def test_session_modulator_disabled_returns_unity():
    mod = SessionModulator({"safety": {"session_aware_sizing": False}})
    assert mod.current().multiplier == 1.0


# ---------- CorrelationFilter ------------------------------------------------

def _ar1_returns(n: int, phi: float = 0.0, noise: float = 0.01, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, noise, n)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return pd.Series(out)


def test_correlation_filter_blocks_compounding_long_long_bet():
    cfg = {"risk": {"correlation_filter": {"enabled": True, "lookback_bars": 60, "corr_max": 0.85}}}
    cf = CorrelationFilter(cfg)
    base = _ar1_returns(60, seed=11)
    near_copy = base + _ar1_returns(60, seed=22, noise=0.001) * 0.05  # ~0.97 corr
    decision = cf.evaluate(
        candidate_symbol="ETHUSDT",
        candidate_direction="long",
        candidate_returns=near_copy,
        open_trades={"BTCUSDT": {"direction": "long", "returns": base}},
    )
    assert decision.allowed is False
    assert decision.max_corr >= 0.85


def test_correlation_filter_allows_hedge_with_inverse_correlation():
    """If candidate is highly correlated but in the opposite direction, that's
    a hedge of existing exposure, not a compounding bet — should pass."""
    cfg = {"risk": {"correlation_filter": {"enabled": True, "lookback_bars": 60, "corr_max": 0.85}}}
    cf = CorrelationFilter(cfg)
    base = _ar1_returns(60, seed=33)
    near_copy = base * 1.0
    decision = cf.evaluate(
        candidate_symbol="ETHUSDT",
        candidate_direction="short",
        candidate_returns=near_copy,
        open_trades={"BTCUSDT": {"direction": "long", "returns": base}},
    )
    assert decision.allowed is True


def test_correlation_filter_disabled_passes_everything():
    cf = CorrelationFilter({"risk": {"correlation_filter": {"enabled": False}}})
    decision = cf.evaluate("X", "long", pd.Series(np.zeros(60)), {"Y": {"direction": "long", "returns": pd.Series(np.zeros(60))}})
    assert decision.allowed is True


# ---------- EdgeDetector mode mapping ----------------------------------------

def test_edge_detector_observer_mode():
    det = EdgeDetector({"edge_detector": {"mode": "observer"}})
    assert det.is_blocking is False
    assert det.affects_sizing is False


def test_edge_detector_soft_gate_mode():
    det = EdgeDetector({"edge_detector": {"mode": "soft_gate"}})
    assert det.is_blocking is False
    assert det.affects_sizing is True


def test_edge_detector_active_gate_mode():
    det = EdgeDetector({"edge_detector": {"mode": "active_gate"}})
    assert det.is_blocking is True
    assert det.affects_sizing is True


def test_edge_detector_legacy_enabled_flag_maps_to_active_gate():
    det = EdgeDetector({"edge_policy": {"enabled": True}})
    assert det.is_blocking is True
    assert det.affects_sizing is True


# ---------- CohortPolicy size_mult -------------------------------------------

def _bd(*, sleeve="trend_following", direction="long", regime="trending_expansion", sm="trending"):
    breakdown = SimpleNamespace(
        symbol="BTC/USDT:USDT",
        direction=direction,
        strategy_sleeve=sleeve,
        regime=SimpleNamespace(value=regime),
        smart_money=SimpleNamespace(phase=SimpleNamespace(value=sm)),
    )
    return breakdown


def test_cohort_policy_strong_attribution_yields_size_boost():
    cfg = {"edge_policy": {"enabled": True, "attribution": {"min_trades": 4}}}
    policy = CohortPolicy(cfg)
    report = {
        "by_sleeve_regime_side": [
            {
                "key": "trend_following|trending_expansion|long",
                "trades": 10,
                "win_rate": 0.7,
                "profit_factor": 2.5,
                "expectancy_usd": 4.0,
            }
        ]
    }
    decision = policy.evaluate(_bd(), [], attribution_report=report)
    assert decision.allowed is True
    assert decision.size_mult > 1.0


def test_cohort_policy_weak_attribution_yields_size_cut():
    cfg = {"edge_policy": {"enabled": True, "attribution": {"min_trades": 4}}}
    policy = CohortPolicy(cfg)
    report = {
        "by_sleeve_regime_side": [
            {
                "key": "trend_following|trending_expansion|long",
                "trades": 10,
                "win_rate": 0.30,
                "profit_factor": 0.6,
                "expectancy_usd": -1.5,
            }
        ]
    }
    decision = policy.evaluate(_bd(), [], attribution_report=report)
    # PF/WR/expectancy are bad enough to attribution-block before size_mult evaluates.
    assert decision.allowed is False
    assert "attribution block" in decision.reason


# ---------- Adverse-cut behaviour --------------------------------------------

def _make_trade_manager_for_cut(min_age_s: float = 5.0):
    """Build a TradeManager with stub deps and force one open trade ready for cut."""
    from src.execution.trade_manager import TradeManager, OpenTrade
    from src.risk.risk_manager import TradeSetup

    cfg = {
        "exit": {
            "tp1_r_multiple": 1.5,
            "tp2_r_multiple": 2.0,
            "tp1_size_pct": 0.5,
            "tp2_size_pct": 0.3,
            "trail_size_pct": 0.2,
            "breakeven_trigger_r": 1.0,
            "trailing_atr_multiplier": 1.5,
            "tp2_trailing_stop_pct": 0.15,
            "max_hold_duration_s": 86400,
            "early_cut": {
                "enabled": True,
                "min_age_s": min_age_s,
                "max_age_s": 7200,
                "mae_r_threshold": 0.65,
                "mfe_r_ceiling": 0.20,
            },
        },
        "ev_model": {"taker_fee_pct": 0.04},
    }

    risk = MagicMock()
    risk.can_open_trade.return_value = (True, "ok")
    risk.check_trade_exposure.return_value = (True, "ok")
    executor = MagicMock()
    executor.cancel_all = AsyncMock()
    executor.close_position_market = AsyncMock()
    client = MagicMock()
    mgr = TradeManager(client, executor, risk, cfg)
    setup = TradeSetup(
        symbol="BTC/USDT:USDT",
        direction="long",
        entry_price=100.0,
        stop_loss=95.0,
        tp1=110.0,
        tp2=120.0,
        tp3=130.0,
        size_contracts=1.0,
        size_usd=100.0,
        leverage=5,
        atr=1.0,
        risk_pct=1.0,
        r_distance=5.0,
        strategy_sleeve="trend_following",
        exit_profile="trend_following",
        tp1_size_pct=0.5,
        tp2_size_pct=0.3,
        trail_size_pct=0.2,
        breakeven_trigger_r=1.0,
        trailing_atr_multiplier=1.5,
        max_hold_duration_s=86400,
    )
    trade = OpenTrade(
        setup=setup,
        entry_order_id="x",
        sl_order_id="y",
        tp1_order_id="z1",
        tp2_order_id="z2",
        opened_at=time.time() - 60.0,        # 1 minute old, > min_age_s
        remaining_contracts=1.0,
    )
    mgr._trades[setup.symbol] = trade
    return mgr, trade


def test_early_adverse_cut_fires_when_mae_high_and_mfe_low():
    mgr, trade = _make_trade_manager_for_cut(min_age_s=5.0)
    # First tick to update MAE -> price 96.5 puts trade at 0.7R adverse, 0R favourable.
    asyncio.run(mgr._check_exits(trade, 96.5))
    assert trade.symbol not in mgr._trades            # closed
    mgr._executor.close_position_market.assert_awaited()


def test_early_adverse_cut_skipped_when_favourable_move_present():
    mgr, trade = _make_trade_manager_for_cut(min_age_s=5.0)
    # First tick: trade went 0.4R favourable.
    asyncio.run(mgr._check_exits(trade, 102.0))
    # Then it sells off — adverse 0.7R but mfe 0.4R already > ceiling 0.20.
    asyncio.run(mgr._check_exits(trade, 96.5))
    # Trade should still be open (no early cut).
    assert trade.symbol in mgr._trades
