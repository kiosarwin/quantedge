"""Edge-case tests for RiskManager.calculate_setup(): zero equity, zero ATR, zero entry_price."""
import pandas as pd

import src.risk.risk_manager as risk_manager_module
from src.risk.risk_manager import RiskManager


def _base_cfg():
    """Minimal config sufficient for calculate_setup()."""
    return {
        "trading": {"max_open_trades": 2},
        "risk": {
            "daily_loss_cap_pct": 5.0,
            "weekly_loss_cap_pct": 8.0,
            "max_drawdown_pct": 15.0,
            "risk_per_trade_pct": 1.0,
            "max_risk_per_trade_pct": 1.25,
            "cooldown_after_loss_streak": 2,
            "cooldown_minutes": 30,
            "max_symbol_risk_pct": 1.0,
            "max_direction_risk_pct": 1.5,
            "default_leverage": 5,
            "max_leverage": 10,
            "stop_loss_atr_multiplier": 1.5,
            "max_stop_distance_pct": 50.0,
            "min_rr_ratio": 2.0,
            "min_risk_usd": 0.0,
            "min_notional_usd": 5.0,
        },
        "exit": {
            "tp1_r_multiple": 1.5,
            "tp2_r_multiple": 2.0,
            "tp1_size_pct": 0.50,
            "tp2_size_pct": 0.30,
            "trail_size_pct": 0.20,
            "breakeven_trigger_r": 1.0,
            "trailing_atr_multiplier": 1.5,
            "max_hold_duration_s": 172800,
        },
        "safety": {"max_consecutive_losses": 5},
        "indicators": {"atr_period": 14},
    }


def test_calculate_setup_zero_equity(monkeypatch):
    """Zero equity should result in None (cannot size a position with zero capital)."""
    monkeypatch.setattr(risk_manager_module, "atr", lambda *_args, **_kwargs: 5.0)
    rm = RiskManager(_base_cfg(), exploration_mode=False)
    rm.update_equity(0.0)
    df = pd.DataFrame({"close": [100.0] * 60})

    setup = rm.calculate_setup(
        symbol="BTC/USDT:USDT",
        direction="long",
        df=df,
        entry_price=100.0,
        ev_result=None,
    )
    assert setup is None


def test_calculate_setup_zero_atr(monkeypatch):
    """Zero ATR should result in None (r_distance will be 0, failing RR check)."""
    monkeypatch.setattr(risk_manager_module, "atr", lambda *_args, **_kwargs: 0.0)
    rm = RiskManager(_base_cfg(), exploration_mode=False)
    rm.update_equity(1000.0)
    df = pd.DataFrame({"close": [100.0] * 60})

    setup = rm.calculate_setup(
        symbol="BTC/USDT:USDT",
        direction="long",
        df=df,
        entry_price=100.0,
        ev_result=None,
    )
    assert setup is None


def test_calculate_setup_zero_entry_price(monkeypatch):
    """Zero entry_price should result in None (invalid geometry for binance_alpha sector)."""
    monkeypatch.setattr(risk_manager_module, "atr", lambda *_args, **_kwargs: 5.0)
    rm = RiskManager(_base_cfg(), exploration_mode=False)
    rm.update_equity(1000.0)
    df = pd.DataFrame({"close": [100.0] * 60})

    # Use a binance_alpha sector symbol where the geometry guard is active
    setup = rm.calculate_setup(
        symbol="ESPORTS/USDT:USDT",
        direction="long",
        df=df,
        entry_price=0.0,
        ev_result=None,
    )
    assert setup is None
    assert "invalid binance_alpha price geometry" in rm.last_setup_rejection_reason
