import pandas as pd

import src.risk.risk_manager as risk_manager_module
from src.risk.risk_manager import RiskManager


def _cfg():
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
        },
        "exit": {},
        "safety": {"max_consecutive_losses": 5},
    }


def test_risk_manager_blocks_weekly_loss_cap():
    rm = RiskManager(_cfg(), exploration_mode=False)
    rm.update_equity(1000.0)
    rm.state.weekly_start_equity = 1000.0
    rm.state.daily_start_equity = 910.0
    rm.state.equity = 910.0
    allowed, reason = rm.can_open_trade()
    assert allowed is False
    assert "weekly loss cap" in reason


def test_risk_manager_triggers_cooldown_after_loss_streak():
    rm = RiskManager(_cfg(), exploration_mode=False)
    rm.update_equity(1000.0)
    rm.on_trade_opened(symbol="BTC/USDT:USDT", direction="long", risk_pct=0.5)
    rm.on_trade_closed(-10.0, symbol="BTC/USDT:USDT", direction="long", risk_pct=0.5)
    rm.on_trade_opened(symbol="ETH/USDT:USDT", direction="long", risk_pct=0.5)
    rm.on_trade_closed(-12.0, symbol="ETH/USDT:USDT", direction="long", risk_pct=0.5)
    allowed, reason = rm.can_open_trade()
    assert allowed is False
    assert "cooldown active" in reason


def test_risk_manager_blocks_symbol_exposure():
    rm = RiskManager(_cfg(), exploration_mode=False)
    rm.update_equity(1000.0)
    rm.on_trade_opened(symbol="BTC/USDT:USDT", direction="long", risk_pct=0.8)
    allowed, reason = rm.check_trade_exposure("BTC/USDT:USDT", "long", 0.3)
    assert allowed is False
    assert "symbol exposure cap" in reason


def test_risk_manager_treats_zero_max_consecutive_losses_as_disabled():
    cfg = _cfg()
    cfg["safety"]["max_consecutive_losses"] = 0
    rm = RiskManager(cfg, exploration_mode=False)
    rm.update_equity(1000.0)
    rm.state.consecutive_losses = 2
    allowed, reason = rm.can_open_trade()
    assert allowed is True
    assert reason == "ok"


def test_calculate_setup_bumps_small_trade_to_minimum_notional(monkeypatch):
    monkeypatch.setattr(risk_manager_module, "atr", lambda *_args, **_kwargs: 5.0)
    cfg = {
        "trading": {"max_open_trades": 2},
        "risk": {
            "daily_loss_cap_pct": 5.0,
            "weekly_loss_cap_pct": 8.0,
            "max_drawdown_pct": 15.0,
            "risk_per_trade_pct": 0.1,
            "max_risk_per_trade_pct": 2.0,
            "cooldown_after_loss_streak": 2,
            "cooldown_minutes": 30,
            "max_symbol_risk_pct": 1.0,
            "max_direction_risk_pct": 1.5,
            "default_leverage": 5,
            "max_leverage": 10,
            "stop_loss_atr_multiplier": 1.5,
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
    rm = RiskManager(cfg, exploration_mode=False)
    rm.update_equity(1000.0)
    df = pd.DataFrame({"close": [100.0] * 60})

    setup = rm.calculate_setup(
        symbol="BTC/USDT:USDT",
        direction="long",
        df=df,
        entry_price=100.0,
        kelly_scale=0.2,
        ev_result=None,
    )

    assert setup is not None
    assert setup.size_usd >= 5.0
    assert rm.last_setup_rejection_reason == ""


def test_calculate_setup_exposes_specific_rejection_reason(monkeypatch):
    monkeypatch.setattr(risk_manager_module, "atr", lambda *_args, **_kwargs: 10.0)
    cfg = {
        "trading": {"max_open_trades": 1},
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
            "min_rr_ratio": 2.0,
            "min_risk_usd": 3.0,
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
    rm = RiskManager(cfg, exploration_mode=False)
    rm.update_equity(100.0)
    df = pd.DataFrame({"close": [100.0] * 60})

    setup = rm.calculate_setup(
        symbol="BTC/USDT:USDT",
        direction="long",
        df=df,
        entry_price=100.0,
        kelly_scale=0.1,
        ev_result=None,
    )

    assert setup is None
    assert "min_risk_usd" in rm.last_setup_rejection_reason


def test_calculate_setup_allows_rr_at_threshold_with_float_noise(monkeypatch):
    monkeypatch.setattr(risk_manager_module, "atr", lambda *_args, **_kwargs: 1.0000000001)
    cfg = {
        "trading": {"max_open_trades": 1},
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
    rm = RiskManager(cfg, exploration_mode=False)
    rm.update_equity(1000.0)
    df = pd.DataFrame({"close": [100.0] * 60})

    setup = rm.calculate_setup(
        symbol="BTC/USDT:USDT",
        direction="long",
        df=df,
        entry_price=100.0,
        kelly_scale=1.0,
        ev_result=None,
    )

    assert setup is not None
    assert rm.last_setup_rejection_reason == ""
