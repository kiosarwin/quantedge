import pandas as pd

import src.risk.risk_manager as risk_manager_module
from src.models.ev_model import EVResult
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


def test_calculate_setup_scales_unproven_ev_fallback_by_probability(monkeypatch):
    monkeypatch.setattr(risk_manager_module, "atr", lambda *_args, **_kwargs: 5.0)
    cfg = {
        "trading": {"max_open_trades": 2},
        "risk": {
            "daily_loss_cap_pct": 5.0,
            "weekly_loss_cap_pct": 8.0,
            "max_drawdown_pct": 15.0,
            "risk_per_trade_pct": 1.5,
            "max_risk_per_trade_pct": 2.5,
            "cooldown_after_loss_streak": 2,
            "cooldown_minutes": 30,
            "max_symbol_risk_pct": 2.5,
            "max_direction_risk_pct": 5.5,
            "default_leverage": 5,
            "max_leverage": 10,
            "stop_loss_atr_multiplier": 1.5,
            "min_rr_ratio": 2.0,
            "min_risk_usd": 0.0,
            "min_notional_usd": 5.0,
        },
        "kelly": {
            "fraction": 0.30,
            "max_kelly_pct": 5.0,
            "min_kelly_pct": 0.2,
            "volatility_scale": False,
            "confidence_scaling": True,
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

    weak = EVResult(
        p_win=0.22,
        ev_net_pct=-1.0,
        conservative_p_win=0.0,
        conservative_ev_net_pct=-1.2,
        confidence=0.1,
        statistical_edge_ok=False,
    )
    strong_probe = EVResult(
        p_win=0.58,
        ev_net_pct=0.80,
        conservative_p_win=0.52,
        conservative_ev_net_pct=0.20,
        confidence=0.2,
        statistical_edge_ok=False,
    )

    weak_setup = rm.calculate_setup(
        symbol="WEAK/USDT:USDT",
        direction="long",
        df=df,
        entry_price=100.0,
        ev_result=weak,
    )
    strong_setup = rm.calculate_setup(
        symbol="STRONG/USDT:USDT",
        direction="long",
        df=df,
        entry_price=100.0,
        ev_result=strong_probe,
    )

    assert weak_setup is not None
    assert strong_setup is not None
    assert weak_setup.risk_pct < 0.80
    assert strong_setup.risk_pct > weak_setup.risk_pct
    assert strong_setup.risk_pct <= cfg["risk"]["risk_per_trade_pct"] * 1.10


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


def test_calculate_setup_uses_sleeve_specific_exit_profile(monkeypatch):
    monkeypatch.setattr(risk_manager_module, "atr", lambda *_args, **_kwargs: 2.0)
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
        "exit_profiles": {
            "compression_breakout": {
                "tp1_r_multiple": 1.4,
                "tp2_r_multiple": 2.3,
                "tp3_r_multiple": 3.2,
                "tp1_size_pct": 0.40,
                "tp2_size_pct": 0.35,
                "trail_size_pct": 0.25,
                "breakeven_trigger_r": 0.9,
                "trailing_atr_multiplier": 1.4,
                "max_hold_duration_s": 172800,
            }
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
        strategy_sleeve="compression_breakout",
        kelly_scale=1.0,
        ev_result=None,
    )

    assert setup is not None
    assert setup.exit_profile == "compression_breakout"
    assert setup.tp1_size_pct == 0.40
    assert setup.tp2_size_pct == 0.35
    assert setup.trailing_atr_multiplier == 1.4

def test_calculate_setup_rejects_negative_short_targets(monkeypatch):
    monkeypatch.setattr(risk_manager_module, "atr", lambda *_args, **_kwargs: 0.065176)
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
            "max_symbol_risk_pct": 1.25,
            "max_direction_risk_pct": 1.5,
            "default_leverage": 5,
            "max_leverage": 10,
            "stop_loss_atr_multiplier": 1.3,
            "max_stop_distance_pct": 50.0,
            "min_rr_ratio": 1.8,
            "min_risk_usd": 0.0,
            "min_notional_usd": 5.0,
        },
        "exit": {
            "tp1_r_multiple": 1.2,
            "tp2_r_multiple": 2.0,
            "tp1_size_pct": 0.40,
            "tp2_size_pct": 0.30,
            "trail_size_pct": 0.30,
            "breakeven_trigger_r": 0.7,
            "trailing_atr_multiplier": 1.0,
            "max_hold_duration_s": 172800,
        },
        "safety": {"max_consecutive_losses": 5},
        "indicators": {"atr_period": 14},
    }
    rm = RiskManager(cfg, exploration_mode=False)
    rm.update_equity(1000.0)
    df = pd.DataFrame({"close": [0.0622] * 60})

    setup = rm.calculate_setup(
        symbol="ESPORTS/USDT:USDT",
        direction="short",
        df=df,
        entry_price=0.0622,
        strategy_sleeve="reversal",
        ev_result=None,
    )

    assert setup is None
    assert "invalid binance_alpha price geometry" in rm.last_setup_rejection_reason


def _risk_setup_cfg(*, max_stop_distance_pct=50.0):
    return {
        "trading": {"max_open_trades": 1},
        "risk": {
            "daily_loss_cap_pct": 5.0,
            "weekly_loss_cap_pct": 8.0,
            "max_drawdown_pct": 15.0,
            "risk_per_trade_pct": 1.0,
            "max_risk_per_trade_pct": 1.25,
            "cooldown_after_loss_streak": 2,
            "cooldown_minutes": 30,
            "max_symbol_risk_pct": 1.25,
            "max_direction_risk_pct": 1.5,
            "default_leverage": 5,
            "max_leverage": 10,
            "stop_loss_atr_multiplier": 1.0,
            "max_stop_distance_pct": max_stop_distance_pct,
            "min_rr_ratio": 1.8,
            "min_risk_usd": 0.0,
            "min_notional_usd": 5.0,
        },
        "exit": {
            "tp1_r_multiple": 1.0,
            "tp2_r_multiple": 2.0,
            "tp1_size_pct": 0.40,
            "tp2_size_pct": 0.30,
            "trail_size_pct": 0.30,
            "breakeven_trigger_r": 0.8,
            "trailing_atr_multiplier": 1.0,
            "max_hold_duration_s": 172800,
        },
        "safety": {"max_consecutive_losses": 5},
        "indicators": {"atr_period": 14},
    }


def test_calculate_setup_rejects_non_finite_binance_alpha_geometry(monkeypatch):
    monkeypatch.setattr(risk_manager_module, "atr", lambda *_args, **_kwargs: float("nan"))
    rm = RiskManager(_risk_setup_cfg(), exploration_mode=False)
    rm.update_equity(1000.0)
    df = pd.DataFrame({"close": [100.0] * 60})

    setup = rm.calculate_setup(
        symbol="ESPORTS/USDT:USDT",
        direction="long",
        df=df,
        entry_price=100.0,
        ev_result=None,
    )

    assert setup is None
    assert "invalid binance_alpha price geometry" in rm.last_setup_rejection_reason


def test_calculate_setup_rejects_extreme_binance_alpha_stop_distance(monkeypatch):
    monkeypatch.setattr(risk_manager_module, "atr", lambda *_args, **_kwargs: 40.0)
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
            "max_symbol_risk_pct": 1.25,
            "max_direction_risk_pct": 1.5,
            "default_leverage": 5,
            "max_leverage": 10,
            "stop_loss_atr_multiplier": 1.0,
            "max_stop_distance_pct": 35.0,
            "min_rr_ratio": 1.8,
            "min_risk_usd": 0.0,
            "min_notional_usd": 5.0,
        },
        "exit": {
            "tp1_r_multiple": 1.0,
            "tp2_r_multiple": 2.0,
            "tp1_size_pct": 0.40,
            "tp2_size_pct": 0.30,
            "trail_size_pct": 0.30,
            "breakeven_trigger_r": 0.8,
            "trailing_atr_multiplier": 1.0,
            "max_hold_duration_s": 172800,
        },
        "safety": {"max_consecutive_losses": 5},
        "indicators": {"atr_period": 14},
    }
    rm = RiskManager(cfg, exploration_mode=False)
    rm.update_equity(1000.0)
    df = pd.DataFrame({"close": [100.0] * 60})

    setup = rm.calculate_setup(
        symbol="ESPORTS/USDT:USDT",
        direction="long",
        df=df,
        entry_price=100.0,
        ev_result=None,
    )

    assert setup is None
    assert "binance_alpha stop distance" in rm.last_setup_rejection_reason


def test_calculate_setup_does_not_apply_alpha_stop_cap_to_other_sectors(monkeypatch):
    monkeypatch.setattr(risk_manager_module, "atr", lambda *_args, **_kwargs: 40.0)
    rm = RiskManager(_risk_setup_cfg(max_stop_distance_pct=35.0), exploration_mode=False)
    rm.update_equity(1000.0)
    df = pd.DataFrame({"close": [100.0] * 60})

    setup = rm.calculate_setup(
        symbol="BTC/USDT:USDT",
        direction="long",
        df=df,
        entry_price=100.0,
        ev_result=None,
    )

    assert setup is not None
    assert rm.last_setup_rejection_reason == ""
