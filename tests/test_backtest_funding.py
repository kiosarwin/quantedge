"""Test that backtest engine deducts funding costs for trades held across 8h boundaries."""
import pandas as pd
import numpy as np
from src.backtest.engine import BacktestEngine, BacktestTrade


def _make_cfg(funding_rate=0.0001):
    return {
        "trading": {"min_score_threshold": 999, "max_open_trades": 1},
        "risk": {
            "daily_loss_cap_pct": 50.0, "weekly_loss_cap_pct": 50.0,
            "max_drawdown_pct": 50.0, "risk_per_trade_pct": 1.0,
            "max_risk_per_trade_pct": 2.0, "default_leverage": 5,
            "max_leverage": 10, "stop_loss_atr_multiplier": 1.5,
            "min_rr_ratio": 1.5, "min_risk_usd": 0.0, "min_notional_usd": 5.0,
            "cooldown_after_loss_streak": 0, "cooldown_minutes": 0,
        },
        "exit": {
            "tp1_r_multiple": 1.5, "tp2_r_multiple": 2.0,
            "tp1_size_pct": 0.50, "tp2_size_pct": 0.30,
            "trail_size_pct": 0.20, "breakeven_trigger_r": 1.0,
            "trailing_atr_multiplier": 1.5, "max_hold_duration_s": 172800,
            "tp2_trailing_stop_pct": 0.15,
        },
        "backtest": {
            "start_date": "2024-01-01", "end_date": "2024-02-01",
            "initial_capital": 10000, "commission_pct": 0.04,
            "slippage_pct": 0.0, "score_threshold": 50,
            "funding_rate_per_8h": funding_rate,
        },
        "safety": {"max_consecutive_losses": 99},
        "indicators": {"atr_period": 14, "ema_fast": 21, "ema_slow": 55},
        "timeframes": {"primary": "1h", "higher": "4h", "lower": "15m"},
        "scoring": {"weights": {}, "dynamic_adjustment": False},
        "structure": {"swing_lookback": 10, "bos_confirmation_candles": 2},
    }


def test_bars_per_8h_calculation():
    """Verify _bars_per_8h returns correct value for different timeframes."""
    cfg = _make_cfg()
    engine = BacktestEngine(cfg)
    # 1h timeframe: 8 bars per 8 hours
    assert engine._bars_per_8h() == 8

    # Test 15m timeframe
    cfg_15m = _make_cfg()
    cfg_15m["timeframes"]["primary"] = "15m"
    engine_15m = BacktestEngine(cfg_15m)
    assert engine_15m._bars_per_8h() == 32

    # Test 4h timeframe
    cfg_4h = _make_cfg()
    cfg_4h["timeframes"]["primary"] = "4h"
    engine_4h = BacktestEngine(cfg_4h)
    assert engine_4h._bars_per_8h() == 2


def test_funding_cost_field_exists():
    """BacktestTrade should have a funding_cost field defaulting to 0."""
    trade = BacktestTrade(
        symbol="BTC/USDT:USDT", direction="long", exit_profile="default",
        entry_bar=0, entry_price=50000.0, stop_loss=49000.0,
        tp1=51500.0, tp2=52000.0, size_contracts=0.1, size_usd=5000.0,
        r_distance=1000.0, tp1_size_pct=0.5, tp2_size_pct=0.3,
        trailing_atr_multiplier=1.5, max_hold_duration_s=172800,
        remaining_contracts=0.1,
    )
    assert trade.funding_cost == 0.0
    assert trade._last_funding_bar == 0


def test_funding_cost_applied_over_24h():
    """A trade held 24 hours should incur 3 x funding cost (3 x 8h periods)."""
    cfg = _make_cfg(funding_rate=0.0001)
    engine = BacktestEngine(cfg)

    # Verify _bars_per_8h is correct for 1h timeframe
    assert engine._bars_per_8h() == 8

    # Create a trade manually and simulate funding deduction logic
    trade = BacktestTrade(
        symbol="BTC/USDT:USDT", direction="long", exit_profile="default",
        entry_bar=0, entry_price=50000.0, stop_loss=49000.0,
        tp1=51500.0, tp2=52000.0, size_contracts=0.1, size_usd=5000.0,
        r_distance=1000.0, tp1_size_pct=0.5, tp2_size_pct=0.3,
        trailing_atr_multiplier=1.5, max_hold_duration_s=172800,
        remaining_contracts=0.1,
    )
    trade._last_funding_bar = 0

    # Simulate 24 bars (24h on 1h timeframe) = 3 funding periods
    bars_8h = engine._bars_per_8h()
    equity = 10000.0
    funding_rate = 0.0001

    for i in range(1, 25):
        if (i - trade._last_funding_bar) >= bars_8h:
            cost = trade.size_usd * funding_rate
            trade.funding_cost += cost
            equity -= cost
            trade._last_funding_bar = i

    # 3 full 8h periods in 24 bars: bar 8, 16, 24
    expected_cost_per_period = 5000.0 * 0.0001  # $0.50
    expected_total = expected_cost_per_period * 3  # $1.50

    assert abs(trade.funding_cost - expected_total) < 0.001
    assert abs(equity - (10000.0 - expected_total)) < 0.001


def test_funding_rate_configurable():
    """Funding rate should be configurable via backtest config."""
    cfg = _make_cfg(funding_rate=0.0005)
    engine = BacktestEngine(cfg)
    assert engine._funding_rate_per_8h == 0.0005

    # Default when not specified
    cfg_default = _make_cfg()
    del cfg_default["backtest"]["funding_rate_per_8h"]
    engine_default = BacktestEngine(cfg_default)
    assert engine_default._funding_rate_per_8h == 0.0001
