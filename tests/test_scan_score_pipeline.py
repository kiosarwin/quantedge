"""Integration test: Scorer.score_many() with synthetic OHLCV data should not crash."""
from src.backtest.data_loader import generate_synthetic_ohlcv
from src.data.market_data import MarketSnapshot
from src.scoring.scorer import Scorer


def _minimal_scorer_cfg():
    """Minimal config dict for Scorer instantiation, based on config.yaml structure."""
    return {
        "trading": {
            "mode": "paper",
            "exploration_mode": False,
            "min_score_threshold": 65,
            "max_open_trades": 2,
        },
        "scoring": {
            "weights": {
                "trend_strength": 20,
                "volume_confirmation": 15,
                "structure_quality": 20,
                "open_interest": 15,
                "funding_sentiment": 10,
                "order_book": 10,
                "volatility": 10,
            },
            "dynamic_adjustment": True,
            "adjustment_rate": 0.02,
        },
        "timeframes": {
            "primary": "1h",
            "higher": "4h",
            "lower": "15m",
            "entry": "5m",
        },
        "indicators": {
            "atr_period": 14,
            "adx_period": 14,
            "ema_fast": 21,
            "ema_slow": 55,
            "ema_trend": 200,
            "rsi_period": 14,
            "volume_spike_multiplier": 2.0,
            "volume_lookback": 20,
            "oi_change_threshold_pct": 1.0,
            "funding_high_threshold": 0.03,
            "funding_low_threshold": -0.01,
            "btc_correlation_lookback": 48,
            "min_btc_correlation": 0.5,
        },
        "structure": {
            "swing_lookback": 10,
            "bos_confirmation_candles": 2,
            "liquidity_sweep_pct": 0.3,
        },
        "order_book": {
            "depth_levels": 10,
            "imbalance_threshold": 1.5,
            "wall_size_multiplier": 5.0,
            "absorption_volume_mult": 3.0,
        },
        "smart_money": {
            "oi_accumulation_threshold_pct": 0.2,
            "oi_distribution_threshold_pct": -0.1,
            "oi_baseline_seconds": 3600,
            "price_stagnation_pct": 0.5,
            "funding_extreme_threshold": 0.05,
            "volume_buildup_periods": 5,
            "min_smart_money_score": 60,
            "ls_long_extreme": 0.70,
            "ls_short_extreme": 0.30,
        },
        "strategy": {
            "trend_min_score": 58,
            "trend_min_structure": 52,
            "trend_long_only": False,
            "trend_require_participation": True,
            "trend_min_alignment": 0.54,
            "breakout_min_alignment": 0.57,
            "reversal_min_alignment": 0.45,
            "enable_compression_breakout": True,
            "compression_min_structure": 55,
            "reversal_max_volatility": 80,
            "allow_long_reversal": True,
            "allow_short_reversal": True,
            "short_reversal_require_distribution": True,
            "short_reversal_min_sm_score": 78,
            "short_reversal_min_structure": 58,
            "short_reversal_min_volume": 35,
            "enable_short_setups": False,
            "short_setup_min_confidence": 0.65,
            "short_setup_min_structure": 50,
            "short_setup_max_volatility": 85,
            "dispersion_warn": 28,
            "dispersion_high": 38,
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
            "mtf_price_action_continuation": {
                "enabled": True,
                "min_quality_score": 72.0,
                "require_participation": True,
                "block_high_dispersion": True,
                "breakout_lookback": 20,
                "pullback_lookback": 8,
            },
            "vwap_pullback_continuation": {
                "enabled": True,
                "min_quality_score": 74.0,
                "require_participation": True,
                "block_high_dispersion": True,
                "vwap_period": 20,
                "pullback_lookback": 6,
                "min_volume_ratio": 1.15,
                "max_atr_distance": 0.85,
            },
            "liquidity_sweep_reversal": {
                "enabled": True,
                "min_quality_score": 78.0,
                "require_participation": True,
                "block_high_dispersion": True,
                "lookback_bars": 24,
                "confirm_bars": 2,
                "pierce_min_pct": 0.0015,
                "reclaim_buffer_atr": 0.10,
                "min_volume_ratio": 1.20,
                "max_volatility": 80,
            },
        },
        "risk": {
            "risk_per_trade_pct": 1.0,
            "max_risk_per_trade_pct": 1.25,
            "daily_loss_cap_pct": 5.0,
            "weekly_loss_cap_pct": 8.0,
            "max_drawdown_pct": 15.0,
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
        "ev_model": {
            "gate_enabled": False,
            "statistical_gate_enabled": False,
            "min_trades_for_ev": 20,
            "taker_fee_pct": 0.04,
            "maker_fee_pct": 0.02,
            "slippage_pct": 0.10,
            "min_ev_pct": 0.03,
        },
        "edge_detector": {
            "mode": "observer",
            "min_sample_for_gate": 40,
            "global_baseline_winrate": 0.50,
        },
        "market_context": {
            "enabled": False,
        },
        "time_series": {
            "enabled": False,
        },
    }


def test_score_many_with_synthetic_data_does_not_crash():
    """Scorer.score_many() with synthetic OHLCV data should return a list without raising."""
    cfg = _minimal_scorer_cfg()
    scorer = Scorer(cfg)

    # Generate synthetic 1h and 4h data for a symbol
    symbol = "BTC/USDT:USDT"
    df_1h = generate_synthetic_ohlcv(symbol, "1h", "2024-01-01", "2024-02-01", seed=42)
    df_4h = generate_synthetic_ohlcv(symbol, "4h", "2024-01-01", "2024-02-01", seed=42)
    df_15m = generate_synthetic_ohlcv(symbol, "15m", "2024-01-01", "2024-02-01", seed=42)
    df_5m = generate_synthetic_ohlcv(symbol, "5m", "2024-01-01", "2024-02-01", seed=42)

    snapshot = MarketSnapshot(
        symbol=symbol,
        candles={
            "1h": df_1h,
            "4h": df_4h,
            "15m": df_15m,
            "5m": df_5m,
        },
        last_price=float(df_1h["close"].iloc[-1]),
    )

    results = scorer.score_many({symbol: snapshot})
    assert isinstance(results, list)


def test_score_many_empty_snapshots():
    """Scorer.score_many() with empty dict should return empty list without raising."""
    cfg = _minimal_scorer_cfg()
    scorer = Scorer(cfg)

    results = scorer.score_many({})
    assert results == []
