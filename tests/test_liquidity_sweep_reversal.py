from types import SimpleNamespace

import pandas as pd

from src.analysis.liquidity_sweep_reversal import detect_liquidity_sweep_reversal
from src.models.cohort_policy import CohortPolicy
from src.models.strategy_router import DispersionState, StrategyRouter
from src.reporting.attribution import build_attribution_report


def _cfg():
    return {
        "strategy": {
            "trend_min_score": 65,
            "trend_min_structure": 58,
            "trend_long_only": False,
            "trend_require_participation": True,
            "trend_min_alignment": 0.54,
            "breakout_min_alignment": 0.57,
            "reversal_min_alignment": 0.45,
            "enable_compression_breakout": False,
            "reversal_max_volatility": 80,
            "allow_long_reversal": True,
            "allow_short_reversal": True,
            "short_reversal_require_distribution": True,
            "short_reversal_min_sm_score": 85,
            "short_reversal_min_structure": 62,
            "short_reversal_min_volume": 45,
            "enable_short_setups": True,
            "short_setup_min_confidence": 0.65,
            "short_setup_min_structure": 50,
            "short_setup_max_volatility": 85,
            "mtf_price_action_continuation": {"enabled": True, "min_quality_score": 72.0},
            "vwap_pullback_continuation": {"enabled": True, "min_quality_score": 74.0},
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
                "max_volatility": 80.0,
            },
        },
        "edge_policy": {
            "enabled": True,
            "allowed_regimes": ["trending_expansion", "distribution"],
            "allowed_strategy_sleeves": ["reversal"],
            "allowed_sm_phases": ["neutral", "liquidity_sweep"],
            "attribution": {"min_trades": 2, "block_below_pf": 0.85, "block_below_wr": 0.35},
        },
    }


def _df_from_closes(closes, volume=1000.0):
    rows = []
    for close in closes:
        open_ = close * 1.002
        rows.append({
            "open": open_,
            "high": max(open_, close) * 1.003,
            "low": min(open_, close) * 0.997,
            "close": close,
            "volume": volume,
        })
    return pd.DataFrame(rows)


def _bearish_sweep_df():
    closes = [100 + (i % 6) * 0.08 for i in range(90)]
    df = _df_from_closes(closes, volume=1000.0)
    recent_high = float(df.iloc[-26:-2]["high"].max())
    df.loc[df.index[-2], "high"] = recent_high * 1.004
    df.loc[df.index[-2], "close"] = recent_high * 1.001
    df.loc[df.index[-1], "open"] = recent_high * 1.002
    df.loc[df.index[-1], "high"] = recent_high * 1.003
    df.loc[df.index[-1], "low"] = recent_high * 0.988
    df.loc[df.index[-1], "close"] = recent_high * 0.991
    df.loc[df.index[-1], "volume"] = 1800.0
    return df


def _breakdown(signal):
    return SimpleNamespace(
        symbol="SOL/USDT:USDT",
        direction="short",
        regime=SimpleNamespace(value="distribution"),
        smart_money=SimpleNamespace(phase=SimpleNamespace(value="liquidity_sweep"), direction_bias="short", score=82.0),
        structure_quality=62.0,
        volume_confirmation=48.0,
        open_interest=48.0,
        funding_sentiment=50.0,
        volatility=42.0,
        trend_strength=42.0,
        short_setup=None,
        mtf_price_action=None,
        vwap_pullback=None,
        liquidity_sweep_reversal=signal,
        feature_vector=SimpleNamespace(
            market_structure="bearish_bos",
            order_flow_imbalance=-22.0,
            momentum_strength=-18.0,
            vwap_distance=-0.25,
            liquidation_pressure=-12.0,
            directional_alignment=lambda _direction: 0.62,
        ),
    )


def test_detect_liquidity_sweep_reversal_bearish_reclaim():
    signal = detect_liquidity_sweep_reversal(_bearish_sweep_df(), _cfg(), direction_hint="none")

    assert signal.is_valid is True
    assert signal.direction == "short"
    assert signal.label == "liquidity_sweep_reversal"
    assert signal.trigger == "buy_side_sweep_reversal"
    assert signal.quality_score >= 78.0


def test_strategy_router_routes_liquidity_sweep_reversal_as_isolated_reversal_setup():
    router = StrategyRouter(_cfg())
    signal = SimpleNamespace(is_valid=True, direction="short", quality_score=84.0)

    decision = router.evaluate(_breakdown(signal), DispersionState(value=0.0, state="normal"))

    assert decision.sleeve == "reversal"
    assert decision.setup_type == "liquidity_sweep_reversal"
    assert decision.passport.setup_type == "liquidity_sweep_reversal"
    assert decision.passport.expected_path == "snapback_then_follow_through"


def test_cohort_policy_isolates_liquidity_sweep_reversal_from_generic_reversal_history():
    losing_generic = [
        SimpleNamespace(
            symbol="OLD/USDT:USDT",
            direction="short",
            regime="distribution",
            strategy_sleeve="reversal",
            setup_type="sweep_reversal",
            scores={"sm_phase": "liquidity_sweep", "setup_type": "sweep_reversal"},
            pnl_usd=-1.0,
            pnl_pct=-1.0,
        )
        for _ in range(3)
    ]
    report = build_attribution_report(losing_generic, min_trades=2)
    breakdown = _breakdown(SimpleNamespace(is_valid=True, direction="short", quality_score=84.0))
    breakdown.strategy_sleeve = "reversal"
    breakdown.setup_type = "liquidity_sweep_reversal"

    decision = CohortPolicy(_cfg()).evaluate(breakdown, [], report)

    assert decision.allowed is True
    assert decision.cohort_key == "liquidity_sweep_reversal|distribution|short"


def test_strategy_router_blocks_liquidity_sweep_reversal_during_high_dispersion():
    router = StrategyRouter(_cfg())
    signal = SimpleNamespace(is_valid=True, direction="short", quality_score=84.0)

    decision = router.evaluate(_breakdown(signal), DispersionState(value=41.0, state="high"))

    assert decision.setup_type != "liquidity_sweep_reversal"
