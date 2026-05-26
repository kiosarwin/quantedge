from types import SimpleNamespace

import pandas as pd

from src.analysis.vwap_pullback import detect_vwap_pullback_continuation
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
            "reversal_max_volatility": 75,
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
            "vwap_pullback_continuation": {
                "enabled": True,
                "min_quality_score": 74.0,
                "require_participation": True,
                "block_high_dispersion": True,
                "vwap_period": 20,
                "pullback_lookback": 6,
                "min_volume_ratio": 1.15,
                "max_atr_distance": 1.20,
            },
        },
        "edge_policy": {
            "enabled": True,
            "allowed_regimes": ["trending_expansion"],
            "allowed_strategy_sleeves": ["trend_following"],
            "allowed_sm_phases": ["neutral", "trending"],
            "attribution": {"min_trades": 2, "block_below_pf": 0.85, "block_below_wr": 0.35},
        },
    }


def _df_from_closes(closes, volume=1000.0):
    rows = []
    for close in closes:
        open_ = close * 0.998
        rows.append({
            "open": open_,
            "high": max(open_, close) * 1.003,
            "low": min(open_, close) * 0.997,
            "close": close,
            "volume": volume,
        })
    return pd.DataFrame(rows)


def _vwap_pullback_primary():
    closes = [100 + i * 0.18 for i in range(90)]
    closes[-7:-1] = [114.8, 114.2, 113.8, 113.5, 113.4, 113.6]
    closes[-1] = 114.7
    df = _df_from_closes(closes)
    df.loc[df.index[-1], "open"] = 113.75
    df.loc[df.index[-1], "low"] = 113.55
    df.loc[df.index[-1], "high"] = 114.95
    df.loc[df.index[-1], "volume"] = 1500.0
    return df


def _breakdown(signal):
    return SimpleNamespace(
        symbol="SOL/USDT:USDT",
        direction="long",
        regime=SimpleNamespace(value="trending_expansion"),
        smart_money=SimpleNamespace(phase=SimpleNamespace(value="neutral"), direction_bias="neutral", score=50.0),
        structure_quality=61.0,
        volume_confirmation=46.0,
        open_interest=46.0,
        funding_sentiment=50.0,
        volatility=30.0,
        trend_strength=66.0,
        short_setup=None,
        mtf_price_action=None,
        vwap_pullback=signal,
        feature_vector=SimpleNamespace(
            market_structure="bullish_bos",
            order_flow_imbalance=22.0,
            momentum_strength=24.0,
            vwap_distance=0.2,
            liquidation_pressure=12.0,
            directional_alignment=lambda _direction: 0.65,
        ),
    )


def test_detect_vwap_pullback_continuation_long_reclaim():
    primary = _vwap_pullback_primary()
    higher = _df_from_closes([95 + i * 0.22 for i in range(70)])

    signal = detect_vwap_pullback_continuation(primary, higher, _cfg(), direction_hint="long")

    assert signal.is_valid is True
    assert signal.direction == "long"
    assert signal.label == "vwap_pullback_continuation"
    assert signal.reclaim_type == "vwap_bull_reclaim"
    assert signal.volume_ratio >= 1.15


def test_strategy_router_routes_vwap_pullback_as_isolated_trend_setup():
    router = StrategyRouter(_cfg())
    signal = SimpleNamespace(is_valid=True, direction="long", quality_score=84.0)

    decision = router.evaluate(_breakdown(signal), DispersionState(value=0.0, state="normal"))

    assert decision.sleeve == "trend_following"
    assert decision.setup_type == "vwap_pullback_continuation"
    assert decision.passport.setup_type == "vwap_pullback_continuation"
    assert decision.passport.expected_path == "vwap_reclaim_continuation"


def test_cohort_policy_isolates_vwap_pullback_from_generic_trend_history():
    losing_generic = [
        SimpleNamespace(
            symbol="OLD/USDT:USDT",
            direction="long",
            regime="trending_expansion",
            strategy_sleeve="trend_following",
            setup_type="trend_continuation",
            scores={"sm_phase": "neutral", "setup_type": "trend_continuation"},
            pnl_usd=-1.0,
            pnl_pct=-1.0,
        )
        for _ in range(3)
    ]
    report = build_attribution_report(losing_generic, min_trades=2)
    breakdown = _breakdown(SimpleNamespace(is_valid=True, direction="long", quality_score=84.0))
    breakdown.strategy_sleeve = "trend_following"
    breakdown.setup_type = "vwap_pullback_continuation"

    decision = CohortPolicy(_cfg()).evaluate(breakdown, [], report)

    assert decision.allowed is True
    assert decision.cohort_key == "vwap_pullback_continuation|trending_expansion|long"


def test_strategy_router_blocks_vwap_pullback_when_participation_is_thin():
    router = StrategyRouter(_cfg())
    signal = SimpleNamespace(is_valid=True, direction="long", quality_score=84.0)
    breakdown = _breakdown(signal)
    breakdown.volume_confirmation = 42.0
    breakdown.open_interest = 42.0

    decision = router.evaluate(breakdown, DispersionState(value=0.0, state="normal"))

    assert decision.sleeve == "neutral"
    assert decision.setup_type == "no_trade"


def test_strategy_router_blocks_vwap_pullback_during_high_dispersion():
    router = StrategyRouter(_cfg())
    signal = SimpleNamespace(is_valid=True, direction="long", quality_score=84.0)

    decision = router.evaluate(_breakdown(signal), DispersionState(value=41.0, state="high"))

    assert decision.sleeve == "neutral"
    assert decision.setup_type == "no_trade"
