from types import SimpleNamespace

import pandas as pd

from src.analysis.mtf_price_action import detect_mtf_price_action_continuation
from src.models.strategy_router import DispersionState, StrategyRouter
from src.reporting.attribution import build_attribution_report
from src.models.cohort_policy import CohortPolicy


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
            "mtf_price_action_continuation": {
                "enabled": True,
                "min_quality_score": 72.0,
                "require_participation": True,
                "block_high_dispersion": True,
                "breakout_lookback": 20,
                "pullback_lookback": 8,
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


def _ohlc(closes):
    rows = []
    for close in closes:
        open_ = close * 0.995
        rows.append({"open": open_, "high": close * 1.005, "low": open_ * 0.995, "close": close, "volume": 1000})
    return pd.DataFrame(rows)


def _long_primary():
    closes = [100 + i * 0.15 for i in range(90)]
    closes[-9:-1] = [111.0, 110.8, 110.6, 110.7, 110.9, 111.0, 111.2, 111.4]
    df = _ohlc(closes)
    prior_high = float(df.iloc[-21:-1]["high"].max())
    df.loc[df.index[-1], "open"] = prior_high * 0.998
    df.loc[df.index[-1], "close"] = prior_high * 1.01
    df.loc[df.index[-1], "high"] = prior_high * 1.012
    df.loc[df.index[-1], "low"] = prior_high * 0.995
    return df


def _breakdown(mtf_signal):
    return SimpleNamespace(
        symbol="SOL/USDT:USDT",
        direction="long",
        regime=SimpleNamespace(value="trending_expansion"),
        smart_money=SimpleNamespace(phase=SimpleNamespace(value="neutral"), direction_bias="neutral", score=50.0),
        structure_quality=59.0,
        volume_confirmation=50.0,
        open_interest=50.0,
        funding_sentiment=50.0,
        volatility=30.0,
        trend_strength=60.0,
        short_setup=None,
        mtf_price_action=mtf_signal,
        feature_vector=SimpleNamespace(
            market_structure="bullish_bos",
            order_flow_imbalance=20.0,
            momentum_strength=25.0,
            vwap_distance=0.5,
            liquidation_pressure=20.0,
            directional_alignment=lambda _direction: 0.66,
        ),
    )


def test_detect_mtf_price_action_continuation_long_breakout():
    primary = _long_primary()
    higher = _ohlc([90 + i * 0.25 for i in range(70)])
    lower = _ohlc([112, 112.1, 112.2, 112.25, 112.3, 112.35, 112.4, 112.6])

    signal = detect_mtf_price_action_continuation(primary, higher, lower, _cfg(), direction_hint="long")

    assert signal.is_valid is True
    assert signal.direction == "long"
    assert signal.label == "mtf_price_action_continuation"
    assert signal.primary_trigger == "breakout_reclaim_high"
    assert signal.quality_score >= 80.0


def test_strategy_router_routes_mtf_price_action_as_isolated_trend_setup():
    router = StrategyRouter(_cfg())
    signal = SimpleNamespace(is_valid=True, direction="long", quality_score=82.0)

    decision = router.evaluate(_breakdown(signal), DispersionState(value=0.0, state="normal"))

    assert decision.sleeve == "trend_following"
    assert decision.setup_type == "mtf_price_action_continuation"
    assert decision.passport.setup_type == "mtf_price_action_continuation"
    assert decision.passport.expected_path == "mtf_impulse_continuation"


def test_attribution_and_cohort_policy_isolate_mtf_setup_from_generic_trend_history():
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
    breakdown = _breakdown(SimpleNamespace(is_valid=True, direction="long", quality_score=82.0))
    breakdown.strategy_sleeve = "trend_following"
    breakdown.setup_type = "mtf_price_action_continuation"

    decision = CohortPolicy(_cfg()).evaluate(breakdown, [], report)

    assert decision.allowed is True
    assert decision.cohort_key == "mtf_price_action_continuation|trending_expansion|long"
    assert report["by_setup_type"][0]["key"] == "trend_continuation"


def test_strategy_router_blocks_mtf_price_action_when_participation_is_thin():
    router = StrategyRouter(_cfg())
    signal = SimpleNamespace(is_valid=True, direction="long", quality_score=82.0)
    breakdown = _breakdown(signal)
    breakdown.volume_confirmation = 40.0
    breakdown.open_interest = 40.0

    decision = router.evaluate(breakdown, DispersionState(value=0.0, state="normal"))

    assert decision.sleeve == "neutral"
    assert decision.setup_type == "no_trade"


def test_strategy_router_blocks_mtf_price_action_during_high_dispersion():
    router = StrategyRouter(_cfg())
    signal = SimpleNamespace(is_valid=True, direction="long", quality_score=82.0)

    decision = router.evaluate(_breakdown(signal), DispersionState(value=41.0, state="high"))

    assert decision.sleeve == "neutral"
    assert decision.setup_type == "no_trade"
