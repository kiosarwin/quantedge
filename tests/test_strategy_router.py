from types import SimpleNamespace

from src.models.strategy_router import DispersionState, StrategyRouter


def _cfg():
    return {
        "strategy": {
            "trend_min_score": 65,
            "trend_min_structure": 58,
            "trend_long_only": True,
            "enable_compression_breakout": False,
            "reversal_max_volatility": 75,
            "allow_long_reversal": True,
            "allow_short_reversal": True,
            "short_reversal_require_distribution": True,
            "short_reversal_min_sm_score": 85,
            "short_reversal_min_structure": 62,
            "short_reversal_min_volume": 45,
            "dispersion_warn": 28,
            "dispersion_high": 38,
        }
    }


def _breakdown(
    direction="long",
    regime="trending_expansion",
    sm_phase="liquidity_sweep",
    sm_bias="long",
    sm_score=90.0,
    structure_quality=70.0,
    volume_confirmation=60.0,
    volatility=25.0,
    trend_strength=72.0,
):
    return SimpleNamespace(
        direction=direction,
        regime=SimpleNamespace(value=regime),
        smart_money=SimpleNamespace(
            phase=SimpleNamespace(value=sm_phase),
            direction_bias=sm_bias,
            score=sm_score,
        ),
        structure_quality=structure_quality,
        volume_confirmation=volume_confirmation,
        volatility=volatility,
        trend_strength=trend_strength,
    )


def test_strategy_router_allows_short_reversal_only_when_conditions_match():
    router = StrategyRouter(_cfg())
    decision = router.evaluate(
        _breakdown(direction="short", regime="distribution", sm_phase="distribution", sm_bias="short"),
        DispersionState(value=0.0, state="normal"),
    )
    assert decision.sleeve == "reversal"
    assert decision.reason.startswith("short reversal sleeve")


def test_strategy_router_blocks_short_trend_following():
    router = StrategyRouter(_cfg())
    decision = router.evaluate(
        _breakdown(direction="short", regime="trending_expansion", sm_phase="neutral", sm_bias="short"),
        DispersionState(value=0.0, state="normal"),
    )
    assert decision.sleeve == "neutral"
    assert "short side restricted" in decision.reason


def test_strategy_router_identifies_short_reversal_candidate():
    router = StrategyRouter(_cfg())
    breakdown = _breakdown(
        direction="short",
        regime="distribution",
        sm_phase="distribution",
        sm_bias="short",
    )
    assert router.is_short_reversal_candidate(breakdown) is True
