from types import SimpleNamespace

from src.models.cohort_policy import CohortPolicy


def _cfg():
    return {
        "trading": {"mode": "paper"},
        "edge_policy": {
            "enabled": True,
            "allowed_regimes": ["trending_expansion", "distribution"],
            "allowed_sm_phases": ["neutral", "trending", "accumulation", "liquidity_sweep", "distribution"],
            "allowed_strategy_sleeves": ["trend_following", "reversal"],
            "long_only": False,
        },
        "paper_validation": {
            "enabled": True,
            "high_conviction_score_trigger": 68.0,
        },
    }


def _breakdown(direction="long", sleeve="reversal", regime="trending_expansion", sm_phase="liquidity_sweep"):
    return SimpleNamespace(
        symbol="MITO/USDT:USDT",
        direction=direction,
        strategy_sleeve=sleeve,
        total_score=73.0,
        regime=SimpleNamespace(value=regime),
        smart_money=SimpleNamespace(phase=SimpleNamespace(value=sm_phase)),
        trend_strength=72.0,
        volume_confirmation=40.0,
        volatility=25.0,
    )


def test_cohort_policy_allows_configured_paper_reversal_override():
    policy = CohortPolicy(_cfg())
    decision = policy.evaluate(_breakdown(), [])
    assert decision.allowed is True


def test_cohort_policy_allows_short_reversal_when_strategy_matches():
    policy = CohortPolicy(_cfg())
    short_breakdown = _breakdown(
        direction="short",
        regime="distribution",
        sm_phase="distribution",
    )
    short_breakdown.smart_money.direction_bias = "short"
    decision = policy.evaluate(short_breakdown, [])
    assert decision.allowed is True


def test_cohort_policy_blocks_paper_reversal_for_symbol_with_bad_local_history():
    policy = CohortPolicy(_cfg())
    bad_history = [
        SimpleNamespace(symbol="MITO/USDT:USDT", pnl_usd=-1.0),
        SimpleNamespace(symbol="MITO/USDT:USDT", pnl_usd=-0.5),
        SimpleNamespace(symbol="MITO/USDT:USDT", pnl_usd=-0.2),
    ]
    decision = policy.evaluate(_breakdown(), bad_history)
    assert decision.allowed is False
    assert "negative cohort" in decision.reason


def test_cohort_policy_uses_attribution_for_blocking_and_bonus():
    policy = CohortPolicy(_cfg())
    breakdown = _breakdown(sleeve="trend_following", regime="trending_expansion", sm_phase="trending")
    report = {
        "by_sleeve_regime_side": [
            {
                "key": "trend_following|trending_expansion|long",
                "trades": 8,
                "win_rate": 0.25,
                "profit_factor": 0.40,
                "expectancy_usd": -1.5,
            }
        ],
        "by_regime_side": [],
        "by_sleeve": [],
        "by_side": [],
    }
    decision = policy.evaluate(breakdown, [], attribution_report=report)
    assert decision.allowed is False
    assert "attribution block" in decision.reason


def test_cohort_policy_attribution_relief_and_ranking_bonus():
    policy = CohortPolicy(_cfg())
    breakdown = _breakdown(sleeve="trend_following", regime="trending_expansion", sm_phase="trending")
    report = {
        "by_sleeve_regime_side": [
            {
                "key": "trend_following|trending_expansion|long",
                "trades": 8,
                "win_rate": 0.75,
                "profit_factor": 1.80,
                "expectancy_usd": 2.1,
            }
        ],
        "by_regime_side": [],
        "by_sleeve": [],
        "by_side": [],
    }
    relief = policy.threshold_relief_with_attribution(breakdown, [], report)
    bonus = policy.ranking_bonus_with_attribution(breakdown, [], report)
    assert relief < 0
    assert bonus > 0
