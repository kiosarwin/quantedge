from types import SimpleNamespace

from src.main import enrich_closed_trade_scores, setup_passport_with_market_context


def _trade(opened_at=1779570000.0):
    setup = SimpleNamespace(
        symbol="EIGEN/USDT:USDT",
        strategy_sleeve="trend_following",
        exit_profile="trend_following",
        setup_passport={
            "asset": "EIGEN",
            "sector": "other",
            "sleeve": "trend_following",
            "regime": "trending_expansion",
            "sm_phase": "neutral",
            "setup_type": "trend_continuation",
            "quality_score": 70.0,
            "reason": "trend sleeve",
        },
    )
    return SimpleNamespace(symbol="EIGEN/USDT:USDT", opened_at=opened_at, setup=setup)


def test_enrich_closed_trade_scores_backfills_restored_trade_context():
    scores = enrich_closed_trade_scores({}, _trade(), {"timeframes": {"primary": "1h"}})

    assert scores["asset"] == "EIGEN"
    assert scores["sector"] == "other"
    assert scores["market_risk_on_state"] == "unknown"
    assert scores["market_rotation_state"] == "unknown"
    assert scores["market_context_confidence"] == 0.0
    assert scores["strategy_sleeve"] == "trend_following"
    assert scores["setup_type"] == "trend_continuation"
    assert scores["regime_code"] == 3.0
    assert scores["hour_of_day"] >= 0
    assert scores["day_of_week"] >= 0
    assert scores["session"] in {"asia", "london", "ny", "overlap_london_ny"}


def test_enrich_closed_trade_scores_keeps_pending_score_values():
    original = {
        "session": "london",
        "hour_of_day": 12,
        "day_of_week": 3,
        "sector": "ai",
        "regime_code": 1.0,
        "market_risk_on_state": "risk_on_alts",
        "market_rotation_state": "alts_outperforming",
        "market_context_confidence": 0.72,
    }
    scores = enrich_closed_trade_scores(original, _trade(), {"timeframes": {"primary": "1h"}})

    assert scores["session"] == "london"
    assert scores["hour_of_day"] == 12
    assert scores["day_of_week"] == 3
    assert scores["sector"] == "ai"
    assert scores["regime_code"] == 1.0
    assert scores["market_risk_on_state"] == "risk_on_alts"
    assert scores["market_rotation_state"] == "alts_outperforming"
    assert scores["market_context_confidence"] == 0.72


def test_setup_passport_with_market_context_persists_entry_snapshot():
    breakdown = SimpleNamespace(
        setup_passport=SimpleNamespace(as_dict=lambda: {"setup_type": "trend_continuation"}),
        market_context=SimpleNamespace(
            risk_on_state="risk_on_alts",
            rotation_state="alts_outperforming",
            btc_trend="up",
            eth_btc_trend="up",
            btc_d_trend="down",
            total_trend="up",
            confidence=0.78,
        ),
        sector_rotation=SimpleNamespace(
            sector="l1",
            state="rotating_in",
            rank=1,
            relative_btc_pct=2.4,
            breadth=0.8,
            confidence=0.71,
        ),
    )

    passport = setup_passport_with_market_context(breakdown)

    assert passport["setup_type"] == "trend_continuation"
    assert passport["market_risk_on_state"] == "risk_on_alts"
    assert passport["market_rotation_state"] == "alts_outperforming"
    assert passport["market_btc_trend"] == "up"
    assert passport["market_eth_btc_trend"] == "up"
    assert passport["market_btc_d_trend"] == "down"
    assert passport["market_total_trend"] == "up"
    assert passport["market_context_confidence"] == 0.78
    assert passport["sector_rotation_state"] == "rotating_in"
    assert passport["sector_rotation_rank"] == 1
    assert passport["sector_rotation_relative_btc_pct"] == 2.4
    assert passport["sector_rotation_confidence"] == 0.71
    assert passport["market_context"] == {
        "risk_on_state": "risk_on_alts",
        "rotation_state": "alts_outperforming",
        "btc_trend": "up",
        "eth_btc_trend": "up",
        "btc_d_trend": "down",
        "total_trend": "up",
        "confidence": 0.78,
    }


def test_enrich_closed_trade_scores_restores_market_context_from_passport():
    trade = _trade()
    trade.setup.setup_passport["market_context"] = {
        "risk_on_state": "mixed",
        "rotation_state": "mixed_rotation",
        "btc_trend": "flat",
        "eth_btc_trend": "flat",
        "btc_d_trend": "unknown",
        "total_trend": "unknown",
        "confidence": 0.38,
    }

    scores = enrich_closed_trade_scores({}, trade, {"timeframes": {"primary": "1h"}})

    assert scores["market_risk_on_state"] == "mixed"
    assert scores["market_rotation_state"] == "mixed_rotation"
    assert scores["market_btc_trend"] == "flat"
    assert scores["market_eth_btc_trend"] == "flat"
    assert scores["market_btc_d_trend"] == "unknown"
    assert scores["market_total_trend"] == "unknown"
    assert scores["market_context_confidence"] == 0.38
