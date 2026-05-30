import pandas as pd

from src.models.market_context import (
    classify_market_context,
    market_context_score_mult,
    unavailable_context,
)


def _candles(values):
    return pd.DataFrame({"close": values})


def test_market_context_classifies_btc_led_alt_rotation():
    context = classify_market_context(
        btc_candles=_candles([100, 103, 106]),
        eth_candles=_candles([10, 10.8, 11.8]),
        btc_d_candles=_candles([55, 54, 53]),
        total_candles=_candles([1000, 1040, 1090]),
        lookback_bars=3,
        trend_threshold_pct=1.0,
    )

    assert context.btc_trend == "up"
    assert context.eth_btc_trend == "up"
    assert context.btc_d_trend == "down"
    assert context.total_trend == "up"
    assert context.risk_on_state == "risk_on_alts"
    assert context.rotation_state == "alts_outperforming"
    assert context.confidence > 0.75


def test_market_context_falls_back_when_inputs_missing():
    context = classify_market_context(lookback_bars=3)

    assert context == unavailable_context()


def test_market_context_multiplier_is_direction_aware():
    risk_off = classify_market_context(
        btc_candles=_candles([100, 98, 95]),
        total_candles=_candles([1000, 970, 940]),
        lookback_bars=3,
        trend_threshold_pct=1.0,
    )

    long_mult, long_reason = market_context_score_mult(risk_off, "long")
    short_mult, short_reason = market_context_score_mult(risk_off, "short")

    assert long_mult < 1.0
    assert "risk_off" in long_reason
    assert short_mult > 1.0
    assert "risk_off" in short_reason


def test_dataset_logger_extracts_market_context_fields():
    from types import SimpleNamespace

    from src.data.dataset_logger import _extract_signal_fields
    from src.models.market_context import MarketContext

    market_context = MarketContext(
        btc_trend="up",
        eth_btc_trend="up",
        btc_d_trend="down",
        total_trend="up",
        risk_on_state="risk_on_alts",
        rotation_state="alts_outperforming",
        confidence=0.82,
    )
    phase = SimpleNamespace(value="neutral")
    bd = SimpleNamespace(
        symbol="ALT/USDT:USDT",
        direction="long",
        total_score=70.0,
        all_gates_passed=True,
        regime_ok=True,
        smart_money_ok=True,
        ev_ok=True,
        trend_strength=70.0,
        volume_confirmation=60.0,
        structure_quality=65.0,
        open_interest=55.0,
        funding_sentiment=50.0,
        order_book=52.0,
        volatility=30.0,
        ev_result=None,
        feature_vector=None,
        smart_money=SimpleNamespace(
            phase=phase,
            score=50.0,
            direction_bias="neutral",
            aligns_with=lambda _direction: False,
        ),
        spot_context=None,
        setup_passport=SimpleNamespace(
            sector="l1",
            rotation_state="risk_on_rotation",
            crowding_state="balanced",
            funding_state="neutral",
            microstructure_state="aligned",
            expected_path="impulse_continuation",
            hold_profile="intraday_to_swing",
        ),
        regime=SimpleNamespace(value="trending_expansion"),
        market_context=market_context,
        sector_rotation=SimpleNamespace(
            state="rotating_in",
            rank=1,
            relative_btc_pct=3.2,
            breadth=0.75,
            confidence=0.68,
        ),
        strategy_reason="trend sleeve",
        strategy_sleeve="trend_following",
        setup_type="trend_continuation",
        setup_quality_score=68.0,
        short_macro_state="strong_broad_unwind",
        short_macro_score=8.5,
        short_macro_reason="short_macro=strong_broad_unwind:btc_down+btc_d_down",
        short_macro_score_mult=1.06,
        short_macro_size_mult=1.08,
        short_macro_threshold_shift=-2.0,
    )
    snap = SimpleNamespace(
        spot_context=None,
        funding_rate=0.0,
        oi_change_pct=0.0,
        ls_ratio=1.0,
        taker_buy_ratio=0.5,
        volume_24h_usdt=100000.0,
        market_context=market_context,
        sector_rotation=SimpleNamespace(
            state="rotating_in",
            rank=1,
            relative_btc_pct=3.2,
            breadth=0.75,
            confidence=0.68,
        ),
    )

    fields = _extract_signal_fields(bd, snap, score_threshold=42.0)

    assert fields["market_btc_trend"] == "up"
    assert fields["market_eth_btc_trend"] == "up"
    assert fields["market_btc_d_trend"] == "down"
    assert fields["market_total_trend"] == "up"
    assert fields["market_risk_on_state"] == "risk_on_alts"
    assert fields["market_rotation_state"] == "alts_outperforming"
    assert fields["market_context_confidence"] == 0.82
    assert fields["sector_rotation_state"] == "rotating_in"
    assert fields["sector_rotation_rank"] == 1
    assert fields["sector_rotation_relative_btc_pct"] == 3.2
    assert fields["sector_rotation_breadth"] == 0.75
    assert fields["sector_rotation_confidence"] == 0.68
    assert fields["short_macro_state"] == "strong_broad_unwind"
    assert fields["short_macro_score"] == 8.5
    assert fields["short_macro_reason"] == "short_macro=strong_broad_unwind:btc_down+btc_d_down"
    assert fields["short_macro_score_mult"] == 1.06
    assert fields["short_macro_size_mult"] == 1.08
    assert fields["short_macro_threshold_shift"] == -2.0


def test_attribution_report_groups_market_context_from_scores():
    from types import SimpleNamespace

    from src.reporting.attribution import build_attribution_report, compact_lines

    trades = [
        SimpleNamespace(
            symbol="ALT/USDT:USDT",
            direction="long",
            regime="trending_expansion",
            pnl_usd=2.0,
            pnl_pct=4.0,
            scores={
                "strategy_sleeve": "trend_following",
                "market_risk_on_state": "risk_on_alts",
                "market_rotation_state": "alts_outperforming",
                "market_btc_trend": "up",
                "market_eth_btc_trend": "up",
            },
        ),
        SimpleNamespace(
            symbol="ALT2/USDT:USDT",
            direction="long",
            regime="trending_expansion",
            pnl_usd=-1.0,
            pnl_pct=-2.0,
            scores={
                "strategy_sleeve": "trend_following",
                "market_risk_on_state": "risk_on_alts",
                "market_rotation_state": "alts_outperforming",
                "market_btc_trend": "up",
                "market_eth_btc_trend": "up",
            },
        ),
    ]

    report = build_attribution_report(trades, min_trades=1)

    assert report["by_market_risk_state"][0]["key"] == "risk_on_alts"
    assert report["by_market_rotation_state"][0]["key"] == "alts_outperforming"
    assert report["by_market_context"][0]["key"] == "risk_on_alts|alts_outperforming"
    assert report["by_market_btc_ethbtc"][0]["key"] == "up|up"
    assert report["headline"]["best_market_context"] == "risk_on_alts|alts_outperforming"
    lines = compact_lines(report)
    assert any(line.startswith("Market") for line in lines)


def test_shadow_ml_trade_log_promotes_market_context_to_trade_record(tmp_path):
    from src.backtest.shadow_engine import ShadowEngine, _ClosedShadowTrade

    cfg = {
        "exit": {},
        "backtest": {"commission_pct": 0.04, "slippage_pct": 0.01, "initial_capital": 1000.0},
        "learning": {"save_path": str(tmp_path / "learner_state.json")},
    }
    engine = ShadowEngine(cfg)
    engine._closed.append(
        _ClosedShadowTrade(
            pnl_usd=1.2,
            pnl_pct=3.4,
            regime="trending_expansion",
            ev_predicted=0.5,
            exit_reason="tp2",
            symbol="ALT/USDT:USDT",
            direction="long",
            sector="l1",
            entry_price=1.0,
            exit_price=1.03,
            opened_at=100.0,
            closed_at=200.0,
            mfe_r=1.4,
            mae_r=0.3,
            scores={
                "strategy_sleeve": "trend_following",
                "exit_profile": "trend_following",
                "session": "asia",
                "hour_of_day": 4,
                "day_of_week": 6,
                "market_risk_on_state": "risk_on_alts",
                "market_rotation_state": "alts_outperforming",
                "market_btc_trend": "up",
                "market_eth_btc_trend": "up",
                "market_btc_d_trend": "down",
                "market_total_trend": "up",
                "market_context_confidence": 0.82,
            },
        )
    )

    records = engine.get_ml_trade_log()

    assert len(records) == 1
    record = records[0]
    assert record.regime == "trending_expansion"
    assert record.strategy_sleeve == "trend_following"
    assert record.exit_profile == "trend_following"
    assert record.session == "asia"
    assert record.hour_of_day == 4
    assert record.day_of_week == 6
    assert record.market_risk_on_state == "risk_on_alts"
    assert record.market_rotation_state == "alts_outperforming"
    assert record.market_btc_trend == "up"
    assert record.market_eth_btc_trend == "up"
    assert record.market_btc_d_trend == "down"
    assert record.market_total_trend == "up"
    assert record.market_context_confidence == 0.82
    assert record.scores["is_shadow"] == 1.0
