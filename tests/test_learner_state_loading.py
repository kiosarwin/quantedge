import json

from src.learning.learner import Learner as FuturesLearner, TradeRecord

def _cfg(save_path: str) -> dict:
    return {
        "learning": {
            "min_trades_to_adjust": 2,
            "lookback_trades": 4,
            "save_path": save_path,
        },
        "scoring": {
            "adjustment_rate": 0.1,
            "weights": {
                "trend_strength": 20,
                "volume_confirmation": 15,
                "structure_quality": 20,
                "open_interest": 15,
                "funding_sentiment": 10,
                "order_book": 10,
                "volatility": 10,
            },
        },
    }

def _state_payload() -> dict:
    return {
        "weights": {
            "trend_strength": 21,
            "volume_confirmation": 14,
            "structure_quality": 20,
            "open_interest": 15,
            "funding_sentiment": 10,
            "order_book": 10,
            "volatility": 10,
        },
        "trade_log": [
            {
                "symbol": "BTC/USDT:USDT",
                "direction": "long",
                "entry_price": 100.0,
                "exit_price": 101.0,
                "pnl_usd": 1.0,
                "pnl_pct": 1.0,
                "reason": "tp2",
                "scores": {},
                "opened_at": 1.0,
                "closed_at": 2.0,
                "regime": "trending_expansion",
                "dispersion_value": 52.24,
                "mfe_r": 1.2,
            }
        ],
    }

def test_futures_learner_ignores_unknown_trade_fields(tmp_path):
    save_path = tmp_path / "learning_state.json"
    save_path.write_text(json.dumps(_state_payload()))

    learner = FuturesLearner(_cfg(str(save_path)))

    assert len(learner._trade_log) == 1
    assert learner._trade_log[0].symbol == "BTC/USDT:USDT"
    assert learner._trade_log[0].dispersion_value == 52.24
    assert learner.current_weights["trend_strength"] == 21.0


def _trade(
    pnl_usd: float,
    trend_strength: float,
    volume_confirmation: float,
    *,
    sector: str,
    session: str,
    hour_of_day: int,
    day_of_week: int,
) -> TradeRecord:
    return TradeRecord(
        symbol="CTX/USDT:USDT",
        direction="long",
        entry_price=100.0,
        exit_price=101.0,
        pnl_usd=pnl_usd,
        pnl_pct=pnl_usd / 100.0,
        reason="tp2" if pnl_usd > 0 else "stop_loss",
        scores={
            "trend_strength": trend_strength,
            "volume_confirmation": volume_confirmation,
            "structure_quality": 50.0,
            "open_interest": 50.0,
            "funding_sentiment": 50.0,
            "order_book": 50.0,
            "volatility": 50.0,
            "sector": sector,
            "session": session,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
        },
        opened_at=1.0,
        closed_at=2.0,
        regime="trending_expansion",
        session=session,
        hour_of_day=hour_of_day,
        day_of_week=day_of_week,
        asset="CTX",
        sector=sector,
    )

def test_futures_learner_uses_sector_session_hour_day_context(tmp_path):
    cfg = _cfg(str(tmp_path / "learning_state.json"))
    cfg["learning"]["min_trades_to_adjust"] = 8
    cfg["learning"]["lookback_trades"] = 8
    cfg["scoring"]["adjustment_rate"] = 0.2
    cfg["scoring"]["weights"] = {
        "trend_strength": 50.0,
        "volume_confirmation": 50.0,
    }
    learner = FuturesLearner(cfg)

    # Global winner/loss averages are balanced across all 8 trades, and all
    # trades share regime+direction. Any surviving tilt comes from context.
    trades = [
        _trade(1.0, 80.0, 20.0, sector="l1", session="london", hour_of_day=9, day_of_week=1),
        _trade(1.0, 80.0, 20.0, sector="l1", session="london", hour_of_day=9, day_of_week=1),
        _trade(1.0, 80.0, 20.0, sector="l1", session="london", hour_of_day=9, day_of_week=1),
        _trade(-1.0, 20.0, 80.0, sector="l1", session="london", hour_of_day=9, day_of_week=1),
        _trade(1.0, 20.0, 80.0, sector="meme", session="ny", hour_of_day=17, day_of_week=2),
        _trade(-1.0, 80.0, 20.0, sector="meme", session="ny", hour_of_day=17, day_of_week=2),
        _trade(-1.0, 80.0, 20.0, sector="meme", session="ny", hour_of_day=17, day_of_week=2),
        _trade(-1.0, 80.0, 20.0, sector="meme", session="ny", hour_of_day=17, day_of_week=2),
    ]
    for trade in trades:
        learner.record_trade(trade)

    assert learner.current_weights["trend_strength"] > 50.0
    assert learner.current_weights["volume_confirmation"] < 50.0

def test_futures_learner_context_keys_fall_back_to_score_metadata():
    trade = TradeRecord(
        symbol="CTX/USDT:USDT",
        direction="long",
        entry_price=100.0,
        exit_price=101.0,
        pnl_usd=1.0,
        pnl_pct=1.0,
        reason="tp2",
        scores={"sector": "AI", "session": "London", "hour_utc": 11.0, "day_of_week": 4},
        opened_at=1.0,
    )

    assert "sector_session_hour_day|ai|london|h08|d4" in FuturesLearner._context_keys(trade)


def test_futures_learner_backfills_legacy_trade_context(tmp_path):
    save_path = tmp_path / "learning_state.json"
    payload = _state_payload()
    payload["trade_log"][0]["session"] = "unknown"
    payload["trade_log"][0]["hour_of_day"] = -1
    payload["trade_log"][0]["day_of_week"] = -1
    payload["trade_log"][0]["asset"] = "unknown"
    payload["trade_log"][0]["sector"] = "unknown"
    save_path.write_text(json.dumps(payload))

    learner = FuturesLearner(_cfg(str(save_path)))
    trade = learner._trade_log[0]

    assert trade.session in {"asia", "london", "ny", "overlap_london_ny"}
    assert trade.hour_of_day >= 0
    assert trade.day_of_week >= 0
    assert trade.asset == "BTC"
    assert trade.sector == "majors"
    assert trade.scores["sector"] == "majors"
    assert trade.scores["hour_of_day"] >= 0

    saved = json.loads(save_path.read_text())
    saved_trade = saved["trade_log"][0]
    assert saved_trade["sector"] == "majors"
    assert saved_trade["hour_of_day"] >= 0

def test_futures_learner_context_keys_include_market_backdrop_without_time_context():
    trade = TradeRecord(
        symbol="ALT/USDT:USDT",
        direction="long",
        entry_price=100.0,
        exit_price=101.0,
        pnl_usd=1.0,
        pnl_pct=1.0,
        reason="tp2",
        scores={
            "market_risk_on_state": "risk_on_alts",
            "market_rotation_state": "alts_outperforming",
            "market_btc_trend": "up",
            "market_eth_btc_trend": "up",
        },
        opened_at=1.0,
        closed_at=2.0,
    )

    keys = FuturesLearner._context_keys(trade)

    assert "market_risk|risk_on_alts" in keys
    assert "market_rotation|alts_outperforming" in keys
    assert "market_context|risk_on_alts|alts_outperforming" in keys
    assert "market_btc_ethbtc|up|up" in keys

def test_futures_learner_load_backfills_market_context_from_scores(tmp_path):
    payload = _state_payload()
    payload["trade_log"][0]["scores"] = {
        "market_risk_on_state": "risk_off",
        "market_rotation_state": "btc_outperforming",
        "market_context_confidence": 0.64,
    }
    save_path = tmp_path / "learning_state.json"
    save_path.write_text(json.dumps(payload))

    learner = FuturesLearner(_cfg(str(save_path)))
    trade = learner._trade_log[0]

    assert trade.market_risk_on_state == "risk_off"
    assert trade.market_rotation_state == "btc_outperforming"
    assert trade.market_context_confidence == 0.64
    saved = json.loads(save_path.read_text())["trade_log"][0]
    assert saved["market_risk_on_state"] == "risk_off"
    assert saved["scores"]["market_rotation_state"] == "btc_outperforming"
