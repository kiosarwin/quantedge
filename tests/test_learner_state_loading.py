import json

from src.backtest.learner import Learner as BacktestLearner
from src.learning.learner import Learner as FuturesLearner


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


def test_backtest_learner_ignores_unknown_trade_fields(tmp_path):
    save_path = tmp_path / "backtest_state.json"
    save_path.write_text(json.dumps(_state_payload()))

    learner = BacktestLearner(_cfg(str(save_path)))

    assert len(learner._trade_log) == 1
    assert learner._trade_log[0].symbol == "BTC/USDT:USDT"
    assert learner.current_weights["trend_strength"] == 21.0
