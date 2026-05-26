from types import SimpleNamespace

import pandas as pd

import src.scoring.scorer as scorer_module
from src.analysis.smart_money import SmartMoneyPhase, SmartMoneySignal
from src.data.market_data import MarketSnapshot
from src.models.regime_classifier import RegimeState
from src.scoring.scorer import Scorer


def _df(rows=60):
    return pd.DataFrame(
        {
            "open": [100.0] * rows,
            "high": [101.0] * rows,
            "low": [99.0] * rows,
            "close": [100.0] * rows,
            "volume": [1000.0] * rows,
        }
    )


def _scorer():
    scorer = Scorer.__new__(Scorer)
    scorer._cfg = {
        "timeframes": {"primary": "1h", "higher": "4h"},
        "indicators": {"volume_lookback": 20},
        "scoring": {
            "weights": {
                "trend_strength": 20,
                "volume_confirmation": 15,
                "structure_quality": 20,
                "open_interest": 15,
                "funding_sentiment": 10,
                "order_book": 10,
                "volatility": 10,
            }
        },
        "trading": {"mode": "paper", "min_score_threshold": 42},
        "paper_validation": {"enabled": True},
        "ev_model": {"gate_enabled": False},
    }
    scorer._weights = dict(scorer._cfg["scoring"]["weights"])
    scorer._trade_log = []
    scorer._score_threshold = 42.0
    scorer._ev_model = SimpleNamespace(
        compute=lambda **_kwargs: SimpleNamespace(
            p_win=0.55,
            ev_net_pct=0.2,
            confidence=0.5,
            trade_count=0,
            conservative_p_win=0.54,
            conservative_ev_net_pct=0.1,
        )
    )
    scorer._edge_detector = SimpleNamespace(evaluate=lambda _ctx: None)
    scorer._strategy_router = SimpleNamespace(
        is_short_reversal_candidate=lambda _bd: False,
        is_long_reversal_candidate=lambda _bd: False,
    )
    return scorer


def test_dedicated_short_setup_overrides_generic_long_direction(monkeypatch):
    short_setup = SimpleNamespace(
        is_valid=True,
        label="liq_sweep",
        confidence=0.82,
        entry_price=100.0,
        stop_loss=102.0,
    )
    monkeypatch.setattr(scorer_module, "detect_short_entry", lambda _df, _cfg: short_setup)
    monkeypatch.setattr(scorer_module, "trade_direction_from_structure", lambda _df, _cfg: "long")
    monkeypatch.setattr(scorer_module, "build_feature_vector", lambda **_kwargs: None)
    monkeypatch.setattr(scorer_module, "classify_four_state", lambda *_args, **_kwargs: RegimeState.TRENDING_EXPANSION)
    monkeypatch.setattr(
        scorer_module,
        "detect_smart_money",
        lambda **_kwargs: SmartMoneySignal(
            phase=SmartMoneyPhase.NEUTRAL,
            score=50.0,
            direction_bias="neutral",
            oi_narrative="",
            funding_narrative="",
            volume_narrative="",
            sweep_narrative="",
            reasoning="neutral",
        ),
    )
    monkeypatch.setattr(scorer_module, "trend_strength_score", lambda *_args, **_kwargs: 65.0)
    monkeypatch.setattr(scorer_module, "volume_ratio", lambda *_args, **_kwargs: 1.2)
    monkeypatch.setattr(scorer_module, "buy_volume_ratio", lambda *_args, **_kwargs: 0.35)
    monkeypatch.setattr(scorer_module, "structure_quality_score", lambda *_args, **_kwargs: 62.0)
    monkeypatch.setattr(scorer_module, "open_interest_score", lambda *_args, **_kwargs: 55.0)
    monkeypatch.setattr(scorer_module, "funding_sentiment_score", lambda *_args, **_kwargs: 50.0)
    monkeypatch.setattr(scorer_module, "order_book_score", lambda *_args, **_kwargs: 52.0)
    monkeypatch.setattr(scorer_module, "volatility_score", lambda *_args, **_kwargs: 45.0)

    snap = MarketSnapshot(symbol="ALT/USDT:USDT", candles={"1h": _df()})

    breakdown = _scorer().score(snap)

    assert breakdown is not None
    assert breakdown.direction == "short"
    assert breakdown.short_setup is short_setup
