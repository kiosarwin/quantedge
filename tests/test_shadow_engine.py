from types import SimpleNamespace

from src.backtest.shadow_engine import ShadowEngine
from src.risk.risk_manager import TradeSetup


def _cfg(tmp_path):
    return {
        "exit": {"tp2_trailing_stop_pct": 0.15},
        "backtest": {
            "commission_pct": 0.04,
            "slippage_pct": 0.0,
            "initial_capital": 1000.0,
        },
        "learning": {"save_path": str(tmp_path / "learning_state.json")},
    }


def _breakdown():
    return SimpleNamespace(
        symbol="ALT/USDT:USDT",
        direction="long",
        total_score=72.0,
        regime=SimpleNamespace(value="trending_expansion"),
        ev_result=SimpleNamespace(ev_net_pct=0.42),
    )


def _setup():
    return TradeSetup(
        symbol="ALT/USDT:USDT",
        direction="long",
        entry_price=1.0,
        stop_loss=0.9,
        tp1=1.2,
        tp2=1.3,
        tp3=1.5,
        size_usd=100.0,
        size_contracts=100.0,
        leverage=1,
        r_distance=0.1,
        strategy_sleeve="trend_following",
        exit_profile="trend_following",
        setup_passport={"sector": "l1"},
    )


def test_shadow_persists_open_and_rejected_records_across_restart(tmp_path):
    cfg = _cfg(tmp_path)
    engine = ShadowEngine(cfg)

    engine.on_signal(_breakdown(), _setup(), scores_dict={"strategy_sleeve": "trend_following"})
    engine.record_rejected(_breakdown(), stage="cohort_policy")

    restored = ShadowEngine(cfg)

    assert restored.metrics().open == 1
    assert "ALT/USDT:USDT" in restored._open
    assert restored._open_meta["ALT/USDT:USDT"]["sector"] == "l1"
    assert len(restored._rejected) == 1
    assert restored._rejected[0]["stage"] == "cohort_policy"


def test_shadow_persists_closed_trade_after_restored_open_exits(tmp_path):
    cfg = _cfg(tmp_path)
    engine = ShadowEngine(cfg)
    engine.on_signal(_breakdown(), _setup(), scores_dict={"strategy_sleeve": "trend_following"})

    restored = ShadowEngine(cfg)
    restored.on_tick({"ALT/USDT:USDT": 0.89})

    metrics = restored.metrics()
    assert metrics.open == 0
    assert metrics.closed == 1
    assert metrics.losses == 1

    reloaded = ShadowEngine(cfg)
    assert reloaded.metrics().closed == 1
    assert reloaded.metrics().open == 0
    assert reloaded.get_ml_trade_log()[0].symbol == "ALT/USDT:USDT"
