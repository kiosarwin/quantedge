"""Tests for the per-cycle rejection tally surfaced to the Telegram cycle report."""
from __future__ import annotations

from src.analysis.rejection_logger import RejectionLogger


def _make_logger(tmp_path) -> RejectionLogger:
    cfg = {
        "rejection_logger": {
            "enabled": True,
            "path": str(tmp_path / "rejections.parquet"),
            "flush_every": 9999,  # never auto-flush during tests
        },
    }
    return RejectionLogger(cfg)


def test_cycle_summary_empty_when_no_rejections(tmp_path):
    rl = _make_logger(tmp_path)
    summary = rl.cycle_summary()
    assert summary == {"total": 0, "stages": []}


def test_cycle_summary_aggregates_by_stage(tmp_path):
    rl = _make_logger(tmp_path)
    # 5 score, 3 gate_sm, 1 risk
    for _ in range(5):
        rl.log(stage="score_threshold", reason="score=42.1 < thresh=65.0")
    for _ in range(3):
        rl.log(stage="gate_sm", reason="sm_phase=neutral score=48")
    rl.log(stage="risk_guard", reason="daily loss cap hit (-4.10%)")

    summary = rl.cycle_summary(top_n=3)
    assert summary["total"] == 9

    stages = summary["stages"]
    assert len(stages) == 3
    # Sorted by count desc — score_threshold (5) > gate_sm (3) > risk_guard (1)
    assert stages[0]["stage"] == "score_threshold"
    assert stages[0]["count"] == 5
    assert stages[0]["top_reason"] == "score=42.1 < thresh=65.0"
    assert stages[1]["stage"] == "gate_sm"
    assert stages[1]["count"] == 3
    assert stages[2]["stage"] == "risk_guard"
    assert stages[2]["count"] == 1


def test_cycle_summary_top_reason_picks_most_frequent(tmp_path):
    rl = _make_logger(tmp_path)
    # Two distinct reasons under same stage; the more frequent one wins.
    for _ in range(2):
        rl.log(stage="gate_sm", reason="sm_phase=neutral")
    for _ in range(5):
        rl.log(stage="gate_sm", reason="sm_phase=conflict")

    summary = rl.cycle_summary()
    stages = summary["stages"]
    assert stages[0]["stage"] == "gate_sm"
    assert stages[0]["count"] == 7
    assert stages[0]["top_reason"] == "sm_phase=conflict"
    assert stages[0]["top_count"] == 5


def test_cycle_summary_reset_clears_tally(tmp_path):
    rl = _make_logger(tmp_path)
    rl.log(stage="score_threshold", reason="r1")
    rl.log(stage="score_threshold", reason="r1")

    first = rl.cycle_summary(reset=True)
    assert first["total"] == 2

    second = rl.cycle_summary()
    assert second == {"total": 0, "stages": []}


def test_cycle_summary_top_n_limits_output(tmp_path):
    rl = _make_logger(tmp_path)
    for stage in ("a", "b", "c", "d", "e"):
        rl.log(stage=stage, reason="why")

    summary = rl.cycle_summary(top_n=2)
    assert summary["total"] == 5
    assert len(summary["stages"]) == 2


def test_cycle_summary_disabled_logger_stays_empty(tmp_path):
    cfg = {"rejection_logger": {"enabled": False}}
    rl = RejectionLogger(cfg)
    rl.log(stage="score_threshold", reason="ignored")
    assert rl.cycle_summary() == {"total": 0, "stages": []}


def test_rejection_record_uses_router_setup_type_and_sleeve(tmp_path):
    from types import SimpleNamespace

    rl = RejectionLogger({
        "rejection_logger": {
            "enabled": True,
            "path": str(tmp_path / "rejections.parquet"),
            "flush_every": 1,
        }
    })
    phase = SimpleNamespace(value="liquidity_sweep")
    breakdown = SimpleNamespace(
        symbol="ZEC/USDT:USDT",
        direction="short",
        total_score=37.2,
        regime=SimpleNamespace(value="trending_expansion"),
        smart_money=SimpleNamespace(phase=phase, score=80.0),
        ev_result=None,
        regime_ok=True,
        smart_money_ok=True,
        ev_ok=True,
        setup_type="liquidity_sweep_reversal",
        strategy_sleeve="reversal",
        setup_quality_score=82.5,
    )

    rl.log(stage="paper_scope", reason="test", breakdown=breakdown, threshold_required=35.0)

    import pandas as pd
    row = pd.read_parquet(tmp_path / "rejections.parquet").iloc[-1]
    assert row["setup_type"] == "liquidity_sweep_reversal"
    assert row["strategy_sleeve"] == "reversal"
    assert row["setup_quality_score"] == 82.5
