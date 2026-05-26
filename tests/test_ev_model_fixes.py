"""
Regression tests for the audit-time EV model fixes:

  * funding-cost is direction-aware (longs pay positive funding, shorts
    receive it — and vice versa); the historic abs() default
    systematically over-estimated cost for the side that earns funding.
  * EdgeMemory.record_outcome (the public API used by main.py after
    the audit fix) actually feeds wins/losses into the per-pair ledger.
"""
from src.models.edge_detector import EdgeDetector, EdgeMemory
from src.models.ev_model import EVModel
from src.models.pwin_engine import PwinContext


_BASE_CFG = {
    "ev_model": {
        "min_trades_for_ev": 20,
        "min_p_win": 0.40,
        "min_ev_pct": 0.05,
        "taker_fee_pct": 0.04,
        "slippage_pct": 0.05,
        "prior_weight_alpha": 8,
        "loss_shrink_factor": 0.5,
    }
}


def test_ev_model_funding_cost_charges_longs_when_funding_positive():
    ev = EVModel(_BASE_CFG)
    long_ctx = PwinContext(direction="long")
    res = ev.compute(trade_log=[], funding_rate=0.0010, pwin_ctx=long_ctx)
    # 0.10% funding rate × 100 = 10bps cost on a long.
    assert res.funding_cost_pct == 0.10


def test_ev_model_funding_cost_credits_shorts_when_funding_positive():
    ev = EVModel(_BASE_CFG)
    short_ctx = PwinContext(direction="short")
    res = ev.compute(trade_log=[], funding_rate=0.0010, pwin_ctx=short_ctx)
    # Shorts receive positive funding -> appears as a *negative* cost
    # (i.e. credit) in the EV ledger so the edge isn't artificially
    # penalised by the historic abs() default.
    assert res.funding_cost_pct == -0.10


def test_ev_model_funding_cost_credits_longs_when_funding_negative():
    ev = EVModel(_BASE_CFG)
    long_ctx = PwinContext(direction="long")
    res = ev.compute(trade_log=[], funding_rate=-0.0008, pwin_ctx=long_ctx)
    # Longs receive negative funding -> credit.
    assert res.funding_cost_pct < 0


def test_edge_memory_record_outcome_increments_sample_ledger():
    mem = EdgeMemory()
    mem.record_signal("BTC/USDT:USDT", "trending_expansion|trending|long", "trending_expansion")
    mem.record_outcome(
        "BTC/USDT:USDT",
        "trending_expansion|trending|long",
        "trending_expansion",
        pnl_pct=0.7,
    )
    bucket = mem.lookup("BTC/USDT:USDT", "trending_expansion|trending|long")
    assert bucket is not None
    assert bucket.samples == 1
    assert bucket.wins == 1
    assert bucket.recent_outcomes == [0.7]


def test_edge_detector_evaluates_with_memory_seeded_by_outcomes():
    det = EdgeDetector({"edge_detector": {"mode": "soft_gate"}})
    # Seed the bucket with three losses to ensure the detector recognises
    # the pair as degraded once the gating threshold is met.
    edge_type = "trending_expansion|trending|long"
    pair = "AAA/USDT:USDT"
    for _ in range(8):
        det.memory.record_outcome(pair, edge_type, "trending_expansion", -1.0)
    # Same context the detector internally derives from EdgeContext.
    from src.models.edge_detector import EdgeContext
    ctx = EdgeContext(
        pair=pair,
        direction="long",
        regime="trending_expansion",
        smart_money_phase="trending",
        structure_quality=60.0,
        trend_strength=70.0,
        volume_zscore=0.0,
        oi_change_pct=0.0,
        funding_rate=0.0,
        volatility=50.0,
        p_win=0.55,
        total_score=70.0,
        score_threshold=60.0,
    )
    result = det.evaluate(ctx)
    # 0% recent WR over 8 samples -> below block threshold (30%) so soft
    # gate must scale size down rather than block hard.
    assert result.action == "SCALE_DOWN"
    assert 0.0 < result.size_mult <= 1.0


def _trade(pnl_pct, direction="long", regime="trending_expansion", sm_phase="trending"):
    class T:
        pass

    t = T()
    t.pnl_pct = pnl_pct
    t.direction = direction
    t.regime = regime
    t.scores = {"sm_phase": sm_phase}
    return t


def test_ev_model_conservative_gate_requires_lower_bound_edge():
    cfg = {
        "ev_model": {
            "statistical_gate_enabled": True,
            "min_trades_for_ev": 20,
            "min_p_win": 0.40,
            "min_ev_pct": 0.03,
            "prior_weight_alpha": 8,
            "loss_shrink_factor": 0.5,
            "confidence_z": 1.0,
            "payoff_haircut": 0.85,
            "loss_inflation": 1.10,
            "cost_buffer_pct": 0.05,
            "min_conservative_ev_pct": 0.0,
        },
        "risk": {"taker_fee_pct": 0.04, "slippage_pct": 0.05},
    }
    ev = EVModel(cfg)
    # Point-estimate looks positive: 12 wins x +1.2%, 8 losses x -0.6%.
    # Conservative lower-bound math should still reject it after haircut/loss/cost buffers.
    trades = [_trade(1.2) for _ in range(12)] + [_trade(-0.6) for _ in range(8)]

    res = ev.compute(
        trades,
        funding_rate=0.0,
        pwin_ctx=PwinContext(
            direction="long",
            regime="trending_expansion",
            sm_phase="trending",
        ),
    )

    assert res.ev_net_pct > 0
    assert res.conservative_ev_net_pct < 0
    assert res.statistical_edge_ok is False
    assert res.is_tradeable is False


def test_ev_model_conservative_gate_passes_strong_sample_edge():
    cfg = {
        "ev_model": {
            "statistical_gate_enabled": True,
            "min_trades_for_ev": 20,
            "min_p_win": 0.40,
            "min_ev_pct": 0.03,
            "prior_weight_alpha": 8,
            "loss_shrink_factor": 0.5,
            "confidence_z": 1.0,
            "payoff_haircut": 0.85,
            "loss_inflation": 1.10,
            "cost_buffer_pct": 0.05,
            "min_conservative_ev_pct": 0.0,
        },
        "risk": {"taker_fee_pct": 0.04, "slippage_pct": 0.05},
    }
    ev = EVModel(cfg)
    trades = [_trade(1.6) for _ in range(16)] + [_trade(-0.5) for _ in range(4)]

    res = ev.compute(
        trades,
        funding_rate=0.0,
        pwin_ctx=PwinContext(
            direction="long",
            regime="trending_expansion",
            sm_phase="trending",
        ),
    )

    assert res.conservative_ev_net_pct > 0
    assert res.statistical_edge_ok is True
    assert res.is_tradeable is True
