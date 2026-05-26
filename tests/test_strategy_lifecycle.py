from types import SimpleNamespace

from src.learning.learner import TradeRecord
from src.models.strategy_lifecycle import StrategyLifecycleManager


def _cfg():
    return {
        "lifecycle": {
            "default_paper_only": True,
            "research_min_trades": 3,
            "paper_validation_min_trades": 5,
            "small_live_min_trades": 8,
            "active_min_trades": 12,
            "min_profit_factor": 1.10,
            "min_expectancy_usd": 0.0,
            "min_win_rate": 0.45,
            "max_outlier_share": 0.8,
            "max_stop_hit_rate": 0.6,
            "disable_profit_factor_below": 0.9,
            "disable_expectancy_usd_below": -0.5,
            "disable_max_drawdown_usd": 20.0,
        }
    }


def _trade(pnl_usd: float, closed_at: float, reason: str = "tp2") -> TradeRecord:
    return TradeRecord(
        symbol="BTC/USDT:USDT",
        direction="long",
        entry_price=100.0,
        exit_price=101.0,
        pnl_usd=pnl_usd,
        pnl_pct=pnl_usd / 100.0,
        reason=reason,
        scores={
            "strategy_sleeve": "trend_following",
            "setup_type": "trend_continuation",
            "session": "london",
            "volatility_bucket": "medium",
            "trend_bucket": "strong",
            "signal_type": "trending_expansion|trending",
            "entry_reason": "trend sleeve",
            "timeframe": "1h",
            "asset": "BTC",
        },
        opened_at=closed_at - 3600,
        closed_at=closed_at,
        regime="trending_expansion",
        strategy_sleeve="trend_following",
        exit_profile="trend_following",
        tp1_hit=True,
        hold_duration_s=3600,
        fees_slippage_pct=0.18,
    )


def test_lifecycle_promotes_healthy_cohort_to_active():
    mgr = StrategyLifecycleManager(_cfg())
    trades = [_trade(2.0, i) for i in range(12)]
    trades += [_trade(-1.0, 100 + i, reason="stop_loss") for i in range(2)]
    report = mgr.build_report(trades)
    cohort = report["cohorts"][0]
    assert cohort["status"] == "ACTIVE"
    assert report["recommendation"] == "TRADE"
    assert report["primary_edge"]["key"] == cohort["key"]
    assert report["primary_edge"]["status"] == "ACTIVE"
    assert report["primary_edge"]["validated"] is True


def test_lifecycle_does_not_promote_losing_cohort_as_primary_edge():
    mgr = StrategyLifecycleManager(_cfg())
    trades = [_trade(-1.0, i, reason="stop_loss") for i in range(6)]
    report = mgr.build_report(trades)

    assert report["recommendation"] == "PAPER_ONLY"
    assert report["primary_edge"] is None


def test_lifecycle_requires_research_sample_before_primary_edge():
    mgr = StrategyLifecycleManager(_cfg())
    report = mgr.build_report([_trade(2.0, 1)])

    assert report["recommendation"] == "PAPER_ONLY"
    assert report["primary_edge"] is None





def test_lifecycle_blocks_unvalidated_live_cohort():
    mgr = StrategyLifecycleManager(_cfg())
    report = mgr.build_report([])
    bd = SimpleNamespace(
        symbol="BTC/USDT:USDT",
        regime=SimpleNamespace(value="trending_expansion"),
        smart_money=SimpleNamespace(phase=SimpleNamespace(value="trending")),
        strategy_sleeve="trend_following",
        strategy_reason="trend sleeve",
        volatility=25.0,
        trend_strength=70.0,
        direction="long",
    )
    decision = mgr.assess_breakdown(bd, report, mode="live", hour_utc=9)
    assert decision.allowed is False
    assert decision.recommendation == "PAPER_ONLY"
    assert decision.status == "RESEARCH"


def test_lifecycle_can_be_disabled():
    cfg = _cfg()
    cfg["lifecycle"]["enabled"] = False
    mgr = StrategyLifecycleManager(cfg)
    report = mgr.build_report([])
    bd = SimpleNamespace(
        symbol="BTC/USDT:USDT",
        regime=SimpleNamespace(value="trending_expansion"),
        smart_money=SimpleNamespace(phase=SimpleNamespace(value="trending")),
        strategy_sleeve="trend_following",
        strategy_reason="trend sleeve",
        volatility=25.0,
        trend_strength=70.0,
        direction="long",
    )
    decision = mgr.assess_breakdown(bd, report, mode="paper", hour_utc=9)
    assert report["recommendation"] == "TRADE"
    assert report.get("primary_edge") is None
    assert decision.allowed is True
    assert decision.reason == "lifecycle filter disabled"


def test_lifecycle_isolates_setup_type_cohorts_from_generic_reversal_history():
    mgr = StrategyLifecycleManager(_cfg())
    generic_losses = [
        TradeRecord(
            symbol="SOL/USDT:USDT",
            direction="short",
            entry_price=100.0,
            exit_price=101.0,
            pnl_usd=-1.0,
            pnl_pct=-1.0,
            reason="stop_loss",
            scores={
                "strategy_sleeve": "reversal",
                "setup_type": "sweep_reversal",
                "sm_phase": "liquidity_sweep",
                "session": "london",
            },
            opened_at=float(i),
            closed_at=float(i + 1),
            regime="distribution",
            strategy_sleeve="reversal",
            exit_profile="reversal",
        )
        for i in range(6)
    ]
    isolated_wins = [
        TradeRecord(
            symbol="SOL/USDT:USDT",
            direction="short",
            entry_price=100.0,
            exit_price=98.0,
            pnl_usd=2.0,
            pnl_pct=2.0,
            reason="tp2",
            scores={
                "strategy_sleeve": "reversal",
                "setup_type": "liquidity_sweep_reversal",
                "sm_phase": "liquidity_sweep",
                "session": "london",
            },
            opened_at=float(20 + i),
            closed_at=float(21 + i),
            regime="distribution",
            strategy_sleeve="reversal",
            exit_profile="reversal",
            tp1_hit=True,
        )
        for i in range(6)
    ]

    report = mgr.build_report(generic_losses + isolated_wins)
    keys = {c["key"]: c for c in report["cohorts"]}

    assert "liquidity_sweep_reversal|distribution|short" in keys
    assert keys["liquidity_sweep_reversal|distribution|short"]["profit_factor"] == float("inf")
    assert report["primary_edge"]["key"] == "liquidity_sweep_reversal|distribution|short"


def test_lifecycle_matches_isolated_setup_breakdown_to_report():
    cfg = _cfg()
    cfg["lifecycle"]["small_live_min_trades"] = 5
    cfg["lifecycle"]["active_min_trades"] = 10
    mgr = StrategyLifecycleManager(cfg)
    trades = []
    for i in range(5):
        t = _trade(2.0, float(i))
        t.direction = "short"
        t.regime = "distribution"
        t.strategy_sleeve = "reversal"
        t.scores["setup_type"] = "liquidity_sweep_reversal"
        t.scores["sm_phase"] = "liquidity_sweep"
        trades.append(t)
    report = mgr.build_report(trades)
    bd = SimpleNamespace(
        symbol="SOL/USDT:USDT",
        regime=SimpleNamespace(value="distribution"),
        smart_money=SimpleNamespace(phase=SimpleNamespace(value="liquidity_sweep")),
        strategy_sleeve="reversal",
        setup_type="liquidity_sweep_reversal",
        direction="short",
    )

    decision = mgr.assess_breakdown(bd, report, mode="live", hour_utc=9)

    assert decision.cohort_key == "liquidity_sweep_reversal|distribution|short"
    assert decision.allowed is True
    assert decision.status == "SMALL_LIVE"
