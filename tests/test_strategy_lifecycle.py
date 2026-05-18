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
    assert decision.allowed is True
    assert decision.reason == "lifecycle filter disabled"
