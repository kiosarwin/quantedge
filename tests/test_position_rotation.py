import time
from types import SimpleNamespace

from src.data.market_data import MarketSnapshot
from src.execution.trade_manager import OpenTrade
from src.main import NinjaTrader
from src.risk.risk_manager import TradeSetup


def _setup(symbol="OLD/USDT:USDT", direction="long", ev_net_pct=0.0, setup_passport=None):
    if setup_passport is None:
        setup_passport = {"setup_type": "trend_continuation", "decision": "eligible"}
    return TradeSetup(
        symbol=symbol,
        direction=direction,
        entry_price=100.0,
        stop_loss=95.0,
        tp1=110.0,
        tp2=115.0,
        tp3=120.0,
        size_usd=10.0,
        size_contracts=0.1,
        leverage=5,
        r_distance=5.0,
        risk_pct=1.0,
        ev_net_pct=ev_net_pct,
        setup_passport=setup_passport,
    )


def _trade(
    symbol="OLD/USDT:USDT",
    opened_age_s=1800,
    mfe_r=0.0,
    tp1_hit=False,
    ev_net_pct=0.0,
    setup_passport=None,
):
    return OpenTrade(
        setup=_setup(symbol=symbol, ev_net_pct=ev_net_pct, setup_passport=setup_passport),
        entry_order_id="e",
        sl_order_id="s",
        tp1_order_id="t1",
        tp2_order_id="t2",
        opened_at=time.time() - opened_age_s,
        remaining_contracts=0.1,
        mfe_r=mfe_r,
        tp1_hit=tp1_hit,
    )


def _bot(open_trades):
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._cfg = {
        "position_rotation": {
            "enabled": True,
            "max_rotations_per_cycle": 1,
            "max_rotations_per_day": 3,
            "min_candidate_score": 72.0,
            "min_score_advantage": 14.0,
            "min_candidate_quality": 58.0,
            "min_candidate_p_win": 0.38,
            "min_candidate_ev_net_pct": -0.75,
            "positive_ev_probe_min_candidate_score": 35.0,
            "positive_ev_probe_min_score_advantage": 0.0,
            "positive_ev_probe_min_candidate_quality": 0.0,
            "min_victim_age_s": 7200,
            "protect_after_tp1": True,
            "protect_mfe_r": 0.80,
            "protect_unrealized_gain_r": 0.30,
            "protect_unrealized_gain_pct": 3.0,
            "protect_winner_min_age_s": 21600,
            "max_victim_adverse_r": 0.90,
        }
    }
    bot._risk = SimpleNamespace(state=SimpleNamespace(day_start_ts=123.0))
    bot._rotation_state = {"day_start_ts": 123.0, "count": 0}
    bot._trade_mgr = SimpleNamespace(open_trades=open_trades)
    return bot


def _candidate(score=82.0, quality=70.0, p_win=0.45, ev_net_pct=-0.10):
    return SimpleNamespace(
        symbol="NEW/USDT:USDT",
        total_score=score,
        setup_quality_score=quality,
        direction="short",
        regime=SimpleNamespace(value="trending_expansion"),
        smart_money=SimpleNamespace(
            phase=SimpleNamespace(value="neutral"),
            score=50.0,
            direction_bias="neutral",
        ),
        regime_ok=True,
        smart_money_ok=True,
        ev_ok=True,
        strategy_sleeve="trend_following",
        ev_result=SimpleNamespace(p_win=p_win, ev_net_pct=ev_net_pct),
    )


def test_position_rotation_selects_weak_aged_trade():
    weak = _trade(opened_age_s=8000, mfe_r=0.05)
    bot = _bot([weak])
    snapshots = {weak.symbol: MarketSnapshot(symbol=weak.symbol, last_price=98.5)}

    victim, price, reason = bot._select_rotation_victim(_candidate(), snapshots, 0)

    assert victim is weak
    assert price == 98.5
    assert "advantage" in reason


def test_position_rotation_protects_tp1_or_strong_winner():
    winner = _trade(opened_age_s=8000, mfe_r=1.10, tp1_hit=True)
    bot = _bot([winner])
    snapshots = {winner.symbol: MarketSnapshot(symbol=winner.symbol, last_price=111.0)}

    victim, _price, reason = bot._select_rotation_victim(_candidate(score=95.0), snapshots, 0)

    assert victim is None
    assert reason == "no eligible weak open trade"


def test_position_rotation_protects_raw_price_winner_even_when_r_is_distorted():
    esports_like = _trade(
        symbol="ESPORTS/USDT:USDT",
        opened_age_s=19000,
        mfe_r=0.07,
    )
    esports_like.setup.direction = "short"
    esports_like.setup.entry_price = 0.0622
    esports_like.setup.stop_loss = 0.14692819
    esports_like.setup.r_distance = 0.08472819
    bot = _bot([esports_like])
    snapshots = {esports_like.symbol: MarketSnapshot(symbol=esports_like.symbol, last_price=0.05652)}

    victim, _price, reason = bot._select_rotation_victim(_candidate(score=99.8), snapshots, 0)

    assert victim is None
    assert reason == "no eligible weak open trade"


def test_position_rotation_requires_victim_to_age_past_two_hours():
    young = _trade(opened_age_s=5200, mfe_r=0.0)
    bot = _bot([young])
    snapshots = {young.symbol: MarketSnapshot(symbol=young.symbol, last_price=99.5)}

    victim, _price, reason = bot._select_rotation_victim(_candidate(score=95.0), snapshots, 0)

    assert victim is None
    assert reason == "no eligible weak open trade"


def test_position_rotation_rejects_candidate_without_large_advantage():
    flat = _trade(opened_age_s=8000, mfe_r=0.20)
    bot = _bot([flat])
    bot._cfg["position_rotation"]["min_score_advantage"] = 30.0
    snapshots = {flat.symbol: MarketSnapshot(symbol=flat.symbol, last_price=100.0)}

    victim, _price, reason = bot._select_rotation_victim(_candidate(score=72.0, quality=58.0), snapshots, 0)

    assert victim is None
    assert "candidate advantage" in reason


def test_position_rotation_uses_probe_thresholds_for_positive_ev_distribution_sweep():
    weak_legacy = _trade(
        opened_age_s=8000,
        mfe_r=0.0,
        ev_net_pct=-0.65,
        setup_passport={},
    )
    bot = _bot([weak_legacy])
    bot._trading = {
        "mode": "paper",
        "min_score_threshold": 42.0,
        "regime_thresholds": {"distribution": 52.0},
    }
    bot._exploration = False
    bot._paper_validation = {
        "enabled": True,
        "threshold_relaxation": 15.0,
        "min_score_floor": 35.0,
        "positive_ev_probe_enabled": True,
        "positive_ev_probe_min_ev_net_pct": 0.50,
        "positive_ev_probe_min_p_win": 0.50,
        "positive_ev_probe_min_score": 30.0,
        "positive_ev_probe_max_score_deficit": 12.0,
        "positive_ev_probe_min_sm_score": 75.0,
        "positive_ev_probe_allowed_directions": ["short"],
        "positive_ev_probe_allowed_regimes": ["distribution"],
        "positive_ev_probe_allowed_sm_phases": ["liquidity_sweep"],
        "positive_ev_probe_allowed_sleeves": ["neutral", "reversal"],
    }
    candidate = _candidate(score=41.5, quality=0.0, p_win=0.51, ev_net_pct=0.68)
    candidate.regime = SimpleNamespace(value="distribution")
    candidate.smart_money = SimpleNamespace(
        phase=SimpleNamespace(value="liquidity_sweep"),
        score=80.0,
        direction_bias="short",
    )
    candidate.strategy_sleeve = "neutral"
    snapshots = {weak_legacy.symbol: MarketSnapshot(symbol=weak_legacy.symbol, last_price=100.0)}

    victim, price, reason = bot._select_rotation_victim(candidate, snapshots, 0)

    assert victim is weak_legacy
    assert price == 100.0
    assert "advantage" in reason
