from copy import deepcopy
from types import SimpleNamespace

import src.main as main_module
from src.main import NinjaTrader, normalize_config
from src.models.fund_manager import FundManager
from src.scoring.scorer import Scorer


def _base_cfg():
    return {
        "exchange": {
            "testnet": True,
            "api_key": "",
            "api_secret": "",
        },
        "trading": {
            "mode": "paper",
            "max_open_trades": 1,
            "top_pairs_to_trade": 1,
        },
        "paper_validation": {
            "enabled": True,
            "max_open_trades_override": 2,
            "top_pairs_to_trade_override": 3,
        },
    }


def test_normalize_config_applies_paper_validation_capacity_overrides():
    cfg = normalize_config(_base_cfg())
    assert cfg["exchange"]["testnet"] is True
    assert cfg["trading"]["max_open_trades"] == 2
    assert cfg["trading"]["top_pairs_to_trade"] == 3


def test_normalize_config_keeps_live_mode_strict():
    cfg = _base_cfg()
    cfg["trading"]["mode"] = "live"
    cfg["exchange"]["testnet"] = False
    cfg["exchange"]["api_key"] = "k"
    cfg["exchange"]["api_secret"] = "s"
    out = normalize_config(cfg)
    assert out["trading"]["max_open_trades"] == 1
    assert out["trading"]["top_pairs_to_trade"] == 1


def test_ninja_trader_initializes_shadow_related_attributes(monkeypatch):
    dummy = lambda *args, **kwargs: SimpleNamespace()
    monkeypatch.setattr(main_module, "BinanceFuturesClient", dummy)
    monkeypatch.setattr(main_module, "MarketDataService", dummy)
    monkeypatch.setattr(main_module, "PairScanner", dummy)
    monkeypatch.setattr(main_module, "Scorer", lambda *args, **kwargs: SimpleNamespace(set_trade_log=lambda *_a, **_k: None, edge_detector=SimpleNamespace(memory=SimpleNamespace(total_samples=lambda: 0))))
    monkeypatch.setattr(main_module, "RiskManager", dummy)
    monkeypatch.setattr(main_module, "Executor", dummy)
    monkeypatch.setattr(main_module, "TradeManager", dummy)
    monkeypatch.setattr(main_module, "Learner", lambda *args, **kwargs: SimpleNamespace(_trade_log=[]))
    monkeypatch.setattr(main_module, "TelegramNotifier", dummy)
    monkeypatch.setattr(main_module, "MLPredictor", dummy)
    monkeypatch.setattr(main_module, "LiveReadiness", dummy)
    monkeypatch.setattr(main_module, "FundManager", dummy)
    monkeypatch.setattr(main_module, "CohortPolicy", dummy)
    monkeypatch.setattr(main_module, "StrategyLifecycleManager", dummy)
    monkeypatch.setattr(main_module, "DatasetLogger", dummy)
    monkeypatch.setattr(main_module, "RejectionLogger", dummy)

    cfg = normalize_config(_base_cfg())
    cfg.update(
        {
            "logging": {"level": "INFO"},
            "learning": {"save_path": "models/test_learning_state.json"},
            "telegram": {"enabled": False},
            "shadow": {"enabled": False, "report_interval_trades": 10},
        }
    )
    bot = NinjaTrader(cfg)
    assert bot._shadow is None
    assert bot._shadow_report_interval == 10
    assert hasattr(bot, "_dataset_logger")
    assert hasattr(bot, "_rej")


def test_paper_ev_relax_mode_allows_high_conviction_signal():
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._trading = {"mode": "paper"}
    bot._exploration = False
    bot._paper_validation = {
        "enabled": True,
        "min_score_floor": 58.0,
        "ev_relax_score_buffer": 2.0,
        "ev_bootstrap_max_deficit_pct": 0.10,
        "ev_probation_max_deficit_pct": 0.05,
        "high_conviction_score_trigger": 68.0,
        "high_conviction_ev_max_deficit_pct": 0.40,
        "high_conviction_allowed_regimes": ["trending_expansion"],
        "high_conviction_allowed_sm_phases": ["liquidity_sweep", "trending"],
    }
    bot._cfg = {"ev_model": {"min_trades_for_ev": 20}}

    breakdown = SimpleNamespace(
        ev_ok=False,
        regime_ok=True,
        smart_money_ok=True,
        total_score=72.0,
        regime=SimpleNamespace(value="trending_expansion"),
        smart_money=SimpleNamespace(phase=SimpleNamespace(value="liquidity_sweep")),
        ev_result=SimpleNamespace(trade_count=19, ev_net_pct=-0.36),
    )

    assert bot._paper_ev_relax_mode(breakdown, 58.0) == "bootstrap"
    assert bot._paper_high_conviction_candidate(breakdown, 58.0) is True


def test_paper_ev_relax_mode_rejects_non_allowlisted_regime():
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._trading = {"mode": "paper"}
    bot._paper_validation = {
        "enabled": True,
        "min_score_floor": 58.0,
        "ev_relax_score_buffer": 2.0,
        "ev_bootstrap_max_deficit_pct": 0.10,
        "ev_probation_max_deficit_pct": 0.05,
        "high_conviction_score_trigger": 68.0,
        "high_conviction_ev_max_deficit_pct": 0.40,
        "high_conviction_allowed_regimes": ["trending_expansion"],
        "high_conviction_allowed_sm_phases": ["liquidity_sweep", "trending"],
    }
    bot._cfg = {"ev_model": {"min_trades_for_ev": 20}}

    breakdown = SimpleNamespace(
        ev_ok=False,
        regime_ok=True,
        smart_money_ok=True,
        total_score=75.0,
        regime=SimpleNamespace(value="distribution"),
        smart_money=SimpleNamespace(phase=SimpleNamespace(value="liquidity_sweep")),
        ev_result=SimpleNamespace(trade_count=19, ev_net_pct=-0.20),
    )

    assert bot._paper_ev_relax_mode(breakdown, 58.0) is None
    assert bot._paper_high_conviction_candidate(breakdown, 58.0) is False


def test_paper_ev_relax_mode_allows_high_conviction_short_reversal():
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._trading = {"mode": "paper"}
    bot._paper_validation = {
        "enabled": True,
        "min_score_floor": 58.0,
        "ev_relax_score_buffer": 2.0,
        "ev_bootstrap_max_deficit_pct": 0.10,
        "ev_probation_max_deficit_pct": 0.05,
        "high_conviction_score_trigger": 68.0,
        "high_conviction_ev_max_deficit_pct": 0.40,
        "high_conviction_allowed_regimes": ["trending_expansion"],
        "high_conviction_allowed_sm_phases": ["liquidity_sweep", "trending"],
        "high_conviction_short_allowed_regimes": ["distribution"],
        "high_conviction_short_allowed_sm_phases": ["distribution", "liquidity_sweep"],
    }
    bot._cfg = {"ev_model": {"min_trades_for_ev": 20}}

    breakdown = SimpleNamespace(
        ev_ok=False,
        regime_ok=True,
        smart_money_ok=True,
        total_score=73.0,
        direction="short",
        strategy_sleeve="reversal",
        regime=SimpleNamespace(value="distribution"),
        smart_money=SimpleNamespace(
            phase=SimpleNamespace(value="distribution"),
            direction_bias="short",
        ),
        ev_result=SimpleNamespace(trade_count=19, ev_net_pct=-0.20),
    )

    assert bot._paper_ev_relax_mode(breakdown, 58.0) == "bootstrap"
    assert bot._paper_high_conviction_candidate(breakdown, 58.0) is True


def test_paper_ev_relax_mode_blocks_negative_generic_probation_sample():
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._trading = {"mode": "paper"}
    bot._paper_validation = {
        "enabled": True,
        "ev_hard_gate_min_trades": 40,
        "min_score_floor": 55.0,
        "ev_relax_score_buffer": 3.0,
        "ev_bootstrap_max_deficit_pct": 0.15,
        "ev_probation_max_deficit_pct": 0.10,
    }
    bot._cfg = {"ev_model": {"min_trades_for_ev": 20}}

    breakdown = SimpleNamespace(
        ev_ok=False,
        regime_ok=True,
        smart_money_ok=True,
        total_score=60.0,
        regime=SimpleNamespace(value="trending_expansion"),
        smart_money=SimpleNamespace(phase=SimpleNamespace(value="neutral")),
        ev_result=SimpleNamespace(trade_count=20, ev_net_pct=-0.10),
    )

    assert bot._paper_ev_relax_mode(breakdown, 58.0) is None


def test_paper_ev_relax_mode_allows_non_negative_generic_probation_sample():
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._trading = {"mode": "paper"}
    bot._paper_validation = {
        "enabled": True,
        "ev_hard_gate_min_trades": 40,
        "min_score_floor": 55.0,
        "ev_relax_score_buffer": 3.0,
        "ev_bootstrap_max_deficit_pct": 0.15,
        "ev_probation_max_deficit_pct": 0.10,
    }
    bot._cfg = {"ev_model": {"min_trades_for_ev": 20}}

    breakdown = SimpleNamespace(
        ev_ok=False,
        regime_ok=True,
        smart_money_ok=True,
        total_score=60.0,
        regime=SimpleNamespace(value="trending_expansion"),
        smart_money=SimpleNamespace(phase=SimpleNamespace(value="neutral")),
        ev_result=SimpleNamespace(trade_count=20, ev_net_pct=0.01),
    )

    assert bot._paper_ev_relax_mode(breakdown, 58.0) == "probation"


def test_paper_ev_relax_mode_blocks_after_hard_gate():
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._trading = {"mode": "paper"}
    bot._paper_validation = {
        "enabled": True,
        "ev_hard_gate_min_trades": 40,
        "min_score_floor": 55.0,
        "ev_relax_score_buffer": 3.0,
        "ev_bootstrap_max_deficit_pct": 0.15,
        "ev_probation_max_deficit_pct": 0.10,
    }
    bot._cfg = {"ev_model": {"min_trades_for_ev": 20}}

    breakdown = SimpleNamespace(
        ev_ok=False,
        regime_ok=True,
        smart_money_ok=True,
        total_score=62.0,
        regime=SimpleNamespace(value="trending_expansion"),
        smart_money=SimpleNamespace(phase=SimpleNamespace(value="neutral")),
        ev_result=SimpleNamespace(trade_count=40, ev_net_pct=-0.05),
    )

    assert bot._paper_ev_relax_mode(breakdown, 58.0) is None


def test_paper_stress_tightening_blocks_negative_ev_relax():
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._trading = {"mode": "paper"}
    bot._cfg = {"ev_model": {"gate_enabled": True}}
    bot._paper_validation = {
        "enabled": True,
        "stress_tighten_daily_loss_pct": -3.0,
        "stress_tighten_consecutive_losses": 3,
        "stress_min_ev_net_pct": 0.0,
    }
    bot._risk = SimpleNamespace(
        state=SimpleNamespace(daily_pnl_pct=-3.5, consecutive_losses=1)
    )
    breakdown = SimpleNamespace(ev_result=SimpleNamespace(ev_net_pct=-0.01))

    assert bot._paper_ev_stress_tightening_active() is True
    assert bot._paper_stress_allows_ev(breakdown) is False


def test_paper_stress_tightening_inactive_when_ev_gate_disabled():
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._trading = {"mode": "paper"}
    bot._cfg = {"ev_model": {"gate_enabled": False}}
    bot._paper_validation = {
        "enabled": True,
        "stress_tighten_daily_loss_pct": -3.0,
        "stress_tighten_consecutive_losses": 3,
        "stress_min_ev_net_pct": 0.0,
    }
    bot._risk = SimpleNamespace(
        state=SimpleNamespace(daily_pnl_pct=-3.5, consecutive_losses=4)
    )
    breakdown = SimpleNamespace(ev_result=SimpleNamespace(ev_net_pct=-1.0))

    assert bot._paper_ev_stress_tightening_active() is False
    assert bot._paper_stress_allows_ev(breakdown) is True


def test_paper_stress_tightening_allows_non_negative_ev():
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._trading = {"mode": "paper"}
    bot._cfg = {"ev_model": {"gate_enabled": True}}
    bot._paper_validation = {
        "enabled": True,
        "stress_tighten_daily_loss_pct": -3.0,
        "stress_tighten_consecutive_losses": 3,
        "stress_min_ev_net_pct": 0.0,
    }
    bot._risk = SimpleNamespace(
        state=SimpleNamespace(daily_pnl_pct=-1.0, consecutive_losses=3)
    )
    breakdown = SimpleNamespace(ev_result=SimpleNamespace(ev_net_pct=0.01))

    assert bot._paper_ev_stress_tightening_active() is True
    assert bot._paper_stress_allows_ev(breakdown) is True


def _ev_observation_floor_bot(gate_enabled=False):
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._trading = {"mode": "paper"}
    bot._cfg = {"ev_model": {"gate_enabled": gate_enabled}}
    bot._paper_validation = {
        "enabled": True,
        "ev_observation_floor_enabled": True,
        "ev_observation_min_p_win": 0.30,
        "ev_observation_max_deficit_pct": 0.75,
    }
    return bot


def test_paper_ev_observation_floor_blocks_low_probability_deep_negative_ev():
    bot = _ev_observation_floor_bot(gate_enabled=False)
    breakdown = SimpleNamespace(ev_result=SimpleNamespace(p_win=0.21, ev_net_pct=-0.98))

    allowed, reason = bot._paper_ev_observation_floor_allows(breakdown)

    assert allowed is False
    assert "p_win=0.210" in reason
    assert "ev_net=-0.980%" in reason


def test_paper_ev_observation_floor_allows_milder_or_higher_probability_ev():
    bot = _ev_observation_floor_bot(gate_enabled=False)

    mild_negative = SimpleNamespace(ev_result=SimpleNamespace(p_win=0.21, ev_net_pct=-0.50))
    higher_probability = SimpleNamespace(ev_result=SimpleNamespace(p_win=0.34, ev_net_pct=-0.98))

    assert bot._paper_ev_observation_floor_allows(mild_negative) == (True, "ok")
    assert bot._paper_ev_observation_floor_allows(higher_probability) == (True, "ok")


def test_paper_ev_observation_floor_inactive_when_ev_gate_enabled():
    bot = _ev_observation_floor_bot(gate_enabled=True)
    breakdown = SimpleNamespace(ev_result=SimpleNamespace(p_win=0.21, ev_net_pct=-0.98))

    assert bot._paper_ev_observation_floor_allows(breakdown) == (True, "EV gate enabled")


def _positive_ev_probe_bot():
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._trading = {"mode": "paper"}
    bot._cfg = {"ev_model": {"min_trades_for_ev": 20}}
    bot._risk = SimpleNamespace(state=SimpleNamespace(daily_pnl_pct=0.0, consecutive_losses=0))
    bot._paper_validation = {
        "enabled": True,
        "positive_ev_probe_enabled": True,
        "positive_ev_probe_min_ev_net_pct": 0.50,
        "positive_ev_probe_min_p_win": 0.50,
        "positive_ev_probe_min_score": 30.0,
        "positive_ev_probe_max_score_deficit": 12.0,
        "positive_ev_probe_min_sm_score": 75.0,
        "positive_ev_probe_allowed_directions": ["short"],
        "positive_ev_probe_allowed_regimes": ["trending_expansion", "distribution"],
        "positive_ev_probe_allowed_sm_phases": ["liquidity_sweep"],
        "positive_ev_probe_allowed_sleeves": ["neutral", "reversal"],
    }
    return bot


def _positive_ev_probe_breakdown(**overrides):
    data = {
        "direction": "short",
        "strategy_sleeve": "neutral",
        "regime_ok": True,
        "smart_money_ok": True,
        "ev_ok": True,
        "total_score": 34.0,
        "regime": SimpleNamespace(value="trending_expansion"),
        "smart_money": SimpleNamespace(
            phase=SimpleNamespace(value="liquidity_sweep"),
            score=80.0,
            direction_bias="short",
        ),
        "ev_result": SimpleNamespace(trade_count=0, ev_net_pct=0.81, p_win=0.54),
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_paper_positive_ev_probe_allows_bounded_liquidity_sweep_short_below_score():
    bot = _positive_ev_probe_bot()
    breakdown = _positive_ev_probe_breakdown()

    assert bot._paper_positive_ev_probe_mode(breakdown, 43.0) == "bootstrap"


def test_paper_positive_ev_probe_allows_distribution_liquidity_sweep_neutral_sleeve():
    bot = _positive_ev_probe_bot()
    breakdown = _positive_ev_probe_breakdown(
        total_score=40.4,
        regime=SimpleNamespace(value="distribution"),
        ev_result=SimpleNamespace(trade_count=0, ev_net_pct=0.68, p_win=0.51),
    )

    assert bot._paper_positive_ev_probe_mode(breakdown, 37.0) == "bootstrap"


def test_paper_positive_ev_probe_uses_pre_gate_score_for_distribution_gate():
    bot = _positive_ev_probe_bot()
    breakdown = _positive_ev_probe_breakdown(
        regime_ok=False,
        total_score=6.2,
        pre_gate_score=40.4,
        regime=SimpleNamespace(value="distribution"),
        ev_result=SimpleNamespace(trade_count=0, ev_net_pct=0.68, p_win=0.51),
    )

    assert bot._paper_positive_ev_probe_mode(breakdown, 37.0) == "bootstrap"


def test_paper_positive_ev_probe_blocks_low_probability_or_deep_score_gap():
    bot = _positive_ev_probe_bot()

    low_pwin = _positive_ev_probe_breakdown(
        ev_result=SimpleNamespace(trade_count=0, ev_net_pct=0.81, p_win=0.49)
    )
    deep_gap = _positive_ev_probe_breakdown(total_score=29.0)

    assert bot._paper_positive_ev_probe_mode(low_pwin, 43.0) is None
    assert bot._paper_positive_ev_probe_mode(deep_gap, 43.0) is None


def test_paper_positive_ev_probe_blocks_during_stress_tightening():
    bot = _positive_ev_probe_bot()
    bot._paper_validation.update(
        {
            "stress_tighten_daily_loss_pct": -3.0,
            "stress_tighten_consecutive_losses": 3,
            "stress_min_ev_net_pct": 1.0,
        }
    )
    bot._risk.state.daily_pnl_pct = -3.5
    breakdown = _positive_ev_probe_breakdown()

    assert bot._paper_positive_ev_probe_mode(breakdown, 43.0) is None


def test_signal_confirmation_counts_positive_ev_probe_below_threshold():
    bot = _positive_ev_probe_bot()
    probe = _positive_ev_probe_breakdown(total_score=34.0)
    weak = _positive_ev_probe_breakdown(
        total_score=34.0,
        ev_result=SimpleNamespace(trade_count=0, ev_net_pct=0.81, p_win=0.49),
    )

    assert bot._signal_confirmation_allowed(probe, 43.0) is True
    assert bot._signal_confirmation_allowed(weak, 43.0) is False


def test_probabilistic_candidate_rank_prefers_high_probability_edge_over_raw_score():
    bot = NinjaTrader.__new__(NinjaTrader)
    high_probability = SimpleNamespace(
        symbol="EDGE/USDT:USDT",
        total_score=43.0,
        setup_quality_score=78.0,
        strategy_ranking_bonus=0.0,
        ev_result=SimpleNamespace(
            p_win=0.58,
            ev_net_pct=0.72,
            conservative_p_win=0.53,
            conservative_ev_net_pct=0.18,
            confidence=0.70,
        ),
    )
    raw_score_only = SimpleNamespace(
        symbol="SCORE/USDT:USDT",
        total_score=46.0,
        setup_quality_score=50.0,
        strategy_ranking_bonus=0.0,
        ev_result=SimpleNamespace(
            p_win=0.40,
            ev_net_pct=-0.64,
            conservative_p_win=0.0,
            conservative_ev_net_pct=-0.92,
            confidence=0.50,
        ),
    )

    ranked = sorted(
        [raw_score_only, high_probability],
        key=lambda b: bot._probabilistic_candidate_rank(
            b, ml_p_win=0.50, regime_scale=1.0, cohort_bonus=0.0
        ),
        reverse=True,
    )

    assert ranked[0].symbol == "EDGE/USDT:USDT"


def test_paper_ml_hard_gate_min_trades_uses_paper_override():
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._trading = {"mode": "paper"}
    bot._paper_validation = {
        "enabled": True,
        "ml_hard_gate_min_trades": 40,
    }
    bot._cfg = {"ml": {"hard_gate_min_trades": 20}}

    assert bot._paper_ml_hard_gate_min_trades() == 40


def test_ml_veto_budget_caps_rejections_and_resets_each_day():
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._cfg = {
        "ml": {
            "max_daily_vetoes": 2,
            "max_veto_rate": 0.50,
            "veto_budget_min_candidates": 2,
        }
    }
    bot._risk = SimpleNamespace(state=SimpleNamespace(day_start_ts=100.0))
    bot._ml_gate_state = {
        "day_start_ts": 100.0,
        "candidate_count": 0,
        "veto_count": 0,
    }

    allowed, reason = bot._ml_veto_budget_allows()
    assert allowed is True
    assert reason == "ok"

    bot._record_ml_veto()
    allowed, reason = bot._ml_veto_budget_allows()
    assert allowed is True
    assert reason == "ok"

    bot._record_ml_veto()
    allowed, reason = bot._ml_veto_budget_allows()
    assert allowed is False
    assert reason == "daily ML veto budget exhausted (2/2)"

    bot._risk.state.day_start_ts = 200.0
    allowed, reason = bot._ml_veto_budget_allows()
    assert allowed is True
    assert reason == "ok"
    assert bot._ml_gate_state["candidate_count"] == 1
    assert bot._ml_gate_state["veto_count"] == 0


def test_paper_trade_scope_allows_only_long_trending_expansion_by_default():
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._trading = {"mode": "paper"}
    bot._paper_validation = {"enabled": True}

    allowed = SimpleNamespace(
        direction="long",
        regime=SimpleNamespace(value="trending_expansion"),
        strategy_sleeve="trend_following",
    )
    blocked_short = SimpleNamespace(
        direction="short",
        regime=SimpleNamespace(value="trending_expansion"),
        strategy_sleeve="trend_following",
    )
    blocked_regime = SimpleNamespace(
        direction="long",
        regime=SimpleNamespace(value="accumulation_compression"),
        strategy_sleeve="trend_following",
    )
    blocked_sleeve = SimpleNamespace(
        direction="long",
        regime=SimpleNamespace(value="trending_expansion"),
        strategy_sleeve="neutral",
    )

    assert bot._paper_trade_scope_allowed(allowed) is True
    assert bot._paper_trade_scope_allowed(blocked_short) is False
    assert bot._paper_trade_scope_allowed(blocked_regime) is False
    assert bot._paper_trade_scope_allowed(blocked_sleeve) is False

    allowed_ok, allowed_reason = bot._paper_trade_scope_check(allowed)
    short_ok, short_reason = bot._paper_trade_scope_check(blocked_short)
    regime_ok, regime_reason = bot._paper_trade_scope_check(blocked_regime)
    sleeve_ok, sleeve_reason = bot._paper_trade_scope_check(blocked_sleeve)

    assert allowed_ok is True
    assert allowed_reason == "paper scope ok"
    assert short_ok is False
    assert short_reason == "direction=short not in ['long']"
    assert regime_ok is False
    assert regime_reason == "regime=accumulation_compression not in ['trending_expansion']"
    assert sleeve_ok is False
    assert sleeve_reason == "sleeve=neutral not in ['trend_following']"


def test_paper_market_context_guard_blocks_negative_rotation_state():
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._trading = {"mode": "paper"}
    bot._paper_validation = {
        "enabled": True,
        "market_context_guard_enabled": True,
        "blocked_market_rotation_states": ["alts_outperforming"],
        "blocked_market_context_states": ["risk_on_alts|mania"],
    }

    blocked = SimpleNamespace(
        market_context=SimpleNamespace(
            rotation_state="alts_outperforming",
            risk_on_state="risk_on_alts",
        )
    )
    allowed = SimpleNamespace(
        market_context=SimpleNamespace(
            rotation_state="mixed_rotation",
            risk_on_state="risk_off",
        )
    )
    context_blocked = SimpleNamespace(
        market_context=SimpleNamespace(
            rotation_state="mania",
            risk_on_state="risk_on_alts",
        )
    )

    assert bot._paper_market_context_guard_check(allowed) == (True, "ok")
    assert bot._paper_market_context_guard_check(blocked) == (
        False,
        "market_rotation_state=alts_outperforming blocked",
    )
    assert bot._paper_market_context_guard_check(context_blocked) == (
        False,
        "market_context=risk_on_alts|mania blocked",
    )


def test_paper_trade_scope_supports_direction_specific_sleeves():
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._trading = {"mode": "paper"}
    bot._paper_validation = {
        "enabled": True,
        "allowed_directions": ["long", "short"],
        "allowed_regimes": ["trending_expansion", "distribution"],
        "allowed_strategy_sleeves": ["trend_following", "reversal", "compression_breakout"],
        "allowed_long_strategy_sleeves": ["trend_following"],
        "allowed_short_strategy_sleeves": ["trend_following", "reversal", "compression_breakout"],
    }

    long_ok = SimpleNamespace(
        direction="long",
        regime=SimpleNamespace(value="trending_expansion"),
        strategy_sleeve="trend_following",
    )
    long_blocked = SimpleNamespace(
        direction="long",
        regime=SimpleNamespace(value="trending_expansion"),
        strategy_sleeve="reversal",
    )
    short_ok = SimpleNamespace(
        direction="short",
        regime=SimpleNamespace(value="distribution"),
        strategy_sleeve="reversal",
    )
    short_blocked = SimpleNamespace(
        direction="short",
        regime=SimpleNamespace(value="distribution"),
        strategy_sleeve="neutral",
    )

    assert bot._paper_trade_scope_allowed(long_ok) is True
    assert bot._paper_trade_scope_allowed(long_blocked) is False
    assert bot._paper_trade_scope_allowed(short_ok) is True
    assert bot._paper_trade_scope_allowed(short_blocked) is False

    _, long_reason = bot._paper_trade_scope_check(long_blocked)
    _, short_reason = bot._paper_trade_scope_check(short_blocked)
    assert long_reason == "sleeve=reversal not in ['trend_following']"
    assert short_reason == "sleeve=neutral not in ['compression_breakout', 'reversal', 'trend_following']"


def test_scorer_paper_soft_ev_allows_small_negative_expectancy_before_hard_gate():
    scorer = Scorer.__new__(Scorer)
    scorer._paper_mode = True
    scorer._cfg = {
        "paper_validation": {
            "enabled": True,
            "ev_hard_gate_min_trades": 40,
            "ev_soft_block_max_deficit_pct": 0.50,
            "ev_bootstrap_max_deficit_pct": 0.15,
            "ev_probation_max_deficit_pct": 0.10,
        },
        "ev_model": {"min_trades_for_ev": 20},
    }
    ev_result = SimpleNamespace(trade_count=20, ev_net_pct=-0.40)

    assert scorer._paper_soft_ev_allowed(ev_result) is False


def test_scorer_paper_soft_ev_blocks_negative_bootstrap_stage():
    scorer = Scorer.__new__(Scorer)
    scorer._paper_mode = True
    scorer._cfg = {
        "paper_validation": {
            "enabled": True,
            "ev_hard_gate_min_trades": 40,
            "ev_bootstrap_max_deficit_pct": 0.15,
            "ev_probation_max_deficit_pct": 0.10,
        },
        "ev_model": {"min_trades_for_ev": 20},
    }
    ev_result = SimpleNamespace(trade_count=19, ev_net_pct=-0.10)

    assert scorer._paper_soft_ev_allowed(ev_result) is False


def test_scorer_paper_soft_ev_blocks_negative_probation_stage():
    scorer = Scorer.__new__(Scorer)
    scorer._paper_mode = True
    scorer._cfg = {
        "paper_validation": {
            "enabled": True,
            "ev_hard_gate_min_trades": 40,
            "ev_bootstrap_max_deficit_pct": 0.15,
            "ev_probation_max_deficit_pct": 0.10,
        },
        "ev_model": {"min_trades_for_ev": 20},
    }
    ev_result = SimpleNamespace(trade_count=20, ev_net_pct=-0.05)

    assert scorer._paper_soft_ev_allowed(ev_result) is False


def test_scorer_paper_soft_ev_allows_non_negative_bootstrap_stage():
    scorer = Scorer.__new__(Scorer)
    scorer._paper_mode = True
    scorer._cfg = {
        "paper_validation": {
            "enabled": True,
            "ev_hard_gate_min_trades": 40,
        },
        "ev_model": {"min_trades_for_ev": 20},
    }
    ev_result = SimpleNamespace(trade_count=19, ev_net_pct=0.01)

    assert scorer._paper_soft_ev_allowed(ev_result) is True


def test_scorer_ev_gate_can_be_disabled_while_preserving_ev_result():
    scorer = Scorer.__new__(Scorer)
    scorer._paper_mode = True
    scorer._ev_bootstrap_enabled = False
    scorer._cfg = {
        "paper_validation": {"enabled": True},
        "ev_model": {"gate_enabled": False, "min_trades_for_ev": 20},
    }
    ev_result = SimpleNamespace(
        trade_count=80,
        ev_net_pct=-1.25,
        is_tradeable=False,
    )

    assert scorer._ev_gate_allows(ev_result) is True


def test_scorer_statistical_ev_gate_blocks_small_sample_point_ev():
    scorer = Scorer.__new__(Scorer)
    scorer._paper_mode = True
    scorer._ev_bootstrap_enabled = False
    scorer._cfg = {
        "paper_validation": {"enabled": True},
        "ev_model": {
            "gate_enabled": True,
            "statistical_gate_enabled": True,
            "min_trades_for_ev": 20,
        },
    }
    ev_result = SimpleNamespace(
        trade_count=8,
        ev_net_pct=0.80,
        is_tradeable=True,
    )

    assert scorer._ev_gate_allows(ev_result) is False


def test_scorer_statistical_ev_gate_requires_tradeable_conservative_edge():
    scorer = Scorer.__new__(Scorer)
    scorer._paper_mode = True
    scorer._ev_bootstrap_enabled = False
    scorer._cfg = {
        "paper_validation": {"enabled": True},
        "ev_model": {
            "gate_enabled": True,
            "statistical_gate_enabled": True,
            "min_trades_for_ev": 20,
        },
    }

    blocked = SimpleNamespace(trade_count=20, ev_net_pct=0.80, is_tradeable=False)
    allowed = SimpleNamespace(trade_count=20, ev_net_pct=0.80, is_tradeable=True)

    assert scorer._ev_gate_allows(blocked) is False
    assert scorer._ev_gate_allows(allowed) is True


def test_scorer_paper_soft_ev_still_blocks_deeply_negative_expectancy():
    scorer = Scorer.__new__(Scorer)
    scorer._paper_mode = True
    scorer._cfg = {
        "paper_validation": {
            "enabled": True,
            "ev_hard_gate_min_trades": 40,
            "ev_bootstrap_max_deficit_pct": 0.15,
            "ev_probation_max_deficit_pct": 0.10,
        },
        "ev_model": {"min_trades_for_ev": 20},
    }
    ev_result = SimpleNamespace(trade_count=20, ev_net_pct=-0.80)

    assert scorer._paper_soft_ev_allowed(ev_result) is False


def test_score_many_keeps_neutral_fallback_conservative():
    scorer = Scorer.__new__(Scorer)
    scorer._strategy_router = SimpleNamespace(
        classify_dispersion=lambda _rows: SimpleNamespace(value=47.17, state="high"),
        evaluate=lambda _bd, _disp: SimpleNamespace(
            sleeve="neutral",
            score_mult=0.8835,
            size_mult=0.765,
            threshold_shift=0.0,
            ranking_bonus=0.0,
            reason="no validated sleeve",
        ),
    )
    scorer._strategy_base_score = lambda _bd, _sleeve: 42.79
    scorer.score = lambda snap: snap

    breakdown = SimpleNamespace(
        symbol="NMR/USDT:USDT",
        legacy_score=53.37,
        base_score=53.37,
        total_score=53.37,
        strategy_sleeve="neutral",
        strategy_reason="",
        strategy_score_mult=1.0,
        strategy_size_mult=1.0,
        strategy_threshold_shift=0.0,
        strategy_ranking_bonus=0.0,
        dispersion_value=0.0,
        dispersion_state="normal",
    )

    [rescored] = scorer.score_many({"NMR": breakdown})

    assert rescored.base_score == 42.79
    assert rescored.total_score == 37.80


def test_score_many_keeps_alpha_sleeves_anchored_to_institutional_score():
    scorer = Scorer.__new__(Scorer)
    scorer._strategy_router = SimpleNamespace(
        classify_dispersion=lambda _rows: SimpleNamespace(value=47.17, state="high"),
        evaluate=lambda _bd, _disp: SimpleNamespace(
            sleeve="trend_following",
            score_mult=0.8835,
            size_mult=0.765,
            threshold_shift=0.0,
            ranking_bonus=0.0,
            reason="trend sleeve",
        ),
    )
    scorer._strategy_base_score = lambda _bd, _sleeve: 42.79
    scorer.score = lambda snap: snap

    breakdown = SimpleNamespace(
        symbol="NMR/USDT:USDT",
        legacy_score=53.37,
        base_score=53.37,
        total_score=53.37,
        strategy_sleeve="neutral",
        strategy_reason="",
        strategy_score_mult=1.0,
        strategy_size_mult=1.0,
        strategy_threshold_shift=0.0,
        strategy_ranking_bonus=0.0,
        dispersion_value=0.0,
        dispersion_state="normal",
    )

    [rescored] = scorer.score_many({"NMR": breakdown})

    assert rescored.base_score == 53.37
    assert rescored.total_score == 47.15


def test_score_many_penalizes_risky_binance_alpha_setup_conditionally():
    scorer = Scorer.__new__(Scorer)
    passport = SimpleNamespace(sector="binance_alpha")
    scorer._strategy_router = SimpleNamespace(
        classify_dispersion=lambda _rows: SimpleNamespace(value=20.0, state="normal"),
        evaluate=lambda _bd, _disp: SimpleNamespace(
            sleeve="trend_following",
            score_mult=1.0,
            size_mult=1.0,
            threshold_shift=0.0,
            ranking_bonus=0.0,
            reason="trend sleeve",
            setup_type="trend_continuation",
            quality_score=60.0,
            passport=passport,
        ),
    )
    scorer._strategy_base_score = lambda bd, _sleeve: bd.legacy_score
    scorer.score = lambda snap: snap

    breakdown = SimpleNamespace(
        symbol="ESPORTS/USDT:USDT",
        legacy_score=50.0,
        base_score=50.0,
        total_score=50.0,
        volume_confirmation=50.0,
        open_interest=49.0,
        volatility=80.0,
        strategy_sleeve="neutral",
        strategy_reason="",
        strategy_score_mult=1.0,
        strategy_size_mult=1.0,
        strategy_threshold_shift=0.0,
        strategy_ranking_bonus=0.0,
        dispersion_value=0.0,
        dispersion_state="normal",
        ev_result=None,
    )

    [rescored] = scorer.score_many({"ESPORTS": breakdown})

    assert rescored.strategy_score_mult == 0.7618
    assert rescored.strategy_size_mult == 0.5419
    assert rescored.total_score == 38.09
    assert "binance_alpha_risk=quality<65+participation<55+volatility>=75" in rescored.strategy_reason


def test_score_applies_sector_rotation_multiplier_to_raw_score():
    scorer = Scorer.__new__(Scorer)
    rotation = SimpleNamespace(state="rotating_in", confidence=1.0, reason="l1 rotating in")
    snapshot = SimpleNamespace(sector_rotation=rotation)

    mult, reason = __import__("src.models.sector_rotation", fromlist=["sector_rotation_score_mult"]).sector_rotation_score_mult(
        snapshot.sector_rotation, "long"
    )

    assert mult == 1.07
    assert "rotating" in reason


def test_score_many_does_not_penalize_clean_binance_alpha_setup():
    scorer = Scorer.__new__(Scorer)
    passport = SimpleNamespace(sector="binance_alpha")
    scorer._strategy_router = SimpleNamespace(
        classify_dispersion=lambda _rows: SimpleNamespace(value=20.0, state="normal"),
        evaluate=lambda _bd, _disp: SimpleNamespace(
            sleeve="trend_following",
            score_mult=1.0,
            size_mult=1.0,
            threshold_shift=0.0,
            ranking_bonus=0.0,
            reason="trend sleeve",
            setup_type="trend_continuation",
            quality_score=72.0,
            passport=passport,
        ),
    )
    scorer._strategy_base_score = lambda bd, _sleeve: bd.legacy_score
    scorer.score = lambda snap: snap

    breakdown = SimpleNamespace(
        symbol="ESPORTS/USDT:USDT",
        legacy_score=50.0,
        base_score=50.0,
        total_score=50.0,
        volume_confirmation=62.0,
        open_interest=60.0,
        volatility=55.0,
        strategy_sleeve="neutral",
        strategy_reason="",
        strategy_score_mult=1.0,
        strategy_size_mult=1.0,
        strategy_threshold_shift=0.0,
        strategy_ranking_bonus=0.0,
        dispersion_value=0.0,
        dispersion_state="normal",
        ev_result=None,
    )

    [rescored] = scorer.score_many({"ESPORTS": breakdown})

    assert rescored.strategy_score_mult == 1.0
    assert rescored.strategy_size_mult == 1.0
    assert rescored.total_score == 50.0
    assert "binance_alpha_risk" not in rescored.strategy_reason


def test_score_many_soft_penalizes_low_probability_negative_ev_candidate():
    scorer = Scorer.__new__(Scorer)
    scorer._strategy_router = SimpleNamespace(
        classify_dispersion=lambda _rows: SimpleNamespace(value=40.0, state="normal"),
        evaluate=lambda _bd, _disp: SimpleNamespace(
            sleeve="trend_following",
            score_mult=1.0,
            size_mult=1.0,
            threshold_shift=0.0,
            ranking_bonus=0.0,
            reason="trend sleeve",
        ),
    )
    scorer._strategy_base_score = lambda bd, _sleeve: bd.legacy_score
    scorer.score = lambda snap: snap

    bad_posterior = SimpleNamespace(
        symbol="BAD/USDT:USDT",
        legacy_score=68.0,
        base_score=68.0,
        total_score=68.0,
        strategy_sleeve="neutral",
        strategy_reason="",
        strategy_score_mult=1.0,
        strategy_size_mult=1.0,
        strategy_threshold_shift=0.0,
        strategy_ranking_bonus=0.0,
        dispersion_value=0.0,
        dispersion_state="normal",
        ev_result=SimpleNamespace(
            p_win=0.22,
            ev_net_pct=-1.00,
            conservative_p_win=0.0,
            conservative_ev_net_pct=-1.10,
            confidence=0.0,
        ),
    )
    good_posterior = SimpleNamespace(
        symbol="GOOD/USDT:USDT",
        legacy_score=54.0,
        base_score=54.0,
        total_score=54.0,
        strategy_sleeve="neutral",
        strategy_reason="",
        strategy_score_mult=1.0,
        strategy_size_mult=1.0,
        strategy_threshold_shift=0.0,
        strategy_ranking_bonus=0.0,
        dispersion_value=0.0,
        dispersion_state="normal",
        ev_result=SimpleNamespace(
            p_win=0.58,
            ev_net_pct=0.70,
            conservative_p_win=0.53,
            conservative_ev_net_pct=0.18,
            confidence=0.70,
        ),
    )

    ranked = scorer.score_many({"BAD": bad_posterior, "GOOD": good_posterior})

    assert bad_posterior.probability_score_mult < 0.60
    assert bad_posterior.total_score < 43.0
    assert good_posterior.probability_score_mult > 1.0
    assert ranked[0].symbol == "GOOD/USDT:USDT"


def test_loss_contributor_guard_penalizes_mixed_missing_rotation_probation():
    scorer = Scorer.__new__(Scorer)
    scorer._cfg = {
        "loss_contributor_guard": {
            "enabled": True,
            "probation_sectors": ["depin", "commodity", "l1"],
            "probation_symbols": ["GRASS"],
        }
    }
    breakdown = SimpleNamespace(
        symbol="GRASS/USDT:USDT",
        direction="long",
        market_context=SimpleNamespace(rotation_state="mixed_rotation"),
        sector_rotation=SimpleNamespace(state="unknown", confidence=0.0),
        setup_passport=SimpleNamespace(sector="depin"),
        ev_result=SimpleNamespace(ev_net_pct=0.25),
    )

    score_mult, size_mult, threshold_shift, reason = scorer._loss_contributor_adjustment(breakdown)

    assert score_mult == 0.8302
    assert size_mult == 0.416
    assert threshold_shift == 14.0
    assert "mixed_rotation" in reason
    assert "sector_rotation_missing" in reason
    assert "probation=depin" in reason


def test_short_macro_overlay_boosts_broad_crypto_unwind_short():
    scorer = Scorer.__new__(Scorer)
    scorer._cfg = {
        "short_macro_overlay": {
            "enabled": True,
            "require_sector_confirmation_for_boost": True,
            "min_sector_confidence": 0.35,
        }
    }
    breakdown = SimpleNamespace(
        direction="short",
        setup_type="trend_continuation",
        market_context=SimpleNamespace(
            btc_trend="down",
            btc_d_trend="down",
            total_trend="down",
            eth_btc_trend="flat",
            risk_on_state="risk_off",
            rotation_state="mixed_rotation",
            confidence=0.8,
        ),
        sector_rotation=SimpleNamespace(state="rotating_out", confidence=0.7),
    )

    score_mult, size_mult, threshold_shift, state, macro_score, reason = scorer._short_macro_overlay_adjustment(breakdown)

    assert state == "strong_broad_unwind"
    assert macro_score == 8.5
    assert score_mult == 1.06
    assert size_mult == 1.08
    assert threshold_shift == -2.0
    assert "btc_down" in reason
    assert "btc_d_down" in reason
    assert "sector_rotating_out" in reason


def test_short_macro_overlay_penalizes_risk_on_or_sector_hostile_short():
    scorer = Scorer.__new__(Scorer)
    scorer._cfg = {"short_macro_overlay": {"enabled": True}}
    breakdown = SimpleNamespace(
        direction="short",
        setup_type="trend_continuation",
        market_context=SimpleNamespace(
            btc_trend="up",
            btc_d_trend="down",
            total_trend="up",
            eth_btc_trend="up",
            risk_on_state="risk_on_alts",
            rotation_state="alts_outperforming",
            confidence=0.8,
        ),
        sector_rotation=SimpleNamespace(state="rotating_in", confidence=0.8),
    )

    score_mult, size_mult, threshold_shift, state, macro_score, reason = scorer._short_macro_overlay_adjustment(breakdown)

    assert state == "short_hostile"
    assert macro_score < 0.0
    assert score_mult == 0.92
    assert size_mult == 0.75
    assert threshold_shift == 4.0
    assert "risk_on_alts" in reason
    assert "sector_rotating_in" in reason


def test_score_many_applies_short_macro_overlay_to_ranked_score_and_telemetry():
    scorer = Scorer.__new__(Scorer)
    scorer._cfg = {
        "short_macro_overlay": {
            "enabled": True,
            "require_sector_confirmation_for_boost": True,
        }
    }
    passport = SimpleNamespace(sector="ai")
    scorer._strategy_router = SimpleNamespace(
        classify_dispersion=lambda _rows: SimpleNamespace(value=20.0, state="normal"),
        evaluate=lambda _bd, _disp: SimpleNamespace(
            sleeve="trend_following",
            score_mult=1.0,
            size_mult=1.0,
            threshold_shift=0.0,
            ranking_bonus=0.0,
            reason="trend sleeve",
            setup_type="trend_continuation",
            quality_score=76.0,
            passport=passport,
        ),
    )
    scorer._strategy_base_score = lambda bd, _sleeve: bd.legacy_score
    scorer.score = lambda snap: snap
    breakdown = SimpleNamespace(
        symbol="ALT/USDT:USDT",
        direction="short",
        legacy_score=50.0,
        base_score=50.0,
        total_score=50.0,
        volume_confirmation=65.0,
        open_interest=62.0,
        volatility=45.0,
        strategy_sleeve="neutral",
        strategy_reason="",
        strategy_score_mult=1.0,
        strategy_size_mult=1.0,
        strategy_threshold_shift=0.0,
        strategy_ranking_bonus=0.0,
        dispersion_value=0.0,
        dispersion_state="normal",
        ev_result=None,
        market_context=SimpleNamespace(
            btc_trend="down",
            btc_d_trend="down",
            total_trend="down",
            eth_btc_trend="flat",
            risk_on_state="risk_off",
            rotation_state="mixed_rotation",
            confidence=0.8,
        ),
        sector_rotation=SimpleNamespace(state="rotating_out", confidence=0.7),
    )

    [rescored] = scorer.score_many({"ALT": breakdown})

    assert rescored.short_macro_state == "strong_broad_unwind"
    assert rescored.short_macro_score == 8.5
    assert rescored.short_macro_score_mult == 1.06
    assert rescored.short_macro_size_mult == 1.08
    assert rescored.short_macro_threshold_shift == -2.0
    assert rescored.strategy_score_mult == 1.06
    assert rescored.strategy_size_mult == 1.08
    assert rescored.strategy_threshold_shift == -2.0
    assert rescored.total_score == 53.0
    assert "short_macro=strong_broad_unwind" in rescored.strategy_reason


def test_loss_contributor_guard_tightens_unsupported_negative_ev_short():
    scorer = Scorer.__new__(Scorer)
    scorer._cfg = {"loss_contributor_guard": {"enabled": True}}
    breakdown = SimpleNamespace(
        symbol="BTC/USDT:USDT",
        direction="short",
        market_context=SimpleNamespace(rotation_state="btc_outperforming"),
        sector_rotation=SimpleNamespace(state="rotating_in", confidence=0.8),
        setup_passport=SimpleNamespace(sector="btc"),
        ev_result=SimpleNamespace(ev_net_pct=-0.01),
    )

    score_mult, size_mult, threshold_shift, reason = scorer._loss_contributor_adjustment(breakdown)

    assert score_mult == 0.846
    assert size_mult == 0.525
    assert threshold_shift == 8.0
    assert "short_ev<=0" in reason
    assert "short_sector_rotation=rotating_in" in reason


def test_start_clears_transient_persisted_kill_switch(monkeypatch):
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._trading = {"mode": "paper"}
    bot._risk = SimpleNamespace(
        state=SimpleNamespace(
            equity=500.0,
            peak_equity=500.0,
            consecutive_losses=0,
            daily_start_equity=500.0,
            kill_switch_reason="",
        ),
        update_equity=lambda *_a, **_k: None,
    )
    bot._client = SimpleNamespace(connect=lambda: None)
    bot._telegram = SimpleNamespace(
        startup=lambda *_a, **_k: None,
        restored_positions=lambda *_a, **_k: None,
    )
    bot._cfg = {}
    bot._starting_equity = 0.0
    bot._equity_curve = []
    bot._consecutive_wins = 0
    bot._running = False
    bot._heartbeat_ts = 0.0
    bot._tg_heartbeat_ts = 0.0
    bot._paper_validation = {}
    bot._safety = {}
    bot._telegram_open_positions = lambda: []

    async def _noop(*_args, **_kwargs):
        return None

    async def _fetch_balance(*_args, **_kwargs):
        return {"USDT": {"free": 500.0}}

    bot._loop = _noop
    bot._shutdown = _noop

    class _PathStub:
        def exists(self):
            return True

        def read_text(self):
            return (
                '{"mode":"paper","equity":429.85,"peak_equity":500.0,'
                '"daily_start_equity":410.0,"day_start_ts":111.0,'
                '"weekly_start_equity":405.0,"week_start_ts":222.0,'
                '"consecutive_losses":2,"starting_equity":500.0,'
                '"equity_curve":[432.55],"consecutive_wins":0,'
                '"kill_switch_reason":"runtime_error_burst"}'
            )

    monkeypatch.setattr(main_module, "Path", lambda *_a, **_k: _PathStub())

    import asyncio

    async def _run():
        await NinjaTrader.start(bot)

    monkeypatch.setattr(bot._client, "connect", _noop)
    monkeypatch.setattr(bot._client, "fetch_balance", _fetch_balance, raising=False)
    monkeypatch.setattr(bot._telegram, "startup", _noop)
    monkeypatch.setattr(bot._telegram, "restored_positions", _noop)

    asyncio.run(_run())

    assert bot._risk.state.kill_switch_reason == ""
    assert bot._risk.state.equity == 429.85
    assert bot._risk.state.daily_start_equity == 410.0
    assert bot._risk.state.day_start_ts == 111.0
    assert bot._risk.state.weekly_start_equity == 405.0
    assert bot._risk.state.week_start_ts == 222.0


def test_start_resets_stale_paper_state_when_reset_token_changes(monkeypatch):
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._trading = {
        "mode": "paper",
        "paper_starting_equity": 70,
        "paper_state_reset_token": "fresh_70_2026_05_21",
    }

    def _reset_day(equity):
        bot._risk.state.daily_start_equity = equity

    def _reset_week(equity):
        bot._risk.state.weekly_start_equity = equity

    bot._risk = SimpleNamespace(
        state=SimpleNamespace(
            equity=70.0,
            peak_equity=70.0,
            consecutive_losses=0,
            daily_start_equity=70.0,
            day_start_ts=0.0,
            weekly_start_equity=70.0,
            week_start_ts=0.0,
            kill_switch_reason="",
            paused_until_ts=0.0,
            reset_day=_reset_day,
            reset_week=_reset_week,
        ),
        update_equity=lambda equity: setattr(bot._risk.state, "equity", equity),
    )
    bot._client = SimpleNamespace(connect=lambda: None)
    bot._telegram = SimpleNamespace(
        startup=lambda *_a, **_k: None,
        restored_positions=lambda *_a, **_k: None,
    )
    bot._cfg = {}
    bot._starting_equity = 0.0
    bot._equity_curve = []
    bot._equity_graph_points = []
    bot._consecutive_wins = 0
    bot._running = False
    bot._heartbeat_ts = 0.0
    bot._tg_heartbeat_ts = 0.0
    bot._paper_validation = {}
    bot._safety = {}
    saved_trade_state = {"called": False}

    def _save_trade_state():
        saved_trade_state["called"] = True

    bot._trade_mgr = SimpleNamespace(
        _trades={"OLD/USDT:USDT": object()},
        _save_state=_save_trade_state,
    )
    bot._telegram_open_positions = lambda: []

    async def _noop(*_args, **_kwargs):
        return None

    class _PathStub:
        def exists(self):
            return True

        def read_text(self):
            return (
                '{"mode":"paper","equity":33.17,"peak_equity":80.0,'
                '"daily_start_equity":80.0,"day_start_ts":111.0,'
                '"weekly_start_equity":80.0,"week_start_ts":222.0,'
                '"consecutive_losses":4,"starting_equity":70.0,'
                '"equity_curve":[80.0,33.17],"consecutive_wins":0,'
                '"kill_switch_reason":"manual","paper_state_reset_token":"old"}'
            )

    monkeypatch.setattr(main_module, "Path", lambda *_a, **_k: _PathStub())
    monkeypatch.setattr(bot._client, "connect", _noop)
    monkeypatch.setattr(bot._telegram, "startup", _noop)
    monkeypatch.setattr(bot._telegram, "restored_positions", _noop)
    bot._loop = _noop
    bot._shutdown = _noop

    import asyncio

    asyncio.run(NinjaTrader.start(bot))

    assert bot._risk.state.equity == 70.0
    assert bot._risk.state.peak_equity == 70.0
    assert bot._risk.state.consecutive_losses == 0
    assert bot._risk.state.kill_switch_reason == ""
    assert bot._equity_curve == [70.0]
    assert bot._risk.state.daily_start_equity == 70.0
    assert bot._risk.state.weekly_start_equity == 70.0
    assert bot._trade_mgr._trades == {}
    assert saved_trade_state["called"] is True
    assert bot._risk.state.open_trade_count == 0
    assert bot._risk.state.open_risk_pct == 0.0
    assert bot._risk.state.symbol_risk_pct == {}
    assert bot._risk.state.direction_risk_pct == {}


def test_fund_manager_does_not_veto_drawdown_only_in_paper_validation():
    fm = FundManager(
        {
            "trading": {"mode": "paper"},
            "paper_validation": {"enabled": True},
        }
    )

    vetoed, reason = fm._veto_check(
        drawdown_pct=14.0,
        daily_pnl_pct=0.0,
        consecutive_losses=2,
        open_risk_pct=0.0,
    )

    assert vetoed is False
    assert reason == ""


def test_fund_manager_does_not_hard_veto_bad_day_loss_streak_in_paper_validation():
    fm = FundManager(
        {
            "trading": {"mode": "paper"},
            "paper_validation": {"enabled": True},
        }
    )

    vetoed, reason = fm._veto_check(
        drawdown_pct=3.2,
        daily_pnl_pct=-3.2,
        consecutive_losses=3,
        open_risk_pct=0.0,
    )

    assert vetoed is False
    assert reason == ""


def test_fund_manager_keeps_bad_day_loss_streak_veto_outside_paper_validation():
    fm = FundManager(
        {
            "trading": {"mode": "paper"},
            "paper_validation": {"enabled": False},
        }
    )

    vetoed, reason = fm._veto_check(
        drawdown_pct=3.2,
        daily_pnl_pct=-3.2,
        consecutive_losses=3,
        open_risk_pct=0.0,
    )

    assert vetoed is True
    assert "consecutive losses on a bad day" in reason


def test_paper_threshold_calibration_collects_mid_30s_trending_samples():
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._trading = {
        "mode": "paper",
        "min_score_threshold": 42.0,
        "regime_thresholds": {"trending_expansion": 50.0, "distribution": 52.0},
    }
    bot._exploration = False
    bot._paper_validation = {
        "enabled": True,
        "threshold_relaxation": 15.0,
        "min_score_floor": 35.0,
    }

    assert bot._threshold_for(SimpleNamespace(value="trending_expansion")) == 35.0
    assert bot._threshold_for(SimpleNamespace(value="distribution")) == 37.0


def test_paper_loss_streak_guard_blocks_weak_samples_without_cooldown():
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._trading = {"mode": "paper"}
    bot._paper_validation = {
        "enabled": True,
        "loss_streak_guard_enabled": True,
        "loss_streak_guard_consecutive_losses": 2,
        "loss_streak_guard_daily_loss_pct": -2.0,
        "loss_streak_guard_streak_daily_pnl_pct": 0.0,
        "loss_streak_guard_streak_drawdown_pct": 2.0,
        "loss_streak_guard_min_score_buffer": 8.0,
        "loss_streak_guard_min_setup_quality": 68.0,
        "loss_streak_guard_min_p_win": 0.42,
        "loss_streak_guard_min_ev_net_pct": 0.0,
    }
    bot._risk = SimpleNamespace(
        state=SimpleNamespace(daily_pnl_pct=-0.5, drawdown_pct=0.5, consecutive_losses=2)
    )
    weak = SimpleNamespace(
        total_score=57.0,
        setup_quality_score=67.0,
        ev_result=SimpleNamespace(p_win=0.41, ev_net_pct=-0.01),
    )

    allowed, reason = bot._paper_loss_streak_guard_check(weak, 50.0)

    assert bot._paper_loss_streak_guard_active() is True
    assert allowed is False
    assert "loss-streak guard" in reason


def test_paper_loss_streak_guard_ignores_stale_streak_on_green_healthy_day():
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._trading = {"mode": "paper"}
    bot._paper_validation = {
        "enabled": True,
        "loss_streak_guard_enabled": True,
        "loss_streak_guard_consecutive_losses": 2,
        "loss_streak_guard_daily_loss_pct": -2.0,
        "loss_streak_guard_streak_daily_pnl_pct": 0.0,
        "loss_streak_guard_streak_drawdown_pct": 2.0,
        "loss_streak_guard_min_score_buffer": 8.0,
        "loss_streak_guard_min_setup_quality": 68.0,
    }
    bot._risk = SimpleNamespace(
        state=SimpleNamespace(daily_pnl_pct=2.7, drawdown_pct=1.2, consecutive_losses=2)
    )
    borderline = SimpleNamespace(
        total_score=55.0,
        setup_quality_score=63.0,
        ev_result=SimpleNamespace(p_win=0.41, ev_net_pct=-0.01),
    )

    assert bot._paper_loss_streak_guard_active() is False
    assert bot._paper_loss_streak_guard_check(borderline, 50.0) == (True, "ok")
    assert bot._paper_loss_streak_size_cap() is None


def test_paper_loss_streak_guard_allows_only_cleaner_samples_and_caps_size():
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._trading = {"mode": "paper"}
    bot._paper_validation = {
        "enabled": True,
        "loss_streak_guard_enabled": True,
        "loss_streak_guard_consecutive_losses": 2,
        "loss_streak_guard_daily_loss_pct": -2.0,
        "loss_streak_guard_min_score_buffer": 8.0,
        "loss_streak_guard_min_setup_quality": 68.0,
        "loss_streak_guard_min_p_win": 0.42,
        "loss_streak_guard_min_ev_net_pct": 0.0,
        "loss_streak_guard_max_total_scale": 0.35,
    }
    bot._risk = SimpleNamespace(state=SimpleNamespace(daily_pnl_pct=-2.1, consecutive_losses=1))
    strong = SimpleNamespace(
        total_score=60.0,
        setup_quality_score=72.0,
        ev_result=SimpleNamespace(p_win=0.45, ev_net_pct=0.05),
    )

    assert bot._paper_loss_streak_guard_check(strong, 50.0) == (True, "ok")
    assert bot._paper_loss_streak_size_cap() == 0.35


def _cohort_exploration_bot():
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._trading = {"mode": "paper"}
    bot._paper_validation = {
        "enabled": True,
        "cohort_exploration_enabled": True,
        "cohort_exploration_max_open": 1,
        "cohort_exploration_max_total_scale": 0.25,
        "cohort_exploration_min_score_buffer": 8.0,
        "cohort_exploration_min_setup_quality": 68.0,
        "cohort_exploration_min_p_win": 0.55,
        "cohort_exploration_min_ev_net_pct": -0.25,
        "cohort_exploration_stop_daily_loss_pct": -1.0,
        "cohort_exploration_stop_consecutive_losses": 2,
        "cohort_exploration_allowed_directions": ["long"],
        "cohort_exploration_allowed_regimes": ["trending_expansion"],
        "cohort_exploration_allowed_sleeves": ["trend_following"],
        "cohort_exploration_allowed_setup_types": ["trend_continuation"],
    }
    bot._risk = SimpleNamespace(state=SimpleNamespace(daily_pnl_pct=0.0, consecutive_losses=0))
    bot._trade_mgr = SimpleNamespace(open_trades=[])
    return bot


def _cohort_exploration_breakdown(**overrides):
    data = {
        "symbol": "XLM/USDT:USDT",
        "direction": "long",
        "regime": SimpleNamespace(value="trending_expansion"),
        "strategy_sleeve": "trend_following",
        "setup_type": "trend_continuation",
        "total_score": 39.0,
        "setup_quality_score": 70.0,
        "ev_result": SimpleNamespace(p_win=0.58, ev_net_pct=-0.12),
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_paper_cohort_exploration_allows_tiny_high_quality_trend_sample():
    bot = _cohort_exploration_bot()
    breakdown = _cohort_exploration_breakdown()

    allowed, reason = bot._paper_cohort_exploration_check(breakdown, 31.0)

    assert allowed is True
    assert "paper cohort exploration" in reason
    assert bot._paper_cohort_exploration_size_cap() == 0.25


def test_paper_cohort_exploration_blocks_neutral_low_quality_or_stress():
    bot = _cohort_exploration_bot()

    neutral = _cohort_exploration_breakdown(strategy_sleeve="neutral")
    low_quality = _cohort_exploration_breakdown(setup_quality_score=67.0)
    low_score = _cohort_exploration_breakdown(total_score=38.9)
    bot._risk.state.daily_pnl_pct = -1.1
    stressed = _cohort_exploration_breakdown()

    assert bot._paper_cohort_exploration_check(neutral, 31.0)[0] is False
    assert bot._paper_cohort_exploration_check(low_quality, 31.0)[0] is False
    assert bot._paper_cohort_exploration_check(low_score, 31.0)[0] is False
    assert bot._paper_cohort_exploration_check(stressed, 31.0)[0] is False


def test_paper_cohort_exploration_counts_only_marked_open_trades():
    bot = _cohort_exploration_bot()
    normal = SimpleNamespace(setup=SimpleNamespace(setup_passport={}))
    exploration = SimpleNamespace(
        setup=SimpleNamespace(setup_passport={"paper_cohort_exploration": True})
    )
    bot._trade_mgr = SimpleNamespace(open_trades=[normal, exploration])

    allowed, reason = bot._paper_cohort_exploration_check(
        _cohort_exploration_breakdown(),
        31.0,
    )

    assert bot._paper_cohort_exploration_open_count() == 1
    assert allowed is False
    assert "max_open" in reason


def test_setup_passport_persists_paper_cohort_exploration_marker():
    breakdown = SimpleNamespace(
        setup_passport=None,
        market_context=None,
        sector_rotation=None,
        paper_cohort_exploration=True,
        paper_cohort_exploration_reason="cohort block; clean sample",
    )

    passport = main_module.setup_passport_with_market_context(breakdown)

    assert passport["paper_cohort_exploration"] is True
    assert passport["paper_cohort_exploration_reason"] == "cohort block; clean sample"


def test_paper_experimental_setup_stress_blocks_unproven_continuation_on_bad_day():
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._trading = {"mode": "paper"}
    bot._paper_validation = {
        "enabled": True,
        "experimental_setup_stress_block_enabled": True,
        "experimental_setup_stress_block_daily_loss_pct": -3.0,
        "experimental_setup_stress_block_consecutive_losses": 3,
        "experimental_setup_stress_block_setup_types": [
            "mtf_price_action_continuation",
            "vwap_pullback_continuation",
        ],
    }
    bot._risk = SimpleNamespace(state=SimpleNamespace(daily_pnl_pct=-3.4, consecutive_losses=1))
    breakdown = SimpleNamespace(setup_type="mtf_price_action_continuation")

    allowed, reason = bot._paper_experimental_setup_stress_check(breakdown)

    assert allowed is False
    assert "experimental setup stress block" in reason
    assert "mtf_price_action_continuation" in reason


def test_paper_experimental_setup_stress_allows_core_or_calm_setups():
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._trading = {"mode": "paper"}
    bot._paper_validation = {
        "enabled": True,
        "experimental_setup_stress_block_enabled": True,
        "experimental_setup_stress_block_daily_loss_pct": -3.0,
        "experimental_setup_stress_block_consecutive_losses": 3,
    }
    bot._risk = SimpleNamespace(state=SimpleNamespace(daily_pnl_pct=-3.4, consecutive_losses=1))

    core = SimpleNamespace(setup_type="trend_continuation")
    calm_experimental = SimpleNamespace(setup_type="vwap_pullback_continuation")

    assert bot._paper_experimental_setup_stress_check(core) == (
        True,
        "not an experimental stress-block setup",
    )

    bot._risk.state.daily_pnl_pct = -1.0
    assert bot._paper_experimental_setup_stress_check(calm_experimental) == (True, "ok")
