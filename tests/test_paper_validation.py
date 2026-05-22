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


def test_paper_ev_relax_mode_uses_extended_paper_hard_gate_window():
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


def test_scorer_paper_soft_ev_allows_bootstrap_stage():
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

    assert scorer._paper_soft_ev_allowed(ev_result) is True


def test_scorer_paper_soft_ev_allows_probation_stage():
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

    assert scorer._paper_soft_ev_allowed(ev_result) is True


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
