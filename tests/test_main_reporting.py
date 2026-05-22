import asyncio
import time
from types import SimpleNamespace

import src.main as main_module
from src.main import NinjaTrader


def test_maybe_send_performance_report_runs_on_interval():
    bot = NinjaTrader.__new__(NinjaTrader)
    sent = {}

    class _Telegram:
        async def cycle_report(self, **kwargs):
            sent.update(kwargs)

    class _ML:
        is_ready = False
        cv_accuracy = 0.0

        @staticmethod
        def kelly_adjustment(trade_log):
            return 1.0

    bot._cfg = {"telegram": {"cycle_report_interval_minutes": 0}}
    bot._safety = {"performance_report_interval_minutes": 5}
    bot._tg_cycle_report_ts = time.time() - 301
    bot._startup_cycle_report_pending = False
    bot._learner = SimpleNamespace(
        _trade_log=[
            SimpleNamespace(
                symbol="BTC/USDT:USDT",
                direction="long",
                pnl_usd=1.25,
                pnl_pct=1.5,
                reason="tp1",
            )
        ]
    )
    bot._fund_mgr = SimpleNamespace(bonus=0.0, _recent_performance_mult=lambda trade_log: 1.0)
    bot._ml = _ML()
    bot._bootstrap_audit_lines = lambda lifecycle_report=None: ["Bootstrap Audit", "Mix: LONG `1` SHORT `0`"]  # type: ignore[assignment]
    bot._risk = SimpleNamespace(
        state=SimpleNamespace(
            equity=80.99,
            peak_equity=81.5,
            drawdown_pct=0.0,
            daily_pnl_pct=1.2,
            open_trade_count=3,
            consecutive_losses=0,
        )
    )
    bot._equity_curve = [80.0, 80.99]
    bot._trading = {"mode": "paper", "regime_thresholds": {"default": 45}}
    bot._starting_equity = 80.0
    bot._consecutive_wins = 1
    bot._telegram = _Telegram()

    asyncio.run(
        bot._maybe_send_fund_manager_report(
            breakdowns=[],
            readiness_report=SimpleNamespace(passed=False),
            jim_status={"ready": False, "trained_on": 0},
            floating_positions=[{"pnl_usd": 1.62}],
            open_positions=[{"symbol": "ETH/USDT:USDT"}],
            total_trades=1,
            win_rate=1.0,
            sharpe=0.0,
            cycle_num=11,
        )
    )

    assert sent["cycle_num"] == 11
    assert sent["equity"] == 80.99
    assert sent["open_positions"] == [{"symbol": "ETH/USDT:USDT"}]
    assert sent["closed_positions"][0].symbol == "BTC/USDT:USDT"
    assert sent["trade_log"] == bot._learner._trade_log
    assert sent["equity_curve"] == bot._equity_curve
    assert sent["starting_equity"] == 80.0
    assert "bootstrap_audit" not in sent


def test_maybe_send_bootstrap_audit_report_runs_on_separate_12h_cadence():
    bot = NinjaTrader.__new__(NinjaTrader)
    sent = {}

    class _Telegram:
        async def bootstrap_audit_report(self, *args, **kwargs):
            sent["args"] = args
            sent["kwargs"] = kwargs

    bot._cfg = {"telegram": {"bootstrap_audit_interval_hours": 12}}
    bot._last_bootstrap_audit_ts = time.time() - (12 * 3600) - 1
    bot._bootstrap_audit_lines = lambda lifecycle_report=None: ["Bootstrap Audit", "Phase: `BOOTSTRAP` | N `1`"]  # type: ignore[assignment]
    bot._telegram = _Telegram()

    asyncio.run(
        bot._maybe_send_bootstrap_audit_report(
            lifecycle_report=SimpleNamespace(counts={}, recommendation="PAPER_ONLY"),
            cycle_num=77,
        )
    )

    assert sent["kwargs"]["cycle_num"] == 77
    assert sent["kwargs"]["interval_hours"] == 12.0
    assert sent["args"][0] == ["Bootstrap Audit", "Phase: `BOOTSTRAP` | N `1`"]


def test_maybe_send_equity_graph_report_runs_on_interval():
    bot = NinjaTrader.__new__(NinjaTrader)
    sent = {}

    class _Telegram:
        async def equity_graph_report(self, points, **kwargs):
            sent["points"] = points
            sent["kwargs"] = kwargs

    bot._cfg = {"telegram": {"equity_graph_interval_minutes": 60}}
    bot._tg_equity_graph_ts = time.time() - (60 * 60) - 1
    bot._equity_graph_points = [
        {"ts": 1716000000.0, "balance": 80.0, "equity": 80.0},
        {"ts": 1716003600.0, "balance": 81.0, "equity": 82.5},
    ]
    bot._trading = {"mode": "paper"}
    bot._telegram = _Telegram()

    asyncio.run(bot._maybe_send_equity_graph_report(cycle_num=19))

    assert sent["kwargs"]["cycle_num"] == 19
    assert sent["kwargs"]["interval_minutes"] == 60.0
    assert sent["kwargs"]["mode"] == "paper"
    assert sent["points"] == bot._equity_graph_points


def test_price_map_includes_open_trades_missing_from_scan(monkeypatch):
    bot = NinjaTrader.__new__(NinjaTrader)

    class _Client:
        async def fetch_ticker(self, symbol):
            return {"last": 42.5 if symbol == "MISSING/USDT:USDT" else 0.0}

    bot._client = _Client()
    bot._trade_mgr = SimpleNamespace(open_symbols=["MISSING/USDT:USDT", "SEEN/USDT:USDT"])
    snapshots = {
        "SEEN/USDT:USDT": SimpleNamespace(last_price=10.0),
    }

    price_map = asyncio.run(bot._price_map_with_open_trades(snapshots))

    assert price_map["SEEN/USDT:USDT"] == 10.0
    assert price_map["MISSING/USDT:USDT"] == 42.5


def test_run_bootstrap_audit_now_sends_once_and_exits(monkeypatch):
    sent = {}

    class _Client:
        async def close(self):
            sent["closed"] = True

    class _Telegram:
        async def bootstrap_audit_report(self, lines, interval_hours):
            sent["lines"] = lines
            sent["interval_hours"] = interval_hours

    class _Lifecycle:
        @staticmethod
        def build_report(trade_log):
            return {"counts": {}, "recommendation": "PAPER_ONLY"}

    class _Bot:
        def __init__(self, cfg):
            self._cfg = cfg
            self._client = _Client()
            self._telegram = _Telegram()
            self._lifecycle = _Lifecycle()
            self._learner = SimpleNamespace(_trade_log=[])

        def _bootstrap_audit_lines(self, lifecycle_report=None):
            return ["Bootstrap Audit", "Phase: `BOOTSTRAP` | N `0`"]

    monkeypatch.setattr(main_module, "NinjaTrader", _Bot)
    monkeypatch.setattr(main_module, "setup_logging", lambda cfg: None)
    monkeypatch.setattr(main_module, "load_config", lambda path: {"telegram": {"bootstrap_audit_interval_hours": 12}, "trading": {"mode": "paper"}, "exchange": {}})
    monkeypatch.setattr(main_module, "normalize_config", lambda cfg: cfg)
    # Bypass the pydantic schema gate — this test uses a deliberately stub
    # config to exercise the bootstrap-audit-now path without the full
    # production yaml.
    monkeypatch.setattr(main_module, "validate_config", lambda cfg: cfg)

    asyncio.run(
        main_module._run(
            SimpleNamespace(
                config="config/config.yaml",
                mode=None,
                testnet=None,
                bootstrap_audit_now=True,
            )
        )
    )

    assert sent["interval_hours"] == 12.0
    assert sent["lines"] == ["Bootstrap Audit", "Phase: `BOOTSTRAP` | N `0`"]
    assert sent["closed"] is True


def test_maybe_send_performance_report_sends_immediately_after_start():
    bot = NinjaTrader.__new__(NinjaTrader)
    sent = {}

    class _Telegram:
        async def cycle_report(self, **kwargs):
            sent.update(kwargs)

    class _ML:
        is_ready = False
        cv_accuracy = 0.0

        @staticmethod
        def kelly_adjustment(trade_log):
            return 1.0

    bot._cfg = {"telegram": {"cycle_report_interval_minutes": 5}}
    bot._safety = {"performance_report_interval_minutes": 5}
    bot._tg_cycle_report_ts = time.time()
    bot._last_bootstrap_audit_ts = time.time()
    bot._startup_cycle_report_pending = True
    bot._learner = SimpleNamespace(_trade_log=[])
    bot._fund_mgr = SimpleNamespace(bonus=0.0, _recent_performance_mult=lambda trade_log: 1.0)
    bot._ml = _ML()
    bot._risk = SimpleNamespace(
        state=SimpleNamespace(
            equity=80.99,
            peak_equity=81.5,
            drawdown_pct=0.0,
            daily_pnl_pct=0.0,
            open_trade_count=0,
            consecutive_losses=0,
        )
    )
    bot._equity_curve = [80.0, 80.99]
    bot._trading = {"mode": "paper", "regime_thresholds": {"default": 45}}
    bot._starting_equity = 80.0
    bot._consecutive_wins = 0
    bot._telegram = _Telegram()

    asyncio.run(
        bot._maybe_send_fund_manager_report(
            breakdowns=[],
            readiness_report=SimpleNamespace(passed=False),
            jim_status={"ready": False, "trained_on": 0},
            floating_positions=[],
            open_positions=[],
            total_trades=0,
            win_rate=0.0,
            sharpe=0.0,
            cycle_num=1,
        )
    )

    assert sent["cycle_num"] == 1
    assert bot._startup_cycle_report_pending is False


def test_effective_account_metrics_include_floating_pnl():
    bot = NinjaTrader.__new__(NinjaTrader)
    bot._risk = SimpleNamespace(
        state=SimpleNamespace(
            equity=80.0,
            peak_equity=82.0,
            daily_start_equity=79.0,
        )
    )

    equity, daily_pnl_pct, drawdown_pct = bot._effective_account_metrics(
        [{"pnl_usd": 2.0}, {"pnl_usd": -0.5}]
    )

    assert equity == 81.5
    assert round(daily_pnl_pct, 2) == round((81.5 - 79.0) / 79.0 * 100, 2)
    assert round(drawdown_pct, 2) == round((82.0 - 81.5) / 82.0 * 100, 2)
