import asyncio
import os
from datetime import datetime
from types import SimpleNamespace

from src.notifications.telegram import TelegramNotifier


class _FakeResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


class _FakeAsyncClient:
    def __init__(self, responses, calls) -> None:
        self._responses = list(responses)
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json=None, data=None, files=None):
        self._calls.append((url, json, data, files))
        return self._responses.pop(0)


def test_notifier_strips_env_values(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", " token ")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", " 12345 ")

    notifier = TelegramNotifier({"telegram": {}})

    assert notifier._token == "token"
    assert notifier._chat_id == "12345"
    assert notifier._enabled is True


def test_notifier_falls_back_to_plain_text(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    notifier = TelegramNotifier({"telegram": {}})

    calls = []
    responses = [
        _FakeResponse(400, "Bad Request: can't parse entities"),
        _FakeResponse(200, '{"ok":true}'),
    ]

    monkeypatch.setattr(
        "src.notifications.telegram.httpx.AsyncClient",
        lambda timeout: _FakeAsyncClient(responses, calls),
    )

    import asyncio

    asyncio.run(notifier.send_raw("*hello*"))

    assert len(calls) == 2
    assert calls[0][1]["parse_mode"] == "Markdown"
    assert "parse_mode" not in calls[1][1]


def test_heartbeat_uses_floating_pnl(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    notifier = TelegramNotifier({"telegram": {}})

    sent = {}

    async def fake_send(message: str) -> None:
        sent["message"] = message

    notifier._send = fake_send  # type: ignore[attr-defined]

    asyncio.run(
        notifier.heartbeat(
            equity=80.0,
            drawdown_pct=1.2,
            daily_pnl_pct=0.5,
            open_trades=2,
            top_signals=["BTC long 88", "ETH long 77"],
            floating_positions=[{"pnl_usd": 5.0}, {"pnl_usd": -1.0}],
            open_positions=[{"symbol": "BTC/USDT"}],
        )
    )

    assert "Equity *$84.00*" in sent["message"]


def test_heartbeat_normalizes_empty_open_and_dd(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    notifier = TelegramNotifier({"telegram": {}})

    sent = {}

    async def fake_send(message: str) -> None:
        sent["message"] = message

    notifier._send = fake_send  # type: ignore[attr-defined]

    asyncio.run(
        notifier.heartbeat(
            equity=80.0,
            drawdown_pct="",
            daily_pnl_pct=None,
            open_trades=None,
            top_signals=[],
            floating_positions=None,
            open_positions=None,
        )
    )

    assert "Open `0`" in sent["message"]
    assert "DD `0.0%`" in sent["message"]


def test_performance_report_uses_fund_manager_header(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    notifier = TelegramNotifier({"telegram": {}})

    sent = {}

    async def fake_send(message: str) -> None:
        sent["message"] = message

    notifier._send = fake_send  # type: ignore[attr-defined]

    asyncio.run(
        notifier.performance_report(
            report={
                "trades": 3,
                "win_rate": 2 / 3,
                "profit_factor": 1.8,
                "avg_rr": 2.1,
                "drawdown_pct": 1.4,
                "ml_accuracy": 0.57,
                "ml_trend": "stable",
                "kelly_factor": 1.0,
            },
            equity=80.99,
            mode="paper",
        )
    )

    assert "JIM SIMONS — FUND MANAGER REPORT" in sent["message"]
    assert "Mode: 📋 PAPER" in sent["message"]


def test_performance_report_shows_empty_open_positions(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    notifier = TelegramNotifier({"telegram": {}})

    sent = {}

    async def fake_send(message: str) -> None:
        sent["message"] = message

    notifier._send = fake_send  # type: ignore[attr-defined]

    asyncio.run(
        notifier.performance_report(
            report={
                "trades": 3,
                "win_rate": 2 / 3,
                "profit_factor": 1.8,
                "avg_rr": 2.1,
                "drawdown_pct": 1.4,
                "ml_accuracy": 0.57,
                "ml_trend": "stable",
                "kelly_factor": 1.0,
            },
            equity=80.99,
            mode="paper",
            open_positions=[],
            recent_closed=[],
        )
    )

    assert "📌 *Open Positions:* _none_" in sent["message"]
    assert "🧭 *By Sleeve:* _none_" in sent["message"]
    assert "🎯 *By Exit:* _none_" in sent["message"]
    assert "🧾 *Recent Executions:* _none_" in sent["message"]


def test_performance_report_uses_floating_equity(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    notifier = TelegramNotifier({"telegram": {}})

    sent = {}

    async def fake_send(message: str) -> None:
        sent["message"] = message

    notifier._send = fake_send  # type: ignore[attr-defined]

    asyncio.run(
        notifier.performance_report(
            report={
                "trades": 3,
                "win_rate": 2 / 3,
                "profit_factor": 1.8,
                "avg_rr": 2.1,
                "drawdown_pct": 1.4,
                "ml_accuracy": 0.57,
                "ml_trend": "stable",
                "kelly_factor": 1.0,
            },
            equity=80.0,
            mode="paper",
            floating_positions=[{"pnl_usd": 5.0}, {"pnl_usd": -1.0}],
            open_positions=[],
            recent_closed=[],
        )
    )

    assert "💰 *Equity:* `$84.00`" in sent["message"]


def test_cycle_report_includes_active_session_in_header(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    notifier = TelegramNotifier({"telegram": {}})

    sent = {}

    async def fake_send(message: str) -> None:
        sent["message"] = message

    notifier._send = fake_send  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "src.notifications.telegram.active_market_session_label",
        lambda: "London",
    )

    asyncio.run(
        notifier.cycle_report(
            breakdowns=[],
            equity=80.0,
            drawdown_pct=0.0,
            daily_pnl_pct=0.0,
            open_trades=0,
            cycle_num=11,
            total_trades=0,
            starting_equity=80.0,
        )
    )

    assert "JIM SIMONS — FUND MANAGER REPORT" in sent["message"]
    assert "Session: *London*" in sent["message"]
    assert "🏦 Balance: *$80.00 USDT*" in sent["message"]
    assert sent["message"].index("🏦 Balance: *$80.00 USDT*") < sent["message"].index("💰 Equity: *$80.00 USDT*")


def test_cycle_report_includes_extra_performance_metrics(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    notifier = TelegramNotifier({"telegram": {}})

    sent = {}

    async def fake_send(message: str) -> None:
        sent["message"] = message

    notifier._send = fake_send  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "src.notifications.telegram.active_market_session_label",
        lambda: "London",
    )

    trade_log = [
        SimpleNamespace(pnl_usd=10.0, pnl_pct=10.0, opened_at=1716000000.0, closed_at=1716086400.0),
        SimpleNamespace(pnl_usd=-5.0, pnl_pct=-5.0, opened_at=1716086400.0, closed_at=1716172800.0),
        SimpleNamespace(pnl_usd=8.0, pnl_pct=8.0, opened_at=1716172800.0, closed_at=1716259200.0),
        SimpleNamespace(pnl_usd=-4.0, pnl_pct=-4.0, opened_at=1716259200.0, closed_at=1716345600.0),
    ]

    asyncio.run(
        notifier.cycle_report(
            breakdowns=[],
            equity=109.0,
            drawdown_pct=2.5,
            daily_pnl_pct=0.0,
            open_trades=0,
            cycle_num=12,
            total_trades=4,
            starting_equity=100.0,
            trade_log=trade_log,
            equity_curve=[100.0, 110.0, 105.0, 109.0],
        )
    )

    assert "PF `" in sent["message"]
    assert "WR `" in sent["message"]
    assert "Z-Score `" in sent["message"]
    assert "GHPR `" in sent["message"]
    assert "CAGR `" in sent["message"]
    assert "MAR `" in sent["message"]
    assert "Sharpe `" in sent["message"]
    assert "Sortino `" in sent["message"]
    assert "Avg W `" in sent["message"]
    assert "Avg L `" in sent["message"]
    assert "Avg W/L `" in sent["message"]
    assert "Recovery `" in sent["message"]


def test_cycle_report_normalizes_string_floating_values(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    notifier = TelegramNotifier({"telegram": {}})

    sent = {}

    async def fake_send(message: str) -> None:
        sent["message"] = message

    notifier._send = fake_send  # type: ignore[attr-defined]

    asyncio.run(
        notifier.cycle_report(
            breakdowns=[],
            equity="80.00",
            drawdown_pct="1.5",
            daily_pnl_pct="0.2",
            open_trades="2",
            cycle_num=3,
            total_trades=1,
            starting_equity=80.0,
            floating_positions=[{"pnl_usd": "1.25"}, {"pnl_usd": None}],
            open_positions=[{"symbol": "BTC/USDT:USDT", "direction": "long", "entry": "1.0", "size_usd": "2.0"}],
        )
    )

    assert "💰 Equity: *$81.25 USDT*" in sent["message"]
    assert "Open Positions" in sent["message"]


def test_cycle_report_shows_no_symbols_scored(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    notifier = TelegramNotifier({"telegram": {}})

    sent = {}

    async def fake_send(message: str) -> None:
        sent["message"] = message

    notifier._send = fake_send  # type: ignore[attr-defined]

    asyncio.run(
        notifier.cycle_report(
            breakdowns=[],
            equity=80.0,
            drawdown_pct=0.0,
            daily_pnl_pct=0.0,
            open_trades=0,
            cycle_num=11,
            total_trades=0,
            starting_equity=80.0,
        )
    )

    assert "No symbols scored this cycle" in sent["message"]


def test_monthly_report_tolerates_sparse_report(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    notifier = TelegramNotifier({"telegram": {}})

    sent = {}

    async def fake_send(message: str) -> None:
        sent["message"] = message

    notifier._send = fake_send  # type: ignore[attr-defined]

    asyncio.run(
        notifier.monthly_report(
            {
                "trades": 4,
                "mode": "paper",
                "monthly_trades": "4",
                "monthly_days": "30",
                "equity": "82.50",
                "peak_equity": "84.00",
                "monthly_return_pct": "3.25",
                "win_rate": "0.5",
                "profit_factor": "1.75",
                "avg_rr": "2.1",
                "drawdown_pct": "1.2",
                "net_pnl": "2.5",
                "best_trade": "1.0",
                "worst_trade": "-0.5",
                "sharpe": "1.4",
                "sortino": "2.2",
                "calmar": "0.8",
            }
        )
    )

    assert "MONTHLY STATEMENT" in sent["message"]
    assert "Equity: *$82.50*" in sent["message"]
    assert "Profit Factor: `1.75`" in sent["message"]


def test_equity_graph_report_sends_png_photo(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    notifier = TelegramNotifier({"telegram": {}})

    calls = []
    responses = [_FakeResponse(200, '{"ok":true}')]

    monkeypatch.setattr(
        "src.notifications.telegram.httpx.AsyncClient",
        lambda timeout: _FakeAsyncClient(responses, calls),
    )

    asyncio.run(
        notifier.equity_graph_report(
            [
                {"ts": 1716000000.0, "balance": 80.0, "equity": 80.0},
                {"ts": 1716003600.0, "balance": 81.0, "equity": 82.5},
            ],
            cycle_num=12,
            interval_minutes=60,
            mode="paper",
        )
    )

    assert calls[0][0] == notifier._photo_url
    assert calls[0][2]["caption"].startswith("JIM SIMONS — BALANCE / EQUITY GRAPH")
    assert "DD:" in calls[0][2]["caption"]
    photo_name, photo_bytes, mime_type = calls[0][3]["photo"]
    assert photo_name == "equity_graph.png"
    assert mime_type == "image/png"
    assert photo_bytes.startswith(b"\x89PNG")


def test_bootstrap_audit_report_uses_separate_header(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    notifier = TelegramNotifier({"telegram": {}})

    sent = {}

    async def fake_send(message: str) -> None:
        sent["message"] = message

    notifier._send = fake_send  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "src.notifications.telegram.active_market_session_label",
        lambda: "Asia",
    )

    asyncio.run(
        notifier.bootstrap_audit_report(
            ["Phase: `BOOTSTRAP` | N `12`", "Overall: WR `50.0%`"],
            cycle_num=77,
            interval_hours=12,
        )
    )

    assert "BOOTSTRAP AUDIT REPORT" in sent["message"]
    assert "`Cadence: 12h`" in sent["message"]
    assert "`Session: Asia`" in sent["message"]
    assert "Separate from Jim Simons report" in sent["message"]


def test_restored_positions_reports_empty_state(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    notifier = TelegramNotifier({"telegram": {}})

    sent = {}

    async def fake_send(message: str) -> None:
        sent["message"] = message

    notifier._send = fake_send  # type: ignore[attr-defined]

    asyncio.run(notifier.restored_positions([], mode="paper"))

    assert "Count: `0`" in sent["message"]
    assert "No open positions to restore." in sent["message"]


def test_shutdown_reports_closed_positions(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    notifier = TelegramNotifier({"telegram": {}})

    sent = {}

    async def fake_send(message: str) -> None:
        sent["message"] = message

    notifier._send = fake_send  # type: ignore[attr-defined]

    asyncio.run(
        notifier.shutdown(
            mode="paper",
            equity=80.0,
            open_trades=2,
            close_positions=True,
        )
    )

    assert "shutting down" in sent["message"].lower()
    assert "Open positions at stop: `2`" in sent["message"]
    assert "Closing open positions before exit." in sent["message"]


def test_shutdown_reports_preserved_positions(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    notifier = TelegramNotifier({"telegram": {}})

    sent = {}

    async def fake_send(message: str) -> None:
        sent["message"] = message

    notifier._send = fake_send  # type: ignore[attr-defined]

    asyncio.run(
        notifier.shutdown(
            mode="live",
            equity=81.5,
            open_trades=1,
            close_positions=False,
        )
    )

    assert "Mode: `LIVE`" in sent["message"]
    assert "Open positions preserved on shutdown." in sent["message"]
