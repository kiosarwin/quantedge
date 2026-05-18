import asyncio
import os
from datetime import datetime

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

    async def post(self, url, json):
        self._calls.append((url, json))
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
