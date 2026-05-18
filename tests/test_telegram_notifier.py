import os

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
