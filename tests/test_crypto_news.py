import asyncio
from types import SimpleNamespace

import pytest

from src.data.crypto_news import build_market_hot_narratives, build_rule_based_market_narratives, fetch_crypto_news_highlights, parse_crypto_news_feed
from src.notifications.telegram import TelegramNotifier


RSS = """
<rss><channel>
  <item>
    <title>Breaking: Bitcoin ETF approved after SEC review</title>
    <link>https://example.com/btc-etf</link>
    <pubDate>Sun, 24 May 2026 08:00:00 GMT</pubDate>
  </item>
  <item>
    <title>Altcoin market update</title>
    <link>https://example.com/alts</link>
    <pubDate>Sun, 24 May 2026 07:00:00 GMT</pubDate>
  </item>
  <item>
    <title>Crypto, banks and policy experts press Congress to modernize banking rules</title>
    <link>https://example.com/policy</link>
    <pubDate>Sun, 24 May 2026 08:30:00 GMT</pubDate>
  </item>
</channel></rss>
"""


def test_parse_crypto_news_feed_ranks_urgent_headline():
    items = parse_crypto_news_feed(RSS, source_url="https://example.com/rss", now_ts=1779613200)

    assert len(items) == 2
    assert items[0].title.startswith("Breaking")
    assert items[0].source == "example.com"
    assert items[0].urgent_score > items[1].urgent_score


def test_parse_crypto_news_feed_filters_broad_policy_without_btc_or_altcoin_scope():
    items = parse_crypto_news_feed(RSS, source_url="https://example.com/rss", now_ts=1779613200)

    titles = [item.title for item in items]
    assert all("banking rules" not in title for title in titles)


class _Resp:
    status_code = 200
    text = RSS


class _Client:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get(self, url, **kwargs):
        return _Resp()


def test_fetch_crypto_news_highlights_dedupes_and_limits(monkeypatch):
    monkeypatch.setattr("src.data.crypto_news.httpx.AsyncClient", _Client)

    items = asyncio.run(
        fetch_crypto_news_highlights(
            {"telegram": {"crypto_news_feeds": ["https://example.com/rss", "https://example.com/rss"]}},
            limit=1,
        )
    )

    assert len(items) == 1
    assert "Bitcoin ETF" in items[0].title


def test_rule_based_market_narratives_group_headlines():
    items = parse_crypto_news_feed(RSS, source_url="https://example.com/rss", now_ts=1779613200)

    narratives = build_rule_based_market_narratives(items, limit=5)

    assert narratives
    assert narratives[0].title == "BTC / ETF flow"
    assert narratives[0].bias == "bullish"
    assert narratives[0].recommended_altcoins
    assert "ETH" in narratives[0].recommended_altcoins
    assert "Bitcoin ETF" in narratives[0].evidence[0]


def test_build_market_hot_narratives_uses_llm_when_configured(monkeypatch):
    item = SimpleNamespace(
        title="Hyperliquid HYPE hits record as perp DEX volume surges",
        url="https://example.com/hype",
        source="example.com",
        urgent_score=12.5,
        published_ts=1779613200,
    )

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '[{"title":"Perp DEX revenue trade", "bias":"bullish", '
                                '"explanation":"HYPE strength is tied to perp DEX activity and token demand.", '
                                '"evidence_ids":[1], "recommended_altcoins":["HYPE"], '
                                '"recommendation_reason":"Perp DEX volume and token demand directly support HYPE.", '
                                '"urgency_score":12.5, "confidence":0.82}]'
                            )
                        }
                    }
                ]
            }

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, json=None):
            return _Resp()

    for env_name in ("OPENAI_API_KEY", "GROQ_API_KEY", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("src.data.crypto_news.httpx.AsyncClient", _Client)

    narratives = asyncio.run(
        build_market_hot_narratives(
            {"telegram": {"crypto_news_llm_enabled": True}},
            [item],
            limit=5,
        )
    )

    assert narratives[0].source == "llm:openrouter"
    assert narratives[0].title == "Perp DEX revenue trade"
    assert narratives[0].recommended_altcoins == ("HYPE",)
    assert "Perp DEX" in narratives[0].recommendation_reason
    assert narratives[0].evidence == ("Hyperliquid HYPE hits record as perp DEX volume surges",)


def test_build_market_hot_narratives_prefers_gemini_when_available(monkeypatch):
    item = SimpleNamespace(
        title="Bitcoin ETF inflows lift crypto risk appetite",
        url="https://example.com/btc",
        source="example.com",
        urgent_score=11.0,
        published_ts=1779613200,
    )
    calls = []
    payloads = []

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '[{"title":"BTC institutional flow", "bias":"bullish", '
                                '"explanation":"ETF inflow headlines point to stronger BTC-led risk appetite.", '
                                '"evidence_ids":[1], "recommended_altcoins":["ETH","SOL"], '
                                '"recommendation_reason":"BTC-led risk-on favors liquid large-cap alt beta.", '
                                '"urgency_score":11.0, "confidence":0.81}]'
                            )
                        }
                    }
                ]
            }

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, json=None):
            calls.append((url, json.get("model")))
            payloads.append(json)
            return _Resp()

    for env_name in ("OPENAI_API_KEY", "GROQ_API_KEY", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setattr("src.data.crypto_news.httpx.AsyncClient", _Client)

    narratives = asyncio.run(
        build_market_hot_narratives(
            {"telegram": {"crypto_news_llm_enabled": True}},
            [item],
            limit=5,
        )
    )

    assert calls == [("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "gemini-2.5-flash")]
    assert payloads[0]["reasoning_effort"] == "none"
    assert payloads[0]["max_tokens"] == 1800
    assert narratives[0].source == "llm:gemini"
    assert narratives[0].title == "BTC institutional flow"
    assert narratives[0].recommended_altcoins == ("ETH", "SOL")


def test_build_market_hot_narratives_falls_back_to_next_public_provider(monkeypatch):
    item = SimpleNamespace(
        title="NEAR rallies as AI crypto narrative heats up",
        url="https://example.com/near",
        source="example.com",
        urgent_score=9.0,
        published_ts=1779613200,
    )
    calls = []

    class _Resp:
        def __init__(self, status_code, content=""):
            self.status_code = status_code
            self._content = content

        def json(self):
            return {"choices": [{"message": {"content": self._content}}]}

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, json=None):
            calls.append((url, json.get("model")))
            if "openrouter" in url:
                return _Resp(429)
            return _Resp(
                200,
                '[{"title":"AI alt rotation", "bias":"bullish", '
                '"explanation":"NEAR headline points to AI-linked altcoin demand.", '
                '"evidence_ids":[1], "recommended_altcoins":["NEAR"], '
                '"recommendation_reason":"NEAR is directly named in the AI rotation headline.", '
                '"urgency_score":9.0, "confidence":0.74}]',
            )

    for env_name in ("OPENAI_API_KEY", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setattr("src.data.crypto_news.httpx.AsyncClient", _Client)

    narratives = asyncio.run(
        build_market_hot_narratives(
            {"telegram": {"crypto_news_llm_enabled": True}},
            [item],
            limit=5,
        )
    )

    assert [call[1] for call in calls] == ["openrouter/free", "llama-3.3-70b-versatile"]
    assert narratives[0].source == "llm:groq"
    assert narratives[0].title == "AI alt rotation"
    assert narratives[0].recommended_altcoins == ("NEAR",)


def test_telegram_crypto_news_report_renders_market_narratives(monkeypatch):
    notifier = TelegramNotifier({"telegram": {}})
    notifier._enabled = True
    sent = {}

    async def fake_send(message):
        sent["message"] = message

    monkeypatch.setattr(notifier, "_send", fake_send)
    item = SimpleNamespace(
        title="Perp DEX revenue trade",
        bias="bullish",
        explanation="Hyperliquid strength is tied to exchange volume and token demand.",
        evidence=("Hyperliquid HYPE hits record as perp DEX volume surges",),
        recommended_altcoins=("HYPE",),
        recommendation_reason="Perp DEX volume and token demand directly support HYPE.",
        source="llm",
        urgency_score=12.5,
        confidence=0.82,
    )

    asyncio.run(notifier.crypto_news_report([item], interval_minutes=60))

    assert "MARKET HOT NARRATIVE" in sent["message"]
    assert "Perp DEX revenue trade" in sent["message"]
    assert "Hyperliquid HYPE hits record" in sent["message"]
    assert "Buy watchlist: `HYPE`" in sent["message"]
    assert "Perp DEX volume" in sent["message"]
    assert "Source `llm`" in sent["message"]
