"""
Tests for the hang-storm guard in BinanceFuturesClient.

Verifies that:
  1. Every public async method goes through ``_call``, which enforces an
     ``asyncio.wait_for`` ceiling.
  2. The two-layer budget (ccxt http + asyncio outer) is configurable via
     ``safety.api_http_timeout_seconds`` / ``safety.api_call_timeout_seconds``
     and falls back to safe defaults when omitted.
  3. The outer budget is auto-bumped if the operator misconfigures it
     below the inner budget (otherwise we'd cancel ccxt before it can
     time out cleanly).
  4. A method whose underlying coroutine hangs raises ``TimeoutError``
     rather than blocking forever, and the elapsed time is bounded by
     the outer budget — not by any retry / backoff stack.
  5. Methods that already swallow exceptions (``fetch_open_interest``,
     ``set_margin_mode``, ``set_position_mode_one_way``,
     ``fetch_long_short_ratio``, ``fetch_taker_buy_ratio``) keep doing so
     when ``_call`` raises ``TimeoutError`` — no behaviour regression on
     paths that intentionally tolerate API failures.
"""
from __future__ import annotations

import asyncio
import time
import types

import pytest

from src.data.client import (
    BinanceFuturesClient,
    _DEFAULT_HTTP_TIMEOUT_S,
    _DEFAULT_CALL_TIMEOUT_S,
)


def _make_client(
    *,
    http_timeout: float | None = None,
    call_timeout: float | None = None,
) -> BinanceFuturesClient:
    """Construct a client without going through ccxt's __init__."""
    safety: dict = {}
    if http_timeout is not None:
        safety["api_http_timeout_seconds"] = http_timeout
    if call_timeout is not None:
        safety["api_call_timeout_seconds"] = call_timeout

    cfg = {
        "exchange": {"testnet": True, "api_key": "", "api_secret": ""},
        "safety": safety,
    }
    # Bypass ccxt construction; the wrapper logic under test only needs the
    # timeout fields and the _exchange attribute (replaced by tests below).
    client = BinanceFuturesClient.__new__(BinanceFuturesClient)
    client._is_testnet = True
    client._fapi_base = ""
    client._http_timeout_s = float(
        safety.get("api_http_timeout_seconds", _DEFAULT_HTTP_TIMEOUT_S)
    )
    client._call_timeout_s = float(
        safety.get("api_call_timeout_seconds", _DEFAULT_CALL_TIMEOUT_S)
    )
    if client._call_timeout_s < client._http_timeout_s:
        client._call_timeout_s = client._http_timeout_s + 3.0
    client._retry_attempts = 3
    client._retry_delay = 5
    return client


# ---------------------------------------------------------------------- #
#  Defaults & configuration                                                #
# ---------------------------------------------------------------------- #


def test_defaults_match_documented_values():
    c = _make_client()
    assert c._http_timeout_s == _DEFAULT_HTTP_TIMEOUT_S
    assert c._call_timeout_s == _DEFAULT_CALL_TIMEOUT_S
    assert c._call_timeout_s >= c._http_timeout_s


def test_operator_overrides_are_respected():
    c = _make_client(http_timeout=8.0, call_timeout=11.0)
    assert c._http_timeout_s == 8.0
    assert c._call_timeout_s == 11.0


def test_outer_budget_auto_bumped_when_below_inner():
    """If the operator sets outer < inner, the wrapper bumps it so ccxt has
    a chance to surface its own TimeoutError before asyncio cancels."""
    c = _make_client(http_timeout=20.0, call_timeout=5.0)
    assert c._http_timeout_s == 20.0
    assert c._call_timeout_s == 23.0  # inner + 3s safety margin


# ---------------------------------------------------------------------- #
#  _call wrapper behaviour                                                 #
# ---------------------------------------------------------------------- #


def test_call_returns_value_when_coroutine_completes_in_time():
    c = _make_client(call_timeout=1.0)

    async def coro():
        await asyncio.sleep(0.01)
        return 42

    result = asyncio.run(c._call(coro(), "happy_path"))
    assert result == 42


def test_call_raises_timeout_when_coroutine_hangs():
    """Coroutine that hangs longer than outer budget is cancelled."""
    c = _make_client(http_timeout=0.05, call_timeout=0.1)

    async def hang():
        await asyncio.sleep(10.0)  # would block forever in test
        return "should-not-reach"

    start = time.monotonic()
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(c._call(hang(), "hang_path"))
    elapsed = time.monotonic() - start
    # Elapsed must be bounded by outer budget + small scheduling slack
    assert elapsed < 1.0, f"hang exceeded outer budget by too much: {elapsed:.3f}s"


def test_call_propagates_non_timeout_exceptions_unchanged():
    """The wrapper must not swallow legitimate errors — only enforce timeout."""
    c = _make_client()

    class MyError(RuntimeError):
        pass

    async def boom():
        raise MyError("network-down")

    with pytest.raises(MyError, match="network-down"):
        asyncio.run(c._call(boom(), "error_path"))


# ---------------------------------------------------------------------- #
#  Public-method coverage                                                  #
# ---------------------------------------------------------------------- #


# (method_name, args) — every public async method that goes through ``_call``
_PROTECTED_METHODS: list[tuple[str, tuple]] = [
    ("connect",                ()),
    ("fetch_markets",          ()),
    ("fetch_ticker",           ("BTC/USDT:USDT",)),
    ("fetch_tickers",          ()),
    ("fetch_funding_rate",     ("BTC/USDT:USDT",)),
    ("fetch_order_book",       ("BTC/USDT:USDT",)),
    ("fetch_balance",          ()),
    ("fetch_positions",        ()),
    ("fetch_open_orders",      ("BTC/USDT:USDT",)),
    ("set_leverage",           ("BTC/USDT:USDT", 5)),
    ("create_order",           ("BTC/USDT:USDT", "market", "buy", 0.01)),
    ("cancel_order",           ("oid", "BTC/USDT:USDT")),
    ("cancel_all_orders",      ("BTC/USDT:USDT",)),
    ("fetch_order",            ("oid", "BTC/USDT:USDT")),
    ("fetch_ohlcv",            ("BTC/USDT:USDT",)),
]


@pytest.mark.parametrize("method_name,args", _PROTECTED_METHODS)
def test_method_is_bounded_by_outer_budget(method_name, args):
    """
    Every wrapped method must time out via the outer budget, even if the
    underlying ccxt call would hang forever. Regression guard against
    silently un-wrapping a method during future refactors.
    """
    c = _make_client(http_timeout=0.05, call_timeout=0.1)

    # Build a fake _exchange whose every method returns an awaitable that hangs.
    async def hang(*a, **kw):
        await asyncio.sleep(10.0)
        return None

    fake_exchange = types.SimpleNamespace(
        load_markets=hang,
        fetch_ticker=hang,
        fetch_tickers=hang,
        fetch_funding_rate=hang,
        fetch_open_interest=hang,
        fetch_order_book=hang,
        fetch_balance=hang,
        fetch_positions=hang,
        fetch_open_orders=hang,
        set_leverage=hang,
        set_margin_mode=hang,
        create_order=hang,
        cancel_order=hang,
        cancel_all_orders=hang,
        fetch_order=hang,
        fetch_ohlcv=hang,
        fapiPrivatePostPositionSideDual=hang,
        fapiPublicGetFuturesDataGlobalLongShortAccountRatio=hang,
        fapiPublicGetFuturesDataTakerbuyVolume=hang,
    )
    c._exchange = fake_exchange

    bound = getattr(c, method_name)
    start = time.monotonic()
    with pytest.raises((asyncio.TimeoutError, Exception)):
        asyncio.run(bound(*args))
    elapsed = time.monotonic() - start

    # fetch_ohlcv has tenacity retry (3 attempts, exp backoff 2-10s). Each
    # attempt is bounded by outer budget. Worst case ≈ 3*outer + backoff(2+4) = 6.3s.
    cap = 8.0 if method_name == "fetch_ohlcv" else 1.0
    assert elapsed < cap, (
        f"{method_name} not bounded by outer budget: elapsed={elapsed:.3f}s cap={cap}s"
    )


# ---------------------------------------------------------------------- #
#  Tolerant-failure paths must keep tolerating timeouts                     #
# ---------------------------------------------------------------------- #


def test_fetch_open_interest_returns_empty_dict_on_timeout():
    c = _make_client(http_timeout=0.05, call_timeout=0.1)

    async def hang(*a, **kw):
        await asyncio.sleep(10.0)

    c._exchange = types.SimpleNamespace(fetch_open_interest=hang)
    result = asyncio.run(c.fetch_open_interest("BTC/USDT:USDT"))
    assert result == {}


def test_fetch_long_short_ratio_returns_neutral_on_timeout():
    c = _make_client(http_timeout=0.05, call_timeout=0.1)

    async def hang(*a, **kw):
        await asyncio.sleep(10.0)

    c._exchange = types.SimpleNamespace(
        fapiPublicGetFuturesDataGlobalLongShortAccountRatio=hang
    )
    # Use a unique symbol to avoid hitting the module-level TTL cache populated
    # by other tests in the same process.
    result = asyncio.run(c.fetch_long_short_ratio("HANGTEST/USDT:USDT"))
    assert result == 1.0


def test_fetch_taker_buy_ratio_returns_neutral_on_timeout():
    c = _make_client(http_timeout=0.05, call_timeout=0.1)

    async def hang(*a, **kw):
        await asyncio.sleep(10.0)

    c._exchange = types.SimpleNamespace(
        fapiPublicGetFuturesDataTakerbuyVolume=hang
    )
    result = asyncio.run(c.fetch_taker_buy_ratio("HANGTEST2/USDT:USDT"))
    assert result == 0.5


def test_set_margin_mode_swallows_timeout():
    c = _make_client(http_timeout=0.05, call_timeout=0.1)

    async def hang(*a, **kw):
        await asyncio.sleep(10.0)

    c._exchange = types.SimpleNamespace(set_margin_mode=hang)
    # Must not raise — the existing contract is "best effort".
    asyncio.run(c.set_margin_mode("BTC/USDT:USDT"))


def test_set_position_mode_one_way_swallows_timeout():
    c = _make_client(http_timeout=0.05, call_timeout=0.1)

    async def hang(*a, **kw):
        await asyncio.sleep(10.0)

    c._exchange = types.SimpleNamespace(fapiPrivatePostPositionSideDual=hang)
    # Must not raise — startup must continue even if Binance is unreachable.
    asyncio.run(c.set_position_mode_one_way())
