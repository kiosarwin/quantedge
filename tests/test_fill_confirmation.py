"""Tests for Executor._confirm_fill method."""
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.execution.executor import Executor


def _make_executor(paper=False):
    """Create an Executor with a mocked client."""
    client = MagicMock()
    client.fetch_order = AsyncMock()
    cfg = {"trading": {"mode": "paper" if paper else "live"}}
    executor = Executor(client, cfg)
    return executor, client


@pytest.mark.asyncio
async def test_confirm_fill_returns_order_on_closed():
    """Returns confirmed order when status is 'closed'."""
    executor, client = _make_executor()
    client.fetch_order.return_value = {
        "id": "123",
        "status": "closed",
        "filled": 1.0,
        "amount": 1.0,
    }
    result = await executor._confirm_fill("BTCUSDT", "123", max_retries=3)
    assert result is not None
    assert result["status"] == "closed"
    assert client.fetch_order.call_count == 1


@pytest.mark.asyncio
async def test_confirm_fill_returns_none_on_rejected():
    """Returns None when order status is 'rejected'."""
    executor, client = _make_executor()
    client.fetch_order.return_value = {
        "id": "456",
        "status": "rejected",
    }
    result = await executor._confirm_fill("ETHUSDT", "456", max_retries=3)
    assert result is None


@pytest.mark.asyncio
async def test_confirm_fill_returns_none_on_cancelled():
    """Returns None when order status is 'cancelled'."""
    executor, client = _make_executor()
    client.fetch_order.return_value = {
        "id": "789",
        "status": "cancelled",
    }
    result = await executor._confirm_fill("ETHUSDT", "789", max_retries=2)
    assert result is None


@pytest.mark.asyncio
async def test_confirm_fill_returns_none_after_max_retries():
    """Returns None if order stays 'open' after all retries."""
    executor, client = _make_executor()
    client.fetch_order.return_value = {
        "id": "111",
        "status": "open",
    }
    result = await executor._confirm_fill("SOLUSDT", "111", max_retries=3)
    assert result is None
    assert client.fetch_order.call_count == 3


@pytest.mark.asyncio
async def test_confirm_fill_partial_fill_still_returns_order():
    """Partial fill (filled < amount) logs warning but still returns order."""
    executor, client = _make_executor()
    client.fetch_order.return_value = {
        "id": "222",
        "status": "closed",
        "filled": 0.5,
        "amount": 1.0,
    }
    result = await executor._confirm_fill("BTCUSDT", "222", max_retries=3)
    assert result is not None
    assert result["filled"] == 0.5


@pytest.mark.asyncio
async def test_confirm_fill_handles_fetch_exception():
    """If fetch_order raises, retries until max_retries then returns None."""
    executor, client = _make_executor()
    client.fetch_order.side_effect = Exception("network error")
    result = await executor._confirm_fill("BTCUSDT", "333", max_retries=2)
    assert result is None
    assert client.fetch_order.call_count == 2
