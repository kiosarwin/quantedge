import asyncio

import pytest

from src.execution.trade_manager import TradeManager
from src.risk.risk_manager import TradeSetup


class _FakeExecutor:
    def __init__(self):
        self.take_profit_calls = []

    async def open_position(self, setup):
        return {"id": "entry-1", "price": setup.entry_price}

    async def place_stop_loss(self, setup, entry_order_id):
        return {"id": "sl-1"}

    async def place_take_profit(self, setup, price, size_pct):
        self.take_profit_calls.append((price, size_pct))
        return {"id": f"tp-{len(self.take_profit_calls)}"}

    async def cancel_all(self, symbol):
        return None

    async def close_position_market(self, symbol, direction, size_contracts):
        return None


class _FakeRisk:
    def can_open_trade(self):
        return True, "ok"

    def check_trade_exposure(self, symbol, direction, risk_pct):
        return True, "ok"

    def on_trade_opened(self, **kwargs):
        return None

    def on_trade_closed(self, *args, **kwargs):
        return None


def test_trade_manager_uses_sleeve_specific_exit_profile(monkeypatch, tmp_path):
    cfg = {"exit": {}}
    executor = _FakeExecutor()
    risk = _FakeRisk()
    monkeypatch.setattr(TradeManager, "STATE_PATH", tmp_path / "open_trades.json")
    manager = TradeManager(client=object(), executor=executor, risk=risk, cfg=cfg)

    setup = TradeSetup(
        symbol="BTC/USDT:USDT",
        direction="long",
        entry_price=100.0,
        stop_loss=95.0,
        tp1=103.0,
        tp2=106.0,
        tp3=110.0,
        size_usd=100.0,
        size_contracts=1.0,
        leverage=5,
        r_distance=5.0,
        strategy_sleeve="compression_breakout",
        atr=2.0,
        risk_pct=1.0,
        exit_profile="compression_breakout",
        tp1_size_pct=0.40,
        tp2_size_pct=0.35,
        trail_size_pct=0.25,
        breakeven_trigger_r=0.9,
        trailing_atr_multiplier=1.4,
        max_hold_duration_s=9999,
    )

    trade = asyncio.run(manager.open(setup))
    assert trade is not None
    assert executor.take_profit_calls == [(103.0, 0.40), (106.0, 0.35)]

    asyncio.run(manager._check_exits(trade, 103.0))
    assert trade.tp1_hit is True
    assert trade.remaining_contracts == pytest.approx(0.60)
    assert trade.setup.stop_loss == pytest.approx(100.0)
    assert trade.trailing_stop == pytest.approx(100.2)


def test_trade_manager_ignores_invalid_zero_prices(monkeypatch, tmp_path):
    cfg = {"exit": {}}
    executor = _FakeExecutor()
    risk = _FakeRisk()
    monkeypatch.setattr(TradeManager, "STATE_PATH", tmp_path / "open_trades.json")
    manager = TradeManager(client=object(), executor=executor, risk=risk, cfg=cfg)

    setup = TradeSetup(
        symbol="ETH/USDT:USDT",
        direction="short",
        entry_price=100.0,
        stop_loss=105.0,
        tp1=97.0,
        tp2=94.0,
        tp3=90.0,
        size_usd=100.0,
        size_contracts=1.0,
        leverage=5,
        r_distance=5.0,
        strategy_sleeve="neutral",
        atr=2.0,
        risk_pct=1.0,
        exit_profile="default",
        tp1_size_pct=0.40,
        tp2_size_pct=0.35,
        trail_size_pct=0.25,
        breakeven_trigger_r=0.9,
        trailing_atr_multiplier=1.4,
        max_hold_duration_s=9999,
    )

    trade = asyncio.run(manager.open(setup))
    assert trade is not None

    asyncio.run(manager.monitor_all({"ETH/USDT:USDT": 0.0}))
    assert "ETH/USDT:USDT" in manager.open_symbols
    assert manager.get_floating_pnl({"ETH/USDT:USDT": 0.0}) == []
