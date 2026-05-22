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


def test_trade_manager_tp2_closes_pct_of_original_size(monkeypatch, tmp_path):
    """TP2 partial exit must close `tp2_size_pct` of the *original* position.

    Historic implementation closed `tp2_size_pct * remaining_contracts`,
    which after a 50%-at-TP1 partial exit only liquidated 15% of the
    original instead of the configured 30%, leaving the trail bucket too
    fat (35%) and diverging from the exchange-side reduceOnly TP order.
    """
    cfg = {"exit": {}, "ev_model": {}}
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
        strategy_sleeve="default",
        atr=2.0,
        risk_pct=1.0,
        exit_profile="default",
        tp1_size_pct=0.50,
        tp2_size_pct=0.30,
        trail_size_pct=0.20,
        breakeven_trigger_r=1.0,
        trailing_atr_multiplier=1.5,
        max_hold_duration_s=9999,
    )

    trade = asyncio.run(manager.open(setup))
    assert trade is not None

    # TP1 at 103 -> close 50% of 1.0 = 0.5; remainder 0.5
    asyncio.run(manager._check_exits(trade, 103.0))
    assert trade.tp1_hit is True
    assert trade.remaining_contracts == pytest.approx(0.50)

    # TP2 at 106 -> must close 30% of *original* (1.0) = 0.30,
    # leaving the configured trail_size_pct = 0.20 to ride to TP3.
    asyncio.run(manager._check_exits(trade, 106.0))
    assert trade.tp2_hit is True
    assert trade.remaining_contracts == pytest.approx(0.20)


def test_trade_manager_stamps_actual_exit_price_for_every_reason(monkeypatch, tmp_path):
    """`OpenTrade.exit_price` must be the actual market price for every exit.

    Specifically covers the early_adverse_cut / max_hold / manual paths
    where the historic mapping fell back to entry_price and produced
    misleading TradeRecord.exit_price downstream.
    """
    cfg = {
        "exit": {
            "early_cut": {
                "enabled": True,
                "min_age_s": 0.0,
                "max_age_s": 99999.0,
                "mae_r_threshold": 0.5,
                "mfe_r_ceiling": 0.20,
            }
        },
        "ev_model": {},
    }
    executor = _FakeExecutor()
    risk = _FakeRisk()
    monkeypatch.setattr(TradeManager, "STATE_PATH", tmp_path / "open_trades.json")
    manager = TradeManager(client=object(), executor=executor, risk=risk, cfg=cfg)

    setup = TradeSetup(
        symbol="ETH/USDT:USDT",
        direction="long",
        entry_price=100.0,
        stop_loss=90.0,
        tp1=110.0,
        tp2=120.0,
        tp3=130.0,
        size_usd=100.0,
        size_contracts=1.0,
        leverage=5,
        r_distance=10.0,
        strategy_sleeve="default",
        atr=5.0,
        risk_pct=1.0,
        exit_profile="default",
        tp1_size_pct=0.50,
        tp2_size_pct=0.30,
        trail_size_pct=0.20,
        breakeven_trigger_r=1.0,
        trailing_atr_multiplier=1.5,
        max_hold_duration_s=9999,
    )

    trade = asyncio.run(manager.open(setup))
    assert trade is not None
    # 0.6R adverse without any favourable move -> early_adverse_cut at 94.0.
    asyncio.run(manager._check_exits(trade, 94.0))
    assert trade.exit_price == pytest.approx(94.0)
    assert trade.symbol not in manager.open_symbols


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
