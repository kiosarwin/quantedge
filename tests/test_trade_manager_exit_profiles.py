import asyncio

import pytest

from src.execution.trade_manager import TradeManager
from src.risk.risk_manager import TradeSetup


class _FakeExecutor:
    def __init__(self):
        self.take_profit_calls = []
        self.move_stop_calls = []
        self.close_calls = []

    async def open_position(self, setup):
        return {"id": "entry-1", "price": setup.entry_price}

    async def place_stop_loss(self, setup, entry_order_id):
        return {"id": "sl-1"}

    async def place_take_profit(self, setup, price, size_pct):
        self.take_profit_calls.append((price, size_pct))
        return {"id": f"tp-{len(self.take_profit_calls)}"}

    async def move_stop_loss(self, symbol, direction, old_sl_id, new_stop_price, size_contracts):
        # Mirror the live executor: cancel old SL and emit a fresh order id.
        self.move_stop_calls.append({
            "symbol": symbol,
            "direction": direction,
            "old_sl_id": old_sl_id,
            "new_stop_price": new_stop_price,
            "size_contracts": size_contracts,
        })
        return {
            "id": f"sl-be-{len(self.move_stop_calls)}",
            "stopPrice": new_stop_price,
            "status": "NEW",
        }

    async def cancel_all(self, symbol):
        return None

    async def close_position_market(self, symbol, direction, size_contracts):
        self.close_calls.append((symbol, direction, size_contracts))
        return None


class _FakeRisk:
    def can_open_trade(self):
        return True, "ok"

    def check_trade_exposure(self, symbol, direction, risk_pct, sector="unknown"):
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


def test_trade_manager_tp1_refreshes_exchange_sl_to_breakeven(monkeypatch, tmp_path):
    """After TP1 partial fill, the exchange-side SL must be replaced.

    Historically the bot only updated the *local* trade.setup.stop_loss
    field; the reduceOnly stop on Binance still triggered at the
    original (wider) SL price, so an offline bot could leak the
    residual at the old stop instead of the configured break-even.
    """
    cfg = {"exit": {}, "ev_model": {}}
    executor = _FakeExecutor()
    risk = _FakeRisk()
    monkeypatch.setattr(TradeManager, "STATE_PATH", tmp_path / "open_trades.json")
    manager = TradeManager(client=object(), executor=executor, risk=risk, cfg=cfg)

    setup = TradeSetup(
        symbol="SOL/USDT:USDT",
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
    assert trade.sl_order_id == "sl-1"

    asyncio.run(manager._check_exits(trade, 103.0))

    # Local invariants — still hold
    assert trade.tp1_hit is True
    assert trade.setup.stop_loss == pytest.approx(100.0)

    # Exchange SL replacement was issued exactly once with:
    #   - the OLD sl id we placed at trade open
    #   - the residual (post-TP1) contract count, not the original
    #   - the new stop price = entry (break-even)
    assert len(executor.move_stop_calls) == 1
    call = executor.move_stop_calls[0]
    assert call["symbol"] == "SOL/USDT:USDT"
    assert call["direction"] == "long"
    assert call["old_sl_id"] == "sl-1"
    assert call["new_stop_price"] == pytest.approx(100.0)
    assert call["size_contracts"] == pytest.approx(0.50)

    # Local sl_order_id now tracks the fresh SL on the exchange.
    assert trade.sl_order_id == "sl-be-1"


def test_trade_manager_tp1_tolerates_move_stop_loss_failure(monkeypatch, tmp_path):
    """If the SL refresh blows up (network glitch, Binance rejection),
    the trade must NOT crash — the local stop_loss is still authoritative
    on the next tick and the bot will market-close if the local SL is
    breached."""
    cfg = {"exit": {}, "ev_model": {}}
    risk = _FakeRisk()

    class _BrokenExecutor(_FakeExecutor):
        async def move_stop_loss(self, *args, **kwargs):
            raise RuntimeError("simulated exchange outage")

    executor = _BrokenExecutor()
    monkeypatch.setattr(TradeManager, "STATE_PATH", tmp_path / "open_trades.json")
    manager = TradeManager(client=object(), executor=executor, risk=risk, cfg=cfg)

    setup = TradeSetup(
        symbol="SOL/USDT:USDT",
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

    # Should not raise even though move_stop_loss explodes.
    asyncio.run(manager._check_exits(trade, 103.0))
    assert trade.tp1_hit is True
    assert trade.setup.stop_loss == pytest.approx(100.0)
    assert trade.symbol in manager.open_symbols
    # Local sl_order_id stays on the original SL — caller should treat
    # the exchange SL as stale (this is the documented degraded mode).
    assert trade.sl_order_id == "sl-1"


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


def test_trade_manager_persists_setup_passport(monkeypatch, tmp_path):
    cfg = {"exit": {}, "ev_model": {}}
    executor = _FakeExecutor()
    risk = _FakeRisk()
    state_path = tmp_path / "open_trades.json"
    monkeypatch.setattr(TradeManager, "STATE_PATH", state_path)
    manager = TradeManager(client=object(), executor=executor, risk=risk, cfg=cfg)

    passport = {
        "decision": "eligible",
        "setup_type": "trend_continuation",
        "sleeve": "trend_following",
        "expected_path": "impulse_continuation",
        "invalidation": "breaks_recent_structure_or_stop",
        "alignment": 0.66,
    }
    setup = TradeSetup(
        symbol="BTC/USDT:USDT",
        direction="long",
        entry_price=100.0,
        stop_loss=95.0,
        tp1=110.0,
        tp2=120.0,
        tp3=130.0,
        size_usd=100.0,
        size_contracts=1.0,
        leverage=5,
        r_distance=5.0,
        strategy_sleeve="trend_following",
        atr=2.0,
        risk_pct=1.0,
        setup_passport=passport,
    )

    trade = asyncio.run(manager.open(setup))
    assert trade is not None

    import json
    payload = json.loads(state_path.read_text())
    saved = payload["trades"][0]["setup"]["setup_passport"]
    assert saved["setup_type"] == "trend_continuation"
    assert saved["expected_path"] == "impulse_continuation"


def test_trade_manager_closes_failed_passport_breakout(monkeypatch, tmp_path):
    cfg = {
        "exit": {
            "passport_monitor": {
                "enabled": True,
                "min_age_s": 0.0,
                "breakout_failure_mae_r": 0.45,
                "breakout_mfe_ceiling_r": 0.25,
            },
            "early_cut": {"enabled": False},
        },
        "ev_model": {},
    }
    executor = _FakeExecutor()
    risk = _FakeRisk()
    monkeypatch.setattr(TradeManager, "STATE_PATH", tmp_path / "open_trades.json")
    manager = TradeManager(client=object(), executor=executor, risk=risk, cfg=cfg)

    setup = TradeSetup(
        symbol="SOL/USDT:USDT",
        direction="long",
        entry_price=100.0,
        stop_loss=90.0,
        tp1=115.0,
        tp2=125.0,
        tp3=140.0,
        size_usd=100.0,
        size_contracts=1.0,
        leverage=5,
        r_distance=10.0,
        strategy_sleeve="compression_breakout",
        atr=4.0,
        risk_pct=1.0,
        setup_passport={
            "decision": "eligible",
            "setup_type": "compression_breakout",
            "expected_path": "range_expansion",
            "invalidation": "failed_breakout_reenters_range",
        },
    )

    trade = asyncio.run(manager.open(setup))
    assert trade is not None

    # 0.5R adverse, no favourable movement: breakout contract failed.
    asyncio.run(manager._check_exits(trade, 95.0))

    assert "SOL/USDT:USDT" not in manager.open_symbols
    assert trade.exit_price == pytest.approx(95.0)
    assert executor.close_calls == [("SOL/USDT:USDT", "long", pytest.approx(1.0))]


def test_trade_manager_closes_failed_experimental_continuation_passport(monkeypatch, tmp_path):
    cfg = {
        "exit": {
            "passport_monitor": {
                "enabled": True,
                "min_age_s": 0.0,
                "experimental_continuation_failure_mae_r": 0.45,
                "experimental_continuation_mfe_ceiling_r": 0.35,
            },
            "early_cut": {"enabled": False},
        },
        "ev_model": {},
    }
    executor = _FakeExecutor()
    risk = _FakeRisk()
    monkeypatch.setattr(TradeManager, "STATE_PATH", tmp_path / "open_trades.json")
    manager = TradeManager(client=object(), executor=executor, risk=risk, cfg=cfg)

    setup = TradeSetup(
        symbol="BCH/USDT:USDT",
        direction="short",
        entry_price=350.0,
        stop_loss=355.0,
        tp1=340.0,
        tp2=332.5,
        tp3=325.0,
        size_usd=100.0,
        size_contracts=1.0,
        leverage=5,
        r_distance=5.0,
        strategy_sleeve="trend_following",
        atr=2.0,
        risk_pct=1.0,
        setup_passport={
            "decision": "eligible",
            "setup_type": "mtf_price_action_continuation",
            "expected_path": "mtf_impulse_continuation",
            "invalidation": "breaks_pullback_structure",
        },
    )

    trade = asyncio.run(manager.open(setup))
    assert trade is not None

    # Short trade: price rallies 0.5R adverse before any meaningful follow-through.
    asyncio.run(manager._check_exits(trade, 352.5))

    assert "BCH/USDT:USDT" not in manager.open_symbols
    assert trade.exit_price == pytest.approx(352.5)
    assert executor.close_calls == [("BCH/USDT:USDT", "short", pytest.approx(1.0))]


def test_trade_manager_keeps_experimental_continuation_after_meaningful_mfe(monkeypatch, tmp_path):
    cfg = {
        "exit": {
            "passport_monitor": {
                "enabled": True,
                "min_age_s": 0.0,
                "experimental_continuation_failure_mae_r": 0.45,
                "experimental_continuation_mfe_ceiling_r": 0.35,
            },
            "early_cut": {"enabled": False},
        },
        "ev_model": {},
    }
    executor = _FakeExecutor()
    risk = _FakeRisk()
    monkeypatch.setattr(TradeManager, "STATE_PATH", tmp_path / "open_trades.json")
    manager = TradeManager(client=object(), executor=executor, risk=risk, cfg=cfg)

    setup = TradeSetup(
        symbol="BCH/USDT:USDT",
        direction="short",
        entry_price=350.0,
        stop_loss=355.0,
        tp1=340.0,
        tp2=332.5,
        tp3=325.0,
        size_usd=100.0,
        size_contracts=1.0,
        leverage=5,
        r_distance=5.0,
        strategy_sleeve="trend_following",
        atr=2.0,
        risk_pct=1.0,
        setup_passport={
            "decision": "eligible",
            "setup_type": "vwap_pullback_continuation",
            "expected_path": "vwap_reclaim_continuation",
            "invalidation": "loses_vwap_and_pullback_extreme",
        },
    )

    trade = asyncio.run(manager.open(setup))
    assert trade is not None

    asyncio.run(manager._check_exits(trade, 348.0))  # 0.4R favourable first
    asyncio.run(manager._check_exits(trade, 352.5))  # then 0.5R adverse

    assert "BCH/USDT:USDT" in manager.open_symbols
    assert trade.mfe_r == pytest.approx(0.4)
    assert trade.mae_r == pytest.approx(0.5)
    assert executor.close_calls == []
