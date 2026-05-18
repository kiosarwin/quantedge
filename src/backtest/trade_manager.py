"""
Trade manager — tracks open position lifecycle: breakeven moves, trailing stop,
TP1/TP2 execution, and panic closes.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable

from src.data.client import BinanceFuturesClient
from src.execution.executor import Executor
from src.risk.risk_manager import RiskManager, TradeSetup

log = logging.getLogger(__name__)


@dataclass
class OpenTrade:
    setup: TradeSetup
    entry_order_id: str
    sl_order_id: str
    tp1_order_id: str
    tp2_order_id: str
    opened_at: float = field(default_factory=time.time)
    tp1_hit: bool = False
    breakeven_moved: bool = False
    trailing_stop: float | None = None
    remaining_contracts: float = 0.0
    realized_pnl: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0

    @property
    def symbol(self) -> str:
        return self.setup.symbol

    @property
    def direction(self) -> str:
        return self.setup.direction

    def current_pnl(self, price: float) -> float:
        mult = 1 if self.direction == "long" else -1
        return mult * (price - self.setup.entry_price) * self.remaining_contracts

    def is_sl_hit(self, price: float) -> bool:
        if self.direction == "long":
            return price <= self.setup.stop_loss
        return price >= self.setup.stop_loss

    def is_tp1_hit(self, price: float) -> bool:
        if self.direction == "long":
            return price >= self.setup.tp1
        return price <= self.setup.tp1

    def is_tp2_hit(self, price: float) -> bool:
        if self.direction == "long":
            return price >= self.setup.tp2
        return price <= self.setup.tp2


CloseCallback = Callable[[OpenTrade, float, str], Awaitable[None]]


class TradeManager:
    def __init__(
        self,
        client: BinanceFuturesClient,
        executor: Executor,
        risk: RiskManager,
        cfg: dict,
        on_close: CloseCallback | None = None,
    ):
        self._client = client
        self._executor = executor
        self._risk = risk
        self._cfg = cfg
        self._exit = cfg["exit"]
        self._trades: dict[str, OpenTrade] = {}
        self._on_close = on_close

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    async def open(self, setup: TradeSetup) -> OpenTrade | None:
        can, reason = self._risk.can_open_trade()
        if not can:
            log.warning("Risk guard blocked trade on %s: %s", setup.symbol, reason)
            return None

        try:
            entry_order = await self._executor.open_position(setup)
        except Exception as exc:
            log.error("[%s] Entry order failed — no position opened: %s", setup.symbol, exc)
            return None

        fill_price = float(entry_order.get("price", setup.entry_price) or setup.entry_price)
        if fill_price == 0:
            fill_price = setup.entry_price

        try:
            sl_order = await self._executor.place_stop_loss(setup, entry_order["id"])
            tp1_order = await self._executor.place_take_profit(
                setup, setup.tp1, setup.tp1_size_pct
            )
            tp2_order = await self._executor.place_take_profit(
                setup, setup.tp2, setup.tp2_size_pct
            )
        except Exception as exc:
            log.error(
                "[%s] SL/TP placement failed — closing entry to prevent orphan: %s",
                setup.symbol, exc,
            )
            try:
                await self._executor.cancel_all(setup.symbol)
                await self._executor.close_position_market(
                    setup.symbol, setup.direction, setup.size_contracts
                )
            except Exception as cancel_exc:
                log.critical(
                    "[%s] ORPHAN POSITION — could not cancel after SL/TP failure: %s",
                    setup.symbol, cancel_exc,
                )
            return None

        trade = OpenTrade(
            setup=setup,
            entry_order_id=entry_order["id"],
            sl_order_id=sl_order["id"],
            tp1_order_id=tp1_order["id"],
            tp2_order_id=tp2_order["id"],
            remaining_contracts=setup.size_contracts,
        )
        self._trades[setup.symbol] = trade
        self._risk.on_trade_opened(risk_pct=setup.risk_pct)
        log.info("Trade opened on %s @ %.4f", setup.symbol, fill_price)
        return trade

    async def monitor_all(self, price_lookup: dict[str, float]) -> None:
        """Called on each price tick — checks exits for every open trade."""
        for symbol, trade in list(self._trades.items()):
            price = price_lookup.get(symbol)
            if price is None:
                continue
            await self._check_exits(trade, price)

    async def close_all(self, reason: str = "manual") -> None:
        for symbol, trade in list(self._trades.items()):
            await self._close_trade(trade, trade.setup.entry_price, reason)

    @property
    def open_symbols(self) -> list[str]:
        return list(self._trades.keys())

    def get_floating_pnl(self, price_map: dict[str, float]) -> list[dict]:
        result = []
        for symbol, trade in self._trades.items():
            price = price_map.get(symbol)
            if price is None:
                continue
            entry = trade.setup.entry_price
            size = trade.setup.size_usd
            if trade.setup.direction == "long":
                pnl_pct = (price - entry) / entry * 100
            else:
                pnl_pct = (entry - price) / entry * 100
            pnl_usd = size * pnl_pct / 100
            risk_usd = trade.setup.r_distance / entry * size if entry > 0 else 0.0
            result.append({
                "symbol": symbol,
                "direction": trade.setup.direction,
                "entry": entry,
                "current": price,
                "pnl_usd": pnl_usd,
                "pnl_pct": pnl_pct,
                "size_usd": size,
                "elapsed_s": time.time() - trade.opened_at,
                "sl": trade.setup.stop_loss,
                "tp1": trade.setup.tp1,
                "risk_pct": trade.setup.risk_pct,
                "risk_usd": round(risk_usd, 2),
            })
        return result

    # ------------------------------------------------------------------ #
    #  Exit logic                                                          #
    # ------------------------------------------------------------------ #

    async def _check_exits(self, trade: OpenTrade, price: float) -> None:
        # Update MFE/MAE in R multiples
        r = trade.setup.r_distance
        if r > 0:
            entry = trade.setup.entry_price
            if trade.direction == "long":
                fav = price - entry
                adv = entry - price
            else:
                fav = entry - price
                adv = price - entry
            if fav > 0:
                trade.mfe_r = max(trade.mfe_r, fav / r)
            if adv > 0:
                trade.mae_r = max(trade.mae_r, adv / r)

        # Max hold duration
        if time.time() - trade.opened_at > trade.setup.max_hold_duration_s:
            await self._close_trade(trade, price, "max_hold")
            return

        # Stop loss
        if trade.is_sl_hit(price):
            await self._close_trade(trade, price, "stop_loss")
            return

        # Trailing stop (after TP1 hit)
        if trade.trailing_stop is not None:
            if trade.direction == "long" and price <= trade.trailing_stop:
                await self._close_trade(trade, price, "trailing_stop")
                return
            if trade.direction == "short" and price >= trade.trailing_stop:
                await self._close_trade(trade, price, "trailing_stop")
                return

        # TP1 hit → move SL to breakeven + start trailing
        if not trade.tp1_hit and trade.is_tp1_hit(price):
            log.info("[%s] TP1 hit @ %.4f", trade.symbol, price)
            trade.tp1_hit = True
            trade.remaining_contracts *= (1 - trade.setup.tp1_size_pct)
            pnl = abs(trade.setup.tp1 - trade.setup.entry_price) * (
                trade.setup.size_contracts * trade.setup.tp1_size_pct
            )
            trade.realized_pnl += pnl
            # Move SL to breakeven
            trade.setup.stop_loss = trade.setup.entry_price
            # Init ATR-based trailing stop
            trail_dist = trade.setup.atr * trade.setup.trailing_atr_multiplier
            if trade.direction == "long":
                trade.trailing_stop = price - trail_dist
            else:
                trade.trailing_stop = price + trail_dist

        # Update trailing stop (ratchet only — never widen)
        if trade.tp1_hit and trade.trailing_stop is not None:
            trail_dist = trade.setup.atr * trade.setup.trailing_atr_multiplier
            if trade.direction == "long":
                new_trail = price - trail_dist
                if new_trail > trade.trailing_stop:
                    trade.trailing_stop = new_trail
            else:
                new_trail = price + trail_dist
                if new_trail < trade.trailing_stop:
                    trade.trailing_stop = new_trail

        # TP2 partial exit — close tp2_size_pct, let remainder trail to TP3
        if not trade.tp1_hit:
            pass  # TP1 not yet hit, nothing to do for TP2 yet
        elif trade.is_tp2_hit(price) and not getattr(trade, "tp2_hit", False):
            trade.tp2_hit = True
            tp2_close_pct = trade.setup.tp2_size_pct
            contracts_to_close = trade.remaining_contracts * tp2_close_pct
            pnl = abs(trade.setup.tp2 - trade.setup.entry_price) * contracts_to_close
            trade.realized_pnl += pnl
            trade.remaining_contracts -= contracts_to_close
            log.info("[%s] TP2 hit @ %.4f — closed %.1f%%, trailing remainder", trade.symbol, price, tp2_close_pct * 100)
            if trade.remaining_contracts <= 0.0:
                await self._close_trade(trade, price, "tp2_full")

        # TP3 exit — close all remaining after TP2 partial
        elif getattr(trade, "tp2_hit", False) and trade.remaining_contracts > 0:
            if trade.direction == "long" and price >= trade.setup.tp3:
                await self._close_trade(trade, price, "tp3")
            elif trade.direction == "short" and price <= trade.setup.tp3:
                await self._close_trade(trade, price, "tp3")

    async def _close_trade(self, trade: OpenTrade, price: float, reason: str) -> None:
        log.info("[%s] Closing %s @ %.4f  reason=%s", trade.symbol, trade.direction, price, reason)
        await self._executor.cancel_all(trade.symbol)
        try:
            await self._executor.close_position_market(
                trade.symbol, trade.direction, trade.remaining_contracts
            )
        except Exception as exc:
            # Exchange may have already closed this (SL/TP filled on exchange).
            # Always clean up local state so the slot is freed.
            log.warning("[%s] close_position_market failed (likely already closed): %s", trade.symbol, exc)

        mult = 1 if trade.direction == "long" else -1
        close_pnl = mult * (price - trade.setup.entry_price) * trade.remaining_contracts

        # Deduct round-trip taker fees from realized PnL
        fee_pct = self._cfg.get("ev_model", {}).get("taker_fee_pct", 0.04) / 100
        fee_cost = trade.setup.size_usd * fee_pct * 2   # entry + exit legs
        total_pnl = trade.realized_pnl + close_pnl - fee_cost

        self._risk.on_trade_closed(total_pnl, risk_pct=trade.setup.risk_pct)

        if self._on_close:
            await self._on_close(trade, total_pnl, reason)

        del self._trades[trade.symbol]
        log.info("[%s] Trade closed  pnl=$%.2f (fees -$%.3f)  reason=%s",
                 trade.symbol, total_pnl, fee_cost, reason)
