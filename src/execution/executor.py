"""
Order executor — places market entries, limit/market TP/SL orders on Binance Futures.
Paper mode mirrors the logic but records fills without sending real orders.
"""
from __future__ import annotations

import logging
import math
import time

from src.data.client import BinanceFuturesClient
from src.risk.risk_manager import TradeSetup

log = logging.getLogger(__name__)

_PAPER_ORDER_ID = 0


def _next_paper_id() -> str:
    global _PAPER_ORDER_ID
    _PAPER_ORDER_ID += 1
    return f"PAPER-{_PAPER_ORDER_ID}"


class Executor:
    def __init__(self, client: BinanceFuturesClient, cfg: dict):
        self._client = client
        self._cfg = cfg
        self._paper = cfg["trading"]["mode"] == "paper"

    async def open_position(self, setup: TradeSetup) -> dict:
        """
        Opens a position, returns a dict with order details.
        Sets leverage and margin mode first.
        """
        log.info(
            "[%s] Opening %s %s  entry=%.4f  sl=%.4f  tp1=%.4f  tp2=%.4f  size=$%.2f",
            "PAPER" if self._paper else "LIVE",
            setup.direction.upper(),
            setup.symbol,
            setup.entry_price,
            setup.stop_loss,
            setup.tp1,
            setup.tp2,
            setup.size_usd,
        )

        if self._paper:
            return self._paper_fill(setup)

        await self._client.set_leverage(setup.symbol, setup.leverage)
        await self._client.set_margin_mode(setup.symbol, "isolated")

        side = "buy" if setup.direction == "long" else "sell"
        amount = self._round_amount(setup.symbol, setup.size_contracts)

        if amount <= 0:
            raise ValueError(
                f"{setup.symbol}: rounded contracts={amount} (raw={setup.size_contracts:.6f}) "
                f"is below min lot size — position too small for this account"
            )

        order = await self._client.create_order(
            setup.symbol, "market", side, amount
        )
        log.info("Entry order filled: %s", order.get("id"))
        return order

    async def place_stop_loss(self, setup: TradeSetup, order_id: str) -> dict:
        if self._paper:
            return {"id": _next_paper_id(), "type": "stop_market", "stopPrice": setup.stop_loss}

        side = "sell" if setup.direction == "long" else "buy"
        amount = self._round_amount(setup.symbol, setup.size_contracts)
        sl_price = self._round_price(setup.symbol, setup.stop_loss)

        params = {
            "stopPrice": sl_price,
            "reduceOnly": True,
            "closePosition": False,
        }
        order = await self._client.create_order(
            setup.symbol, "stop_market", side, amount, params=params
        )
        status = order.get("status", "")
        if status not in ("NEW", "PARTIALLY_FILLED", "open", "new"):
            raise RuntimeError(f"SL order rejected by exchange — status={status!r}")
        log.info("SL order placed: %s @ %.4f", order.get("id"), setup.stop_loss)
        return order

    async def place_take_profit(
        self, setup: TradeSetup, tp_price: float, size_pct: float
    ) -> dict:
        if self._paper:
            return {"id": _next_paper_id(), "type": "take_profit_market", "stopPrice": tp_price}

        side = "sell" if setup.direction == "long" else "buy"
        amount = self._round_amount(setup.symbol, setup.size_contracts * size_pct)
        rounded_tp = self._round_price(setup.symbol, tp_price)

        params = {
            "stopPrice": rounded_tp,
            "reduceOnly": True,
        }
        order = await self._client.create_order(
            setup.symbol, "take_profit_market", side, amount, params=params
        )
        status = order.get("status", "")
        if status not in ("NEW", "PARTIALLY_FILLED", "open", "new"):
            raise RuntimeError(f"TP order rejected by exchange — status={status!r} @ {tp_price}")
        log.info("TP order placed: %s @ %.4f (%.0f%%)", order.get("id"), tp_price, size_pct * 100)
        return order

    async def close_position_market(self, symbol: str, direction: str, contracts: float) -> dict:
        if self._paper:
            return {"id": _next_paper_id(), "type": "market", "status": "closed"}

        side = "sell" if direction == "long" else "buy"
        amount = self._round_amount(symbol, contracts)
        params = {"reduceOnly": True}
        order = await self._client.create_order(symbol, "market", side, amount, params=params)
        log.info("Position closed market: %s", order.get("id"))
        return order

    async def cancel_all(self, symbol: str) -> None:
        if self._paper:
            return
        try:
            await self._client.cancel_all_orders(symbol)
        except Exception as exc:
            log.warning("cancel_all failed for %s: %s", symbol, exc)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _paper_fill(self, setup: TradeSetup) -> dict:
        return {
            "id": _next_paper_id(),
            "symbol": setup.symbol,
            "side": "buy" if setup.direction == "long" else "sell",
            "price": setup.entry_price,
            "amount": setup.size_contracts,
            "cost": setup.size_usd,
            "status": "closed",
            "timestamp": int(time.time() * 1000),
        }

    def _round_amount(self, symbol: str, amount: float) -> float:
        """Floor quantity to exchange lot stepSize via ccxt."""
        return self._client.amount_to_precision(symbol, amount)

    def _round_price(self, symbol: str, price: float) -> float:
        """Round price to exchange tickSize via ccxt."""
        return self._client.price_to_precision(symbol, price)
