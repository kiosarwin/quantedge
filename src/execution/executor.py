"""
Order executor — places market entries, limit/market TP/SL orders on Binance Futures.
Paper mode mirrors the logic but records fills without sending real orders.
"""
from __future__ import annotations

import asyncio
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
        # Confirm fill in live mode
        confirmed = await self._confirm_fill(setup.symbol, order.get("id", ""))
        if confirmed is not None:
            order = confirmed
        if confirmed is None:
            log.error("[%s] Entry order %s not confirmed/rejected", setup.symbol, order.get("id"))
            raise RuntimeError(
                f"{setup.symbol}: entry order {order.get('id')} rejected or unconfirmed"
            )
        log.info("Entry order confirmed: %s", order.get("id"))
        return order

    async def _confirm_fill(self, symbol: str, order_id: str, max_retries: int = 5) -> dict | None:
        """Poll exchange for fill confirmation after live market order."""
        for attempt in range(max_retries):
            await asyncio.sleep(1.0)
            try:
                order = await self._client.fetch_order(order_id, symbol)
            except Exception as exc:
                log.warning("Fill check attempt %d/%d for %s failed: %s", attempt + 1, max_retries, order_id, exc)
                continue
            status = order.get("status", "")
            if status == "closed":
                # Check for partial fill
                filled = float(order.get("filled", 0) or 0)
                amount = float(order.get("amount", 0) or 0)
                if 0 < filled < amount:
                    log.warning(
                        "Order %s partially filled: %.6f / %.6f on %s",
                        order_id, filled, amount, symbol,
                    )
                return order
            if status in ("canceled", "cancelled", "rejected", "expired"):
                log.error("Order %s %s on %s: %s", order_id, status, symbol, order)
                return None
        log.warning("Order %s not confirmed after %d attempts on %s", order_id, max_retries, symbol)
        return None

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

    async def move_stop_loss(
        self,
        symbol: str,
        direction: str,
        old_sl_id: str,
        new_stop_price: float,
        size_contracts: float,
    ) -> dict | None:
        """Cancel the existing SL and place a fresh one at ``new_stop_price``.

        Used by ``TradeManager`` after a TP1 partial fill to anchor the
        residual to break-even on the *exchange* — not just locally — so
        an offline bot cannot bleed the residual at the original wider
        stop.  Returns the new order dict on success, or ``None`` if the
        replacement could not be placed; in either case the caller should
        treat ``None`` / a missing id as "exchange SL is now stale" and
        rely on the local stop_loss field.

        Failure modes tolerated:
          * old SL already filled / cancelled — proceed to place new SL
          * new SL rejected by exchange — log critical and return None
            (the local SL still triggers via market close on next tick)

        Paper mode short-circuits to a synthetic order id so unit tests
        and dry-runs work the same way.
        """
        if self._paper:
            return {
                "id": _next_paper_id(),
                "type": "stop_market",
                "stopPrice": new_stop_price,
            }

        # Step 1: cancel the old SL.  We swallow "already cancelled / filled"
        # errors because that's exactly the race we're trying to be robust
        # against — never let stale exchange state block placing the new
        # protective stop.
        if old_sl_id:
            try:
                await self._client.cancel_order(old_sl_id, symbol)
            except Exception as exc:
                log.warning(
                    "[%s] move_stop_loss: cancel of old SL %s failed (probably already gone): %s",
                    symbol, old_sl_id, exc,
                )

        # Step 2: place the new reduceOnly SL sized to the residual.
        side = "sell" if direction == "long" else "buy"
        amount = self._round_amount(symbol, size_contracts)
        if amount <= 0:
            log.warning(
                "[%s] move_stop_loss: residual size %.6f rounds to zero — "
                "no exchange SL refresh (local stop still active)",
                symbol, size_contracts,
            )
            return None

        sl_price = self._round_price(symbol, new_stop_price)
        params = {
            "stopPrice": sl_price,
            "reduceOnly": True,
            "closePosition": False,
        }
        try:
            order = await self._client.create_order(
                symbol, "stop_market", side, amount, params=params
            )
        except Exception as exc:
            log.critical(
                "[%s] move_stop_loss: failed to place new SL @ %.4f for %.6f contracts — "
                "exchange SL is now MISSING; relying on local stop. err=%s",
                symbol, sl_price, amount, exc,
            )
            return None

        status = order.get("status", "")
        if status not in ("NEW", "PARTIALLY_FILLED", "open", "new"):
            log.critical(
                "[%s] move_stop_loss: new SL rejected by exchange — status=%r",
                symbol, status,
            )
            return None

        log.info(
            "[%s] SL moved on exchange: %s @ %.4f (size %.6f)",
            symbol, order.get("id"), sl_price, amount,
        )
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
