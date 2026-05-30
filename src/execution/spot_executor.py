"""
Spot Executor — places Binance Spot buy/sell orders.

Live mode:
  - Entry: market buy (quoteOrderQty = USDT amount)
  - Exit:  market sell when SL/TP hits (managed by trade_manager)
  - Alternative: limit buy + OCO sell (SL + TP1 combined)

Paper mode:
  - Simulates fills at current price with no API calls

No leverage. No shorts. Pure Spot.
"""
from __future__ import annotations

import logging
import time

from src.data.spot_client import BinanceSpotClient
from src.risk.spot_risk_manager import SpotTradeSetup

log = logging.getLogger(__name__)

_PAPER_ORDER_ID = 0


def _next_paper_id() -> str:
    global _PAPER_ORDER_ID
    _PAPER_ORDER_ID += 1
    return f"PAPER-{_PAPER_ORDER_ID}"


class SpotExecutor:
    def __init__(self, client: BinanceSpotClient, cfg: dict):
        self._client = client
        self._cfg = cfg
        self._paper = cfg["trading"]["mode"] == "paper"

    # ------------------------------------------------------------------ #
    #  Entry                                                              #
    # ------------------------------------------------------------------ #

    async def open_position(self, setup: SpotTradeSetup) -> dict:
        """
        Execute a spot market buy.  Returns order dict with filled price.
        """
        mode = "PAPER" if self._paper else "LIVE"
        log.info(
            "[%s] BUY %s  entry=%.6f  size_usdt=$%.2f  qty=%.6f  sl=%.6f  tp1=%.6f  tp2=%.6f  tp3=%.6f",
            mode,
            setup.symbol,
            setup.entry_price,
            setup.size_usdt,
            setup.base_qty,
            setup.stop_loss,
            setup.tp1,
            setup.tp2,
            setup.tp3,
        )

        if self._paper:
            return self._paper_buy(setup)

        # Live: market buy using quote amount (USDT)
        try:
            order = await self._client.create_market_buy(setup.symbol, setup.size_usdt)
            log.info("Entry order filled: %s", order.get("id"))
            return order
        except Exception as exc:
            log.error("Entry order failed for %s: %s", setup.symbol, exc)
            raise

    # ------------------------------------------------------------------ #
    #  Exit                                                               #
    # ------------------------------------------------------------------ #

    async def close_position(
        self,
        symbol: str,
        base_qty: float,
        reason: str = "manual",
    ) -> dict:
        """
        Market sell the entire remaining position.
        """
        log.info(
            "[%s] SELL %s  qty=%.6f  reason=%s",
            "PAPER" if self._paper else "LIVE",
            symbol,
            base_qty,
            reason,
        )

        if self._paper:
            return {"id": _next_paper_id(), "status": "closed", "reason": reason}

        try:
            order = await self._client.create_market_sell(symbol, base_qty)
            log.info("Exit order filled: %s", order.get("id"))
            return order
        except Exception as exc:
            log.error("Exit order failed for %s qty=%.6f: %s", symbol, base_qty, exc)
            raise

    async def partial_sell(
        self,
        symbol: str,
        base_qty: float,
        tp_label: str,
    ) -> dict:
        """Sell a portion of the position (partial TP)."""
        log.info(
            "[%s] PARTIAL SELL %s  qty=%.6f  (%s)",
            "PAPER" if self._paper else "LIVE",
            symbol,
            base_qty,
            tp_label,
        )

        if self._paper:
            return {"id": _next_paper_id(), "status": "closed", "tp": tp_label}

        try:
            order = await self._client.create_market_sell(symbol, base_qty)
            return order
        except Exception as exc:
            log.error("Partial sell failed %s: %s", symbol, exc)
            raise

    async def cancel_all(self, symbol: str) -> None:
        if not self._paper:
            try:
                await self._client.cancel_all_orders(symbol)
            except Exception as exc:
                log.warning("cancel_all failed for %s: %s", symbol, exc)

    # ------------------------------------------------------------------ #
    #  Paper helpers                                                      #
    # ------------------------------------------------------------------ #

    def _paper_buy(self, setup: SpotTradeSetup) -> dict:
        actual_qty = setup.size_usdt / setup.entry_price
        return {
            "id": _next_paper_id(),
            "symbol": setup.symbol,
            "side": "buy",
            "price": setup.entry_price,
            "amount": actual_qty,
            "cost": setup.size_usdt,
            "status": "closed",
            "timestamp": int(time.time() * 1000),
        }
