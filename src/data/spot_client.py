"""
Binance Spot exchange client — async REST wrapper using ccxt.

Spot-specific behaviour vs the Futures client:
  * defaultType = "spot"
  * No leverage / margin-mode calls
  * Supports OCO orders (stop-loss + take-profit combined)
  * Balance is per-asset (e.g. ETH, BTC) not a single margin pool
"""
from __future__ import annotations

import logging
import math
from typing import Any

import ccxt.async_support as ccxt
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

log = logging.getLogger(__name__)


class BinanceSpotClient:
    """Thin async wrapper around ccxt's binance (spot)."""

    def __init__(self, cfg: dict):
        ex_cfg = cfg["exchange"]
        safety_cfg = cfg.get("safety", {})

        options: dict[str, Any] = {
            "defaultType": "spot",
            "recvWindow": ex_cfg.get("recv_window", 10000),
            "fetchCurrencies": False,
        }
        if ex_cfg.get("testnet", False):
            options["sandboxMode"] = True

        self._exchange = ccxt.binance(
            {
                "apiKey": ex_cfg.get("api_key", ""),
                "secret": ex_cfg.get("api_secret", ""),
                "options": options,
                "enableRateLimit": True,
            }
        )
        self._retry_attempts = safety_cfg.get("api_retry_attempts", 3)
        self._retry_delay = safety_cfg.get("api_retry_delay_seconds", 5)

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    async def connect(self) -> None:
        await self._exchange.load_markets()
        log.info(
            "Connected to Binance Spot (testnet=%s)",
            self._exchange.options.get("sandboxMode", False),
        )

    async def close(self) -> None:
        await self._exchange.close()
        log.info("Binance Spot connection closed.")

    # ------------------------------------------------------------------ #
    #  Market helpers                                                      #
    # ------------------------------------------------------------------ #

    @property
    def markets(self) -> dict:
        return self._exchange.markets or {}

    async def fetch_markets(self) -> dict:
        return await self._exchange.load_markets(reload=True)

    # ------------------------------------------------------------------ #
    #  OHLCV                                                              #
    # ------------------------------------------------------------------ #

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 300,
        since: int | None = None,
    ) -> list[list]:
        """Returns list of [ts, open, high, low, close, volume]."""
        return await self._exchange.fetch_ohlcv(
            symbol, timeframe, since=since, limit=limit
        )

    # ------------------------------------------------------------------ #
    #  Tickers                                                            #
    # ------------------------------------------------------------------ #

    async def fetch_ticker(self, symbol: str) -> dict:
        return await self._exchange.fetch_ticker(symbol)

    async def fetch_tickers(self, symbols: list[str] | None = None) -> dict:
        return await self._exchange.fetch_tickers(symbols)

    # ------------------------------------------------------------------ #
    #  Account                                                            #
    # ------------------------------------------------------------------ #

    async def fetch_balance(self) -> dict:
        """Returns balance dict; access USDT via balance['USDT']['free']."""
        return await self._exchange.fetch_balance()

    async def fetch_usdt_balance(self) -> float:
        bal = await self.fetch_balance()
        return float(bal.get("USDT", {}).get("free", 0.0))

    async def fetch_asset_balance(self, asset: str) -> float:
        bal = await self.fetch_balance()
        return float(bal.get(asset, {}).get("free", 0.0))

    async def fetch_open_orders(self, symbol: str) -> list[dict]:
        return await self._exchange.fetch_open_orders(symbol)

    # ------------------------------------------------------------------ #
    #  Order management                                                   #
    # ------------------------------------------------------------------ #

    async def create_market_buy(self, symbol: str, quote_amount: float) -> dict:
        """
        Market buy using a USDT (quote) amount.
        Binance Spot supports 'quoteOrderQty' for this.
        """
        params = {"quoteOrderQty": round(quote_amount, 2)}
        return await self._exchange.create_order(
            symbol, "market", "buy", None, None, params
        )

    async def create_limit_buy(
        self, symbol: str, base_amount: float, price: float
    ) -> dict:
        """Limit buy order."""
        amount = self._round_amount(symbol, base_amount)
        return await self._exchange.create_order(
            symbol, "limit", "buy", amount, price
        )

    async def create_market_sell(self, symbol: str, base_amount: float) -> dict:
        """Market sell of a base currency amount."""
        amount = self._round_amount(symbol, base_amount)
        return await self._exchange.create_order(
            symbol, "market", "sell", amount
        )

    async def create_limit_sell(
        self, symbol: str, base_amount: float, price: float
    ) -> dict:
        """Limit sell at a specific price."""
        amount = self._round_amount(symbol, base_amount)
        return await self._exchange.create_order(
            symbol, "limit", "sell", amount, price
        )

    async def create_oco_sell(
        self,
        symbol: str,
        base_amount: float,
        take_profit_price: float,
        stop_price: float,
        stop_limit_price: float | None = None,
    ) -> dict:
        """
        OCO (One-Cancels-the-Other) sell order:
          - Limit sell at take_profit_price (hits on rally)
          - Stop-limit sell at stop_price / stop_limit_price (hits on drop)

        stop_limit_price defaults to stop_price * 0.999 (slightly below trigger).
        """
        amount = self._round_amount(symbol, base_amount)
        slp = stop_limit_price or round(stop_price * 0.999, self._price_decimals(symbol))
        params = {
            "stopPrice": stop_price,
            "stopLimitPrice": slp,
            "stopLimitTimeInForce": "GTC",
        }
        return await self._exchange.create_order(
            symbol, "oco", "sell", amount, take_profit_price, params
        )

    async def cancel_order(self, order_id: str, symbol: str) -> dict:
        return await self._exchange.cancel_order(order_id, symbol)

    async def cancel_all_orders(self, symbol: str) -> list:
        return await self._exchange.cancel_all_orders(symbol)

    async def fetch_order(self, order_id: str, symbol: str) -> dict:
        return await self._exchange.fetch_order(order_id, symbol)

    # ------------------------------------------------------------------ #
    #  Precision helpers                                                  #
    # ------------------------------------------------------------------ #

    def symbol_precision(self, symbol: str) -> tuple[int, int]:
        """Returns (amount_decimal_places, price_decimal_places)."""
        market = self.markets.get(symbol, {})
        precision = market.get("precision", {})
        return (
            int(precision.get("amount", 4)),
            int(precision.get("price", 2)),
        )

    def _price_decimals(self, symbol: str) -> int:
        return self.symbol_precision(symbol)[1]

    def _round_amount(self, symbol: str, amount: float) -> float:
        prec, _ = self.symbol_precision(symbol)
        factor = 10 ** prec
        return math.floor(amount * factor) / factor

    def min_notional(self, symbol: str) -> float:
        market = self.markets.get(symbol, {})
        limits = market.get("limits", {})
        return limits.get("cost", {}).get("min", 10.0)

    def is_spot_market(self, symbol: str) -> bool:
        market = self.markets.get(symbol, {})
        return market.get("spot", False) and market.get("active", False)
