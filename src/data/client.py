"""
Binance Futures exchange client — REST + WebSocket wrapper.

Hang-storm protection (added 2026-05-22):
    Every async call to the exchange goes through ``_call``, which enforces a
    hard ``asyncio.wait_for`` ceiling on the coroutine. This is layered on top
    of ccxt's own ``timeout`` (passed to the underlying aiohttp session) so a
    stuck rate-limiter, lost socket, or session-pool deadlock cannot freeze
    the whole bot. Both budgets are operator-tunable via ``safety``:

        safety.api_http_timeout_seconds   -> ccxt / aiohttp ClientTimeout (inner)
        safety.api_call_timeout_seconds   -> asyncio.wait_for ceiling (outer)

    The outer budget should be slightly larger than the inner so ccxt has a
    chance to time out cleanly before we cancel.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any, Awaitable, TypeVar

import ccxt.async_support as ccxt
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

log = logging.getLogger(__name__)

# TTL caches for high-frequency public endpoints (60s — data resolves at 5m anyway)
_RATIO_TTL = 60.0
_ls_cache:    dict[str, tuple[float, float]] = {}
_taker_cache: dict[str, tuple[float, float]] = {}

# Default timeouts (seconds). Overridable via safety.api_*_timeout_seconds in
# config.yaml. Outer must be >= inner so ccxt has time to surface its own
# TimeoutError before asyncio cancels the coroutine.
_DEFAULT_HTTP_TIMEOUT_S = 12.0   # inner: aiohttp ClientTimeout
_DEFAULT_CALL_TIMEOUT_S = 15.0   # outer: asyncio.wait_for budget

T = TypeVar("T")



class BinanceFuturesClient:
    """Thin async wrapper around ccxt's binance_futures."""

    def __init__(self, cfg: dict):
        ex_cfg = cfg["exchange"]
        safety_cfg = cfg.get("safety", {})
        self._is_testnet = bool(ex_cfg.get("testnet", True))
        self._fapi_base = ex_cfg.get("fapi_base_url", "").rstrip("/")

        # Hang-storm protection: explicit budgets for ccxt (inner) and asyncio
        # (outer). Read from safety config with safe defaults. Outer must be
        # >= inner so ccxt surfaces its own TimeoutError first when possible.
        self._http_timeout_s = float(
            safety_cfg.get("api_http_timeout_seconds", _DEFAULT_HTTP_TIMEOUT_S)
        )
        self._call_timeout_s = float(
            safety_cfg.get("api_call_timeout_seconds", _DEFAULT_CALL_TIMEOUT_S)
        )
        if self._call_timeout_s < self._http_timeout_s:
            log.warning(
                "api_call_timeout_seconds (%.1f) < api_http_timeout_seconds (%.1f); "
                "outer budget should be >= inner. Bumping outer to inner + 3s.",
                self._call_timeout_s, self._http_timeout_s,
            )
            self._call_timeout_s = self._http_timeout_s + 3.0

        options: dict[str, Any] = {
            "defaultType": "future",
            "recvWindow": ex_cfg.get("recv_window", 10000),
            "fetchCurrencies": False,
            "fetchMarkets": ["linear"],
        }

        url_overrides: dict[str, Any] = {}
        if self._is_testnet:
            options["sandboxMode"] = True
            url_overrides["urls"] = {
                "api": {
                    "fapiPublic": "https://testnet.binancefuture.com/fapi/v1",
                    "fapiPublicV2": "https://testnet.binancefuture.com/fapi/v2",
                    "fapiPublicV3": "https://testnet.binancefuture.com/fapi/v3",
                    "fapiPrivate": "https://testnet.binancefuture.com/fapi/v1",
                    "fapiPrivateV2": "https://testnet.binancefuture.com/fapi/v2",
                    "fapiPrivateV3": "https://testnet.binancefuture.com/fapi/v3",
                    "fapiData": "https://testnet.binancefuture.com/futures/data",
                }
            }
        elif self._fapi_base:
            url_overrides["urls"] = {
                "api": {
                    "fapiPublic":   f"{self._fapi_base}/fapi/v1",
                    "fapiPublicV2": f"{self._fapi_base}/fapi/v2",
                    "fapiPublicV3": f"{self._fapi_base}/fapi/v3",
                    "fapiPrivate":  f"{self._fapi_base}/fapi/v1",
                    "fapiPrivateV2":f"{self._fapi_base}/fapi/v2",
                    "fapiPrivateV3":f"{self._fapi_base}/fapi/v3",
                    "fapiData":     f"{self._fapi_base}/futures/data",
                }
            }
            log.info("FAPI URL override → %s", self._fapi_base)

        self._exchange = ccxt.binance(
            {
                "apiKey": ex_cfg.get("api_key", ""),
                "secret": ex_cfg.get("api_secret", ""),
                "options": options,
                "enableRateLimit": True,
                # Inner ceiling: aiohttp ClientTimeout total per HTTP request.
                # Hang-storm prevention — must be set explicitly, ccxt's default
                # has been observed to mis-apply under flaky networks.
                "timeout": int(self._http_timeout_s * 1000),
                **url_overrides,
            }
        )
        if self._is_testnet:
            try:
                self._exchange.set_sandbox_mode(True)
            except Exception:
                pass

        self._retry_attempts = safety_cfg.get("api_retry_attempts", 3)
        self._retry_delay = safety_cfg.get("api_retry_delay_seconds", 5)

    # ------------------------------------------------------------------ #
    #  Hang-storm guard                                                    #
    # ------------------------------------------------------------------ #

    async def _call(self, coro: Awaitable[T], label: str) -> T:
        """
        Run *coro* with a hard ``asyncio.wait_for`` ceiling.

        This is the outer layer of the two-layer hang-storm guard:
            * ccxt's ``timeout`` enforces the inner HTTP budget.
            * ``_call`` enforces the outer wallclock budget so a stuck rate
              limiter or deadlocked aiohttp session pool cannot freeze the bot.

        On timeout, raises ``asyncio.TimeoutError`` (subclass of ``Exception``)
        so existing per-call ``try/except`` blocks and ``tenacity`` retry
        decorators behave unchanged. The label is logged so operators can
        trace which endpoint is unhealthy.
        """
        try:
            return await asyncio.wait_for(coro, timeout=self._call_timeout_s)
        except asyncio.TimeoutError:
            log.warning(
                "Binance API call '%s' exceeded outer budget %.1fs — aborted",
                label, self._call_timeout_s,
            )
            raise

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    async def connect(self) -> None:
        await self._call(self._exchange.load_markets(), "load_markets")
        log.info("Connected to Binance Futures (testnet=%s)", self._is_testnet)

    async def close(self) -> None:
        # close() must not be wrapped — it's the cleanup path; if it hangs the
        # caller is already shutting down and will be killed externally.
        await self._exchange.close()
        log.info("Exchange connection closed.")

    # ------------------------------------------------------------------ #
    #  Market helpers                                                       #
    # ------------------------------------------------------------------ #

    @property
    def markets(self) -> dict:
        return self._exchange.markets or {}

    async def fetch_markets(self) -> dict:
        return await self._call(self._exchange.load_markets(reload=True), "load_markets(reload)")

    # ------------------------------------------------------------------ #
    #  OHLCV                                                               #
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
        return await self._call(
            self._exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit),
            f"fetch_ohlcv({symbol},{timeframe})",
        )

    # ------------------------------------------------------------------ #
    #  Ticker / Funding / OI                                               #
    # ------------------------------------------------------------------ #

    async def fetch_ticker(self, symbol: str) -> dict:
        return await self._call(self._exchange.fetch_ticker(symbol), f"fetch_ticker({symbol})")

    async def fetch_tickers(self, symbols: list[str] | None = None) -> dict:
        raw = await self._call(self._exchange.fetch_tickers(symbols), "fetch_tickers")
        if not isinstance(raw, dict):
            log.warning("fetch_tickers returned %s instead of dict (possible geo-block); returning empty", type(raw).__name__)
            return {}
        return raw

    async def fetch_funding_rate(self, symbol: str) -> dict:
        """Returns {'fundingRate': float, 'nextFundingTime': int, ...}"""
        return await self._call(
            self._exchange.fetch_funding_rate(symbol), f"fetch_funding_rate({symbol})"
        )

    async def fetch_open_interest(self, symbol: str) -> dict:
        """Returns {'openInterestValue': float, ...}"""
        try:
            return await self._call(
                self._exchange.fetch_open_interest(symbol), f"fetch_open_interest({symbol})"
            )
        except Exception as exc:
            log.warning("OI fetch failed for %s: %s", symbol, exc)
            return {}

    async def fetch_long_short_ratio(self, symbol: str, period: str = "5m", limit: int = 1) -> float:
        """
        Returns global long/short account ratio (> 1 = more longs than shorts).
        Uses Binance FAPI public endpoint — no auth required.
        Falls back to 1.0 (neutral) on failure. Cached 60s.
        """
        cached = _ls_cache.get(symbol)
        if cached and time.time() - cached[1] < _RATIO_TTL:
            return cached[0]
        raw_sym = symbol.replace("/", "").replace(":USDT", "")
        try:
            data = await self._call(
                self._exchange.fapiPublicGetFuturesDataGlobalLongShortAccountRatio({
                    "symbol": raw_sym,
                    "period": period,
                    "limit": limit,
                }),
                f"long_short_ratio({raw_sym})",
            )
            if data and isinstance(data, list):
                ratio = float(data[-1].get("longShortRatio", 1.0))
                _ls_cache[symbol] = (ratio, time.time())
                return ratio
        except Exception as exc:
            log.debug("L/S ratio fetch failed for %s: %s", symbol, exc)
        return 1.0

    async def fetch_taker_buy_ratio(self, symbol: str, period: str = "5m", limit: int = 1) -> float:
        """
        Returns taker buy volume / (buy + sell) volume (0.5 = neutral).
        Uses Binance FAPI public taker buy volume endpoint.
        Falls back to 0.5 on failure. Cached 60s.
        """
        cached = _taker_cache.get(symbol)
        if cached and time.time() - cached[1] < _RATIO_TTL:
            return cached[0]
        raw_sym = symbol.replace("/", "").replace(":USDT", "")
        try:
            data = await self._call(
                self._exchange.fapiPublicGetFuturesDataTakerbuyVolume({
                    "symbol": raw_sym,
                    "period": period,
                    "limit": limit,
                }),
                f"taker_buy_ratio({raw_sym})",
            )
            if data and isinstance(data, list):
                entry = data[-1]
                buy_vol = float(entry.get("buyVol", 0))
                sell_vol = float(entry.get("sellVol", 0))
                total = buy_vol + sell_vol
                ratio = buy_vol / total if total > 0 else 0.5
                _taker_cache[symbol] = (ratio, time.time())
                return ratio
        except Exception as exc:
            log.debug("Taker buy ratio fetch failed for %s: %s", symbol, exc)
        return 0.5

    # ------------------------------------------------------------------ #
    #  Order book                                                          #
    # ------------------------------------------------------------------ #

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        return await self._call(
            self._exchange.fetch_order_book(symbol, limit=limit),
            f"fetch_order_book({symbol})",
        )

    # ------------------------------------------------------------------ #
    #  Account                                                             #
    # ------------------------------------------------------------------ #

    async def fetch_balance(self) -> dict:
        return await self._call(self._exchange.fetch_balance(), "fetch_balance")

    async def fetch_positions(self, symbols: list[str] | None = None) -> list[dict]:
        return await self._call(self._exchange.fetch_positions(symbols), "fetch_positions")

    async def fetch_open_orders(self, symbol: str) -> list[dict]:
        return await self._call(
            self._exchange.fetch_open_orders(symbol), f"fetch_open_orders({symbol})"
        )

    # ------------------------------------------------------------------ #
    #  Order management                                                    #
    # ------------------------------------------------------------------ #

    async def set_leverage(self, symbol: str, leverage: int) -> dict:
        return await self._call(
            self._exchange.set_leverage(leverage, symbol),
            f"set_leverage({symbol},{leverage})",
        )

    async def set_margin_mode(self, symbol: str, mode: str = "isolated") -> None:
        try:
            await self._call(
                self._exchange.set_margin_mode(mode, symbol),
                f"set_margin_mode({symbol},{mode})",
            )
        except Exception as exc:
            # Already set or not supported in paper mode
            log.debug("set_margin_mode %s for %s: %s", mode, symbol, exc)

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: float | None = None,
        params: dict | None = None,
    ) -> dict:
        return await self._call(
            self._exchange.create_order(symbol, order_type, side, amount, price, params or {}),
            f"create_order({symbol},{order_type},{side})",
        )

    async def cancel_order(self, order_id: str, symbol: str) -> dict:
        return await self._call(
            self._exchange.cancel_order(order_id, symbol),
            f"cancel_order({symbol},{order_id})",
        )

    async def cancel_all_orders(self, symbol: str) -> list:
        return await self._call(
            self._exchange.cancel_all_orders(symbol), f"cancel_all_orders({symbol})"
        )

    async def fetch_order(self, order_id: str, symbol: str) -> dict:
        return await self._call(
            self._exchange.fetch_order(order_id, symbol),
            f"fetch_order({symbol},{order_id})",
        )

    # ------------------------------------------------------------------ #
    #  Convenience                                                         #
    # ------------------------------------------------------------------ #

    def symbol_precision(self, symbol: str) -> tuple[int, int]:
        """Returns (amount_decimal_places, price_decimal_places) for a symbol."""
        market = self.markets.get(symbol, {})
        precision = market.get("precision", {})
        return (
            precision.get("amount", 3),
            precision.get("price", 2),
        )

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        """Truncate (floor) amount to exchange lot step size using ccxt."""
        try:
            return float(self._exchange.amount_to_precision(symbol, amount))
        except Exception:
            amt_decimals, _ = self.symbol_precision(symbol)
            step = 10 ** (-amt_decimals)
            return math.floor(amount / step) * step

    def price_to_precision(self, symbol: str, price: float) -> float:
        """Round price to exchange tick size using ccxt."""
        try:
            return float(self._exchange.price_to_precision(symbol, price))
        except Exception:
            _, price_decimals = self.symbol_precision(symbol)
            return round(price, price_decimals)

    def min_lot_size(self, symbol: str) -> float:
        """Returns the exchange minimum order quantity (stepSize / minQty)."""
        market = self.markets.get(symbol, {})
        limits = market.get("limits", {})
        return float(limits.get("amount", {}).get("min") or 0.0)

    def min_notional(self, symbol: str) -> float:
        """Returns the exchange minimum notional (cost) for the symbol."""
        market = self.markets.get(symbol, {})
        limits = market.get("limits", {})
        return float(limits.get("cost", {}).get("min") or 5.0)

    async def set_position_mode_one_way(self) -> None:
        """Ensure One-Way position mode (not Hedge). Must be called on startup."""
        try:
            await self._call(
                self._exchange.fapiPrivatePostPositionSideDual({"dualSidePosition": "false"}),
                "set_position_mode_one_way",
            )
            log.info("Position mode set to One-Way")
        except Exception as exc:
            # Already in One-Way mode raises 'No need to change position side'
            log.debug("set_position_mode_one_way: %s", exc)
