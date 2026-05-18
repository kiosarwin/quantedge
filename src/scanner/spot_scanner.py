"""
Spot Pair Scanner — discovers and ranks Binance Spot USDT pairs.

Filters:
  * USDT quote asset
  * Active spot market
  * Minimum 24H volume threshold
  * Not a stablecoin
  * Not blacklisted

Output: sorted list of symbols (highest volume first), capped at max_pairs.
"""
from __future__ import annotations

import logging

from src.data.spot_client import BinanceSpotClient

log = logging.getLogger(__name__)


class SpotScanner:
    def __init__(self, client: BinanceSpotClient, cfg: dict):
        self._client = client
        self._filters = cfg["filters"]
        self._trading = cfg["trading"]

    async def scan(self) -> list[str]:
        """Return filtered and sorted list of tradeable Spot symbols."""
        try:
            tickers = await self._client.fetch_tickers()
        except Exception as exc:
            log.error("Failed to fetch tickers: %s", exc)
            return []

        markets = self._client.markets
        candidates: list[tuple[str, float]] = []

        for symbol, ticker in tickers.items():
            market = markets.get(symbol, {})

            if not self._is_valid_spot_market(symbol, market):
                continue

            volume_usdt = float(ticker.get("quoteVolume", 0) or 0)
            if volume_usdt < self._filters["min_24h_volume_usdt"]:
                continue

            last_price = float(ticker.get("last", 0) or 0)
            if last_price < self._filters["min_price_usdt"]:
                continue

            candidates.append((symbol, volume_usdt))

        candidates.sort(key=lambda x: x[1], reverse=True)

        # Cap to configured max for scoring efficiency
        max_scan = self._trading.get("max_pairs_to_scan", 60)
        result = [sym for sym, _ in candidates[:max_scan]]
        log.info("SpotScanner: %d candidates (from %d total tickers)", len(result), len(tickers))
        return result

    def _is_valid_spot_market(self, symbol: str, market: dict) -> bool:
        # Must end in USDT (spot pair)
        if not symbol.endswith("/USDT") and not symbol.endswith("USDT"):
            return False

        # Must be a spot market
        if market.get("type") not in ("spot",):
            return False
        if not market.get("active", False):
            return False
        if not market.get("spot", False):
            return False

        base = market.get("base", "")
        quote = market.get("quote", "")

        if quote != "USDT":
            return False

        # Skip stablecoins (USDC, BUSD, etc.)
        stablecoins = set(self._filters.get("stablecoins", []))
        if base in stablecoins:
            return False

        # Skip blacklisted pairs
        blacklist = set(self._filters.get("blacklisted_pairs", []))
        clean_symbol = symbol.replace("/", "")
        if symbol in blacklist or clean_symbol in blacklist:
            return False

        # Skip leveraged tokens (3L, 3S, UP, DOWN)
        suffixes = ("3L", "3S", "UP", "DOWN", "BULL", "BEAR")
        if any(base.endswith(s) for s in suffixes):
            return False

        return True
