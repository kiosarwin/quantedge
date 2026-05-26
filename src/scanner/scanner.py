"""
Pair scanner — discovers tradeable Binance Futures USDT pairs and applies filters.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.data.client import BinanceFuturesClient

log = logging.getLogger(__name__)


class PairScanner:
    def __init__(self, client: BinanceFuturesClient, cfg: dict):
        self._client = client
        self._filters = cfg["filters"]
        self._trading = cfg["trading"]

    async def scan(self, priority_symbols: list[str] | None = None) -> list[str]:
        """Return filtered list of tradeable symbols."""
        try:
            tickers = await self._client.fetch_tickers()
        except Exception as exc:
            log.warning("Scanner ticker fetch failed: %s", exc)
            return []
        markets = self._client.markets

        candidates: list[tuple[str, float]] = []
        priority_symbols = priority_symbols or []

        for symbol, ticker in tickers.items():
            market = markets.get(symbol, {})

            # Must be a linear USDT-margined perpetual future
            if not self._is_valid_market(symbol, market):
                continue

            volume_usdt = float(ticker.get("quoteVolume", 0) or 0)
            if volume_usdt < self._filters["min_24h_volume_usdt"]:
                continue

            last_price = float(ticker.get("last", 0) or 0)
            if last_price < self._filters["min_price_usdt"]:
                continue

            candidates.append((symbol, volume_usdt))

        # Sort by volume descending, then let the learned edge queue override
        # position. A symbol with validated/probabilistic edge should not be
        # buried simply because BTC/ETH majors have larger quote volume.
        candidates.sort(key=lambda x: x[1], reverse=True)
        volume_ranked = [sym for sym, _ in candidates]

        priority_hits: list[str] = []
        seen_priority: set[str] = set()
        for symbol in priority_symbols:
            if symbol in seen_priority:
                continue
            seen_priority.add(symbol)
            ticker = tickers.get(symbol)
            market = markets.get(symbol, {})
            if ticker is None:
                continue
            if not self._is_valid_market(symbol, market):
                continue
            last_price = float(ticker.get("last", 0) or 0)
            if last_price < self._filters["min_price_usdt"]:
                continue
            priority_hits.append(symbol)

        priority_set = set(priority_hits)
        result = priority_hits + [sym for sym in volume_ranked if sym not in priority_set]
        log.info("Scanner found %d eligible pairs", len(result))
        return result

    def _is_valid_market(self, symbol: str, market: dict) -> bool:
        if not symbol.endswith("/USDT:USDT") and not symbol.endswith("USDT"):
            return False

        # Only swap/perpetual futures
        if market.get("type") not in ("swap", "future"):
            return False
        if not market.get("active", False):
            return False
        if not market.get("linear", True):
            return False

        base = market.get("base", "")
        quote = market.get("quote", "")
        settle = market.get("settle", quote)

        if quote != "USDT" and settle != "USDT":
            return False

        # Blacklist check
        if symbol in self._filters.get("blacklisted_pairs", []):
            return False
        if base in self._filters.get("stablecoins", []):
            return False

        # Reject non-ASCII symbols (honeypots/scam tokens with unicode names)
        if not symbol.isascii():
            return False

        return True
