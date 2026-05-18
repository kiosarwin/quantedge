"""
Spot market data aggregator — fetches and caches OHLCV for Spot pairs.

Spot has no funding rate, no open interest, no margin.
We add BTC price data so the BTC safety gate can use it.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import pandas as pd

from .spot_client import BinanceSpotClient

log = logging.getLogger(__name__)

_OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass
class SpotSnapshot:
    symbol: str
    candles: dict[str, pd.DataFrame] = field(default_factory=dict)
    last_price: float = 0.0
    volume_24h_usdt: float = 0.0
    fetched_at: float = field(default_factory=time.time)

    def candles_for(self, tf: str) -> pd.DataFrame:
        return self.candles.get(tf, pd.DataFrame())

    @property
    def is_valid(self) -> bool:
        return self.last_price > 0 and bool(self.candles)


class SpotMarketDataService:
    """Fetches and lightly caches OHLCV + ticker for Spot symbols."""

    def __init__(self, client: BinanceSpotClient, cfg: dict):
        self._client = client
        self._cfg = cfg
        tf_cfg = cfg["timeframes"]
        self._timeframes: list[str] = list({
            tf_cfg["primary"],
            tf_cfg.get("secondary", tf_cfg["primary"]),
            tf_cfg["entry"],
            tf_cfg.get("higher", tf_cfg["primary"]),
        })
        # symbol → {timeframe → (df, fetched_at)}
        self._candle_cache: dict[str, dict[str, tuple[pd.DataFrame, float]]] = {}
        self._candle_ttl = cfg["trading"].get("candle_refresh_seconds", 60)

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    async def fetch_snapshot(self, symbol: str) -> SpotSnapshot:
        snap = SpotSnapshot(symbol=symbol)
        candles, ticker = await asyncio.gather(
            self._fetch_candles(symbol),
            self._client.fetch_ticker(symbol),
            return_exceptions=True,
        )
        if isinstance(candles, Exception):
            log.warning("Candle fetch failed %s: %s", symbol, candles)
            candles = {}
        if isinstance(ticker, Exception):
            log.warning("Ticker fetch failed %s: %s", symbol, ticker)
            ticker = {}

        snap.candles = candles
        snap.last_price = float(ticker.get("last", 0) or 0)
        snap.volume_24h_usdt = float(ticker.get("quoteVolume", 0) or 0)
        return snap

    async def fetch_snapshots(self, symbols: list[str]) -> dict[str, SpotSnapshot]:
        tasks = [self.fetch_snapshot(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: dict[str, SpotSnapshot] = {}
        for sym, res in zip(symbols, results):
            if isinstance(res, Exception):
                log.warning("Snapshot failed for %s: %s", sym, res)
            elif res.is_valid:
                out[sym] = res
        return out

    async def fetch_btc_candles(self, timeframe: str, limit: int = 300) -> pd.DataFrame:
        """Fetch BTC/USDT candles — used by BTC safety gate."""
        btc_symbol = self._cfg["btc"]["symbol"]
        raw = await self._client.fetch_ohlcv(btc_symbol, timeframe, limit=limit)
        return _ohlcv_to_df(raw) if raw else pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  Internal                                                           #
    # ------------------------------------------------------------------ #

    async def _fetch_candles(self, symbol: str) -> dict[str, pd.DataFrame]:
        now = time.time()
        tasks: dict[str, object] = {}

        for tf in self._timeframes:
            cache = self._candle_cache.get(symbol, {}).get(tf)
            if cache and (now - cache[1]) < self._candle_ttl:
                continue
            tasks[tf] = self._client.fetch_ohlcv(symbol, tf, limit=300)

        if tasks:
            fetched = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for tf, result in zip(tasks.keys(), fetched):
                if isinstance(result, Exception):
                    log.warning("OHLCV fetch failed %s %s: %s", symbol, tf, result)
                    continue
                df = _ohlcv_to_df(result)
                self._candle_cache.setdefault(symbol, {})[tf] = (df, now)

        return {
            tf: self._candle_cache.get(symbol, {}).get(tf, (pd.DataFrame(),))[0]
            for tf in self._timeframes
        }


def _ohlcv_to_df(raw: list[list]) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=_OHLCV_COLS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df
