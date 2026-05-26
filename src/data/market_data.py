"""
Market data aggregator — fetches and caches OHLCV, order book, funding, OI.
"""
from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .client import BinanceFuturesClient
from src.analysis.spot_context import SpotContext, fetch_spot_context  # noqa: F401 (SpotContext used in type hint)
from src.models.market_context import MarketContext, classify_market_context, unavailable_context
from src.models.sector_rotation import SectorRotation
from src.models.strategy_passport import register_market_sectors
from src.models.time_series import TimeSeriesDiagnostics, compute_time_series_diagnostics

log = logging.getLogger(__name__)

_OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass
class MarketSnapshot:
    symbol: str
    candles: dict[str, pd.DataFrame] = field(default_factory=dict)   # timeframe → df
    order_book: dict = field(default_factory=dict)
    funding_rate: float = 0.0
    next_funding_ms: int = 0
    open_interest_usd: float = 0.0
    oi_change_pct: float = 0.0          # vs previous fetch
    last_price: float = 0.0
    volume_24h_usdt: float = 0.0
    ls_ratio: float = 1.0               # global long/short account ratio
    taker_buy_ratio: float = 0.5        # taker buy vol / total (0.5 = neutral)
    spot_context: SpotContext | None = None
    market_context: MarketContext | None = None
    sector_rotation: SectorRotation | None = None
    time_series: TimeSeriesDiagnostics | None = None
    fetched_at: float = field(default_factory=time.time)

    def candles_for(self, tf: str) -> pd.DataFrame:
        return self.candles.get(tf, pd.DataFrame())


class MarketDataService:
    """Fetches and lightly caches market data for a list of symbols."""

    def __init__(self, client: BinanceFuturesClient, cfg: dict):
        self._client = client
        self._cfg = cfg
        tf_cfg = cfg["timeframes"]
        self._timeframes: list[str] = list({
            tf_cfg["primary"],
            tf_cfg["higher"],
            tf_cfg.get("lower") or tf_cfg.get("secondary"),
            tf_cfg["entry"],
        })
        self._ob_depth: int = cfg.get("order_book", {}).get("depth_levels", 20)

        # symbol → {timeframe → (df, fetched_at)}
        self._candle_cache: dict[str, dict[str, tuple[pd.DataFrame, float]]] = {}
        self._candle_ttl = cfg["trading"].get("candle_refresh_seconds", 30)

        # symbol → deque of (oi_value, fetched_at)
        self._oi_history: dict[str, collections.deque] = {}
        self._oi_baseline_seconds: int = cfg.get("smart_money", {}).get("oi_baseline_seconds", 3600)

        safety_cfg = cfg.get("safety", {}) or {}
        self._snapshot_concurrency = max(
            1, int(safety_cfg.get("market_data_snapshot_concurrency", 8) or 8)
        )
        self._snapshot_timeout_s = max(
            1.0, float(safety_cfg.get("market_data_snapshot_timeout_seconds", 25.0) or 25.0)
        )
        ts_cfg = cfg.get("time_series", {}) or {}
        self._time_series_enabled = bool(ts_cfg.get("enabled", True))
        self._time_series_lookback = max(30, int(ts_cfg.get("lookback_bars", 96) or 96))
        self._time_series_ewma_lambda = min(0.99, max(0.01, float(ts_cfg.get("ewma_lambda", 0.94) or 0.94)))
        self._time_series_min_returns = max(20, int(ts_cfg.get("min_returns", 30) or 30))
        self._time_series_shock_z = max(1.0, float(ts_cfg.get("shock_z_threshold", 2.8) or 2.8))
        self._time_series_vol_ratio = max(1.05, float(ts_cfg.get("vol_ratio_threshold", 1.65) or 1.65))
        self._time_series_markov_window = max(3, int(ts_cfg.get("markov_window", 20) or 20))
        self._time_series_markov_threshold_pct = max(0.05, float(ts_cfg.get("markov_threshold_pct", 2.0) or 2.0))

        context_cfg = cfg.get("market_context", {}) or {}
        self._market_context_enabled = bool(context_cfg.get("enabled", True))
        self._market_context_ttl = max(
            30.0, float(context_cfg.get("ttl_seconds", 300.0) or 300.0)
        )
        self._market_context_lookback = max(
            2, int(context_cfg.get("lookback_bars", 48) or 48)
        )
        self._market_context_threshold = max(
            0.05, float(context_cfg.get("trend_threshold_pct", 1.0) or 1.0)
        )
        self._market_context_symbols = {
            "btc": str(context_cfg.get("btc_symbol", "BTC/USDT:USDT")),
            "eth": str(context_cfg.get("eth_symbol", "ETH/USDT:USDT")),
            "btc_d": str(context_cfg.get("btc_d_symbol", "")),
            "total": str(context_cfg.get("total_symbol", "")),
        }
        self._market_context_cache: tuple[MarketContext, float] | None = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    async def fetch_snapshot(self, symbol: str) -> MarketSnapshot:
        snap = MarketSnapshot(symbol=symbol)

        candles, ticker, ob, funding, oi, ls_ratio, taker_ratio = await self._fetch_all(symbol)

        snap.candles = candles
        snap.order_book = ob
        snap.last_price = float(ticker.get("last", 0))
        snap.volume_24h_usdt = float(ticker.get("quoteVolume", 0))

        if funding:
            snap.funding_rate = float(funding.get("fundingRate", 0))
            snap.next_funding_ms = int(funding.get("nextFundingTime", 0) or 0)

        current_oi = float(oi.get("openInterestValue", 0) or 0)
        snap.open_interest_usd = current_oi
        now_ts = time.time()

        history = self._oi_history.setdefault(symbol, collections.deque())
        history.append((current_oi, now_ts))

        # Prune entries older than baseline window + 10 min buffer
        cutoff = now_ts - (self._oi_baseline_seconds + 600)
        while history and history[0][1] < cutoff:
            history.popleft()

        # Baseline = oldest entry at least 15 min old, else oldest available
        baseline_oi = 0.0
        target_ts = now_ts - self._oi_baseline_seconds
        for val, ts in history:
            if ts <= target_ts:
                baseline_oi = val
            else:
                break
        if baseline_oi == 0.0 and len(history) > 1:
            baseline_oi = history[0][0]

        if baseline_oi > 0 and current_oi > 0:
            snap.oi_change_pct = (current_oi - baseline_oi) / baseline_oi * 100

        snap.ls_ratio = ls_ratio
        snap.taker_buy_ratio = taker_ratio

        if self._time_series_enabled:
            try:
                primary_tf = self._cfg["timeframes"]["primary"]
                snap.time_series = compute_time_series_diagnostics(
                    snap.candles_for(primary_tf),
                    lookback=self._time_series_lookback,
                    ewma_lambda=self._time_series_ewma_lambda,
                    min_returns=self._time_series_min_returns,
                    shock_z_threshold=self._time_series_shock_z,
                    vol_ratio_threshold=self._time_series_vol_ratio,
                    markov_window=self._time_series_markov_window,
                    markov_threshold_pct=self._time_series_markov_threshold_pct,
                )
            except Exception as exc:
                log.debug("Time-series diagnostics failed for %s: %s", symbol, exc)

        # Raw spot data — direction-neutral. Scorer applies directional mult via compute_spot_mult().
        if snap.last_price > 0:
            try:
                snap.spot_context = await fetch_spot_context(symbol, snap.last_price)
            except Exception:
                pass

        return snap

    async def fetch_snapshots(self, symbols: list[str]) -> dict[str, MarketSnapshot]:
        import asyncio

        client = getattr(self, "_client", None)
        register_market_sectors(getattr(client, "markets", {}) or {})
        semaphore = asyncio.Semaphore(self._snapshot_concurrency)
        market_context = await self.fetch_market_context()

        async def _bounded_fetch(symbol: str):
            async with semaphore:
                try:
                    return await asyncio.wait_for(
                        self.fetch_snapshot(symbol),
                        timeout=self._snapshot_timeout_s,
                    )
                except asyncio.TimeoutError:
                    log.warning(
                        "Snapshot timed out for %s after %.1fs",
                        symbol,
                        self._snapshot_timeout_s,
                    )
                    return None

        tasks = [_bounded_fetch(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: dict[str, MarketSnapshot] = {}
        for sym, res in zip(symbols, results):
            if isinstance(res, Exception):
                log.warning("Snapshot failed for %s: %s", sym, res)
            elif res is not None:
                res.market_context = market_context
                # Backward-compatible hints for the EV/P(win) context.
                res.btc_state = market_context.btc_trend
                res.btcd_state = market_context.btc_d_trend
                out[sym] = res
        return out

    async def fetch_market_context(self) -> MarketContext:
        if not self._market_context_enabled:
            return unavailable_context()
        now = time.time()
        if self._market_context_cache and now - self._market_context_cache[1] < self._market_context_ttl:
            return self._market_context_cache[0]

        async def _optional_candles(symbol: str) -> pd.DataFrame:
            if not symbol:
                return pd.DataFrame()
            try:
                raw = await self._client.fetch_ohlcv(
                    symbol,
                    self._cfg["timeframes"]["primary"],
                    limit=self._market_context_lookback,
                )
                return _ohlcv_to_df(raw)
            except Exception as exc:
                log.debug("Market context fetch failed for %s: %s", symbol, exc)
                return pd.DataFrame()

        import asyncio

        btc_df, eth_df, btc_d_df, total_df = await asyncio.gather(
            _optional_candles(self._market_context_symbols["btc"]),
            _optional_candles(self._market_context_symbols["eth"]),
            _optional_candles(self._market_context_symbols["btc_d"]),
            _optional_candles(self._market_context_symbols["total"]),
        )
        context = classify_market_context(
            btc_candles=btc_df,
            eth_candles=eth_df,
            btc_d_candles=btc_d_df,
            total_candles=total_df,
            lookback_bars=self._market_context_lookback,
            trend_threshold_pct=self._market_context_threshold,
        )
        self._market_context_cache = (context, now)
        log.info(
            "Market context btc=%s eth_btc=%s btc_d=%s total=%s risk=%s rotation=%s conf=%.2f",
            context.btc_trend,
            context.eth_btc_trend,
            context.btc_d_trend,
            context.total_trend,
            context.risk_on_state,
            context.rotation_state,
            context.confidence,
        )
        return context

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    async def _fetch_all(self, symbol: str):
        import asyncio
        deriv_cfg = self._cfg.get("derivatives", {})
        period = deriv_cfg.get("ls_ratio_period", "5m")

        candles, ticker, ob, funding, oi, ls, taker = await asyncio.gather(
            self._fetch_candles(symbol),
            self._client.fetch_ticker(symbol),
            self._client.fetch_order_book(symbol, self._ob_depth),
            self._client.fetch_funding_rate(symbol),
            self._client.fetch_open_interest(symbol),
            self._client.fetch_long_short_ratio(symbol, period),
            self._client.fetch_taker_buy_ratio(symbol, period),
            return_exceptions=True,
        )
        return (
            candles if not isinstance(candles, Exception) else {},
            ticker if not isinstance(ticker, Exception) else {},
            ob if not isinstance(ob, Exception) else {},
            funding if not isinstance(funding, Exception) else {},
            oi if not isinstance(oi, Exception) else {},
            ls if not isinstance(ls, Exception) else 1.0,
            taker if not isinstance(taker, Exception) else 0.5,
        )

    async def _fetch_candles(self, symbol: str) -> dict[str, pd.DataFrame]:
        import asyncio
        now = time.time()
        tasks = {}
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
    df = pd.DataFrame(raw, columns=_OHLCV_COLS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df
