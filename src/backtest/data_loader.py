"""
Historical OHLCV data loader.
Attempts to download from Binance; falls back to realistic synthetic data
when the exchange is unreachable (e.g. ISP-level blocking).
Synthetic data uses Geometric Brownian Motion with calibrated volatility/drift.
"""
from __future__ import annotations

import logging
import ssl
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_COLS = ["timestamp", "open", "high", "low", "close", "volume"]
_CACHE_DIR = Path("data/cache")

# Realistic crypto start-price & daily vol estimates (2023 approximate)
_ASSET_PARAMS: dict[str, dict] = {
    "BTC": {"start_price": 16500.0, "daily_vol": 0.028, "drift": 0.0012},
    "ETH": {"start_price": 1200.0,  "daily_vol": 0.032, "drift": 0.0010},
    "SOL": {"start_price": 10.0,    "daily_vol": 0.055, "drift": 0.0020},
    "BNB": {"start_price": 240.0,   "daily_vol": 0.025, "drift": 0.0005},
}
_DEFAULT_PARAMS = {"start_price": 1.0, "daily_vol": 0.04, "drift": 0.0008}

_TF_MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
               "1h": 60, "2h": 120, "4h": 240, "1d": 1440}


def _cache_path(symbol: str, tf: str, start: str, end: str) -> Path:
    safe = symbol.replace("/", "_").replace(":", "_")
    return _CACHE_DIR / f"{safe}_{tf}_{start}_{end}.parquet"


def generate_synthetic_ohlcv(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data using Geometric Brownian Motion.
    Produces realistic-looking crypto candles with trends, volatility clusters,
    and volume spikes.
    """
    base = symbol.split("/")[0].replace(":", "").upper()
    # Strip suffixes like USDT
    for suffix in ["USDT", "BTC", "ETH", "BNB"]:
        if base.endswith(suffix) and len(base) > len(suffix):
            base = base[:-len(suffix)]
            break

    params = _ASSET_PARAMS.get(base, _DEFAULT_PARAMS)
    tf_min = _TF_MINUTES.get(timeframe, 60)

    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")
    index = pd.date_range(start=start_ts, end=end_ts, freq=f"{tf_min}min", tz="UTC")[:-1]
    n = len(index)

    if n == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"], index=index)

    rng = np.random.default_rng(seed or abs(hash(symbol + timeframe)) % (2**32))

    # Per-bar vol = daily_vol / sqrt(1440/tf_min)
    bar_vol = params["daily_vol"] / np.sqrt(1440 / tf_min)
    bar_drift = params["drift"] / (1440 / tf_min)

    # GARCH-like volatility clustering
    vol_multipliers = np.ones(n)
    for i in range(1, n):
        shock = abs(rng.standard_normal()) * 0.1
        vol_multipliers[i] = 0.95 * vol_multipliers[i - 1] + 0.05 + shock

    returns = rng.standard_normal(n) * bar_vol * vol_multipliers + bar_drift
    # Occasional large moves (regime shifts / news events)
    jumps = (rng.uniform(size=n) < 0.002) * rng.normal(0, bar_vol * 5, n)
    returns += jumps

    # Build close prices
    log_prices = np.cumsum(returns)
    closes = params["start_price"] * np.exp(log_prices)

    # Build OHLC from closes
    bar_range_pct = bar_vol * vol_multipliers * rng.uniform(0.5, 2.0, n)
    opens = np.empty(n)
    opens[0] = closes[0] * (1 + rng.normal(0, bar_vol * 0.3))
    for i in range(1, n):
        opens[i] = closes[i - 1] * (1 + rng.normal(0, bar_vol * 0.1))

    highs = np.maximum(opens, closes) * (1 + bar_range_pct * rng.uniform(0.3, 1.0, n))
    lows = np.minimum(opens, closes) * (1 - bar_range_pct * rng.uniform(0.3, 1.0, n))

    # Sanity: lows <= open/close <= highs
    lows = np.minimum(lows, np.minimum(opens, closes) * 0.999)
    highs = np.maximum(highs, np.maximum(opens, closes) * 1.001)

    # Volume: correlated with price moves, occasional spikes
    base_vol = 1000.0 * params["start_price"]
    vol_noise = rng.lognormal(0, 0.5, n)
    vol_spike = 1.0 + (rng.uniform(size=n) < 0.03) * rng.uniform(2, 8, n)
    volumes = base_vol * vol_noise * vol_spike * (1 + abs(returns) / bar_vol * 2)

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }, index=index)
    df.index.name = "timestamp"
    return df


def _load_or_generate(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    cache_file = _cache_path(symbol, timeframe, start_date, end_date)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        log.debug("Cache hit: %s", cache_file.name)
        return pd.read_parquet(cache_file)

    log.info("Generating synthetic data for %s %s  %s → %s", symbol, timeframe, start_date, end_date)
    df = generate_synthetic_ohlcv(symbol, timeframe, start_date, end_date)
    try:
        df.to_parquet(cache_file)
    except Exception as exc:
        log.debug("Cache save failed (non-fatal): %s", exc)
    log.info("Generated %d bars for %s %s", len(df), symbol, timeframe)
    return df


async def fetch_historical(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    testnet: bool = False,
) -> pd.DataFrame:
    """
    Fetch real OHLCV from Binance FAPI with pagination and disk cache.
    Falls back to synthetic data only if Binance is unreachable.
    """
    cache_file = _cache_path(symbol, timeframe, start_date, end_date)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        log.info("Cache hit: %s", cache_file.name)
        return pd.read_parquet(cache_file)

    raw_sym = symbol.replace("/", "").replace(":USDT", "")  # BTC/USDT:USDT → BTCUSDT
    start_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    end_ms   = int(pd.Timestamp(end_date,   tz="UTC").timestamp() * 1000)
    base_url = "https://fapi.binance.com/fapi/v1/klines"

    try:
        import httpx
        all_rows: list[list] = []
        current_start = start_ms

        async with httpx.AsyncClient(timeout=30) as client:
            while current_start < end_ms:
                r = await client.get(base_url, params={
                    "symbol":    raw_sym,
                    "interval":  timeframe,
                    "startTime": current_start,
                    "endTime":   end_ms,
                    "limit":     1500,
                })
                r.raise_for_status()
                batch = r.json()
                if not batch:
                    break
                all_rows.extend(batch)
                last_open_ms = int(batch[-1][0])
                if last_open_ms <= current_start:
                    break
                current_start = last_open_ms + 1

        if not all_rows:
            raise ValueError("Empty response from Binance")

        df = pd.DataFrame(all_rows, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        df = df.astype(float)
        df = df[df.index < pd.Timestamp(end_date, tz="UTC")]

        try:
            df.to_parquet(cache_file)
        except Exception as exc:
            log.debug("Cache save failed (non-fatal): %s", exc)

        log.info("Fetched %d real bars for %s %s from Binance", len(df), symbol, timeframe)
        return df

    except Exception as exc:
        log.warning("Binance fetch failed for %s %s — using synthetic data: %s", symbol, timeframe, exc)
        return _load_or_generate(symbol, timeframe, start_date, end_date)


async def fetch_all_symbols(
    symbols: list[str],
    timeframes: list[str],
    start_date: str,
    end_date: str,
    testnet: bool = False,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Returns {symbol: {timeframe: df}}"""
    results: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol in symbols:
        results[symbol] = {}
        for tf in timeframes:
            try:
                df = await fetch_historical(symbol, tf, start_date, end_date, testnet)
                results[symbol][tf] = df
            except Exception as exc:
                log.warning("Failed to get data for %s %s: %s", symbol, tf, exc)
                results[symbol][tf] = pd.DataFrame()
    return results
