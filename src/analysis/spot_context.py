"""
Spot market context — basis, spot volume confirmation, Coinbase premium.

Three independent signals that sharpen futures edge:
  1. Basis (futures - spot) / spot: crowded contango = crowded longs = risk.
  2. Spot volume spike: real money moving in spot confirms the futures signal.
  3. Coinbase premium (BTC/ETH/SOL): US institutional flow is directional alpha.

fetch_spot_context() collects raw data (direction-neutral).
compute_spot_mult() applies the directional multiplier — called in scorer.py
after direction is determined. Result is capped 0.75–1.15, never hard-blocks.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Coinbase product IDs for supported symbols
_COINBASE_MAP: dict[str, str] = {
    "BTC/USDT:USDT": "BTC-USD",
    "ETH/USDT:USDT": "ETH-USD",
    "SOL/USDT:USDT": "SOL-USD",
    "BNB/USDT:USDT": "BNB-USDT",
}

_vol_ema: dict[str, float] = {}
_VOL_EMA_ALPHA = 0.15

_spot_cache: dict[str, tuple[float, float, float]] = {}
_SPOT_TTL = 90.0

_cb_cache: dict[str, tuple[float, float]] = {}
_CB_TTL  = 90.0


@dataclass
class SpotContext:
    symbol: str
    spot_price: float
    basis_pct: float            # (futures - spot) / spot * 100
    spot_volume_ratio: float    # current 24h vol / EMA baseline (1.0 = average)
    coinbase_premium_pct: float # (coinbase - binance_spot) / binance_spot * 100


async def fetch_spot_context(symbol: str, futures_price: float) -> SpotContext | None:
    """
    Fetch raw spot data for a futures symbol.
    Direction-neutral — call compute_spot_mult() separately once direction is known.
    Returns None silently if unavailable.
    """
    spot_symbol = symbol.split(":")[0]   # BTC/USDT:USDT → BTC/USDT

    try:
        spot_price, spot_vol_24h = await _fetch_binance_spot(spot_symbol)
        if spot_price <= 0:
            return None

        basis_pct = (futures_price - spot_price) / spot_price * 100
        vol_ratio = _update_vol_ema(symbol, spot_vol_24h)

        cb_premium = 0.0
        cb_product = _COINBASE_MAP.get(symbol)
        if cb_product:
            try:
                import asyncio
                cb_price = await asyncio.wait_for(_fetch_coinbase(cb_product), timeout=2.0)
                if cb_price and spot_price > 0:
                    cb_premium = (cb_price - spot_price) / spot_price * 100
            except Exception:
                pass  # Coinbase optional — never block the scan

        return SpotContext(
            symbol=symbol,
            spot_price=spot_price,
            basis_pct=basis_pct,
            spot_volume_ratio=vol_ratio,
            coinbase_premium_pct=cb_premium,
        )

    except Exception as exc:
        log.debug("spot_context failed for %s: %s", symbol, exc)
        return None


def compute_spot_mult(ctx: SpotContext, direction: str) -> tuple[float, list[str]]:
    """
    Compute score multiplier from spot context given the actual trade direction.
    Returns (multiplier, reasoning_list). Multiplier is capped [0.75, 1.15].
    """
    mult = 1.0
    reasoning: list[str] = []

    # ── Basis ────────────────────────────────────────────────────────
    if direction == "long":
        if ctx.basis_pct > 0.50:
            mult *= 0.80
            reasoning.append(f"extreme contango {ctx.basis_pct:+.2f}% — longs crowded")
        elif ctx.basis_pct > 0.30:
            mult *= 0.90
            reasoning.append(f"elevated basis {ctx.basis_pct:+.2f}% — caution long")
        elif ctx.basis_pct > 0.15:
            mult *= 0.96
            reasoning.append(f"mild contango {ctx.basis_pct:+.2f}%")
        elif ctx.basis_pct < -0.10:
            mult *= 1.06
            reasoning.append(f"futures discount {ctx.basis_pct:+.2f}% — cheap long entry")
    else:  # short
        if ctx.basis_pct > 0.40:
            mult *= 1.06
            reasoning.append(f"contango {ctx.basis_pct:+.2f}% — favours short")
        elif ctx.basis_pct < -0.15:
            mult *= 0.90
            reasoning.append(f"backwardation {ctx.basis_pct:+.2f}% — short squeeze risk")

    # ── Spot volume ──────────────────────────────────────────────────
    if ctx.spot_volume_ratio > 2.0:
        mult *= 1.08 if direction == "long" else 1.05
        reasoning.append(f"spot vol spike {ctx.spot_volume_ratio:.1f}x — real conviction")
    elif ctx.spot_volume_ratio > 1.5:
        mult *= 1.04
        reasoning.append(f"elevated spot vol {ctx.spot_volume_ratio:.1f}x")
    elif 0 < ctx.spot_volume_ratio < 0.4:
        mult *= 0.94
        reasoning.append(f"thin spot vol {ctx.spot_volume_ratio:.1f}x — low conviction")

    # ── Coinbase premium ─────────────────────────────────────────────
    if ctx.coinbase_premium_pct != 0.0:
        if direction == "long" and ctx.coinbase_premium_pct > 0.15:
            mult *= 1.07
            reasoning.append(f"CB premium +{ctx.coinbase_premium_pct:.2f}% — US institutions buying")
        elif direction == "long" and ctx.coinbase_premium_pct < -0.15:
            mult *= 0.92
            reasoning.append(f"CB discount {ctx.coinbase_premium_pct:.2f}% — US institutions selling")
        elif direction == "short" and ctx.coinbase_premium_pct < -0.15:
            mult *= 1.07
            reasoning.append(f"CB discount {ctx.coinbase_premium_pct:.2f}% — US selling, confirms short")
        elif direction == "short" and ctx.coinbase_premium_pct > 0.15:
            mult *= 0.92
            reasoning.append(f"CB premium +{ctx.coinbase_premium_pct:.2f}% — US buying, counter-short risk")

    return round(max(0.75, min(1.15, mult)), 4), reasoning


# ── Data fetchers ─────────────────────────────────────────────────────────

async def _fetch_binance_spot(spot_symbol: str) -> tuple[float, float]:
    """Returns (last_price, quote_volume_24h) from Binance spot public API."""
    cached = _spot_cache.get(spot_symbol)
    if cached and time.time() - cached[1] < _SPOT_TTL:
        return cached[0], cached[2]

    binance_sym = spot_symbol.replace("/", "")  # "BTC/USDT" → "BTCUSDT"
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(
                "https://api.binance.com/api/v3/ticker/24hr",
                params={"symbol": binance_sym},
            )
            if r.status_code == 200:
                data = r.json()
                price = float(data.get("lastPrice") or 0)
                vol   = float(data.get("quoteVolume") or 0)
                _spot_cache[spot_symbol] = (price, time.time(), vol)
                return price, vol
    except Exception as exc:
        log.debug("Binance spot ticker failed %s: %s", spot_symbol, exc)
    return 0.0, 0.0


async def _fetch_coinbase(product: str) -> float | None:
    """Returns last price from Coinbase public REST API. No auth required."""
    cached = _cb_cache.get(product)
    if cached and time.time() - cached[1] < _CB_TTL:
        return cached[0]

    try:
        import httpx
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get(
                f"https://api.exchange.coinbase.com/products/{product}/ticker",
                headers={"User-Agent": "NinjaTrader/1.0"},
            )
            if r.status_code == 200:
                price = float(r.json().get("price", 0))
                _cb_cache[product] = (price, time.time())
                return price
    except Exception as exc:
        log.debug("Coinbase fetch failed %s: %s", product, exc)
    return None


def _update_vol_ema(symbol: str, vol: float) -> float:
    """EMA of spot 24h volume. Returns current/ema ratio (1.0 = average)."""
    if vol <= 0:
        return 1.0
    prev = _vol_ema.get(symbol)
    if prev is None or prev <= 0:
        _vol_ema[symbol] = vol
        return 1.0
    new_ema = prev * (1 - _VOL_EMA_ALPHA) + vol * _VOL_EMA_ALPHA
    _vol_ema[symbol] = new_ema
    return round(vol / new_ema, 3)
