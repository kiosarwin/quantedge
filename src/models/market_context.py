"""Broad crypto market context for pair-level admission."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class MarketContext:
    btc_trend: str = "unknown"
    eth_btc_trend: str = "unknown"
    btc_d_trend: str = "unknown"
    total_trend: str = "unknown"
    risk_on_state: str = "unknown"
    rotation_state: str = "unknown"
    confidence: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def unavailable_context() -> MarketContext:
    return MarketContext()


def _trend_from_series(values: pd.Series, threshold_pct: float) -> tuple[str, float]:
    clean = values.dropna().astype(float)
    if len(clean) < 2:
        return "unknown", 0.0
    first = float(clean.iloc[0])
    last = float(clean.iloc[-1])
    if first <= 0.0 or last <= 0.0:
        return "unknown", 0.0
    pct = (last - first) / first * 100.0
    if pct >= threshold_pct:
        return "up", abs(pct)
    if pct <= -threshold_pct:
        return "down", abs(pct)
    return "flat", abs(pct)


def classify_market_context(
    *,
    btc_candles: pd.DataFrame | None = None,
    eth_candles: pd.DataFrame | None = None,
    btc_d_candles: pd.DataFrame | None = None,
    total_candles: pd.DataFrame | None = None,
    lookback_bars: int = 48,
    trend_threshold_pct: float = 1.0,
) -> MarketContext:
    """Classify BTC-led backdrop from the broad series that are available."""
    lookback_bars = max(2, int(lookback_bars or 48))
    threshold = max(0.05, float(trend_threshold_pct or 1.0))

    def closes(df: pd.DataFrame | None) -> pd.Series:
        if df is None or df.empty or "close" not in df:
            return pd.Series(dtype=float)
        return df["close"].tail(lookback_bars)

    btc_close = closes(btc_candles)
    eth_close = closes(eth_candles)
    btc_d_close = closes(btc_d_candles)
    total_close = closes(total_candles)

    btc_trend, btc_mag = _trend_from_series(btc_close, threshold)
    btc_d_trend, btc_d_mag = _trend_from_series(btc_d_close, threshold)
    total_trend, total_mag = _trend_from_series(total_close, threshold)

    eth_btc_trend = "unknown"
    eth_btc_mag = 0.0
    if len(btc_close) >= 2 and len(eth_close) >= 2:
        aligned = pd.concat([eth_close.rename("eth"), btc_close.rename("btc")], axis=1).dropna()
        if len(aligned) >= 2:
            ratio = aligned["eth"] / aligned["btc"]
            eth_btc_trend, eth_btc_mag = _trend_from_series(ratio, threshold * 0.5)

    available = [btc_trend, eth_btc_trend, btc_d_trend, total_trend]
    known_count = sum(1 for value in available if value != "unknown")
    mag = min(1.0, (btc_mag + eth_btc_mag + btc_d_mag + total_mag) / 12.0)
    confidence = round(min(1.0, known_count / 4.0 * 0.70 + mag * 0.30), 3)

    if known_count == 0:
        risk_on_state = "unknown"
    elif btc_trend == "down" and total_trend in {"down", "unknown", "flat"}:
        risk_on_state = "risk_off"
    elif btc_trend == "up" and eth_btc_trend == "up":
        risk_on_state = "risk_on_alts"
    elif btc_trend == "up":
        risk_on_state = "risk_on_btc"
    elif total_trend == "up" and btc_d_trend == "down":
        risk_on_state = "risk_on_alts"
    elif btc_trend == "flat" and total_trend == "flat":
        risk_on_state = "neutral"
    else:
        risk_on_state = "mixed"

    if eth_btc_trend == "up" and btc_d_trend in {"down", "unknown", "flat"}:
        rotation_state = "alts_outperforming"
    elif eth_btc_trend == "down" and btc_d_trend in {"up", "unknown", "flat"}:
        rotation_state = "btc_outperforming"
    elif btc_d_trend == "down" and total_trend == "up":
        rotation_state = "broad_alt_rotation"
    elif btc_d_trend == "up" and btc_trend != "down":
        rotation_state = "btc_dominance_bid"
    elif known_count == 0:
        rotation_state = "unknown"
    else:
        rotation_state = "mixed_rotation"

    return MarketContext(
        btc_trend=btc_trend,
        eth_btc_trend=eth_btc_trend,
        btc_d_trend=btc_d_trend,
        total_trend=total_trend,
        risk_on_state=risk_on_state,
        rotation_state=rotation_state,
        confidence=confidence,
    )


def market_context_score_mult(context: MarketContext | None, direction: str) -> tuple[float, str]:
    """Return a soft score multiplier and reason for a pair direction."""
    if context is None or context.confidence <= 0.0:
        return 1.0, "market_context_unavailable"

    direction = str(direction or "").lower()
    state = context.risk_on_state
    rotation = context.rotation_state

    mult = 1.0
    reasons: list[str] = []
    if direction == "long":
        if state in {"risk_on_alts", "risk_on_btc"}:
            mult *= 1.05
            reasons.append(state)
        elif state == "risk_off":
            mult *= 0.90
            reasons.append("risk_off_against_long")
        elif state == "mixed":
            mult *= 0.97
            reasons.append("mixed_backdrop")
        if rotation in {"alts_outperforming", "broad_alt_rotation"}:
            mult *= 1.03
            reasons.append(rotation)
        elif rotation == "btc_outperforming":
            mult *= 0.97
            reasons.append(rotation)
    elif direction == "short":
        if state == "risk_off":
            mult *= 1.06
            reasons.append("risk_off_supports_short")
        elif state in {"risk_on_alts", "risk_on_btc"}:
            mult *= 0.92
            reasons.append(f"{state}_against_short")
        elif state == "mixed":
            mult *= 0.98
            reasons.append("mixed_backdrop")
        if rotation in {"alts_outperforming", "broad_alt_rotation"}:
            mult *= 0.97
            reasons.append(rotation)

    confidence_blend = max(0.0, min(1.0, context.confidence))
    blended = 1.0 + (mult - 1.0) * confidence_blend
    return round(max(0.85, min(1.10, blended)), 4), ",".join(reasons) or state
