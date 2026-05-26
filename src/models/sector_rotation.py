"""Cross-sectional sector rotation from already-fetched pair snapshots."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.models.strategy_passport import asset_from_symbol, sector_for_asset


@dataclass(frozen=True)
class SectorRotation:
    sector: str = "unknown"
    state: str = "unknown"
    rank: int = 0
    sector_return_pct: float = 0.0
    relative_btc_pct: float = 0.0
    breadth: float = 0.0
    volume_accel: float = 1.0
    oi_change_pct: float = 0.0
    confidence: float = 0.0
    reason: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def unavailable_sector_rotation(sector: str = "unknown") -> SectorRotation:
    return SectorRotation(sector=sector)


def _primary_candles(snapshot) -> pd.DataFrame:
    candles = getattr(snapshot, "candles", {}) or {}
    if "1h" in candles:
        df = candles.get("1h")
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    for df in candles.values():
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    return pd.DataFrame()


def _return_pct(df: pd.DataFrame, lookback_bars: int) -> float | None:
    if df is None or df.empty or "close" not in df:
        return None
    close = df["close"].dropna().astype(float).tail(max(2, int(lookback_bars or 24)))
    if len(close) < 2:
        return None
    first = float(close.iloc[0])
    last = float(close.iloc[-1])
    if first <= 0.0 or last <= 0.0:
        return None
    return (last - first) / first * 100.0


def _volume_accel(df: pd.DataFrame) -> float:
    if df is None or df.empty or "volume" not in df:
        return 1.0
    vol = df["volume"].dropna().astype(float).tail(30)
    if len(vol) < 8:
        return 1.0
    recent = float(vol.tail(3).mean())
    base = float(vol.iloc[:-3].tail(24).mean())
    if base <= 0.0 or recent <= 0.0:
        return 1.0
    return max(0.1, min(5.0, recent / base))


def _state_for_sector(relative_btc_pct: float, breadth: float, volume_accel: float, oi_change_pct: float) -> str:
    if relative_btc_pct >= 1.5 and breadth >= 0.55 and (volume_accel >= 1.05 or oi_change_pct >= 0.0):
        return "rotating_in"
    if relative_btc_pct <= -1.5 and breadth <= 0.45:
        return "rotating_out"
    if relative_btc_pct >= 0.75 and breadth >= 0.50:
        return "firming"
    if relative_btc_pct <= -0.75 and breadth <= 0.50:
        return "weakening"
    if abs(relative_btc_pct) < 0.75 and 0.40 <= breadth <= 0.60:
        return "neutral"
    return "mixed"


def _confidence(symbol_count: int, relative_btc_pct: float, breadth: float, volume_accel: float) -> float:
    sample = min(1.0, max(0.25, symbol_count / 4.0))
    rel = min(1.0, abs(relative_btc_pct) / 5.0)
    breadth_edge = min(1.0, abs(breadth - 0.5) * 2.0)
    vol = min(1.0, abs(volume_accel - 1.0) / 1.0)
    return round(max(0.0, min(1.0, sample * (0.45 + rel * 0.30 + breadth_edge * 0.15 + vol * 0.10))), 3)


def compute_sector_rotation(
    snapshots: dict[str, object],
    *,
    lookback_bars: int = 24,
    btc_symbol: str = "BTC/USDT:USDT",
) -> dict[str, SectorRotation]:
    rows: list[dict] = []
    btc_return = 0.0
    btc_snap = snapshots.get(btc_symbol)
    if btc_snap is not None:
        btc_ret = _return_pct(_primary_candles(btc_snap), lookback_bars)
        if btc_ret is not None:
            btc_return = btc_ret

    for symbol, snap in (snapshots or {}).items():
        asset = asset_from_symbol(symbol)
        sector = sector_for_asset(asset)
        if sector in {"other", "unknown"}:
            continue
        df = _primary_candles(snap)
        ret = _return_pct(df, lookback_bars)
        if ret is None:
            continue
        rows.append(
            {
                "symbol": symbol,
                "sector": sector,
                "return_pct": float(ret),
                "volume_accel": _volume_accel(df),
                "oi_change_pct": float(getattr(snap, "oi_change_pct", 0.0) or 0.0),
            }
        )

    if not rows:
        return {}

    sector_rows: dict[str, list[dict]] = {}
    for row in rows:
        sector_rows.setdefault(row["sector"], []).append(row)

    scored: list[tuple[str, float, SectorRotation]] = []
    for sector, group in sector_rows.items():
        count = len(group)
        sector_return = sum(row["return_pct"] for row in group) / count
        relative = sector_return - btc_return
        breadth = sum(1 for row in group if row["return_pct"] > 0.0) / count
        volume_accel = sum(row["volume_accel"] for row in group) / count
        oi_change = sum(row["oi_change_pct"] for row in group) / count
        state = _state_for_sector(relative, breadth, volume_accel, oi_change)
        conf = _confidence(count, relative, breadth, volume_accel)
        score = relative + (breadth - 0.5) * 2.0 + min(2.0, max(-1.0, volume_accel - 1.0)) + oi_change * 0.05
        rotation = SectorRotation(
            sector=sector,
            state=state,
            rank=0,
            sector_return_pct=round(sector_return, 3),
            relative_btc_pct=round(relative, 3),
            breadth=round(breadth, 3),
            volume_accel=round(volume_accel, 3),
            oi_change_pct=round(oi_change, 3),
            confidence=conf,
            reason=(
                f"sector={sector} state={state} rel_btc={relative:.2f}% "
                f"breadth={breadth:.2f} vol={volume_accel:.2f} oi={oi_change:.2f}%"
            ),
        )
        scored.append((sector, score, rotation))

    scored.sort(key=lambda item: item[1], reverse=True)
    out: dict[str, SectorRotation] = {}
    for idx, (sector, _score, rotation) in enumerate(scored, start=1):
        out[sector] = SectorRotation(**{**rotation.as_dict(), "rank": idx})
    return out


def attach_sector_rotation(snapshots: dict[str, object], rotations: dict[str, SectorRotation]) -> None:
    for symbol, snap in (snapshots or {}).items():
        sector = sector_for_asset(asset_from_symbol(symbol))
        setattr(snap, "sector_rotation", rotations.get(sector, unavailable_sector_rotation(sector)))


def sector_rotation_score_mult(rotation: SectorRotation | None, direction: str) -> tuple[float, str]:
    if rotation is None or rotation.confidence <= 0.0 or rotation.state in {"unknown", "neutral"}:
        return 1.0, "sector_rotation_unavailable"

    direction = str(direction or "").lower()
    state = rotation.state
    mult = 1.0
    if direction == "long":
        if state == "rotating_in":
            mult = 1.07
        elif state == "firming":
            mult = 1.03
        elif state == "rotating_out":
            mult = 0.91
        elif state == "weakening":
            mult = 0.96
    elif direction == "short":
        if state == "rotating_out":
            mult = 1.06
        elif state == "weakening":
            mult = 1.03
        elif state == "rotating_in":
            mult = 0.93
        elif state == "firming":
            mult = 0.97

    blended = 1.0 + (mult - 1.0) * max(0.0, min(1.0, rotation.confidence))
    return round(max(0.90, min(1.08, blended)), 4), rotation.reason
