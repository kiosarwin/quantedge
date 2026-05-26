from types import SimpleNamespace

import pandas as pd

from src.data.market_data import MarketSnapshot
from src.models.sector_rotation import (
    attach_sector_rotation,
    compute_sector_rotation,
    sector_rotation_score_mult,
)


def _snap(symbol, closes, volumes=None, oi=0.0):
    if volumes is None:
        volumes = [100.0 for _ in closes]
    return MarketSnapshot(
        symbol=symbol,
        candles={"1h": pd.DataFrame({"close": closes, "volume": volumes})},
        oi_change_pct=oi,
    )


def test_compute_sector_rotation_detects_rotating_in_sector_vs_btc():
    snapshots = {
        "BTC/USDT:USDT": _snap("BTC/USDT:USDT", [100, 101]),
        "SOL/USDT:USDT": _snap("SOL/USDT:USDT", [9.8, 10.0, 11.0], [100, 100, 170], oi=1.0),
        "SUI/USDT:USDT": _snap("SUI/USDT:USDT", [4.9, 5.0, 5.45], [100, 100, 160], oi=0.5),
        "DOGE/USDT:USDT": _snap("DOGE/USDT:USDT", [1, 0.98]),
    }

    rotations = compute_sector_rotation(snapshots, lookback_bars=2)

    assert rotations["l1"].state == "rotating_in"
    assert rotations["l1"].relative_btc_pct > 7.0
    assert rotations["l1"].breadth == 1.0
    assert rotations["l1"].rank == 1


def test_attach_sector_rotation_sets_snapshot_context():
    snapshots = {
        "BTC/USDT:USDT": _snap("BTC/USDT:USDT", [100, 101]),
        "SOL/USDT:USDT": _snap("SOL/USDT:USDT", [10, 11]),
    }
    rotations = compute_sector_rotation(snapshots, lookback_bars=2)

    attach_sector_rotation(snapshots, rotations)

    assert snapshots["SOL/USDT:USDT"].sector_rotation.sector == "l1"
    assert snapshots["SOL/USDT:USDT"].sector_rotation.state in {"rotating_in", "firming"}


def test_sector_rotation_multiplier_is_direction_aware():
    rotation = SimpleNamespace(state="rotating_in", confidence=0.80, reason="l1 rotating in")

    long_mult, long_reason = sector_rotation_score_mult(rotation, "long")
    short_mult, short_reason = sector_rotation_score_mult(rotation, "short")

    assert long_mult > 1.0
    assert short_mult < 1.0
    assert "rotating" in long_reason
    assert "rotating" in short_reason
