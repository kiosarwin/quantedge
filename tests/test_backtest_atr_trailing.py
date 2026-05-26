"""
Regression tests for the backtest/shadow ATR-based trailing stop (B12).

Live's TradeManager sets the post-TP1 trail distance from the entry-time
ATR snapshot:

    trail_dist = trade.setup.atr * trade.setup.trailing_atr_multiplier
    new_trailing_stop = current_price - trail_dist  (long; '+' for short)

Before this fix, BacktestTrade had no ATR field and the engine fell back to
``tp2_trailing_stop_pct`` (default 0.15 -> a 15%-of-price trail), which is
wildly wider than live's typical 1.5x ATR distance. As a result, backtest
P&L diverged from live by giving runners much more room to bleed before
the trail closed them.

The fix:

  * BacktestTrade.atr captures setup.atr at entry.
  * _check_exits uses ``atr * trailing_atr_multiplier`` when the snapshot
    is non-zero, falling back to %-based trail only for legacy / zero ATR.

These tests freeze that behaviour at the unit level so a future refactor
can't quietly drop ATR plumbing again.
"""
from __future__ import annotations

import pandas as pd
from types import SimpleNamespace

from src.backtest.engine import BacktestEngine, BacktestTrade


def _engine_with_trail_pct(trail_pct: float = 0.15) -> BacktestEngine:
    """Bare-bones engine — only the bits ``_check_exits`` actually reads."""
    cfg = {
        "backtest": {"commission_pct": 0.04, "slippage_pct": 0.02, "initial_capital": 10000.0},
        "exit": {"tp2_trailing_stop_pct": trail_pct, "tp3_r_multiple": 3.0},
        "trading": {"min_score_threshold": 60.0},
        "risk": {},
        "indicators": {},
        "scoring": {},
    }
    eng = BacktestEngine.__new__(BacktestEngine)
    eng._cfg = cfg
    eng._bt_cfg = cfg["backtest"]
    eng._exit_cfg = cfg["exit"]
    eng._commission_pct = cfg["backtest"]["commission_pct"] / 100
    eng._slippage_pct = cfg["backtest"]["slippage_pct"] / 100
    eng._threshold = cfg["trading"]["min_score_threshold"]
    eng._safety = {}
    return eng


def _make_trade(*, atr: float, direction: str = "long") -> BacktestTrade:
    """Build a minimal trade with TP1 already poised but not yet hit."""
    return BacktestTrade(
        symbol="BTC/USDT:USDT",
        direction=direction,
        exit_profile="default",
        entry_bar=0,
        entry_price=100.0,
        stop_loss=95.0 if direction == "long" else 105.0,
        tp1=110.0 if direction == "long" else 90.0,
        tp2=120.0 if direction == "long" else 80.0,
        size_contracts=1.0,
        size_usd=100.0,
        r_distance=5.0,
        tp1_size_pct=0.50,
        tp2_size_pct=0.30,
        trailing_atr_multiplier=1.5,
        max_hold_duration_s=86400,
        atr=atr,
        remaining_contracts=1.0,
    )


def test_atr_trailing_uses_atr_when_available_long():
    """ATR=2.0, trail_mult=1.5 -> trail_dist = 3.0.  TP1=110 -> initial
    trail at 110 - 3 = 107, NOT 110 * (1 - 0.15) = 93.5 (legacy %-based)."""
    eng = _engine_with_trail_pct(0.15)
    trade = _make_trade(atr=2.0, direction="long")

    # Bar that hits TP1 exactly (high=tp1, close=tp1, no SL hit).
    eng._check_exits(trade, high=110.0, low=100.0, close=110.0, bar_idx=1)

    assert trade.tp1_hit is True
    assert trade.trailing_stop is not None
    # ATR mode: 110 - (2.0 * 1.5) = 107.0
    assert abs(trade.trailing_stop - 107.0) < 1e-9


def test_atr_trailing_falls_back_to_pct_when_atr_missing_long():
    """No ATR snapshot (legacy state) -> fall back to %-based trail."""
    eng = _engine_with_trail_pct(0.15)
    trade = _make_trade(atr=0.0, direction="long")

    eng._check_exits(trade, high=110.0, low=100.0, close=110.0, bar_idx=1)

    assert trade.tp1_hit is True
    assert trade.trailing_stop is not None
    # Pct mode: 110 * (1 - 0.15) = 93.5
    assert abs(trade.trailing_stop - 93.5) < 1e-9


def test_atr_trailing_ratchets_up_with_price_long():
    eng = _engine_with_trail_pct(0.15)
    trade = _make_trade(atr=2.0, direction="long")

    # Bar 1: TP1 just hit at 110 -> trail = 107.
    eng._check_exits(trade, high=110.0, low=100.0, close=110.0, bar_idx=1)
    assert abs(trade.trailing_stop - 107.0) < 1e-9

    # Bar 2: price closes at 115 -> trail ratchets to 115 - 3 = 112.
    eng._check_exits(trade, high=115.0, low=109.0, close=115.0, bar_idx=2)
    assert abs(trade.trailing_stop - 112.0) < 1e-9

    # Bar 3: price retraces close=112 -> trail would be 109, but ratchet
    # only widens never tightens, so trail stays at 112.
    eng._check_exits(trade, high=113.0, low=111.0, close=112.0, bar_idx=3)
    assert abs(trade.trailing_stop - 112.0) < 1e-9


def test_atr_trailing_ratchets_down_for_shorts():
    """Short side: trail = entry + atr*mult, ratchets DOWN as price falls."""
    eng = _engine_with_trail_pct(0.15)
    trade = _make_trade(atr=2.0, direction="short")

    # Bar 1: TP1 (90) hit on a dip; close also at 90 -> trail = 90 + 3 = 93.
    eng._check_exits(trade, high=100.0, low=90.0, close=90.0, bar_idx=1)
    assert trade.tp1_hit is True
    assert abs(trade.trailing_stop - 93.0) < 1e-9

    # Bar 2: price closes lower at 85 -> trail ratchets DOWN to 85 + 3 = 88.
    eng._check_exits(trade, high=91.0, low=84.0, close=85.0, bar_idx=2)
    assert abs(trade.trailing_stop - 88.0) < 1e-9

    # Bar 3: price rebounds to 88 -> trail stays at 88 (ratchet only).
    eng._check_exits(trade, high=89.0, low=86.0, close=88.0, bar_idx=3)
    assert abs(trade.trailing_stop - 88.0) < 1e-9


def test_atr_trailing_stop_hit_closes_trade():
    eng = _engine_with_trail_pct(0.15)
    trade = _make_trade(atr=2.0, direction="long")

    # Bar 1: TP1 hit -> trail at 107.
    eng._check_exits(trade, high=110.0, low=100.0, close=110.0, bar_idx=1)
    assert trade.tp1_hit is True

    # Bar 2: price collapses, low goes through 107 -> trailing close at 107.
    _, closed = eng._check_exits(trade, high=109.0, low=106.0, close=108.0, bar_idx=2)
    assert closed is True
    assert trade.exit_reason == "trailing_stop"
    assert abs(trade.exit_price - 107.0) < 1e-9


def test_backtest_score_bar_uses_runtime_scorer_snapshot_with_lower_tf():
    seen = {}

    class FakeScorer:
        def score_many(self, snapshots):
            seen.update(snapshots)
            return [SimpleNamespace(symbol="BTC/USDT:USDT", total_score=72.0)]

    eng = BacktestEngine.__new__(BacktestEngine)
    eng._cfg = {"timeframes": {"primary": "1h", "higher": "4h", "lower": "15m", "entry": "5m"}}
    eng._scorer = FakeScorer()

    idx = pd.date_range("2026-01-01", periods=220, freq="h", tz="UTC")
    primary = pd.DataFrame({"open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 100}, index=idx)
    higher = primary.iloc[::4].copy()
    lower_idx = pd.date_range("2026-01-01", periods=880, freq="15min", tz="UTC")
    lower = pd.DataFrame({"open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 100}, index=lower_idx)

    bd = eng._score_bar("BTC/USDT:USDT", primary, higher, lower)

    assert bd.total_score == 72.0
    snap = seen["BTC/USDT:USDT"]
    assert set(snap.candles) == {"1h", "4h", "15m"}
    assert snap.last_price == 1.5
    assert snap.ls_ratio == 1.0
    assert snap.taker_buy_ratio == 0.5
