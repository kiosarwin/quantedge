"""
Portfolio-level volatility targeting.

This is the FOUNDATION of risk parity portfolios used by Bridgewater
(All Weather), AQR, Man Group AHL, and most systematic CTA funds.

The principle is simple but powerful:
  Target a fixed annualized volatility (e.g., 40% for crypto futures).
  When realized vol < target → portfolio is underutilized → SIZE UP.
  When realized vol > target → portfolio is overheated → SIZE DOWN.

This automatically:
  - Prevents tail risk during volatile periods
  - Captures full edge during quiet periods
  - Stabilizes the equity curve (Sharpe-optimal scaling)
  - Eliminates the need to manually tune position size with market state

Reference: Moreira & Muir (2017) "Volatility-Managed Portfolios", J. Finance
           Hocquard et al. (2013) "A Constant-Volatility Framework"
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from pathlib import Path

log = logging.getLogger(__name__)


class VolTargeter:
    """
    Targets a fixed annualized portfolio volatility by scaling position sizes
    inversely to realized vol over recent history.
    """

    STATE_PATH = Path("models/vol_targeter.json")

    def __init__(
        self,
        target_annual_vol: float = 0.40,
        lookback: int = 30,
        min_mult: float = 0.50,
        max_mult: float = 1.50,
    ):
        """
        Args:
            target_annual_vol: 0.40 = 40% annual vol (aggressive for crypto, sustainable)
            lookback: number of daily returns for realized vol estimate
            min_mult / max_mult: hard bounds on the multiplier
        """
        self._target = float(target_annual_vol)
        self._lookback = int(lookback)
        self._min_mult = float(min_mult)
        self._max_mult = float(max_mult)
        self._returns: deque = deque(maxlen=lookback)
        self._last_record_ts: float = 0.0
        self._load()

    def record_return(self, daily_pnl_pct: float, force: bool = False) -> None:
        """
        Record a daily return. Pass daily_pnl_pct as percentage (e.g., 1.5 for +1.5%).
        Throttled to one record per ~20h unless forced.
        """
        now = time.time()
        if not force and now - self._last_record_ts < 20 * 3600:
            return
        self._returns.append(float(daily_pnl_pct) / 100.0)
        self._last_record_ts = now
        self._save()

    def realized_vol(self) -> float:
        """Annualized realized vol = std(daily_returns) × sqrt(365)."""
        if len(self._returns) < 5:
            return self._target  # no data yet — assume on-target
        rets = list(self._returns)
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / max(1, len(rets) - 1)
        daily_vol = math.sqrt(max(var, 1e-12))
        return daily_vol * math.sqrt(365)

    def size_multiplier(self) -> float:
        """
        Returns the size multiplier to apply to all positions.
          target / realized > 1: market is calm → size up
          target / realized < 1: market is hot → size down
        Bounded by [min_mult, max_mult].
        """
        realized = self.realized_vol()
        if realized < 1e-6:
            return self._max_mult
        scale = self._target / realized
        return max(self._min_mult, min(self._max_mult, scale))

    def report(self) -> dict:
        return {
            "target_annual_vol": round(self._target, 4),
            "realized_annual_vol": round(self.realized_vol(), 4),
            "size_multiplier": round(self.size_multiplier(), 3),
            "n_observations": len(self._returns),
            "regime": (
                "calm" if self.size_multiplier() > 1.05
                else "elevated" if self.size_multiplier() < 0.95
                else "normal"
            ),
        }

    def _save(self) -> None:
        try:
            self.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "returns": list(self._returns),
                "last_record_ts": self._last_record_ts,
                "target": self._target,
            }
            self.STATE_PATH.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            log.warning("VolTargeter save failed: %s", exc)

    def _load(self) -> None:
        if not self.STATE_PATH.exists():
            return
        try:
            data = json.loads(self.STATE_PATH.read_text())
            for r in data.get("returns", []):
                self._returns.append(float(r))
            self._last_record_ts = float(data.get("last_record_ts", 0.0))
            log.info("VolTargeter loaded: %d returns, realized=%.3f%%, mult=%.2fx",
                     len(self._returns), self.realized_vol() * 100, self.size_multiplier())
        except Exception as exc:
            log.warning("VolTargeter load failed: %s", exc)
