"""
BTC Safety Gate — the most critical filter in any altcoin spot strategy.

When BTC is dumping, altcoins almost always follow.  This module:
  1. Monitors BTC's 4H candle return vs a dump threshold (-3% default)
  2. Checks if BTC price is above/below its EMA50 (trend bias)
  3. Raises a flag that halts all new ALTCOIN entries

State machine:
  SAFE   → new entries permitted
  HALTED → BTC danger detected; no new entries until SAFE again

Recovery: BTC must reclaim the EMA50 AND bounce +1.5% from its low
before the gate re-opens.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from src.analysis.indicators import ema

log = logging.getLogger(__name__)


class BTCState(str, Enum):
    SAFE = "safe"
    HALTED = "halted"


@dataclass
class BTCStatus:
    state: BTCState = BTCState.SAFE
    last_price: float = 0.0
    ema50: float = 0.0
    candle_change_pct: float = 0.0
    low_since_halt: float = float("inf")
    halted_at: float = 0.0
    reason: str = ""
    checked_at: float = field(default_factory=time.time)

    @property
    def is_safe(self) -> bool:
        return self.state == BTCState.SAFE

    def __str__(self) -> str:
        return (
            f"BTC [{self.state.value.upper()}]  "
            f"price={self.last_price:.2f}  ema50={self.ema50:.2f}  "
            f"4h_chg={self.candle_change_pct:+.2f}%  reason={self.reason}"
        )


class BTCGuard:
    """
    Evaluate current BTC market conditions and decide whether altcoin
    entries should be allowed.

    Usage:
        guard = BTCGuard(cfg)
        status = guard.evaluate(btc_df_4h)
        if not status.is_safe:
            skip_new_entries()
    """

    def __init__(self, cfg: dict):
        btc_cfg = cfg.get("btc", {})
        self._enabled: bool = btc_cfg.get("safety_enabled", True)
        self._dump_threshold: float = btc_cfg.get("dump_threshold_4h_pct", -3.0)
        self._recovery_threshold: float = btc_cfg.get("recovery_threshold_pct", 1.5)
        self._ema_period: int = btc_cfg.get("ema_period", 50)
        self._below_ema_halt: bool = btc_cfg.get("below_ema_halt", True)
        self._status = BTCStatus()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def evaluate(self, df: pd.DataFrame) -> BTCStatus:
        """
        Evaluate BTC conditions using the provided OHLCV DataFrame.

        Args:
            df: OHLCV DataFrame for BTC (recommended: 4H timeframe).

        Returns:
            Updated BTCStatus.
        """
        if not self._enabled or df.empty or len(df) < max(self._ema_period, 2):
            self._status.state = BTCState.SAFE
            self._status.reason = "guard_disabled_or_no_data"
            return self._status

        last_close = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2])
        candle_change_pct = (last_close - prev_close) / prev_close * 100

        ema50_series = ema(df["close"], self._ema_period)
        ema50 = float(ema50_series.iloc[-1])

        self._status.last_price = last_close
        self._status.ema50 = ema50
        self._status.candle_change_pct = candle_change_pct
        self._status.checked_at = time.time()

        # ── Detect danger ─────────────────────────────────────────────
        dumping = candle_change_pct <= self._dump_threshold
        below_ema = self._below_ema_halt and (last_close < ema50)

        if self._status.state == BTCState.SAFE:
            if dumping:
                self._halt(f"BTC dropped {candle_change_pct:.2f}% (threshold {self._dump_threshold:.1f}%)")
            elif below_ema:
                self._halt(f"BTC below EMA{self._ema_period} ({last_close:.2f} < {ema50:.2f})")

        # ── Evaluate recovery ─────────────────────────────────────────
        elif self._status.state == BTCState.HALTED:
            # Track lowest price since halt for recovery measurement
            if last_close < self._status.low_since_halt:
                self._status.low_since_halt = last_close

            recovery_pct = (last_close - self._status.low_since_halt) / self._status.low_since_halt * 100
            back_above_ema = last_close > ema50

            if back_above_ema and recovery_pct >= self._recovery_threshold:
                self._resume(
                    f"BTC recovered {recovery_pct:.2f}% from low, back above EMA{self._ema_period}"
                )
            else:
                log.debug(
                    "BTC still halted — recovery=%.2f%% (need %.1f%%) above_ema=%s",
                    recovery_pct,
                    self._recovery_threshold,
                    back_above_ema,
                )

        log.info("%s", self._status)
        return self._status

    @property
    def is_safe(self) -> bool:
        return self._status.is_safe

    @property
    def status(self) -> BTCStatus:
        return self._status

    # ------------------------------------------------------------------ #
    #  State transitions                                                  #
    # ------------------------------------------------------------------ #

    def _halt(self, reason: str) -> None:
        log.warning("BTC GUARD HALTED: %s", reason)
        self._status.state = BTCState.HALTED
        self._status.reason = reason
        self._status.halted_at = time.time()
        self._status.low_since_halt = self._status.last_price

    def _resume(self, reason: str) -> None:
        log.info("BTC GUARD RESUMED: %s", reason)
        self._status.state = BTCState.SAFE
        self._status.reason = reason
        self._status.low_since_halt = float("inf")
