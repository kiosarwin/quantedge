"""
Spot Risk Manager

Position sizing for Spot (no leverage):
  - Allocates 10–25% of equity per trade
  - Stop-loss based on ATR × multiplier (below structure)
  - Take-profit at +10%, +18%, +25% (partial exits)
  - Enforces daily loss cap and max drawdown guard
  - No shorting — all positions are LONG (buy base, hold, sell)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import pandas as pd

from src.analysis.indicators import atr

log = logging.getLogger(__name__)


@dataclass
class SpotTradeSetup:
    symbol: str
    direction: str = "long"         # Spot only = long
    entry_price: float = 0.0
    stop_loss: float = 0.0          # Price to exit if wrong
    tp1: float = 0.0                # +10%
    tp2: float = 0.0                # +18%
    tp3: float = 0.0                # +25%
    size_usdt: float = 0.0          # USDT amount to spend (BUY)
    base_qty: float = 0.0           # Base currency amount (e.g. ETH)
    atr_value: float = 0.0
    risk_reward: float = 0.0
    entry_strategy: str = "none"


@dataclass
class SpotPortfolioState:
    equity: float = 0.0
    peak_equity: float = 0.0
    daily_start_equity: float = 0.0
    day_start_ts: float = field(default_factory=time.time)
    consecutive_losses: int = 0
    open_trade_count: int = 0

    def reset_day(self, equity: float) -> None:
        self.daily_start_equity = equity
        self.day_start_ts = time.time()

    @property
    def daily_pnl_pct(self) -> float:
        if not self.daily_start_equity:
            return 0.0
        return (self.equity - self.daily_start_equity) / self.daily_start_equity * 100

    @property
    def drawdown_pct(self) -> float:
        if not self.peak_equity:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity * 100


class SpotRiskManager:
    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._risk = cfg["risk"]
        self._capital = cfg.get("capital", {})
        self._exit = cfg["exit"]
        self.state = SpotPortfolioState()

    # ------------------------------------------------------------------ #
    #  Guards                                                             #
    # ------------------------------------------------------------------ #

    def can_open_trade(self) -> tuple[bool, str]:
        """Return (allowed, reason)."""
        s = self.state
        max_open = self._cfg["trading"]["max_open_trades"]
        if s.open_trade_count >= max_open:
            return False, f"max_open_trades ({max_open}) reached"

        if s.daily_pnl_pct < -self._risk["daily_loss_cap_pct"]:
            return False, f"daily loss cap hit ({s.daily_pnl_pct:.1f}%)"

        if s.drawdown_pct >= self._risk["max_drawdown_pct"]:
            return False, f"max drawdown hit ({s.drawdown_pct:.1f}%)"

        max_consec = self._risk.get("max_consecutive_losses", 4)
        if s.consecutive_losses >= max_consec:
            return False, f"consecutive losses limit ({s.consecutive_losses})"

        return True, "ok"

    # ------------------------------------------------------------------ #
    #  Setup calculation                                                  #
    # ------------------------------------------------------------------ #

    def calculate_setup(
        self,
        symbol: str,
        df: pd.DataFrame,
        entry_price: float,
        strategy_sl: float | None = None,
        entry_strategy: str = "none",
    ) -> SpotTradeSetup | None:
        """
        Calculate a complete trade setup for a Spot long entry.

        Args:
            symbol:      Trading pair
            df:          OHLCV DataFrame for ATR calculation
            entry_price: Expected fill price
            strategy_sl: Stop-loss suggested by the entry strategy (optional)
                         If None, uses ATR-based SL
            entry_strategy: Name of the entry strategy used

        Returns:
            SpotTradeSetup or None if setup is invalid
        """
        ind_cfg = self._cfg["indicators"]
        current_atr = atr(df, ind_cfg.get("atr_period", 14))

        # ── Stop-loss ─────────────────────────────────────────────────
        if strategy_sl is not None and strategy_sl > 0:
            sl = strategy_sl
        else:
            atr_mult = ind_cfg.get("atr_sl_multiplier", 2.0)
            sl = entry_price - current_atr * atr_mult

        if sl <= 0 or sl >= entry_price:
            log.debug("%s invalid SL: %.6f >= entry %.6f", symbol, sl, entry_price)
            return None

        # ── Take-profit levels ────────────────────────────────────────
        tp1 = entry_price * (1 + self._exit.get("tp1_pct", 0.10))
        tp2 = entry_price * (1 + self._exit.get("tp2_pct", 0.18))
        tp3 = entry_price * (1 + self._exit.get("tp3_pct", 0.25))

        # ── R:R check ─────────────────────────────────────────────────
        risk_per_unit = entry_price - sl
        reward_tp2 = tp2 - entry_price
        rr = reward_tp2 / risk_per_unit if risk_per_unit > 0 else 0.0

        min_rr = self._risk.get("min_rr_ratio", 2.0)
        if rr < min_rr:
            log.debug(
                "%s R:R too low: %.2f (need %.1f)  entry=%.6f sl=%.6f tp2=%.6f",
                symbol, rr, min_rr, entry_price, sl, tp2,
            )
            return None

        # ── Position sizing (% of equity) ────────────────────────────
        equity = self.state.equity
        if equity <= 0:
            log.warning("Equity is 0 — cannot size trade for %s", symbol)
            return None

        min_pct = self._capital.get("min_position_pct", 10) / 100
        max_pct = self._capital.get("max_position_pct", 25) / 100

        # Target: allocate based on signal strength (use midpoint for now)
        # Adaptive sizing: scale between min and max based on R:R quality
        rr_factor = min(1.0, (rr - min_rr) / (min_rr * 2))
        position_pct = min_pct + (max_pct - min_pct) * rr_factor
        size_usdt = equity * position_pct

        # Enforce min notional ($10 to be safe)
        if size_usdt < 10.0:
            log.debug("%s position too small: $%.2f", symbol, size_usdt)
            return None

        base_qty = size_usdt / entry_price

        return SpotTradeSetup(
            symbol=symbol,
            direction="long",
            entry_price=entry_price,
            stop_loss=round(sl, 8),
            tp1=round(tp1, 8),
            tp2=round(tp2, 8),
            tp3=round(tp3, 8),
            size_usdt=round(size_usdt, 2),
            base_qty=round(base_qty, 6),
            atr_value=round(current_atr, 8),
            risk_reward=round(rr, 2),
            entry_strategy=entry_strategy,
        )

    # ------------------------------------------------------------------ #
    #  State updates                                                      #
    # ------------------------------------------------------------------ #

    def update_equity(self, equity: float) -> None:
        self.state.equity = equity
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity
        # Day reset
        if time.time() - self.state.day_start_ts > 86400:
            self.state.reset_day(equity)

    def on_trade_opened(self) -> None:
        self.state.open_trade_count += 1

    def on_trade_closed(self, pnl_pct: float) -> None:
        """pnl_pct: positive = win, negative = loss."""
        self.state.open_trade_count = max(0, self.state.open_trade_count - 1)
        if pnl_pct < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0
