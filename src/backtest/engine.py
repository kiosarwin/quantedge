"""
Walk-forward backtesting engine.

For each bar on the primary timeframe:
  1. Build a DataFrame slice up to (not including) the current bar
  2. Score the symbol
  3. Enter trades when score >= threshold and no position is open
  4. Check exits (SL, TP1, trailing, TP2, max hold) at close of each bar
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.analysis.indicators import atr, trend_strength_score, volatility_score, volume_ratio, buy_volume_ratio, is_extreme_volatility
from src.analysis.structure import structure_quality_score, trade_direction_from_structure
from src.analysis.sentiment import funding_sentiment_score, open_interest_score
from src.scoring.scorer import Scorer, SignalBreakdown
from src.risk.risk_manager import RiskManager, TradeSetup

log = logging.getLogger(__name__)

# Warm-up bars needed before we have enough indicator data
_MIN_BARS = 210


@dataclass
class BacktestTrade:
    symbol: str
    direction: str
    entry_bar: int
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    size_contracts: float
    size_usd: float
    r_distance: float
    exit_bar: int = 0
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    tp1_hit: bool = False
    trailing_stop: float | None = None
    remaining_contracts: float = 0.0
    entry_score: float = 0.0
    mfe_r: float = 0.0   # max favorable excursion in R multiples
    mae_r: float = 0.0   # max adverse excursion in R multiples

    def current_pnl(self, price: float) -> float:
        mult = 1 if self.direction == "long" else -1
        return mult * (price - self.entry_price) * self.remaining_contracts

    def is_sl_hit(self, low: float, high: float) -> bool:
        if self.direction == "long":
            return low <= self.stop_loss
        return high >= self.stop_loss

    def is_tp1_hit(self, low: float, high: float) -> bool:
        if self.direction == "long":
            return high >= self.tp1
        return low <= self.tp1

    def is_tp2_hit(self, low: float, high: float) -> bool:
        if self.direction == "long":
            return high >= self.tp2
        return low <= self.tp2


@dataclass
class BacktestResult:
    symbol: str
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    initial_capital: float = 10000.0

    @property
    def closed_trades(self) -> list[BacktestTrade]:
        return [t for t in self.trades if t.exit_bar > 0]

    @property
    def wins(self) -> list[BacktestTrade]:
        return [t for t in self.closed_trades if t.pnl_usd > 0]

    @property
    def losses(self) -> list[BacktestTrade]:
        return [t for t in self.closed_trades if t.pnl_usd <= 0]

    @property
    def win_rate(self) -> float:
        n = len(self.closed_trades)
        return len(self.wins) / n * 100 if n else 0.0

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl_usd for t in self.closed_trades)

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.pnl_usd for t in self.wins)
        gross_loss = abs(sum(t.pnl_usd for t in self.losses))
        return gross_win / gross_loss if gross_loss > 0 else float("inf")

    @property
    def avg_win(self) -> float:
        return sum(t.pnl_usd for t in self.wins) / len(self.wins) if self.wins else 0.0

    @property
    def avg_loss(self) -> float:
        return sum(t.pnl_usd for t in self.losses) / len(self.losses) if self.losses else 0.0

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        peak = self.initial_capital
        max_dd = 0.0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1] if self.equity_curve else self.initial_capital

    @property
    def return_pct(self) -> float:
        return (self.final_equity - self.initial_capital) / self.initial_capital * 100


class BacktestEngine:
    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._bt_cfg = cfg["backtest"]
        self._exit_cfg = cfg["exit"]
        self._scorer = Scorer(cfg)
        self._risk = RiskManager(cfg)
        self._commission_pct = self._bt_cfg["commission_pct"] / 100
        self._slippage_pct = self._bt_cfg["slippage_pct"] / 100
        self._threshold = cfg["backtest"].get("score_threshold", cfg["trading"]["min_score_threshold"])
        self._safety = cfg.get("safety", {})

    def run(
        self,
        symbol: str,
        candles: dict[str, pd.DataFrame],
        initial_capital: float | None = None,
    ) -> BacktestResult:
        capital = initial_capital or self._bt_cfg["initial_capital"]
        self._risk.update_equity(capital)
        self._risk.state.peak_equity = capital
        self._risk.state.daily_start_equity = capital
        self._risk.state.open_trade_count = 0
        self._risk.state.consecutive_losses = 0
        self._risk.state.open_risk_pct = 0.0

        tf_cfg = self._cfg["timeframes"]
        primary_tf = tf_cfg["primary"]
        higher_tf = tf_cfg["higher"]

        df = candles.get(primary_tf)
        df_high = candles.get(higher_tf, pd.DataFrame())

        if df is None or df.empty:
            log.warning("[%s] No primary candles", symbol)
            return BacktestResult(symbol=symbol, initial_capital=capital)

        result = BacktestResult(symbol=symbol, initial_capital=capital)
        result.equity_curve.append(capital)

        open_trade: BacktestTrade | None = None
        equity = capital

        for i in range(_MIN_BARS, len(df)):
            bar = df.iloc[i]
            bar_open = float(bar["open"])
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])

            # Slice up to (not including) this bar — no lookahead
            df_slice = df.iloc[:i]
            bar_ts = df.index[i]
            df_high_slice = df_high.loc[df_high.index < bar_ts] if not df_high.empty else df_high

            # ── Manage open trade ──────────────────────────────────────────
            if open_trade is not None:
                open_trade, closed = self._check_exits(open_trade, bar_high, bar_low, bar_close, i)
                if closed:
                    commission = abs(open_trade.pnl_usd) * self._commission_pct * 2
                    open_trade.pnl_usd -= commission
                    equity += open_trade.pnl_usd
                    self._risk.on_trade_closed(open_trade.pnl_usd)
                    result.trades.append(open_trade)
                    result.equity_curve.append(equity)
                    open_trade = None
                    continue

            # ── Look for new entry ─────────────────────────────────────────
            if open_trade is None and self._risk.can_open_trade()[0]:
                bd = self._score_bar(symbol, df_slice, df_high_slice)
                if bd and bd.total_score >= self._threshold:
                    # Apply slippage to entry
                    slip = bar_close * self._slippage_pct
                    entry_price = bar_close + slip if bd.direction == "long" else bar_close - slip

                    self._risk.update_equity(equity)
                    setup = self._risk.calculate_setup(symbol, bd.direction, df_slice, entry_price)
                    if setup:
                        commission_entry = setup.size_usd * self._commission_pct
                        equity -= commission_entry

                        open_trade = BacktestTrade(
                            symbol=symbol,
                            direction=bd.direction,
                            entry_bar=i,
                            entry_price=entry_price,
                            stop_loss=setup.stop_loss,
                            tp1=setup.tp1,
                            tp2=setup.tp2,
                            size_contracts=setup.size_contracts,
                            size_usd=setup.size_usd,
                            r_distance=setup.r_distance,
                            remaining_contracts=setup.size_contracts,
                            entry_score=bd.total_score,
                        )
                        self._risk.on_trade_opened()

            result.equity_curve.append(equity)

        # Force-close any still-open trade at last bar
        if open_trade is not None:
            last = df.iloc[-1]
            last_close = float(last["close"])
            mult = 1 if open_trade.direction == "long" else -1
            open_trade.pnl_usd = mult * (last_close - open_trade.entry_price) * open_trade.remaining_contracts
            open_trade.pnl_usd -= open_trade.pnl_usd * self._commission_pct * 2
            open_trade.exit_price = last_close
            open_trade.exit_reason = "end_of_data"
            open_trade.exit_bar = len(df) - 1
            equity += open_trade.pnl_usd
            result.trades.append(open_trade)
            result.equity_curve.append(equity)

        result.equity_curve.append(equity)
        return result

    # ------------------------------------------------------------------ #
    #  Scoring on a historical slice                                       #
    # ------------------------------------------------------------------ #

    def _score_bar(
        self, symbol: str, df: pd.DataFrame, df_high: pd.DataFrame
    ) -> SignalBreakdown | None:
        if len(df) < _MIN_BARS:
            return None

        # Order book and live signals not available in backtest → neutral (50)
        class _FakeSnap:
            pass

        direction = trade_direction_from_structure(df, self._cfg)
        if direction == "none":
            if not df_high.empty and len(df_high) >= 50:
                direction = trade_direction_from_structure(df_high, self._cfg)
        if direction == "none":
            return None

        try:
            ts = trend_strength_score(df, self._cfg)
        except Exception:
            ts = 0.0

        try:
            ind = self._cfg["indicators"]
            vol_r = volume_ratio(df, ind["volume_lookback"])
            buy_r = buy_volume_ratio(df, ind["volume_lookback"])
            vol_score = min(100.0, vol_r / ind["volume_spike_multiplier"] * 50)
            if direction == "long":
                vol_score = vol_score * 0.5 + buy_r * 100 * 0.5
            else:
                vol_score = vol_score * 0.5 + (1 - buy_r) * 100 * 0.5
        except Exception:
            vol_score = 0.0

        try:
            sq = structure_quality_score(df, self._cfg)
        except Exception:
            sq = 0.0

        oi_score = 50.0      # no historical OI
        fs_score = 50.0      # no historical funding
        ob_score = 50.0      # no historical order book

        try:
            vs = volatility_score(df, self._cfg)
        except Exception:
            vs = 50.0

        w = self._scorer._weights
        total_weight = sum(w.values()) or 1.0
        raw_score = (
            ts       * w.get("trend_strength", 20) +
            vol_score * w.get("volume_confirmation", 15) +
            sq       * w.get("structure_quality", 20) +
            oi_score * w.get("open_interest", 15) +
            fs_score * w.get("funding_sentiment", 10) +
            ob_score * w.get("order_book", 10) +
            vs       * w.get("volatility", 10)
        ) / total_weight

        final_score = round(raw_score, 2)
        return SignalBreakdown(
            symbol=symbol,
            direction=direction,
            total_score=final_score,
            base_score=final_score,
            trend_strength=ts,
            volume_confirmation=vol_score,
            structure_quality=sq,
            open_interest=oi_score,
            funding_sentiment=fs_score,
            order_book=ob_score,
            volatility=vs,
            weights_used=dict(w),
        )

    # ------------------------------------------------------------------ #
    #  Exit logic (bar-level simulation)                                   #
    # ------------------------------------------------------------------ #

    def _check_exits(
        self,
        trade: BacktestTrade,
        high: float,
        low: float,
        close: float,
        bar_idx: int,
    ) -> tuple[BacktestTrade, bool]:
        exit_cfg = self._exit_cfg
        trail_pct = exit_cfg.get("tp2_trailing_stop_pct", 0.15)
        max_hold = exit_cfg.get("max_hold_duration_s", 3600)

        # Update MFE/MAE in R multiples before exit logic
        if trade.r_distance > 0:
            if trade.direction == "long":
                fav = high - trade.entry_price
                adv = trade.entry_price - low
            else:
                fav = trade.entry_price - low
                adv = high - trade.entry_price
            trade.mfe_r = max(trade.mfe_r, fav / trade.r_distance)
            trade.mae_r = max(trade.mae_r, adv / trade.r_distance)

        # TP1 hit first on this bar
        if not trade.tp1_hit and trade.is_tp1_hit(low, high):
            tp1_pnl = abs(trade.tp1 - trade.entry_price) * (trade.size_contracts * exit_cfg["tp1_size_pct"])
            trade.pnl_usd = tp1_pnl if trade.direction == "long" else tp1_pnl
            trade.remaining_contracts *= (1 - exit_cfg["tp1_size_pct"])
            trade.stop_loss = trade.entry_price   # move to breakeven
            trade.tp1_hit = True
            if trade.direction == "long":
                trade.trailing_stop = trade.tp1 * (1 - trail_pct)
            else:
                trade.trailing_stop = trade.tp1 * (1 + trail_pct)

        # Update trailing stop
        if trade.tp1_hit and trade.trailing_stop is not None:
            if trade.direction == "long":
                new_trail = close * (1 - trail_pct)
                if new_trail > trade.trailing_stop:
                    trade.trailing_stop = new_trail
            else:
                new_trail = close * (1 + trail_pct)
                if new_trail < trade.trailing_stop:
                    trade.trailing_stop = new_trail

        # SL hit
        if trade.is_sl_hit(low, high):
            mult = 1 if trade.direction == "long" else -1
            close_pnl = mult * (trade.stop_loss - trade.entry_price) * trade.remaining_contracts
            trade.pnl_usd += close_pnl
            trade.exit_price = trade.stop_loss
            trade.exit_reason = "stop_loss"
            trade.exit_bar = bar_idx
            return trade, True

        # Trailing stop hit
        if trade.trailing_stop is not None:
            ts_hit = (trade.direction == "long" and low <= trade.trailing_stop) or \
                     (trade.direction == "short" and high >= trade.trailing_stop)
            if ts_hit:
                mult = 1 if trade.direction == "long" else -1
                close_pnl = mult * (trade.trailing_stop - trade.entry_price) * trade.remaining_contracts
                trade.pnl_usd += close_pnl
                trade.exit_price = trade.trailing_stop
                trade.exit_reason = "trailing_stop"
                trade.exit_bar = bar_idx
                return trade, True

        # TP2 hit
        if trade.is_tp2_hit(low, high):
            mult = 1 if trade.direction == "long" else -1
            close_pnl = mult * (trade.tp2 - trade.entry_price) * trade.remaining_contracts
            trade.pnl_usd += close_pnl
            trade.exit_price = trade.tp2
            trade.exit_reason = "tp2"
            trade.exit_bar = bar_idx
            return trade, True

        return trade, False
