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
from dataclasses import dataclass, field

import pandas as pd

from src.scoring.scorer import Scorer, SignalBreakdown
from src.risk.risk_manager import RiskManager
from src.data.market_data import MarketSnapshot

log = logging.getLogger(__name__)

# Warm-up bars needed before we have enough indicator data
_MIN_BARS = 210


@dataclass
class BacktestTrade:
    symbol: str
    direction: str
    exit_profile: str
    entry_bar: int
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    size_contracts: float
    size_usd: float
    r_distance: float
    tp1_size_pct: float
    tp2_size_pct: float
    trailing_atr_multiplier: float
    max_hold_duration_s: int
    # Snapshot of the symbol's ATR at trade-entry time. Used by the
    # backtest exit logic to mirror live's ATR-based trailing stop:
    # ``trail_dist = atr * trailing_atr_multiplier`` is constant for
    # the life of the trade, only the ratchet target moves with price.
    # Defaults to 0.0 so legacy serialized trades (engine state, shadow
    # state) load cleanly; the backtest then falls back to the legacy
    # %-based trail.
    atr: float = 0.0
    exit_bar: int = 0
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    tp1_hit: bool = False
    trailing_stop: float | None = None
    remaining_contracts: float = 0.0
    entry_score: float = 0.0
    strategy_sleeve: str = "neutral"
    setup_type: str = "unknown"
    mfe_r: float = 0.0   # max favorable excursion in R multiples
    mae_r: float = 0.0   # max adverse excursion in R multiples
    funding_cost: float = 0.0  # cumulative funding costs deducted
    _last_funding_bar: int = 0  # bar index of last funding charge

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
        self._funding_rate_per_8h = self._bt_cfg.get("funding_rate_per_8h", 0.0001)  # 0.01% per 8h default

    def _bars_per_8h(self) -> int:
        """Number of bars in 8 hours based on primary timeframe."""
        tf = self._cfg["timeframes"]["primary"]
        tf_minutes = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
                      "1h": 60, "2h": 120, "4h": 240, "1d": 1440}.get(tf, 60)
        return max(1, 480 // tf_minutes)  # 480 minutes = 8 hours

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
        lower_tf = tf_cfg.get("lower") or tf_cfg.get("secondary") or tf_cfg.get("entry")

        df = candles.get(primary_tf)
        df_high = candles.get(higher_tf, pd.DataFrame())
        df_lower = candles.get(lower_tf, pd.DataFrame()) if lower_tf else pd.DataFrame()

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
            df_lower_slice = df_lower.loc[df_lower.index < bar_ts] if not df_lower.empty else df_lower

            # ── Manage open trade ──────────────────────────────────────────
            if open_trade is not None:
                open_trade, closed = self._check_exits(open_trade, bar_high, bar_low, bar_close, i)
                if closed:
                    # Round-trip commission is on the entry+exit notional, not
                    # on PnL — historic implementation incorrectly scaled with
                    # |pnl| which double-counted on big winners and ignored
                    # break-even trades. Entry commission was already debited
                    # at trade open, so we only charge the exit leg here.
                    commission = open_trade.size_usd * self._commission_pct
                    open_trade.pnl_usd -= commission
                    equity += open_trade.pnl_usd
                    self._risk.on_trade_closed(open_trade.pnl_usd)
                    result.trades.append(open_trade)
                    result.equity_curve.append(equity)
                    open_trade = None
                    continue

            # H11: Deduct funding costs for positions held across 8h boundaries
            if open_trade is not None:
                bars_8h = self._bars_per_8h()
                if (i - open_trade._last_funding_bar) >= bars_8h:
                    cost = open_trade.size_usd * self._funding_rate_per_8h
                    open_trade.funding_cost += cost
                    equity -= cost
                    open_trade._last_funding_bar = i

            # ── Look for new entry ─────────────────────────────────────────
            if open_trade is None and self._risk.can_open_trade()[0]:
                bd = self._score_bar(symbol, df_slice, df_high_slice, df_lower_slice)
                if bd and bd.total_score >= self._threshold:
                    # Apply slippage to entry
                    slip = bar_close * self._slippage_pct
                    entry_price = bar_close + slip if bd.direction == "long" else bar_close - slip

                    self._risk.update_equity(equity)
                    setup = self._risk.calculate_setup(
                        symbol,
                        bd.direction,
                        df_slice,
                        entry_price,
                        strategy_sleeve=str(getattr(bd, "strategy_sleeve", "neutral") or "neutral"),
                        ev_result=getattr(bd, "ev_result", None),
                        setup_passport=(
                            bd.setup_passport.as_dict()
                            if getattr(bd, "setup_passport", None) is not None
                            and hasattr(bd.setup_passport, "as_dict")
                            else None
                        ),
                    )
                    if setup:
                        commission_entry = setup.size_usd * self._commission_pct
                        equity -= commission_entry

                        open_trade = BacktestTrade(
                            symbol=symbol,
                            direction=bd.direction,
                            exit_profile=setup.exit_profile,
                            entry_bar=i,
                            entry_price=entry_price,
                            stop_loss=setup.stop_loss,
                            tp1=setup.tp1,
                            tp2=setup.tp2,
                            size_contracts=setup.size_contracts,
                            size_usd=setup.size_usd,
                            r_distance=setup.r_distance,
                            tp1_size_pct=setup.tp1_size_pct,
                            tp2_size_pct=setup.tp2_size_pct,
                            trailing_atr_multiplier=setup.trailing_atr_multiplier,
                            max_hold_duration_s=setup.max_hold_duration_s,
                            atr=float(setup.atr or 0.0),
                            remaining_contracts=setup.size_contracts,
                            entry_score=bd.total_score,
                            strategy_sleeve=str(getattr(bd, "strategy_sleeve", "neutral") or "neutral"),
                            setup_type=str(getattr(bd, "setup_type", "unknown") or "unknown"),
                        )
                        open_trade._last_funding_bar = i
                        self._risk.on_trade_opened()

            result.equity_curve.append(equity)

        # Force-close any still-open trade at last bar
        if open_trade is not None:
            last = df.iloc[-1]
            last_close = float(last["close"])
            mult = 1 if open_trade.direction == "long" else -1
            open_trade.pnl_usd = mult * (last_close - open_trade.entry_price) * open_trade.remaining_contracts
            # Exit-leg commission on notional (entry leg already debited).
            open_trade.pnl_usd -= open_trade.size_usd * self._commission_pct
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
        self,
        symbol: str,
        df: pd.DataFrame,
        df_high: pd.DataFrame,
        df_lower: pd.DataFrame | None = None,
    ) -> SignalBreakdown | None:
        if len(df) < _MIN_BARS:
            return None

        tf_cfg = self._cfg["timeframes"]
        candles = {tf_cfg["primary"]: df, tf_cfg["higher"]: df_high}
        lower_tf = tf_cfg.get("lower") or tf_cfg.get("secondary") or tf_cfg.get("entry")
        if lower_tf:
            candles[lower_tf] = df_lower if df_lower is not None else pd.DataFrame()

        snap = MarketSnapshot(
            symbol=symbol,
            candles=candles,
            order_book={"bids": [], "asks": []},
            funding_rate=0.0,
            open_interest_usd=0.0,
            oi_change_pct=0.0,
            last_price=float(df["close"].iloc[-1]),
            volume_24h_usdt=0.0,
            ls_ratio=1.0,
            taker_buy_ratio=0.5,
        )
        scored = self._scorer.score_many({symbol: snap})
        return scored[0] if scored else None

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
        # ATR-based trailing mirrors live behaviour: ``trail_dist =
        # atr * trailing_atr_multiplier`` is fixed at entry, only the
        # ratchet target moves with price. Falls back to the legacy
        # %-based trail (``tp2_trailing_stop_pct``) when ATR isn't
        # known — keeps backwards compatibility with older state
        # snapshots and shadow trades that didn't capture ATR.
        atr_mult = float(trade.trailing_atr_multiplier or 0.0)
        trail_dist = float(trade.atr or 0.0) * atr_mult
        atr_mode = trail_dist > 0.0
        trail_pct = self._exit_cfg.get("tp2_trailing_stop_pct", 0.15)

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

        mult = 1 if trade.direction == "long" else -1

        # ------------------------------------------------------------------
        # Order-of-operations within a bar
        # ------------------------------------------------------------------
        # We process the bar in the same priority that live does:
        #   1. SL hit  (worst case first to avoid optimistic bias)
        #   2. Trailing stop hit
        #   3. TP1 partial — only if not yet hit
        #   4. TP2 partial — only after TP1
        #   5. TP3 full close — only after TP2
        # ------------------------------------------------------------------

        # 1. SL hit (always closes the residual)
        if trade.is_sl_hit(low, high):
            close_pnl = mult * (trade.stop_loss - trade.entry_price) * trade.remaining_contracts
            trade.pnl_usd += close_pnl
            trade.exit_price = trade.stop_loss
            trade.exit_reason = "stop_loss"
            trade.exit_bar = bar_idx
            return trade, True

        # 2. Trailing stop hit (closes residual)
        if trade.tp1_hit and trade.trailing_stop is not None:
            ts_hit = (
                (trade.direction == "long" and low <= trade.trailing_stop)
                or (trade.direction == "short" and high >= trade.trailing_stop)
            )
            if ts_hit:
                close_pnl = mult * (trade.trailing_stop - trade.entry_price) * trade.remaining_contracts
                trade.pnl_usd += close_pnl
                trade.exit_price = trade.trailing_stop
                trade.exit_reason = "trailing_stop"
                trade.exit_bar = bar_idx
                return trade, True

        # 3. TP1 partial (closes tp1_size_pct of original size)
        if not trade.tp1_hit and trade.is_tp1_hit(low, high):
            tp1_qty = trade.size_contracts * trade.tp1_size_pct
            tp1_pnl = mult * (trade.tp1 - trade.entry_price) * tp1_qty
            trade.pnl_usd += tp1_pnl
            trade.remaining_contracts -= tp1_qty
            trade.tp1_hit = True
            # Move SL to breakeven
            trade.stop_loss = trade.entry_price
            # Initial trailing stop — at the TP1 fill price minus
            # (or plus) the trail distance, mirroring live.
            if atr_mode:
                if trade.direction == "long":
                    trade.trailing_stop = trade.tp1 - trail_dist
                else:
                    trade.trailing_stop = trade.tp1 + trail_dist
            else:
                if trade.direction == "long":
                    trade.trailing_stop = trade.tp1 * (1 - trail_pct)
                else:
                    trade.trailing_stop = trade.tp1 * (1 + trail_pct)

        # 3b. Update trailing stop after TP1 (ratchet only — never widen).
        # Live recomputes trail_dist from the (constant) entry-time ATR
        # and the current tick price; backtest does the same with the
        # bar close as a tick proxy.
        if trade.tp1_hit and trade.trailing_stop is not None:
            if atr_mode:
                if trade.direction == "long":
                    new_trail = close - trail_dist
                    if new_trail > trade.trailing_stop:
                        trade.trailing_stop = new_trail
                else:
                    new_trail = close + trail_dist
                    if new_trail < trade.trailing_stop:
                        trade.trailing_stop = new_trail
            else:
                if trade.direction == "long":
                    new_trail = close * (1 - trail_pct)
                    if new_trail > trade.trailing_stop:
                        trade.trailing_stop = new_trail
                else:
                    new_trail = close * (1 + trail_pct)
                    if new_trail < trade.trailing_stop:
                        trade.trailing_stop = new_trail

        # 4. TP2 partial — closes tp2_size_pct of ORIGINAL position (matches
        #    live + exchange-placed reduceOnly TP order). Earlier versions
        #    used remaining*tp2_size_pct which left a much larger residual to
        #    trail than the configured trail_size_pct.
        tp3 = trade.entry_price + mult * trade.r_distance * float(
            self._exit_cfg.get("tp3_r_multiple", 3.0)
        )
        tp2_already_hit = getattr(trade, "_tp2_hit", False)
        if trade.tp1_hit and not tp2_already_hit and trade.is_tp2_hit(low, high):
            tp2_qty = min(
                trade.remaining_contracts,
                trade.size_contracts * trade.tp2_size_pct,
            )
            tp2_pnl = mult * (trade.tp2 - trade.entry_price) * tp2_qty
            trade.pnl_usd += tp2_pnl
            trade.remaining_contracts -= tp2_qty
            setattr(trade, "_tp2_hit", True)
            if trade.remaining_contracts <= 1e-9:
                trade.exit_price = trade.tp2
                trade.exit_reason = "tp2_full"
                trade.exit_bar = bar_idx
                return trade, True

        # 5. TP3 closes the rest of the runner
        tp3_hit = (
            (trade.direction == "long" and high >= tp3)
            or (trade.direction == "short" and low <= tp3)
        )
        if getattr(trade, "_tp2_hit", False) and tp3_hit and trade.remaining_contracts > 0:
            close_pnl = mult * (tp3 - trade.entry_price) * trade.remaining_contracts
            trade.pnl_usd += close_pnl
            trade.exit_price = tp3
            trade.exit_reason = "tp3"
            trade.exit_bar = bar_idx
            return trade, True

        return trade, False
