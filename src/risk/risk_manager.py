"""
Risk manager — position sizing, SL/TP calculation, daily loss / drawdown guards.
Integrates fractional Kelly sizing when EV model data is available.

Professional-grade additions:
  - Historical VaR / CVaR (Conditional VaR) from equity curve
  - Dynamic drawdown management with graduated size reduction
  - Recovery period logic with gradual re-entry
  - Portfolio heat tracking and risk budget allocation
  - Risk-adjusted performance metrics (Sharpe, Sortino, Calmar)
  - Dynamic leverage adjustment based on volatility regime
"""
from __future__ import annotations

import collections
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from src.analysis.indicators import atr
from src.models.strategy_passport import asset_from_symbol, sector_for_asset

if TYPE_CHECKING:
    from src.models.ev_model import EVResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Portfolio Risk Metrics — professional-grade analytics
# ---------------------------------------------------------------------------

@dataclass
class PortfolioRiskMetrics:
    """Snapshot of portfolio-level risk analytics computed from equity curve."""
    var_95: float = 0.0          # 95% Value-at-Risk (% of equity)
    var_99: float = 0.0          # 99% Value-at-Risk (% of equity)
    cvar_95: float = 0.0         # 95% Conditional VaR (expected shortfall)
    cvar_99: float = 0.0         # 99% Conditional VaR
    sharpe_ratio: float = 0.0    # annualized Sharpe (risk-free = 0)
    sortino_ratio: float = 0.0   # annualized Sortino (downside deviation)
    calmar_ratio: float = 0.0    # annualized return / max drawdown
    max_drawdown_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    recovery_factor: float = 0.0 # net profit / max drawdown
    portfolio_heat_pct: float = 0.0  # total open risk as % of equity
    tail_risk_score: float = 0.0 # 0-100, higher = fatter tails
    volatility_annualized: float = 0.0
    downside_deviation: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    payoff_ratio: float = 0.0    # avg_win / abs(avg_loss)
    kelly_criterion: float = 0.0 # theoretical optimal bet fraction
    sample_size: int = 0


class RiskAnalytics:
    """Computes institutional-grade risk metrics from trade history and equity curve.

    Used by the CRO (RiskManager) to make dynamic sizing and drawdown decisions.
    """

    def __init__(self, lookback: int = 100):
        self._lookback = lookback
        self._equity_history: collections.deque[float] = collections.deque(maxlen=lookback + 1)
        self._trade_returns: collections.deque[float] = collections.deque(maxlen=lookback)
        self._daily_returns: collections.deque[float] = collections.deque(maxlen=252)

    def record_equity(self, equity: float) -> None:
        """Append equity point for rolling metrics."""
        if equity > 0:
            self._equity_history.append(equity)

    def record_trade_return(self, pnl_pct: float) -> None:
        """Record a closed trade return (% of equity at entry)."""
        self._trade_returns.append(pnl_pct)

    def compute(self, peak_equity: float, current_equity: float) -> PortfolioRiskMetrics:
        """Compute all risk metrics from current state."""
        m = PortfolioRiskMetrics()
        m.current_drawdown_pct = (
            (peak_equity - current_equity) / peak_equity * 100.0
            if peak_equity > 0 else 0.0
        )

        if len(self._trade_returns) < 3:
            m.sample_size = len(self._trade_returns)
            return m

        returns = list(self._trade_returns)
        m.sample_size = len(returns)

        # --- Basic stats ---
        m.win_rate = sum(1 for r in returns if r > 0) / len(returns) * 100.0
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        m.avg_win = statistics.mean(wins) if wins else 0.0
        m.avg_loss = statistics.mean(losses) if losses else 0.0
        m.payoff_ratio = abs(m.avg_win / m.avg_loss) if m.avg_loss != 0 else 0.0
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # --- Kelly criterion ---
        if m.win_rate > 0 and m.payoff_ratio > 0:
            p = m.win_rate / 100.0
            b = m.payoff_ratio
            m.kelly_criterion = max(0.0, (p * b - (1 - p)) / b)

        # --- Volatility metrics ---
        if len(returns) >= 2:
            m.volatility_annualized = statistics.stdev(returns) * math.sqrt(252)
            downside = [r for r in returns if r < 0]
            m.downside_deviation = statistics.stdev(downside) * math.sqrt(252) if len(downside) >= 2 else 0.0

        # --- Risk-adjusted returns ---
        if m.volatility_annualized > 0:
            mean_annual = statistics.mean(returns) * 252
            m.sharpe_ratio = mean_annual / m.volatility_annualized
        if m.downside_deviation > 0:
            mean_annual = statistics.mean(returns) * 252
            m.sortino_ratio = mean_annual / m.downside_deviation

        # --- VaR / CVaR (historical simulation) ---
        sorted_returns = sorted(returns)
        n = len(sorted_returns)
        idx_95 = max(0, int(n * 0.05))
        idx_99 = max(0, int(n * 0.01))
        m.var_95 = abs(sorted_returns[idx_95])
        m.var_99 = abs(sorted_returns[idx_99])
        tail_95 = sorted_returns[:idx_95 + 1]
        tail_99 = sorted_returns[:idx_99 + 1]
        m.cvar_95 = abs(statistics.mean(tail_95)) if tail_95 else m.var_95
        m.cvar_99 = abs(statistics.mean(tail_99)) if tail_99 else m.var_99

        # --- Tail risk score (0-100) ---
        # Compares CVaR to VaR — higher ratio = fatter tails
        if m.var_95 > 0:
            tail_ratio = m.cvar_95 / m.var_95
            m.tail_risk_score = min(100.0, max(0.0, (tail_ratio - 1.0) * 200.0))

        # --- Drawdown from equity curve ---
        if len(self._equity_history) >= 2:
            eq = list(self._equity_history)
            peak = eq[0]
            max_dd = 0.0
            for e in eq:
                if e > peak:
                    peak = e
                dd = (peak - e) / peak * 100.0 if peak > 0 else 0.0
                max_dd = max(max_dd, dd)
            m.max_drawdown_pct = max_dd

        # --- Calmar ratio ---
        if m.max_drawdown_pct > 0:
            mean_annual = statistics.mean(returns) * 252
            m.calmar_ratio = mean_annual / (m.max_drawdown_pct / 100.0)

        # --- Recovery factor ---
        if m.max_drawdown_pct > 0:
            net_profit = sum(returns)
            m.recovery_factor = net_profit / (m.max_drawdown_pct / 100.0)

        return m

    def get_drawdown_state(self, peak_equity: float, current_equity: float) -> tuple[float, str]:
        """Return (drawdown_pct, state) where state is 'normal', 'warning', 'critical'."""
        dd = (peak_equity - current_equity) / peak_equity * 100.0 if peak_equity > 0 else 0.0
        if dd >= 15.0:
            return dd, "critical"
        if dd >= 8.0:
            return dd, "warning"
        return dd, "normal"


@dataclass
class TradeSetup:
    symbol: str
    direction: str          # 'long' | 'short'
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float              # extended target (3R)
    size_usd: float
    size_contracts: float
    leverage: int
    r_distance: float       # $ distance to 1R
    strategy_sleeve: str = "neutral"
    atr: float = 0.0        # ATR at entry (for ATR-based trailing stop)
    risk_pct: float = 0.0   # actual risk % of equity used
    ev_net_pct: float = 0.0  # EV net % carried forward for trade card
    exit_profile: str = "default"
    tp1_size_pct: float = 0.50
    tp2_size_pct: float = 0.30
    trail_size_pct: float = 0.20
    breakeven_trigger_r: float = 1.0
    trailing_atr_multiplier: float = 1.5
    max_hold_duration_s: int = 172800
    # Dedicated short-setup label and confidence (Phase D / Liq Sweep) so
    # downstream surfaces (Telegram, dataset_logger, attribution) can
    # report which post-distribution edge fired. Empty when no dedicated
    # short setup was active. Defaults preserve backward compat for state
    # loaded from older `data/open_trades.json`.
    short_setup_label: str = ""
    short_setup_confidence: float = 0.0
    setup_passport: dict = field(default_factory=dict)


@dataclass
class PortfolioState:
    equity: float = 0.0
    daily_start_equity: float = 0.0
    day_start_ts: float = field(default_factory=time.time)
    weekly_start_equity: float = 0.0
    week_start_ts: float = field(default_factory=time.time)
    consecutive_losses: int = 0
    open_trade_count: int = 0
    open_risk_pct: float = 0.0   # sum of risk_pct for all currently open trades
    peak_equity: float = 0.0
    paused_until_ts: float = 0.0
    kill_switch_reason: str = ""
    symbol_risk_pct: dict[str, float] = field(default_factory=dict)
    direction_risk_pct: dict[str, float] = field(default_factory=dict)
    sector_risk_pct: dict[str, float] = field(default_factory=dict)
    # --- Dynamic drawdown management ---
    drawdown_state: str = "normal"      # 'normal' | 'warning' | 'critical'
    recovery_mode: bool = False         # True when recovering from drawdown
    recovery_start_ts: float = 0.0      # when recovery mode began
    recovery_start_equity: float = 0.0  # H2: equity at recovery start
    recovery_stage: int = 0             # 0-3, gradual re-entry stages
    peak_drawdown_pct: float = 0.0      # worst drawdown observed this cycle
    drawdown_enter_ts: float = 0.0      # when drawdown state escalated
    risk_budget_multiplier: float = 1.0 # 0.0-1.0, reduced during drawdown

    def reset_day(self, equity: float) -> None:
        self.daily_start_equity = equity
        self.day_start_ts = time.time()

    def reset_week(self, equity: float) -> None:
        self.weekly_start_equity = equity
        self.week_start_ts = time.time()

    @property
    def daily_pnl_pct(self) -> float:
        if self.daily_start_equity == 0:
            return 0.0
        return (self.equity - self.daily_start_equity) / self.daily_start_equity * 100

    @property
    def weekly_pnl_pct(self) -> float:
        if self.weekly_start_equity == 0:
            return 0.0
        return (self.equity - self.weekly_start_equity) / self.weekly_start_equity * 100

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity == 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity * 100

    def update_drawdown_state(self, warning_threshold: float = 8.0, critical_threshold: float = 15.0) -> None:
        """Update drawdown state machine and risk budget multiplier."""
        dd = self.drawdown_pct
        self.peak_drawdown_pct = max(self.peak_drawdown_pct, dd)

        prev_state = self.drawdown_state
        if dd >= critical_threshold:
            self.drawdown_state = "critical"
        elif dd >= warning_threshold:
            self.drawdown_state = "warning"
        else:
            self.drawdown_state = "normal"

        # State transition: enter drawdown management
        if self.drawdown_state != prev_state and self.drawdown_state != "normal":
            self.drawdown_enter_ts = time.time()
            log.warning(
                "Drawdown state escalated: %s → %s (%.2f%%)",
                prev_state, self.drawdown_state, dd,
            )

        # Risk budget multiplier: graduated reduction
        if self.drawdown_state == "critical":
            self.risk_budget_multiplier = 0.25  # 75% reduction
        elif self.drawdown_state == "warning":
            self.risk_budget_multiplier = 0.50  # 50% reduction
        else:
            # Recovery: gradual ramp-up
            if self.recovery_mode:
                elapsed = time.time() - self.recovery_start_ts
                # H2 audit: require both time elapsed AND equity recovering 30% of drawdown per stage
                time_stage = int(elapsed / 3600)  # minimum 1 hour per stage
                # Equity recovery: what fraction of the drawdown has been reclaimed?
                drawdown_at_start = self.peak_equity - self.recovery_start_equity if self.peak_equity > 0 else 0.0
                equity_recovered = self.equity - self.recovery_start_equity if self.recovery_start_equity > 0 else 0.0
                recovery_pct = equity_recovered / drawdown_at_start if drawdown_at_start > 0 else 1.0
                # Each stage requires reclaiming 30% of the drawdown
                equity_stage = int(recovery_pct / 0.30) if recovery_pct >= 0 else 0
                # Advance only as far as BOTH conditions allow
                stage = min(3, min(time_stage, equity_stage))
                self.recovery_stage = stage
                self.risk_budget_multiplier = [0.60, 0.70, 0.85, 1.00][stage]
                if stage >= 3:
                    self.recovery_mode = False
                    self.peak_drawdown_pct = 0.0
                    log.info("Recovery complete — full risk budget restored")
            else:
                self.risk_budget_multiplier = 1.0

        # Enter recovery mode when exiting critical/warning
        if prev_state in ("warning", "critical") and self.drawdown_state == "normal":
            self.recovery_mode = True
            self.recovery_start_ts = time.time()
            self.recovery_start_equity = self.equity  # H2: track equity at recovery start
            self.recovery_stage = 0
            log.info("Entering recovery mode — risk budget at 60%%, ramping up over 3 hours")


class RiskManager:
    def __init__(self, cfg: dict, exploration_mode: bool = False):
        self._cfg = cfg
        self._risk = cfg["risk"]
        self._exit = cfg["exit"]
        self._safety = cfg.get("safety", {})
        self._exploration = bool(exploration_mode)
        self.state = PortfolioState()

        from src.risk.kelly_sizer import KellySizer
        self._kelly = KellySizer(cfg)
        self._last_setup_rejection_reason = ""

        # Professional-grade risk analytics engine
        self._analytics = RiskAnalytics(lookback=200)

    @property
    def last_setup_rejection_reason(self) -> str:
        return self._last_setup_rejection_reason

    def _reject_setup(self, reason: str) -> None:
        self._last_setup_rejection_reason = reason

    def _max_risk_usd(self, equity: float) -> float:
        return equity * (self._risk["max_risk_per_trade_pct"] / 100)

    # ------------------------------------------------------------------ #
    #  Guards                                                              #
    # ------------------------------------------------------------------ #

    def can_open_trade(self) -> tuple[bool, str]:
        """Returns (allowed, reason)."""
        s = self.state

        max_open = self._cfg["trading"]["max_open_trades"]
        if s.open_trade_count >= max_open:
            return False, f"max_open_trades ({max_open}) reached"

        if s.kill_switch_reason:
            return False, f"kill switch active ({s.kill_switch_reason})"

        if s.paused_until_ts > time.time():
            remaining = int(s.paused_until_ts - time.time())
            return False, f"cooldown active ({remaining}s remaining)"

        if not self._exploration:
            if abs(s.daily_pnl_pct) >= self._risk["daily_loss_cap_pct"] and s.daily_pnl_pct < 0:
                return False, f"daily loss cap hit ({s.daily_pnl_pct:.2f}%)"

            weekly_cap = float(self._risk.get("weekly_loss_cap_pct", 0.0) or 0.0)
            if weekly_cap > 0 and s.weekly_pnl_pct <= -weekly_cap:
                return False, f"weekly loss cap hit ({s.weekly_pnl_pct:.2f}%)"

            if s.drawdown_pct >= self._risk["max_drawdown_pct"]:
                return False, f"max drawdown hit ({s.drawdown_pct:.2f}%)"

            max_consecutive_losses = int(
                self._cfg["safety"].get("max_consecutive_losses", 0) or 0
            )
            if max_consecutive_losses > 0 and s.consecutive_losses >= max_consecutive_losses:
                return False, f"consecutive losses limit ({s.consecutive_losses})"

        max_total_risk = self._risk.get("max_risk_per_trade_pct", 2.0) * self._cfg["trading"]["max_open_trades"]
        if s.open_risk_pct >= max_total_risk:
            return False, f"aggregate risk cap hit ({s.open_risk_pct:.1f}% >= {max_total_risk:.1f}%)"

        return True, "ok"

    def check_trade_exposure(self, symbol: str, direction: str, risk_pct: float, sector: str = "") -> tuple[bool, str]:
        max_symbol_risk = float(self._risk.get("max_symbol_risk_pct", 0.0) or 0.0)
        if max_symbol_risk > 0:
            current = float(self.state.symbol_risk_pct.get(symbol, 0.0))
            if current + risk_pct > max_symbol_risk:
                return False, f"symbol exposure cap hit for {symbol} ({current + risk_pct:.2f}% > {max_symbol_risk:.2f}%)"

        max_direction_risk = float(self._risk.get("max_direction_risk_pct", 0.0) or 0.0)
        if max_direction_risk > 0:
            current = float(self.state.direction_risk_pct.get(direction, 0.0))
            if current + risk_pct > max_direction_risk:
                return False, f"direction exposure cap hit for {direction} ({current + risk_pct:.2f}% > {max_direction_risk:.2f}%)"

        max_sector_risk = float(self._risk.get("max_sector_risk_pct", 0.0) or 0.0)
        if max_sector_risk > 0 and sector:
            current = float(self.state.sector_risk_pct.get(sector, 0.0))
            if current + risk_pct > max_sector_risk:
                return False, f"sector exposure cap hit for {sector} ({current + risk_pct:.2f}% > {max_sector_risk:.2f}%)"

        return True, "ok"

    def trigger_kill_switch(self, reason: str) -> None:
        self.state.kill_switch_reason = reason

    def clear_kill_switch(self) -> None:
        self.state.kill_switch_reason = ""

    # ------------------------------------------------------------------ #
    #  Trade setup calculation                                            #
    # ------------------------------------------------------------------ #

    def calculate_setup(
        self,
        symbol: str,
        direction: str,
        df: pd.DataFrame,
        entry_price: float,
        strategy_sleeve: str = "neutral",
        leverage: int | None = None,
        ev_result: "EVResult | None" = None,
        kelly_scale: float = 1.0,
        backtest: bool = False,
        min_lot_size: float = 0.0,
        min_notional_usd: float = 0.0,
        short_setup_label: str = "",
        short_setup_confidence: float = 0.0,
        setup_passport: dict | None = None,
    ) -> TradeSetup | None:
        """
        Returns a fully computed TradeSetup or None if risk/reward insufficient.
        Uses fractional Kelly sizing when EV result is provided.
        """
        self._last_setup_rejection_reason = ""

        # Exploration mode: flat % sizing, no Kelly, no FM scaling
        if self._exploration:
            ev_result = None
            kelly_scale = 1.0

        ind_cfg = self._cfg["indicators"]
        current_atr = atr(df, ind_cfg["atr_period"])

        lev = leverage or self._risk["default_leverage"]
        lev = min(lev, self._risk["max_leverage"])
        exit_profile_name, exit_profile = self._resolve_exit_profile(strategy_sleeve)

        atr_mult = self._risk.get("stop_loss_atr_multiplier", 1.5)
        tp1_r = float(exit_profile.get("tp1_r_multiple", self._exit["tp1_r_multiple"]))
        tp2_r = float(exit_profile.get("tp2_r_multiple", self._exit["tp2_r_multiple"]))
        tp3_r = float(exit_profile.get("tp3_r_multiple", 3.0))
        if direction == "long":
            stop_loss = entry_price - current_atr * atr_mult
            tp1 = entry_price + current_atr * atr_mult * tp1_r
            tp2 = entry_price + current_atr * atr_mult * tp2_r
            tp3 = entry_price + current_atr * atr_mult * tp3_r
        else:
            stop_loss = entry_price + current_atr * atr_mult
            tp1 = entry_price - current_atr * atr_mult * tp1_r
            tp2 = entry_price - current_atr * atr_mult * tp2_r
            tp3 = entry_price - current_atr * atr_mult * tp3_r

        r_distance = abs(entry_price - stop_loss)
        r_distance_pct = (r_distance / entry_price * 100.0) if entry_price > 0 else 0.0
        max_stop_distance_pct = float(self._risk.get("max_stop_distance_pct", 50.0) or 50.0)
        setup_sector = str((setup_passport or {}).get("sector") or sector_for_asset(asset_from_symbol(symbol)))
        is_binance_alpha = setup_sector == "binance_alpha"
        geometry_values = (entry_price, current_atr, stop_loss, tp1, tp2, tp3, r_distance, r_distance_pct)
        invalid_geometry = (
            not all(math.isfinite(float(v)) for v in geometry_values)
            or entry_price <= 0
            or current_atr <= 0
            or stop_loss <= 0
            or tp1 <= 0
            or tp2 <= 0
            or tp3 <= 0
            or r_distance <= 0
        )
        if is_binance_alpha and invalid_geometry:
            self._reject_setup(
                f"invalid binance_alpha price geometry: entry={entry_price:.8f} atr={current_atr:.8f} "
                f"sl={stop_loss:.8f} tp1={tp1:.8f} tp2={tp2:.8f} tp3={tp3:.8f}"
            )
            log.info(
                "%s skipped — invalid binance_alpha setup geometry entry=%.8f atr=%.8f sl=%.8f tp1=%.8f tp2=%.8f tp3=%.8f",
                symbol, entry_price, current_atr, stop_loss, tp1, tp2, tp3,
            )
            return None
        if is_binance_alpha and r_distance_pct > max_stop_distance_pct:
            self._reject_setup(
                f"binance_alpha stop distance {r_distance_pct:.1f}% exceeds max {max_stop_distance_pct:.1f}%"
            )
            log.info(
                "%s skipped — binance_alpha stop distance %.1f%% exceeds max %.1f%%",
                symbol, r_distance_pct, max_stop_distance_pct,
            )
            return None

        # Reward/risk check against TP2
        reward = abs(tp2 - entry_price)
        rr_ratio = reward / r_distance if r_distance else 0.0
        min_rr_ratio = float(self._risk["min_rr_ratio"])
        rr_epsilon = 1e-9
        if r_distance == 0 or rr_ratio + rr_epsilon < min_rr_ratio:
            self._reject_setup(
                f"rr={rr_ratio:.2f} below minimum {min_rr_ratio:.2f}"
            )
            log.debug(
                "%s RR too low: %.2f (need %.1f)",
                symbol, rr_ratio, min_rr_ratio,
            )
            return None

        equity = self.state.equity

        # ── Kelly sizing (preferred) ──────────────────────────────────
        size_usd = 0.0
        risk_pct = 0.0
        if ev_result is not None:
            kelly_result = self._kelly.size(
                ev=ev_result,
                equity=equity,
                entry_price=entry_price,
                stop_loss=stop_loss,
                df=df,
                cfg=self._cfg,
            )
            if kelly_result.is_valid:
                size_usd = kelly_result.size_usd
                risk_pct = kelly_result.final_risk_pct
                log.debug("[%s] Kelly size: $%.2f (risk=%.2f%%)", symbol, size_usd, risk_pct)

        # ── Fallback to fixed % sizing ────────────────────────────────
        if size_usd == 0.0:
            risk_usd = equity * (self._risk["risk_per_trade_pct"] / 100)
            max_risk_usd = self._max_risk_usd(equity)
            risk_usd = min(risk_usd, max_risk_usd)
            size_usd = (risk_usd / r_distance) * entry_price
            risk_pct = self._risk["risk_per_trade_pct"]

        # ── Apply ML fund manager Kelly scaling ──────────────────────
        if kelly_scale != 1.0:
            size_usd *= kelly_scale
            risk_usd_scaled = (size_usd * r_distance) / entry_price
            max_risk_usd = self._max_risk_usd(equity)
            if risk_usd_scaled > max_risk_usd:
                size_usd = (max_risk_usd / r_distance) * entry_price
            risk_pct = min(risk_pct * kelly_scale, self._risk["max_risk_per_trade_pct"])

        # ── Dynamic drawdown-aware sizing (professional CRO layer) ────
        dd_multiplier = self.get_dynamic_risk_multiplier()
        if dd_multiplier < 1.0:
            size_usd *= dd_multiplier
            risk_pct *= dd_multiplier
            log.info(
                "%s drawdown-adjusted: multiplier=%.2f → size=$%.2f risk=%.2f%%",
                symbol, dd_multiplier, size_usd, risk_pct,
            )

        # ── VaR-adjusted sizing (institutional risk scaling) ─────────
        if not backtest and not self._exploration:
            size_usd = self.var_adjusted_position_size(size_usd)
            risk_pct = (size_usd * r_distance / entry_price) / equity * 100 if equity > 0 else risk_pct

        # ── Minimum risk-per-trade floor ($5 default) ────────────────────
        # Ensures each trade risks at least a meaningful dollar amount.
        min_risk_usd = self._risk.get("min_risk_usd", 5.0)
        current_risk_usd = (size_usd * r_distance) / entry_price
        if current_risk_usd < min_risk_usd and not backtest:
            bumped_size = (min_risk_usd / r_distance) * entry_price
            bumped_risk_pct = min_risk_usd / equity * 100
            max_risk_usd = self._max_risk_usd(equity)
            if min_risk_usd > max_risk_usd:
                self._reject_setup(
                    f"min_risk_usd {min_risk_usd:.2f} exceeds max allowed risk {max_risk_usd:.2f}"
                )
                log.info(
                    "%s skipped — min risk $%.2f exceeds max risk $%.2f at current equity",
                    symbol, min_risk_usd, max_risk_usd,
                )
                return None
            log.info("%s: bumped to min risk $%.2f → position $%.2f", symbol, min_risk_usd, bumped_size)
            size_usd = bumped_size
            risk_pct = min(bumped_risk_pct, self._risk["max_risk_per_trade_pct"])

        # ── Binance Futures minimum notional enforcement ─────────────────
        effective_min_notional = min_notional_usd if min_notional_usd > 0 else self._risk.get("min_notional_usd", 5.0)
        if size_usd < effective_min_notional and not backtest:
            bumped_risk_usd = (effective_min_notional * r_distance) / entry_price
            max_risk_usd = self._max_risk_usd(equity)
            if bumped_risk_usd <= max_risk_usd:
                log.info(
                    "%s: bumped to minimum notional $%.2f — risk %.2f%%",
                    symbol, effective_min_notional, bumped_risk_usd / equity * 100,
                )
                size_usd = float(effective_min_notional)
                risk_pct = max(risk_pct, bumped_risk_usd / equity * 100)
            else:
                self._reject_setup(
                    f"minimum notional {effective_min_notional:.2f} needs risk {bumped_risk_usd:.2f} above max {max_risk_usd:.2f}"
                )
                log.info(
                    "%s skipped — position $%.2f below minimum notional $%.2f",
                    symbol, size_usd, effective_min_notional,
                )
                return None

        # Ensure margin required is at least $1 (prevents dust trades)
        margin_required = size_usd / lev
        if margin_required < 1.0:
            bumped_size = float(lev)
            bumped_risk_usd = (bumped_size * r_distance) / entry_price
            max_risk_usd = self._max_risk_usd(equity)
            if bumped_risk_usd <= max_risk_usd:
                log.info(
                    "%s: bumped to minimum margin requirement $1.00 — position $%.2f",
                    symbol, bumped_size,
                )
                size_usd = bumped_size
                risk_pct = max(risk_pct, bumped_risk_usd / equity * 100)
                margin_required = size_usd / lev
            else:
                self._reject_setup(
                    f"minimum margin $1.00 needs risk {bumped_risk_usd:.2f} above max {max_risk_usd:.2f}"
                )
                log.info("%s margin too small: $%.2f — skipping", symbol, margin_required)
                return None

        size_contracts = size_usd / entry_price

        # ── Min lot size enforcement (live mode) ─────────────────────────
        # Exchange rejects orders below their minimum lot size (e.g. 0.001 BTC).
        if not backtest and min_lot_size > 0 and size_contracts < min_lot_size:
            bumped_usd = min_lot_size * entry_price
            bumped_risk_pct = (bumped_usd * r_distance / entry_price) / equity * 100
            if bumped_risk_pct > self._risk["max_risk_per_trade_pct"] * 2:
                self._reject_setup(
                    f"minimum lot {min_lot_size:.6f} requires risk {bumped_risk_pct:.2f}% above tolerance"
                )
                log.info(
                    "%s skipped — min lot $%.2f (%.4f contracts) exceeds max risk at this equity",
                    symbol, bumped_usd, min_lot_size,
                )
                return None
            log.info(
                "%s: bumped to min lot %.4f ($%.2f) — risk %.2f%%",
                symbol, min_lot_size, bumped_usd, bumped_risk_pct,
            )
            size_contracts = min_lot_size
            size_usd = bumped_usd
            risk_pct = bumped_risk_pct

        return TradeSetup(
            symbol=symbol,
            direction=direction,
            strategy_sleeve=str(strategy_sleeve or "neutral"),
            entry_price=entry_price,
            stop_loss=round(stop_loss, 8),
            tp1=round(tp1, 8),
            tp2=round(tp2, 8),
            tp3=round(tp3, 8),
            size_usd=round(size_usd, 2),
            size_contracts=round(size_contracts, 6),
            leverage=lev,
            r_distance=round(r_distance, 8),
            atr=round(current_atr, 8),
            risk_pct=round(risk_pct, 4),
            ev_net_pct=round(ev_result.ev_net_pct, 4) if ev_result else 0.0,
            exit_profile=exit_profile_name,
            tp1_size_pct=float(exit_profile.get("tp1_size_pct", self._exit["tp1_size_pct"])),
            tp2_size_pct=float(exit_profile.get("tp2_size_pct", self._exit["tp2_size_pct"])),
            trail_size_pct=float(exit_profile.get("trail_size_pct", self._exit["trail_size_pct"])),
            breakeven_trigger_r=float(exit_profile.get("breakeven_trigger_r", self._exit.get("breakeven_trigger_r", 1.0))),
            trailing_atr_multiplier=float(exit_profile.get("trailing_atr_multiplier", self._exit.get("trailing_atr_multiplier", 1.5))),
            max_hold_duration_s=int(exit_profile.get(
                "max_hold_duration_s_short" if direction == "short" else "max_hold_duration_s",
                exit_profile.get("max_hold_duration_s", self._exit.get("max_hold_duration_s", 172800)),
            )),
            short_setup_label=str(short_setup_label or ""),
            short_setup_confidence=float(short_setup_confidence or 0.0),
            setup_passport=dict(setup_passport or {}),
        )

    def _resolve_exit_profile(self, strategy_sleeve: str) -> tuple[str, dict]:
        profiles = self._cfg.get("exit_profiles", {})
        sleeve = str(strategy_sleeve or "neutral")
        if sleeve in profiles and isinstance(profiles[sleeve], dict):
            return sleeve, profiles[sleeve]
        return "default", self._exit

    # ------------------------------------------------------------------ #
    #  State updates                                                       #
    # ------------------------------------------------------------------ #

    def update_equity(self, equity: float) -> None:
        self.state.equity = equity
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity

        if self.state.daily_start_equity == 0.0:
            self.state.reset_day(equity)
        elif time.time() - self.state.day_start_ts > 86400:
            self.state.reset_day(equity)
        if self.state.weekly_start_equity == 0.0:
            self.state.reset_week(equity)
        elif time.time() - self.state.week_start_ts > 86400 * 7:
            self.state.reset_week(equity)

    def on_trade_opened(self, symbol: str = "", direction: str = "", risk_pct: float = 0.0, sector: str = "") -> None:
        self.state.open_trade_count += 1
        self.state.open_risk_pct += risk_pct
        if symbol:
            self.state.symbol_risk_pct[symbol] = self.state.symbol_risk_pct.get(symbol, 0.0) + risk_pct
        if direction:
            self.state.direction_risk_pct[direction] = self.state.direction_risk_pct.get(direction, 0.0) + risk_pct
        if sector:
            self.state.sector_risk_pct[sector] = self.state.sector_risk_pct.get(sector, 0.0) + risk_pct

    def on_trade_closed(self, pnl: float, symbol: str = "", direction: str = "", risk_pct: float = 0.0, sector: str = "") -> None:
        self.state.open_trade_count = max(0, self.state.open_trade_count - 1)
        self.state.open_risk_pct = max(0.0, self.state.open_risk_pct - risk_pct)
        if symbol and symbol in self.state.symbol_risk_pct:
            self.state.symbol_risk_pct[symbol] = max(0.0, self.state.symbol_risk_pct[symbol] - risk_pct)
            if self.state.symbol_risk_pct[symbol] == 0.0:
                self.state.symbol_risk_pct.pop(symbol, None)
        if direction and direction in self.state.direction_risk_pct:
            self.state.direction_risk_pct[direction] = max(0.0, self.state.direction_risk_pct[direction] - risk_pct)
            if self.state.direction_risk_pct[direction] == 0.0:
                self.state.direction_risk_pct.pop(direction, None)
        if sector and sector in self.state.sector_risk_pct:
            self.state.sector_risk_pct[sector] = max(0.0, self.state.sector_risk_pct[sector] - risk_pct)
            if self.state.sector_risk_pct[sector] == 0.0:
                self.state.sector_risk_pct.pop(sector, None)
        if pnl < 0:
            self.state.consecutive_losses += 1
            cooldown_after = int(self._risk.get("cooldown_after_loss_streak", 0) or 0)
            cooldown_minutes = int(self._risk.get("cooldown_minutes", 0) or 0)
            if cooldown_after > 0 and cooldown_minutes > 0 and self.state.consecutive_losses >= cooldown_after:
                self.state.paused_until_ts = max(
                    self.state.paused_until_ts,
                    time.time() + cooldown_minutes * 60,
                )
        else:
            self.state.consecutive_losses = 0

        # Record trade return for analytics
        equity_before = self.state.equity
        if equity_before > 0:
            self._analytics.record_trade_return(pnl / equity_before * 100.0)

        self.update_equity(self.state.equity + pnl)

    # ------------------------------------------------------------------ #
    #  Professional-grade risk analytics                                   #
    # ------------------------------------------------------------------ #

    def get_risk_metrics(self) -> PortfolioRiskMetrics:
        """Compute and return current portfolio risk metrics."""
        return self._analytics.compute(self.state.peak_equity, self.state.equity)

    def get_drawdown_state(self) -> tuple[float, str]:
        """Return (drawdown_pct, state) from analytics."""
        return self._analytics.get_drawdown_state(self.state.peak_equity, self.state.equity)

    def get_dynamic_risk_multiplier(self) -> float:
        """Return risk budget multiplier based on drawdown state.

        Professional CRO behavior:
        - Normal: 100% risk budget
        - Warning (8-15% DD): 50% risk budget
        - Critical (>15% DD): 25% risk budget
        - Recovery: gradual ramp 60% → 70% → 85% → 100%
        """
        self.state.update_drawdown_state(
            warning_threshold=float(self._risk.get("drawdown_warning_pct", 8.0)),
            critical_threshold=float(self._risk.get("drawdown_critical_pct", 15.0)),
        )
        return self.state.risk_budget_multiplier

    def var_adjusted_position_size(self, base_size_usd: float, confidence_level: float = 0.95) -> float:
        """Adjust position size based on historical VaR.

        If recent VaR is high relative to normal, reduce size proportionally.
        This is how institutional desks scale down during volatile periods.
        """
        metrics = self._analytics.compute(self.state.peak_equity, self.state.equity)
        if metrics.sample_size < 10:
            return base_size_usd

        var_pct = metrics.var_95 if confidence_level <= 0.95 else metrics.var_99
        # Target: VaR should not exceed 2% of equity per trade
        target_var = 2.0
        if var_pct <= target_var:
            return base_size_usd

        # Scale down proportionally
        scale = target_var / var_pct
        scale = max(0.25, min(1.0, scale))  # floor at 25%
        adjusted = base_size_usd * scale
        log.info(
            "VaR-adjusted sizing: base=$%.2f → $%.2f (VaR=%.2f%%, scale=%.2f)",
            base_size_usd, adjusted, var_pct, scale,
        )
        return adjusted

    def should_reduce_exposure(self) -> tuple[bool, str]:
        """Check if portfolio exposure should be reduced based on risk state.

        Returns (should_reduce, reason).
        """
        dd, state = self.get_drawdown_state()
        if state == "critical":
            return True, f"critical drawdown ({dd:.1f}%)"

        metrics = self._analytics.compute(self.state.peak_equity, self.state.equity)
        if metrics.sample_size >= 10:
            # High tail risk: reduce exposure
            if metrics.tail_risk_score > 70:
                return True, f"high tail risk ({metrics.tail_risk_score:.0f}/100)"
            # VaR spike: reduce exposure
            if metrics.var_99 > 5.0:
                return True, f"VaR99 spike ({metrics.var_99:.1f}%)"

        return False, "ok"

    def portfolio_heat_pct(self) -> float:
        """Current portfolio heat as % of equity (total open risk)."""
        return self.state.open_risk_pct
