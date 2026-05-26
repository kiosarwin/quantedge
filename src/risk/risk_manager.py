"""
Risk manager — position sizing, SL/TP calculation, daily loss / drawdown guards.
Integrates fractional Kelly sizing when EV model data is available.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from src.analysis.indicators import atr
from src.models.strategy_passport import asset_from_symbol, sector_for_asset

if TYPE_CHECKING:
    from src.models.ev_model import EVResult

log = logging.getLogger(__name__)


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
            max_hold_duration_s=int(exit_profile.get("max_hold_duration_s", self._exit.get("max_hold_duration_s", 172800))),
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
        self.update_equity(self.state.equity + pnl)
