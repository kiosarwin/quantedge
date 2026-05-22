"""
Trade manager — tracks open position lifecycle: breakeven moves, trailing stop,
TP1/TP2 execution, and panic closes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Awaitable

from src.data.client import BinanceFuturesClient
from src.execution.executor import Executor
from src.risk.risk_manager import RiskManager, TradeSetup

log = logging.getLogger(__name__)


@dataclass
class OpenTrade:
    setup: TradeSetup
    entry_order_id: str
    sl_order_id: str
    tp1_order_id: str
    tp2_order_id: str
    opened_at: float = field(default_factory=time.time)
    tp1_hit: bool = False
    tp2_hit: bool = False
    breakeven_moved: bool = False
    trailing_stop: float | None = None
    remaining_contracts: float = 0.0
    realized_pnl: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    mfe_peak_ts: float = 0.0        # timestamp when mfe_r last improved
    drawdown_total_s: float = 0.0   # cumulative seconds spent in adverse territory
    _adverse_since: float = 0.0     # internal: timestamp adverse run started
    # Last market price at which the trade was actually closed.  Set by
    # TradeManager._close_trade just before invoking the on_close callback
    # so downstream consumers (TradeRecord, attribution, dataset_logger)
    # see the real exit price for *every* exit reason — not just the
    # subset (tp1/tp2/tp3/sl/trail) that maps cleanly back to a setup
    # field.  0.0 means "not yet closed".
    exit_price: float = 0.0

    @property
    def symbol(self) -> str:
        return self.setup.symbol

    @property
    def direction(self) -> str:
        return self.setup.direction

    def current_pnl(self, price: float) -> float:
        mult = 1 if self.direction == "long" else -1
        return mult * (price - self.setup.entry_price) * self.remaining_contracts

    def is_sl_hit(self, price: float) -> bool:
        if self.direction == "long":
            return price <= self.setup.stop_loss
        return price >= self.setup.stop_loss

    def is_tp1_hit(self, price: float) -> bool:
        if self.direction == "long":
            return price >= self.setup.tp1
        return price <= self.setup.tp1

    def is_tp2_hit(self, price: float) -> bool:
        if self.direction == "long":
            return price >= self.setup.tp2
        return price <= self.setup.tp2


CloseCallback = Callable[[OpenTrade, float, str], Awaitable[None]]
ProgressCallback = Callable[[OpenTrade, str, float], Awaitable[None]]


class TradeManager:
    STATE_PATH = Path("data/open_trades.json")

    def __init__(
        self,
        client: BinanceFuturesClient,
        executor: Executor,
        risk: RiskManager,
        cfg: dict,
        on_close: CloseCallback | None = None,
        on_progress: ProgressCallback | None = None,
    ):
        self._client = client
        self._executor = executor
        self._risk = risk
        self._cfg = cfg
        self._exit = cfg["exit"]
        self._trades: dict[str, OpenTrade] = {}
        self._on_close = on_close
        self._on_progress = on_progress
        self._load_state()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    async def open(self, setup: TradeSetup) -> OpenTrade | None:
        can, reason = self._risk.can_open_trade()
        if not can:
            log.warning("Risk guard blocked trade on %s: %s", setup.symbol, reason)
            return None
        can, reason = self._risk.check_trade_exposure(
            setup.symbol, setup.direction, setup.risk_pct
        )
        if not can:
            log.warning("Exposure guard blocked trade on %s: %s", setup.symbol, reason)
            return None

        try:
            entry_order = await self._executor.open_position(setup)
        except Exception as exc:
            log.error("[%s] Entry order failed — no position opened: %s", setup.symbol, exc)
            return None

        fill_price = float(entry_order.get("price", setup.entry_price) or setup.entry_price)
        if fill_price == 0:
            fill_price = setup.entry_price

        try:
            sl_order = await self._executor.place_stop_loss(setup, entry_order["id"])
            tp1_order = await self._executor.place_take_profit(
                setup, setup.tp1, setup.tp1_size_pct
            )
            tp2_order = await self._executor.place_take_profit(
                setup, setup.tp2, setup.tp2_size_pct
            )
        except Exception as exc:
            log.error(
                "[%s] SL/TP placement failed — closing entry to prevent orphan: %s",
                setup.symbol, exc,
            )
            try:
                await self._executor.cancel_all(setup.symbol)
                await self._executor.close_position_market(
                    setup.symbol, setup.direction, setup.size_contracts
                )
            except Exception as cancel_exc:
                log.critical(
                    "[%s] ORPHAN POSITION — could not cancel after SL/TP failure: %s",
                    setup.symbol, cancel_exc,
                )
            return None

        trade = OpenTrade(
            setup=setup,
            entry_order_id=entry_order["id"],
            sl_order_id=sl_order["id"],
            tp1_order_id=tp1_order["id"],
            tp2_order_id=tp2_order["id"],
            remaining_contracts=setup.size_contracts,
        )
        self._trades[setup.symbol] = trade
        self._risk.on_trade_opened(
            symbol=setup.symbol, direction=setup.direction, risk_pct=setup.risk_pct
        )
        self._save_state()
        log.info("Trade opened on %s @ %.4f", setup.symbol, fill_price)
        return trade

    async def monitor_all(self, price_lookup: dict[str, float]) -> None:
        """Called on each price tick — checks exits for every open trade."""
        for symbol, trade in list(self._trades.items()):
            price = price_lookup.get(symbol)
            if price is None or price <= 0:
                continue
            await self._check_exits(trade, price)

    async def close_all(self, reason: str = "manual") -> None:
        for symbol, trade in list(self._trades.items()):
            await self._close_trade(trade, trade.setup.entry_price, reason)

    @property
    def open_symbols(self) -> list[str]:
        return list(self._trades.keys())

    def get_floating_pnl(self, price_map: dict[str, float]) -> list[dict]:
        result = []
        for symbol, trade in self._trades.items():
            price = price_map.get(symbol)
            if price is None or price <= 0:
                continue
            entry = trade.setup.entry_price
            size = trade.setup.size_usd
            if trade.setup.direction == "long":
                pnl_pct = (price - entry) / entry * 100
            else:
                pnl_pct = (entry - price) / entry * 100
            pnl_usd = size * pnl_pct / 100
            risk_usd = trade.setup.r_distance / entry * size if entry > 0 else 0.0
            result.append({
                "symbol": symbol,
                "direction": trade.setup.direction,
                "entry": entry,
                "current": price,
                "pnl_usd": pnl_usd,
                "pnl_pct": pnl_pct,
                "size_usd": size,
                "elapsed_s": time.time() - trade.opened_at,
                "sl": trade.setup.stop_loss,
                "tp1": trade.setup.tp1,
                "risk_pct": trade.setup.risk_pct,
                "risk_usd": round(risk_usd, 2),
            })
        return result

    # ------------------------------------------------------------------ #
    #  Exit logic                                                          #
    # ------------------------------------------------------------------ #

    async def _check_exits(self, trade: OpenTrade, price: float) -> None:
        if price <= 0:
            log.warning("[%s] Ignoring invalid price tick %.4f", trade.symbol, price)
            return

        # Update MFE/MAE and time-tracking
        now = time.time()
        r = trade.setup.r_distance
        if r > 0:
            entry = trade.setup.entry_price
            if trade.direction == "long":
                fav = price - entry
                adv = entry - price
            else:
                fav = entry - price
                adv = price - entry
            if fav > 0:
                new_mfe = fav / r
                if new_mfe > trade.mfe_r:
                    trade.mfe_r = new_mfe
                    trade.mfe_peak_ts = now
            if adv > 0:
                trade.mae_r = max(trade.mae_r, adv / r)
                if trade._adverse_since == 0.0:
                    trade._adverse_since = now
            else:
                if trade._adverse_since > 0.0:
                    trade.drawdown_total_s += now - trade._adverse_since
                    trade._adverse_since = 0.0

        # Max hold duration
        if time.time() - trade.opened_at > trade.setup.max_hold_duration_s:
            await self._close_trade(trade, price, "max_hold")
            return

        # Stop loss
        if trade.is_sl_hit(price):
            await self._close_trade(trade, price, "stop_loss")
            return

        # ── Adverse-move early-cut ────────────────────────────────────
        # Most losing trades go offside fast and stay offside.  If the
        # trade has been running long enough, has bled past `early_cut_mae_r`
        # of its risk, AND has shown no meaningful favourable excursion,
        # we cut at current price instead of waiting for full SL.  This
        # converts ~1R losers into ~0.6R losers without affecting winners.
        if not trade.tp1_hit:
            ec = self._cfg.get("exit", {}).get("early_cut", {}) or {}
            if ec.get("enabled", True):
                # Note: cannot use `dict.get(k, default) or default` here
                # because legitimate 0 / 0.0 values are falsy and would
                # silently revert to the historical defaults.
                def _ec_float(key: str, default: float) -> float:
                    val = ec.get(key)
                    return float(val) if val is not None else float(default)

                min_age_s = _ec_float("min_age_s", 600)         # ≥10 min default
                max_age_s = _ec_float("max_age_s", 7200)        # ≤2 h default
                mae_threshold = _ec_float("mae_r_threshold", 0.65)
                mfe_ceiling = _ec_float("mfe_r_ceiling", 0.20)
                age = now - trade.opened_at
                if (
                    min_age_s <= age <= max_age_s
                    and trade.mae_r >= mae_threshold
                    and trade.mfe_r <= mfe_ceiling
                ):
                    await self._close_trade(trade, price, "early_adverse_cut")
                    return

        # Trailing stop (after TP1 hit)
        if trade.trailing_stop is not None:
            if trade.direction == "long" and price <= trade.trailing_stop:
                await self._close_trade(trade, price, "trailing_stop")
                return
            if trade.direction == "short" and price >= trade.trailing_stop:
                await self._close_trade(trade, price, "trailing_stop")
                return

        # TP1 hit → move SL to breakeven + start trailing
        if not trade.tp1_hit and trade.is_tp1_hit(price):
            log.info("[%s] TP1 hit @ %.4f", trade.symbol, price)
            trade.tp1_hit = True
            trade.remaining_contracts *= (1 - trade.setup.tp1_size_pct)
            pnl = abs(trade.setup.tp1 - trade.setup.entry_price) * (trade.setup.size_contracts * trade.setup.tp1_size_pct)
            trade.realized_pnl += pnl
            # Move SL to breakeven
            trade.setup.stop_loss = trade.setup.entry_price
            # Init ATR-based trailing stop
            trail_dist = trade.setup.atr * trade.setup.trailing_atr_multiplier
            if trade.direction == "long":
                trade.trailing_stop = price - trail_dist
            else:
                trade.trailing_stop = price + trail_dist
            if self._on_progress:
                await self._on_progress(trade, "tp1", price)
            self._save_state()

        # Update trailing stop (ratchet only — never widen)
        if trade.tp1_hit and trade.trailing_stop is not None:
            trail_dist = trade.setup.atr * trade.setup.trailing_atr_multiplier
            if trade.direction == "long":
                new_trail = price - trail_dist
                if new_trail > trade.trailing_stop:
                    trade.trailing_stop = new_trail
                    if self._on_progress:
                        await self._on_progress(trade, "trail_update", price)
            else:
                new_trail = price + trail_dist
                if new_trail < trade.trailing_stop:
                    trade.trailing_stop = new_trail
                    if self._on_progress:
                        await self._on_progress(trade, "trail_update", price)

        # TP2 partial exit — close tp2_size_pct of ORIGINAL position, let
        # the configured trail_size_pct trail to TP3.  The exchange-placed
        # reduceOnly TP order is sized as `setup.size_contracts *
        # tp2_size_pct`, so local accounting MUST close the same fraction
        # of the original position to stay in sync with exchange state.
        if not trade.tp1_hit:
            pass  # TP1 not yet hit, nothing to do for TP2 yet
        elif trade.is_tp2_hit(price) and not trade.tp2_hit:
            trade.tp2_hit = True
            contracts_to_close = min(
                trade.remaining_contracts,
                trade.setup.size_contracts * trade.setup.tp2_size_pct,
            )
            pnl = abs(trade.setup.tp2 - trade.setup.entry_price) * contracts_to_close
            trade.realized_pnl += pnl
            trade.remaining_contracts -= contracts_to_close
            log.info(
                "[%s] TP2 hit @ %.4f — closed %.1f%% of original, trailing remainder %.4f contracts",
                trade.symbol, price, trade.setup.tp2_size_pct * 100, trade.remaining_contracts,
            )
            if self._on_progress:
                await self._on_progress(trade, "tp2", price)
            self._save_state()
            if trade.remaining_contracts <= 0.0:
                await self._close_trade(trade, price, "tp2_full")

        # TP3 exit — close all remaining after TP2 partial
        elif trade.tp2_hit and trade.remaining_contracts > 0:
            if trade.direction == "long" and price >= trade.setup.tp3:
                await self._close_trade(trade, price, "tp3")
            elif trade.direction == "short" and price <= trade.setup.tp3:
                await self._close_trade(trade, price, "tp3")

    async def _close_trade(self, trade: OpenTrade, price: float, reason: str) -> None:
        log.info("[%s] Closing %s @ %.4f  reason=%s", trade.symbol, trade.direction, price, reason)
        await self._executor.cancel_all(trade.symbol)
        try:
            await self._executor.close_position_market(
                trade.symbol, trade.direction, trade.remaining_contracts
            )
        except Exception as exc:
            # Exchange may have already closed this (SL/TP filled on exchange).
            # Always clean up local state so the slot is freed.
            log.warning("[%s] close_position_market failed (likely already closed): %s", trade.symbol, exc)

        mult = 1 if trade.direction == "long" else -1
        close_pnl = mult * (price - trade.setup.entry_price) * trade.remaining_contracts

        # Deduct round-trip taker fees from realized PnL
        fee_pct = self._cfg.get("ev_model", {}).get("taker_fee_pct", 0.04) / 100
        fee_cost = trade.setup.size_usd * fee_pct * 2   # entry + exit legs
        total_pnl = trade.realized_pnl + close_pnl - fee_cost

        # Stamp the actual market close price on the trade BEFORE calling the
        # on_close callback so downstream consumers can read trade.exit_price
        # for every exit reason — including early_adverse_cut, max_hold,
        # manual, end_of_data, etc., which previously fell back to entry_price.
        trade.exit_price = float(price)

        self._risk.on_trade_closed(
            total_pnl,
            symbol=trade.setup.symbol,
            direction=trade.setup.direction,
            risk_pct=trade.setup.risk_pct,
        )

        if self._on_close:
            await self._on_close(trade, total_pnl, reason)

        del self._trades[trade.symbol]
        self._save_state()
        log.info("[%s] Trade closed  pnl=$%.2f (fees -$%.3f)  reason=%s",
                 trade.symbol, total_pnl, fee_cost, reason)

    # ------------------------------------------------------------------ #
    #  State persistence — survive bot/watchdog restarts                   #
    # ------------------------------------------------------------------ #

    def _save_state(self) -> None:
        try:
            self.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "trades": [self._serialize_trade(t) for t in self._trades.values()],
                "updated_at": time.time(),
            }
            self.STATE_PATH.write_text(json.dumps(payload))
        except Exception as exc:
            log.warning("Trade state save failed: %s", exc)

    def _load_state(self) -> None:
        if not self.STATE_PATH.exists():
            return
        try:
            data = json.loads(self.STATE_PATH.read_text())
            upgraded = False
            for td in data.get("trades", []):
                trade, was_upgraded = self._deserialize_trade(td)
                if trade is not None:
                    self._trades[trade.symbol] = trade
                    upgraded = upgraded or was_upgraded
            if self._trades:
                # Sync portfolio heat with restored positions so risk guards
                # do not think the book is empty after a restart.
                for trade in self._trades.values():
                    self._risk.on_trade_opened(
                        symbol=trade.setup.symbol,
                        direction=trade.setup.direction,
                        risk_pct=trade.setup.risk_pct,
                    )
                log.info("Restored %d open trades: %s", len(self._trades), list(self._trades.keys()))
                if upgraded:
                    self._save_state()
        except Exception as exc:
            log.warning("Trade state load failed: %s", exc)

    @staticmethod
    def _serialize_trade(trade: "OpenTrade") -> dict:
        return {
            "setup": asdict(trade.setup),
            "entry_order_id": trade.entry_order_id,
            "sl_order_id": trade.sl_order_id,
            "tp1_order_id": trade.tp1_order_id,
            "tp2_order_id": trade.tp2_order_id,
            "opened_at": trade.opened_at,
            "tp1_hit": trade.tp1_hit,
            "tp2_hit": trade.tp2_hit,
            "breakeven_moved": trade.breakeven_moved,
            "trailing_stop": trade.trailing_stop,
            "remaining_contracts": trade.remaining_contracts,
            "realized_pnl": trade.realized_pnl,
            "mfe_r": trade.mfe_r,
            "mae_r": trade.mae_r,
            "mfe_peak_ts": trade.mfe_peak_ts,
            "drawdown_total_s": trade.drawdown_total_s,
            "exit_price": trade.exit_price,
        }

    @staticmethod
    def _deserialize_trade(td: dict) -> tuple["OpenTrade | None", bool]:
        try:
            setup_payload = dict(td["setup"])
            upgraded = False
            defaults = {
                "strategy_sleeve": "neutral",
                "exit_profile": "default",
                "tp1_size_pct": 0.50,
                "tp2_size_pct": 0.30,
                "trail_size_pct": 0.20,
                "breakeven_trigger_r": 1.0,
                "trailing_atr_multiplier": 1.5,
                "max_hold_duration_s": 172800,
                "short_setup_label": "",
                "short_setup_confidence": 0.0,
            }
            for key, value in defaults.items():
                if key not in setup_payload:
                    setup_payload[key] = value
                    upgraded = True
            setup = TradeSetup(**setup_payload)
            return OpenTrade(
                setup=setup,
                entry_order_id=td["entry_order_id"],
                sl_order_id=td["sl_order_id"],
                tp1_order_id=td["tp1_order_id"],
                tp2_order_id=td["tp2_order_id"],
                opened_at=td.get("opened_at", time.time()),
                tp1_hit=td.get("tp1_hit", False),
                tp2_hit=td.get("tp2_hit", False),
                breakeven_moved=td.get("breakeven_moved", False),
                trailing_stop=td.get("trailing_stop"),
                remaining_contracts=td.get("remaining_contracts", setup.size_contracts),
                realized_pnl=td.get("realized_pnl", 0.0),
                mfe_r=td.get("mfe_r", 0.0),
                mae_r=td.get("mae_r", 0.0),
                mfe_peak_ts=td.get("mfe_peak_ts", 0.0),
                drawdown_total_s=td.get("drawdown_total_s", 0.0),
                exit_price=td.get("exit_price", 0.0),
            ), upgraded
        except Exception as exc:
            log.warning("Failed to deserialize trade %s: %s", td.get("setup", {}).get("symbol", "?"), exc)
            return None, False
