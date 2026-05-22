"""
Spot Trade Manager — monitors open Spot positions and executes exits.

Exit logic per trade:
  TP1 (+10%): sell 40% → move SL to breakeven, start trailing stop
  TP2 (+18%): sell 35%
  TP3 (+25%): sell remaining 25%
  SL:         sell all if price drops to stop-loss
  Trailing:   5% trailing stop from peak price (after TP1 hit)
  Timeout:    force-close after max_hold_hours

Early exit triggers:
  - Volume collapses (< 50% of average) after TP1 hit
  - BTC guard flips to HALTED mid-trade (optional — sell if configured)

Open trades are persisted to ``data/spot_open_trades.json`` so that a
process restart does not orphan a live spot position (spot has no
exchange-side stop orders — all stop logic runs in-process).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable, Awaitable

from src.data.spot_client import BinanceSpotClient
from src.execution.spot_executor import SpotExecutor
from src.risk.spot_risk_manager import SpotRiskManager, SpotTradeSetup

log = logging.getLogger(__name__)


@dataclass
class SpotTrade:
    setup: SpotTradeSetup
    entry_order_id: str
    opened_at: float = field(default_factory=time.time)

    # Position tracking
    remaining_qty: float = 0.0     # base currency remaining
    cost_basis: float = 0.0        # average entry price (USDT per unit)
    realized_pnl_usdt: float = 0.0

    # Exit state
    tp1_hit: bool = False
    tp2_hit: bool = False
    trailing_stop: float | None = None
    peak_price: float = 0.0

    @property
    def symbol(self) -> str:
        return self.setup.symbol

    @property
    def entry_price(self) -> float:
        return self.setup.entry_price

    def current_pnl_pct(self, price: float) -> float:
        if not self.entry_price:
            return 0.0
        return (price - self.entry_price) / self.entry_price * 100

    def current_pnl_usdt(self, price: float) -> float:
        return self.realized_pnl_usdt + (price - self.cost_basis) * self.remaining_qty

    def is_sl_hit(self, price: float) -> bool:
        return price <= self.setup.stop_loss

    def is_tp1_hit(self, price: float) -> bool:
        return not self.tp1_hit and price >= self.setup.tp1

    def is_tp2_hit(self, price: float) -> bool:
        return self.tp1_hit and not self.tp2_hit and price >= self.setup.tp2

    def is_tp3_hit(self, price: float) -> bool:
        return self.tp2_hit and price >= self.setup.tp3

    def is_trailing_hit(self, price: float) -> bool:
        return self.trailing_stop is not None and price <= self.trailing_stop

    def is_timed_out(self, max_hold_hours: float) -> bool:
        return (time.time() - self.opened_at) > max_hold_hours * 3600


CloseCallback = Callable[["SpotTrade", float, str], Awaitable[None]]


class SpotTradeManager:
    STATE_PATH = Path("data/spot_open_trades.json")

    def __init__(
        self,
        client: BinanceSpotClient,
        executor: SpotExecutor,
        risk: SpotRiskManager,
        cfg: dict,
        on_close: CloseCallback | None = None,
    ):
        self._client = client
        self._executor = executor
        self._risk = risk
        self._cfg = cfg
        self._exit = cfg["exit"]
        self._trades: dict[str, SpotTrade] = {}
        self._on_close = on_close
        self._max_hold_hours: float = self._exit.get("max_hold_hours", 120)
        self._trail_pct: float = self._exit.get("trailing_stop_pct", 0.05)
        self._load_state()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    async def open(self, setup: SpotTradeSetup) -> SpotTrade | None:
        can, reason = self._risk.can_open_trade()
        if not can:
            log.warning("Risk guard blocked trade on %s: %s", setup.symbol, reason)
            return None

        if setup.symbol in self._trades:
            log.warning("Already in trade for %s — skipping", setup.symbol)
            return None

        try:
            order = await self._executor.open_position(setup)
        except Exception as exc:
            log.error("Failed to open position %s: %s", setup.symbol, exc)
            return None

        fill_price = float(order.get("price", setup.entry_price) or setup.entry_price)
        fill_qty = float(order.get("amount", setup.base_qty) or setup.base_qty)

        trade = SpotTrade(
            setup=setup,
            entry_order_id=order["id"],
            remaining_qty=fill_qty,
            cost_basis=fill_price,
            peak_price=fill_price,
        )
        self._trades[setup.symbol] = trade
        self._risk.on_trade_opened()
        self._save_state()

        log.info(
            "OPENED %s  price=%.6f  qty=%.6f ($%.2f)  sl=%.6f  tp1=%.6f  strategy=%s",
            setup.symbol, fill_price, fill_qty, fill_qty * fill_price,
            setup.stop_loss, setup.tp1, setup.entry_strategy,
        )
        return trade

    async def monitor_all(self, price_map: dict[str, float]) -> None:
        """Check exits for every open trade. Call on each price tick."""
        for symbol, trade in list(self._trades.items()):
            price = price_map.get(symbol)
            if price is None or price <= 0:
                continue
            # Update peak price for trailing stop
            if price > trade.peak_price:
                trade.peak_price = price
            await self._check_exits(trade, price)

    async def close_all(self, reason: str = "manual") -> None:
        for symbol, trade in list(self._trades.items()):
            current_price = trade.peak_price  # best available estimate
            await self._close_trade(trade, current_price, reason)

    @property
    def open_symbols(self) -> list[str]:
        return list(self._trades.keys())

    @property
    def open_count(self) -> int:
        return len(self._trades)

    def get_trade(self, symbol: str) -> SpotTrade | None:
        return self._trades.get(symbol)

    # ------------------------------------------------------------------ #
    #  Exit logic                                                         #
    # ------------------------------------------------------------------ #

    async def _check_exits(self, trade: SpotTrade, price: float) -> None:
        # ── Timeout ───────────────────────────────────────────────────
        if trade.is_timed_out(self._max_hold_hours):
            await self._close_trade(trade, price, "timeout")
            return

        # ── Stop-loss ─────────────────────────────────────────────────
        if trade.is_sl_hit(price):
            await self._close_trade(trade, price, "stop_loss")
            return

        # ── Trailing stop (active after TP1) ──────────────────────────
        if trade.is_trailing_hit(price):
            await self._close_trade(trade, price, "trailing_stop")
            return

        # ── TP1 ───────────────────────────────────────────────────────
        if trade.is_tp1_hit(price):
            await self._hit_tp1(trade, price)

        # ── TP2 ───────────────────────────────────────────────────────
        if trade.is_tp2_hit(price):
            await self._hit_tp2(trade, price)

        # ── TP3 ───────────────────────────────────────────────────────
        if trade.is_tp3_hit(price):
            await self._close_trade(trade, price, "tp3")
            return

        # ── Update trailing stop ──────────────────────────────────────
        if trade.tp1_hit and trade.trailing_stop is not None:
            new_trail = price * (1 - self._trail_pct)
            if new_trail > trade.trailing_stop:
                trade.trailing_stop = new_trail

    async def _hit_tp1(self, trade: SpotTrade, price: float) -> None:
        tp1_size_pct = self._exit.get("tp1_size_pct", 0.40)
        sell_qty = trade.remaining_qty * tp1_size_pct

        try:
            await self._executor.partial_sell(trade.symbol, sell_qty, "tp1")
        except Exception as exc:
            log.error("TP1 partial sell failed %s: %s", trade.symbol, exc)
            return

        pnl = (price - trade.cost_basis) * sell_qty
        trade.realized_pnl_usdt += pnl
        trade.remaining_qty -= sell_qty
        trade.remaining_qty = max(0.0, trade.remaining_qty)
        trade.tp1_hit = True

        # Move SL to breakeven
        trade.setup.stop_loss = trade.entry_price

        # Start trailing stop
        trade.trailing_stop = price * (1 - self._trail_pct)
        self._save_state()

        log.info(
            "TP1 hit %s @ %.6f  sold=%.6f  pnl=+$%.2f  new_sl=%.6f (breakeven)",
            trade.symbol, price, sell_qty, pnl, trade.setup.stop_loss,
        )

    async def _hit_tp2(self, trade: SpotTrade, price: float) -> None:
        tp2_size_pct = self._exit.get("tp2_size_pct", 0.35)
        # As fraction of REMAINING qty (not original)
        sell_qty = trade.remaining_qty * tp2_size_pct

        try:
            await self._executor.partial_sell(trade.symbol, sell_qty, "tp2")
        except Exception as exc:
            log.error("TP2 partial sell failed %s: %s", trade.symbol, exc)
            return

        pnl = (price - trade.cost_basis) * sell_qty
        trade.realized_pnl_usdt += pnl
        trade.remaining_qty -= sell_qty
        trade.remaining_qty = max(0.0, trade.remaining_qty)
        trade.tp2_hit = True
        self._save_state()

        log.info(
            "TP2 hit %s @ %.6f  sold=%.6f  pnl=+$%.2f  remaining=%.6f",
            trade.symbol, price, sell_qty, pnl, trade.remaining_qty,
        )

    async def _close_trade(self, trade: SpotTrade, price: float, reason: str) -> None:
        """Sell all remaining position and record the close."""
        if trade.remaining_qty <= 0:
            del self._trades[trade.symbol]
            return

        try:
            await self._executor.close_position(trade.symbol, trade.remaining_qty, reason)
        except Exception as exc:
            log.error("Close failed %s: %s — removing from tracker anyway", trade.symbol, exc)

        close_pnl = (price - trade.cost_basis) * trade.remaining_qty
        total_pnl = trade.realized_pnl_usdt + close_pnl
        total_invested = trade.setup.size_usdt
        pnl_pct = total_pnl / total_invested * 100 if total_invested > 0 else 0.0

        self._risk.on_trade_closed(pnl_pct)

        if self._on_close:
            await self._on_close(trade, total_pnl, reason)

        del self._trades[trade.symbol]
        self._save_state()

        emoji = "✅" if total_pnl > 0 else "❌"
        log.info(
            "%s CLOSED %s @ %.6f  total_pnl=[%s$%.2f] (%.1f%%)  reason=%s",
            emoji, trade.symbol, price,
            "+" if total_pnl >= 0 else "", total_pnl, pnl_pct, reason,
        )

    # ------------------------------------------------------------------ #
    #  State persistence — survive bot/watchdog restarts                 #
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
            log.warning("Spot trade state save failed: %s", exc)

    def _load_state(self) -> None:
        if not self.STATE_PATH.exists():
            return
        try:
            data = json.loads(self.STATE_PATH.read_text())
            for td in data.get("trades", []):
                trade = self._deserialize_trade(td)
                if trade is not None:
                    self._trades[trade.symbol] = trade
                    self._risk.on_trade_opened()
            if self._trades:
                log.info(
                    "Restored %d open spot trades: %s",
                    len(self._trades), list(self._trades.keys()),
                )
        except Exception as exc:
            log.warning("Spot trade state load failed: %s", exc)

    @staticmethod
    def _serialize_trade(trade: "SpotTrade") -> dict:
        return {
            "setup": asdict(trade.setup),
            "entry_order_id": trade.entry_order_id,
            "opened_at": trade.opened_at,
            "remaining_qty": trade.remaining_qty,
            "cost_basis": trade.cost_basis,
            "realized_pnl_usdt": trade.realized_pnl_usdt,
            "tp1_hit": trade.tp1_hit,
            "tp2_hit": trade.tp2_hit,
            "trailing_stop": trade.trailing_stop,
            "peak_price": trade.peak_price,
        }

    @staticmethod
    def _deserialize_trade(td: dict) -> "SpotTrade | None":
        try:
            allowed = {f.name for f in fields(SpotTradeSetup)}
            setup_payload = {k: v for k, v in (td.get("setup") or {}).items() if k in allowed}
            setup = SpotTradeSetup(**setup_payload)
            return SpotTrade(
                setup=setup,
                entry_order_id=td.get("entry_order_id", ""),
                opened_at=float(td.get("opened_at", time.time())),
                remaining_qty=float(td.get("remaining_qty", setup.base_qty)),
                cost_basis=float(td.get("cost_basis", setup.entry_price)),
                realized_pnl_usdt=float(td.get("realized_pnl_usdt", 0.0)),
                tp1_hit=bool(td.get("tp1_hit", False)),
                tp2_hit=bool(td.get("tp2_hit", False)),
                trailing_stop=td.get("trailing_stop"),
                peak_price=float(td.get("peak_price", setup.entry_price)),
            )
        except Exception as exc:
            log.warning(
                "Failed to deserialize spot trade %s: %s",
                (td.get("setup") or {}).get("symbol", "?"), exc,
            )
            return None
