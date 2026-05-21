"""
Telegram notification service.

Sends trade alerts, heartbeat summaries, and circuit-breaker warnings
to a configured Telegram chat via the Bot API.

Uses httpx for async HTTP — no external Telegram library required.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime
from io import BytesIO
from typing import TYPE_CHECKING

import httpx
from src.session_clock import active_market_session_label

if TYPE_CHECKING:
    from src.risk.risk_manager import TradeSetup
    from src.scoring.scorer import SignalBreakdown

log = logging.getLogger(__name__)

_BASE = "https://api.telegram.org/bot{token}/sendMessage"
_PHOTO_BASE = "https://api.telegram.org/bot{token}/sendPhoto"
_MAX_MESSAGE_LEN = 4000


class TelegramNotifier:
    def __init__(self, cfg: dict):
        tg = cfg.get("telegram", {})
        self._token: str = (os.getenv("TELEGRAM_TOKEN") or tg.get("token", "")).strip()
        self._chat_id: str = str(os.getenv("TELEGRAM_CHAT_ID") or tg.get("chat_id", "")).strip()
        self._enabled: bool = bool(self._token and self._chat_id)
        self._url = _BASE.format(token=self._token)
        self._photo_url = _PHOTO_BASE.format(token=self._token)

        if self._enabled:
            log.info("Telegram notifier enabled → chat_id=%s", self._chat_id)
        else:
            log.warning("Telegram notifier disabled — token/chat_id not configured")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            if value in ("", None):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        try:
            if value in ("", None):
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    async def trade_opened(
        self,
        bd: "SignalBreakdown",
        setup: "TradeSetup",
        jim_notes: list[str] | None = None,
    ) -> None:
        if not self._enabled:
            return
        direction = setup.direction.upper()
        arrow = "📈" if setup.direction == "long" else "📉"
        sl_pct  = abs(setup.stop_loss - setup.entry_price) / setup.entry_price * 100
        tp1_pct = abs(setup.tp1 - setup.entry_price) / setup.entry_price * 100
        tp2_pct = abs(setup.tp2 - setup.entry_price) / setup.entry_price * 100

        ev_line = ""
        if bd.ev_result:
            ev = bd.ev_result
            ev_line = f"\n📊 P(win): `{ev.p_win:.1%}`  |  Expectancy: `{ev.ev_net_pct:+.3f}%`"

        sm_line = ""
        if bd.smart_money:
            sm_line = f"\n🧠 Smart Money: `{bd.smart_money.phase.value}`"

        edge_line = ""
        edge = getattr(bd, "edge_result", None)
        if edge:
            edge_line = (
                f"\n🛡️ *EDGE STATUS:* `{edge.status}`"
                f"\n   SCORE: `{edge.score:.0f}/100`  |  CONF: `{edge.confidence:.2f}`"
                f"\n   REGIME COVERAGE: `{edge.regime_coverage}`"
                f"\n   DECISION: `{edge.action}`  |  SIZING: `{edge.size_mult:.2f}x`"
            )

        jim_line = ""
        if jim_notes:
            jim_line = f"\n💭 Jim's take: _{', '.join(jim_notes)}_"

        risk_usd = setup.r_distance / setup.entry_price * setup.size_usd if setup.entry_price > 0 else 0.0
        sym = bd.symbol.replace("_", " ")
        msg = (
            f"{arrow} *Boss, I just pulled the trigger!*\n"
            f"══════════════════════\n"
            f"🎰 *{sym}* — going *{direction}*\n"
            f"Score: `{bd.total_score:.1f}/100`  |  Regime: `{bd.regime.label}`\n"
            f"Strategy: `{bd.strategy_sleeve}`  |  Exit: `{setup.exit_profile}`\n"
            f"──────────────────────\n"
            f"🎯 Entry:  `${setup.entry_price:,.4f}`\n"
            f"🛑 Stop:   `${setup.stop_loss:,.4f}`  (-{sl_pct:.2f}%)\n"
            f"🥇 TP1:    `${setup.tp1:,.4f}`  (+{tp1_pct:.2f}%)  — {setup.tp1_size_pct:.0%} out\n"
            f"🥈 TP2:    `${setup.tp2:,.4f}`  (+{tp2_pct:.2f}%)  — {setup.tp2_size_pct:.0%} out\n"
            f"🏆 TP3:    `${setup.tp3:,.4f}`  — trailing {setup.trail_size_pct:.0%}\n"
            f"──────────────────────\n"
            f"💼 Risk: `{setup.risk_pct:.2f}%`  (${risk_usd:.2f} at risk)  |  Position: `${setup.size_usd:,.2f}`"
            f"{ev_line}"
            f"{sm_line}"
            f"{edge_line}"
            f"{jim_line}\n"
            f"🧮 _Jim placed this trade. — The model never sleeps._ 🥷"
        )
        await self._send(msg)

    async def trade_closed(
        self,
        symbol: str,
        direction: str,
        pnl_usd: float,
        pnl_pct: float,
        reason: str,
        entry: float,
        exit_price: float,
    ) -> None:
        if not self._enabled:
            return
        sym = symbol.replace("_", " ")
        sign = "+" if pnl_usd >= 0 else ""

        if pnl_usd > 0:
            if pnl_pct > 2.0:
                headline = f"💰 *Boss, we CRUSHED it on {sym}!*"
                closer   = "📦 Profits locked. Compounding in progress. 😎"
            else:
                headline = f"✅ *Boss, trade closed green on {sym}.*"
                closer   = "📦 Small win is still a win. Every dollar counts. 💪"
        elif pnl_usd == 0:
            headline = f"😐 *Boss, we broke even on {sym}.*"
            closer   = "🤷 Could be worse. Risk was controlled."
        else:
            if reason == "stop_loss":
                headline = f"🛑 *Boss, stop loss hit on {sym}. We cut it clean.*"
                closer   = "🧠 Protecting capital is the job. We live to trade another day."
            elif "trailing" in reason:
                headline = f"📉 *Boss, trailing stop caught us on {sym}.*"
                closer   = "⚖️ We took profit on the way up. No complaints."
            else:
                headline = f"❌ *Boss, this one didn't work out — {sym}.*"
                closer   = "📋 Logging it. Jim's model is learning from this. Won't repeat. 🧮"

        reason_clean = reason.replace("_", " ")
        msg = (
            f"{headline}\n"
            f"══════════════════════\n"
            f"Direction: `{direction.upper()}`  |  Exit: `{reason_clean}`\n"
            f"Entry: `${entry:,.4f}` → Exit: `${exit_price:,.4f}`\n"
            f"──────────────────────\n"
            f"P&L: *{sign}${pnl_usd:.2f}* ({sign}{pnl_pct:.2f}% of equity)\n"
            f"──────────────────────\n"
            f"{closer}"
        )
        await self._send(msg)

    async def trade_progress(
        self,
        symbol: str,
        direction: str,
        event: str,
        price: float,
        remaining_contracts: float,
        trailing_stop: float | None = None,
    ) -> None:
        if not self._enabled:
            return
        sym = symbol.replace("_", " ")
        if event == "tp1":
            headline = f"🥇 *TP1 hit on {sym}*"
            detail = "Partial profit taken. Stop moved to breakeven and trailing is active."
        elif event == "tp2":
            headline = f"🥈 *TP2 hit on {sym}*"
            detail = "More size closed. Letting the remainder run."
        elif event == "trail_update":
            headline = f"🧭 *Trailing stop updated on {sym}*"
            detail = "Open position remains active with a tighter stop."
        else:
            headline = f"📌 *Position update on {sym}*"
            detail = "Execution state changed."
        trail_line = f"\nTrailing stop: `${trailing_stop:,.4f}`" if trailing_stop is not None else ""
        await self._send(
            f"{headline}\n"
            f"══════════════════════\n"
            f"Direction: `{direction.upper()}`  |  Event: `{event}`\n"
            f"Price: `${price:,.4f}`\n"
            f"Remaining contracts: `{remaining_contracts:.6f}`"
            f"{trail_line}\n"
            f"──────────────────────\n"
            f"{detail}"
        )

    async def circuit_breaker(self, reason: str, equity: float, drawdown_pct: float) -> None:
        if not self._enabled:
            return
        reason_clean = reason.replace("_", " ")
        await self._send(
            f"🚨 *JIM SIMONS — Circuit Breaker Triggered*\n"
            f"══════════════════════\n"
            f"Reason: `{reason_clean}`\n"
            f"Equity: `${equity:,.2f}`  |  Drawdown: `{drawdown_pct:.1f}%`\n"
            f"──────────────────────\n"
            f"*All trading halted. Capital protected.*\n"
            f"_The model never sleeps, Boss. — Jim_ 🧮🥷"
        )

    async def heartbeat(
        self,
        equity: float,
        drawdown_pct: float,
        daily_pnl_pct: float,
        open_trades: int,
        top_signals: list[str],
        floating_positions: list[dict] | None = None,
        open_positions: list[dict] | None = None,
    ) -> None:
        if not self._enabled:
            return
        open_trades = self._safe_int(open_trades)
        drawdown_pct = self._safe_float(drawdown_pct)
        daily_pnl_pct = self._safe_float(daily_pnl_pct)
        status = "🟢 Active" if open_trades > 0 else "⏳ Scanning"
        top_line = ", ".join(top_signals[:3]) if top_signals else "none"
        floating_total = sum(self._safe_float(p.get("pnl_usd", 0.0)) for p in floating_positions or [])
        effective_equity = equity + floating_total
        msg = (
            f"🧮 *Heartbeat*\n"
            f"{status} | Equity *${effective_equity:,.2f}* | Open `{open_trades}`\n"
            f"P&L `{daily_pnl_pct:+.2f}%` | DD `{drawdown_pct:.1f}%`\n"
            f"Top: {top_line}"
        )
        await self._send(msg)

    async def cycle_report(
        self,
        breakdowns: list["SignalBreakdown"],
        equity: float,
        drawdown_pct: float,
        daily_pnl_pct: float,
        open_trades: int,
        cycle_num: int,
        ml_ready: bool = False,
        ml_accuracy: float = 0.0,
        paper_trades: int = 0,
        readiness_passed: bool = False,
        consecutive_wins: int = 0,
        consecutive_losses: int = 0,
        fm_scale: float = 1.0,
        jim_status: dict | None = None,
        floating_positions: list[dict] | None = None,
        open_positions: list[dict] | None = None,
        closed_positions: list | None = None,
        win_rate: float = 0.0,
        sharpe: float = 0.0,
        total_trades: int = 0,
        trade_log: list | None = None,
        equity_curve: list[float] | None = None,
        starting_equity: float = 0.0,
        regime_thresholds: dict | None = None,
        jim_bonus: float = 0.0,
        bootstrap_audit: list[str] | None = None,
    ) -> None:
        if not self._enabled:
            return
        import datetime
        now = datetime.datetime.now(datetime.UTC).strftime("%H:%M:%S UTC")
        active_session = self._active_session_label()
        equity = self._safe_float(equity)
        drawdown_pct = self._safe_float(drawdown_pct)
        daily_pnl_pct = self._safe_float(daily_pnl_pct)
        open_trades = self._safe_int(open_trades)
        cycle_num = self._safe_int(cycle_num)
        ml_accuracy = self._safe_float(ml_accuracy)
        paper_trades = self._safe_int(paper_trades)
        consecutive_wins = self._safe_int(consecutive_wins)
        consecutive_losses = self._safe_int(consecutive_losses)
        fm_scale = self._safe_float(fm_scale, 1.0)
        win_rate = self._safe_float(win_rate)
        sharpe = self._safe_float(sharpe)
        total_trades = self._safe_int(total_trades)
        starting_equity = self._safe_float(starting_equity)

        pnl_emoji = "📈" if daily_pnl_pct >= 0 else "📉"
        dd_emoji  = "🔴" if drawdown_pct > 8 else "🟡" if drawdown_pct > 3 else "🟢"
        sharpe_str = f"`{sharpe:.2f}`" if total_trades >= 5 and sharpe != 0.0 else "—"
        wr_str = f"`{win_rate:.1%}`" if total_trades > 0 else "`0.0%`"

        # FM scale line
        if fm_scale >= 1.20:
            fm_line = f"⚖️ FM Scale: `{fm_scale:.2f}x` — sizing up, Boss. 🚀"
        elif fm_scale >= 1.05:
            fm_line = f"⚖️ FM Scale: `{fm_scale:.2f}x` — slightly up."
        elif fm_scale <= 0.75:
            fm_line = f"⚖️ FM Scale: `{fm_scale:.2f}x` — protecting capital. 🛡️"
        elif fm_scale <= 0.92:
            fm_line = f"⚖️ FM Scale: `{fm_scale:.2f}x` — cautious."
        else:
            fm_line = f"⚖️ FM Scale: `{fm_scale:.2f}x` — normal size."

        # Streak line
        if consecutive_wins >= 4:
            streak_line = f"🔥 *{consecutive_wins} wins in a row!* Hot streak."
        elif consecutive_wins >= 2:
            streak_line = f"✅ {consecutive_wins} consecutive wins. Good momentum."
        elif consecutive_losses >= 2:
            streak_line = f"🛡️ {consecutive_losses} losses in a row — size reduced automatically."
        elif consecutive_losses == 1:
            streak_line = f"⚡ Last trade was a loss — staying cautious."
        else:
            streak_line = f"➡️ Neutral — waiting for the next setup."

        # Jim ML block
        jim_block = None
        if jim_status is not None:
            if jim_status.get("ready"):
                acc_pct = jim_status.get("accuracy", 0) * 100
                trend = {"improving": "📈", "declining": "📉", "stable": "➡️"}.get(
                    jim_status.get("accuracy_trend", "stable"), "➡️"
                )
                thresh_pct = jim_status.get("threshold", 0.55) * 100
                kelly = jim_status.get("kelly_scale", 1.0)
                top_f = jim_status.get("top_feature") or "warming up"
                regime_summary = jim_status.get("regime_summary", {})
                regime_str = "  ".join(
                    f"`{k[:5]}:{v['win_rate']:.0%}`"
                    for k, v in regime_summary.items() if v.get("total", 0) >= 3
                ) or "warming up (need more samples)"
                jim_block = (
                    f"🧮 *Jim's Model:* acc `{acc_pct:.1f}%` {trend}  "
                    f"gate `{thresh_pct:.0f}%`  Kelly `{kelly:.2f}x`\n"
                    f"  Top feature: `{top_f}`  |  By regime: {regime_str}"
                )
            else:
                needed = 30 - (jim_status.get("trained_on", 0) or 0)
                jim_block = f"🧮 *Jim's Model:* _warming up — {needed} more trades to activate_"

        if jim_bonus >= 0.30:
            bonus_line = f"🏆 Jim's Bonus: `+{jim_bonus:.2f}x` — *Elite tier* (cap `{2.0+jim_bonus:.2f}x`)"
        elif jim_bonus >= 0.15:
            bonus_line = f"🎖️ Jim's Bonus: `+{jim_bonus:.2f}x` — on a roll (cap `{2.0+jim_bonus:.2f}x`)"
        elif jim_bonus > 0:
            bonus_line = f"✨ Jim's Bonus: `+{jim_bonus:.2f}x` — building track record"
        else:
            bonus_line = f"Jim's Bonus: `+0.00x` — earn it by winning"

        floating_total = sum(
            self._safe_float(p.get("pnl_usd", 0.0))
            for p in (floating_positions or [])
        )
        balance = self._safe_float(equity)
        effective_equity = balance + floating_total
        total_pnl_usd = effective_equity - starting_equity if starting_equity > 0 else 0.0
        total_pnl_pct = (total_pnl_usd / starting_equity * 100) if starting_equity > 0 else 0.0
        total_sign = "+" if total_pnl_usd >= 0 else ""
        total_emoji = "📈" if total_pnl_usd >= 0 else "📉"
        perf = self._trade_performance_metrics(
            trade_log or [],
            equity_curve or [],
            starting_equity=starting_equity,
            equity=effective_equity,
        )
        perf_lines = [
            f"PF `{perf['profit_factor']:.2f}`  WR `{perf['win_rate']:.1%}`  Z-Score `{perf['z_score']:+.2f}`  GHPR `{perf['ghpr_pct']:+.2f}%/trade`",
            f"CAGR `{perf['cagr_pct']:.1f}%`  MAR `{perf['mar']:.1f}`  Sharpe `{perf['sharpe']:.2f}`  Sortino `{perf['sortino']:.2f}`",
            f"Avg W `{perf['avg_win_pct']:+.2f}%`  Avg L `{perf['avg_loss_pct']:+.2f}%`  Avg W/L `{perf['avg_wl']:.2f}`  Exp `{perf['expectancy_pct']:+.2f}%`",
            f"Recovery `{perf['recovery_factor']:.2f}`  MaxDD `{perf['max_dd_pct']:.1f}%`  Trades `{perf['trades']}`",
        ]

        lines = [
            f"🧮 *JIM SIMONS — FUND MANAGER REPORT*  #{cycle_num}",
            f"`{now}`",
            f"🌍 Session: *{active_session}*",
            f"══════════════════════",
            f"🏦 Balance: *${balance:,.2f} USDT*",
            f"💰 Equity: *${effective_equity:,.2f} USDT*",
            f"{total_emoji} Total PnL: *{total_sign}${total_pnl_usd:.2f} ({total_sign}{total_pnl_pct:.2f}%)*",
            f"{pnl_emoji} Daily P&L: *{daily_pnl_pct:+.2f}%*  {dd_emoji} Drawdown: *{drawdown_pct:.1f}%*",
            f"📊 Win Rate: {wr_str} ({total_trades} trades)  📐 Sharpe: {sharpe_str}",
            *perf_lines,
            f"──────────────────────",
            bonus_line,
            f"{streak_line}",
            f"{fm_line}",
        ]
        if jim_block:
            lines.append(jim_block)

        # ── Open Positions ────────────────────────────────────────────
        lines.append(f"──────────────────────")
        lines.append(f"🗂️ *Open Positions:*")
        position_rows = floating_positions or open_positions or []
        if position_rows:
            total_float = sum(
                self._safe_float(p.get("pnl_usd", 0.0))
                for p in position_rows
            )
            for p in position_rows:
                direction = str(p.get("direction", "") or "").lower()
                d_emoji = "📈" if direction == "long" else "📉" if direction == "short" else "📍"
                pnl_usd = self._safe_float(p.get("pnl_usd", 0.0))
                pnl_pct = self._safe_float(p.get("pnl_pct", 0.0))
                sign = "+" if pnl_usd >= 0 else ""
                color = "🟢" if pnl_usd >= 0 else "🔴"
                elapsed_s = self._safe_float(p.get("elapsed_s", time.time() - p.get("opened_at", time.time())))
                elapsed_m = int(elapsed_s // 60)
                sym = str(p.get("symbol", "?")).split(":")[0]
                entry = self._safe_float(p.get("entry", 0.0))
                current = self._safe_float(p.get("current", entry))
                sl = self._safe_float(p.get("sl", 0.0))
                tp1 = self._safe_float(p.get("tp1", 0.0))
                risk_pct = self._safe_float(p.get("risk_pct", 0.0))
                risk_usd = self._safe_float(p.get("risk_usd", 0.0))
                size_usd = self._safe_float(p.get("size_usd", 0.0))
                strategy = p.get("strategy_sleeve", "neutral")
                exit_profile = p.get("exit_profile", "default")
                lines.append(
                    f"{color} {d_emoji} `{sym}` {direction.upper() or 'UNKNOWN'}\n"
                    f"  ${entry:,.4f} → ${current:,.4f}  "
                    f"({sign}{pnl_pct:+.2f}%)  {sign}${pnl_usd:.2f}  |  {elapsed_m}m\n"
                    f"  SL `${sl:,.4f}`  TP1 `${tp1:,.4f}`  Size `${size_usd:,.2f}`\n"
                    f"  Strategy `{strategy}`  Exit `{exit_profile}`  Risk `{risk_pct:.2f}%` (${risk_usd:.2f})"
                )
            float_sign = "+" if total_float >= 0 else ""
            float_color = "🟢" if total_float >= 0 else "🔴"
            lines.append(f"{float_color} Floating total: *{float_sign}${total_float:.2f} USDT*")
        else:
            lines.append(f"⏳ No open positions — scanning the market...")

        # ── Closed Positions ─────────────────────────────────────────
        lines.append(f"──────────────────────")
        lines.append(f"📋 *Closed Positions (last 5):*")
        recent = list(closed_positions or [])[-5:]
        if recent:
            for t in reversed(recent):
                sign = "+" if t.pnl_usd >= 0 else ""
                color = "🟢" if t.pnl_usd >= 0 else "🔴"
                d_emoji = "📈" if t.direction == "long" else "📉"
                sym = t.symbol.split(":")[0].replace("/", "")
                reason_clean = t.reason.replace("_", " ")
                lines.append(
                    f"{color} {d_emoji} `{sym}` {t.direction.upper()} | "
                    f"{sign}${t.pnl_usd:.2f} ({sign}{t.pnl_pct:.2f}%) | {reason_clean}"
                )
        else:
            lines.append(f"_No closed trades yet._")

        if bootstrap_audit:
            lines.append(f"──────────────────────")
            lines.append(f"🧪 *Bootstrap Audit:*")
            lines.extend(bootstrap_audit)

        # ── Scan Results ─────────────────────────────────────────────
        _rt = regime_thresholds or {}
        _regime_key_map = {
            "trending_expansion": "trending_expansion",
            "accumulation_compression": "accumulation_compression",
        }
        def _threshold_for_bd(b) -> int:
            key = b.regime.value if b.regime else ""
            return int(_rt.get(key, _rt.get("default", 65)))

        lines.append(f"──────────────────────")
        lines.append(f"🔍 *Scan Results* (top {min(len(breakdowns),5)} of {len(breakdowns)} scored | score/threshold):")
        if not breakdowns:
            lines.append(f"_No symbols scored this cycle — scanner returned no eligible setups._")
            lines.append("Legend: `R` regime, `S` smart-money alignment, `X` net expectancy after fees/cost.")
            lines.append(f"──────────────────────")
            lines.append(f"🗺️ *Roadmap:*")
            if readiness_passed:
                lines.append(f"🚀 *ALL CRITERIA PASSED — Switching to LIVE!*")
            elif paper_trades >= 20 and ml_ready:
                lines.append(f"🤖 *Phase 2 — ML Active*  acc `{ml_accuracy:.1%}`")
                lines.append(f"   Monitoring 7 live-readiness criteria...")
            else:
                done = min(paper_trades, 20)
                bar_filled = int(done / 20 * 10)
                bar = "▓" * bar_filled + "░" * (10 - bar_filled)
                lines.append(
                    f"📍 Phase 1 — Bootstrap  [{bar}]  `{done}/20` paper trades\n"
                    f"   Building trade history for ML training..."
                )
            lines.append(f"══════════════════════")
            lines.append(f"_The model never sleeps, Boss. — Jim_ 🧮🥷")
            await self._send("\n".join(lines))
            return
        fire_count = 0
        for b in breakdowns[:5]:
            gates = f"R{'✓' if b.regime_ok else '✗'} S{'✓' if b.smart_money_ok else '✗'} X{'✓' if b.ev_ok else '✗'}"
            regime_short = {
                "trending_expansion": "Trend↑",
                "accumulation_compression": "Accum",
                "distribution": "Dist",
                "chaos": "Chaos",
            }.get(b.regime.value if b.regime else "", "?")
            d_arrow = "📈" if b.direction == "long" else "📉"
            ev_str = f"Exp:`{b.ev_result.ev_net_pct:+.2f}%`" if b.ev_result else ""
            sym = b.symbol.split(":")[0].replace("/", "")
            thr = _threshold_for_bd(b)
            score_str = f"`{b.total_score:.1f}/{thr}`"
            if b.all_gates_passed:
                box = "🟩"
                flag = "  🔥"
                fire_count += 1
            elif b.total_score >= thr - 5:
                box = "🟨"
                flag = ""
            else:
                box = "🟥"
                flag = ""
            lines.append(
                f"{box} {d_arrow} `{sym}` {score_str} | {regime_short} | {gates}{flag}  {ev_str}"
            )
        if fire_count == 0:
            lines.append(f"_No signals above threshold yet — waiting for setup._")
        lines.append("Legend: `R` regime, `S` smart-money alignment, `X` net expectancy after fees/cost.")

        # ── Roadmap ───────────────────────────────────────────────────
        lines.append(f"──────────────────────")
        lines.append(f"🗺️ *Roadmap:*")
        if readiness_passed:
            lines.append(f"🚀 *ALL CRITERIA PASSED — Switching to LIVE!*")
        elif paper_trades >= 20 and ml_ready:
            lines.append(f"🤖 *Phase 2 — ML Active*  acc `{ml_accuracy:.1%}`")
            lines.append(f"   Monitoring 7 live-readiness criteria...")
        else:
            done = min(paper_trades, 20)
            bar_filled = int(done / 20 * 10)
            bar = "▓" * bar_filled + "░" * (10 - bar_filled)
            lines.append(
                f"📍 Phase 1 — Bootstrap  [{bar}]  `{done}/20` paper trades\n"
                f"   Building trade history for ML training..."
            )

        lines.append(f"══════════════════════")
        lines.append(f"_The model never sleeps, Boss. — Jim_ 🧮🥷")
        await self._send("\n".join(lines))

    async def bootstrap_audit_report(
        self,
        report_lines: list[str],
        *,
        cycle_num: int = 0,
        interval_hours: float = 12.0,
    ) -> None:
        """Separate bootstrap audit report, sent on its own cadence."""
        if not self._enabled or not report_lines:
            return
        now = datetime.now(UTC).strftime("%H:%M:%S UTC")
        header = "🧪 *BOOTSTRAP AUDIT REPORT*"
        if cycle_num > 0:
            header += f"  #{cycle_num}"
        cadence = f"`Cadence: {interval_hours:.0f}h`"
        lines = [
            header,
            f"`{now}`",
            cadence,
            f"`Session: {self._active_session_label()}`",
            "══════════════════════",
            *report_lines,
            "══════════════════════",
            "_Separate from Jim Simons report — bootstrap audit only._",
        ]
        await self._send("\n".join(lines))

    async def startup(self, mode: str, equity: float) -> None:
        if not self._enabled:
            return
        mode_line = "📋 Paper mode — training run" if mode == "paper" else "🟢 *LIVE MODE — real money active*"
        await self._send(
            f"🥷 *Ninja Trader is online, Boss.*\n"
            f"══════════════════════\n"
            f"Mode: `{mode.upper()}`  |  Capital: `${equity:,.2f}`\n"
            f"{mode_line}\n"
            f"──────────────────────\n"
            f"Scanning Binance Futures. Hunting for edge.\n"
            f"🧮 *Jim Simons* is watching every signal.\n"
            f"_The model never sleeps, Boss. — Jim_ 🧮🥷"
        )

    async def shutdown(
        self,
        *,
        mode: str,
        equity: float,
        open_trades: int,
        close_positions: bool,
    ) -> None:
        if not self._enabled:
            return
        mode_label = "paper" if mode == "paper" else "live"
        status_line = "Closing open positions before exit." if close_positions else "Open positions preserved on shutdown."
        await self._send(
            f"🛑 *Ninja Trader is shutting down*\n"
            f"══════════════════════\n"
            f"Mode: `{mode_label.upper()}`  |  Equity: `${equity:,.2f}`\n"
            f"Open positions at stop: `{open_trades}`\n"
            f"{status_line}\n"
            f"──────────────────────\n"
            f"_Shutdown complete. Monitoring paused._"
        )

    async def restored_positions(self, positions: list[dict], mode: str) -> None:
        if not self._enabled:
            return
        if not positions:
            mode_label = "paper" if mode == "paper" else "live"
            await self._send(
                f"📌 *Restored Open Positions*\n"
                f"══════════════════════\n"
                f"Mode: `{mode_label.upper()}`  |  Count: `0`\n"
                f"No open positions to restore.\n"
                f"──────────────────────\n"
                f"_State synced. Monitoring continues._"
            )
            return
        mode_label = "paper" if mode == "paper" else "live"
        rows = []
        for pos in positions[:8]:
            rows.append(
                f"  • {pos.get('symbol', '?')}  {str(pos.get('direction', '?')).upper()}  "
                f"entry `${float(pos.get('entry', 0.0)):.4f}`  "
                f"SL `${float(pos.get('sl', 0.0)):.4f}`  "
                f"size `${float(pos.get('size_usd', 0.0)):.2f}`"
            )
        joined_rows = "\n".join(rows)
        await self._send(
            f"📌 *Restored Open Positions*\n"
            f"══════════════════════\n"
            f"Mode: `{mode_label.upper()}`  |  Count: `{len(positions)}`\n"
            f"These positions were already active when the bot came online.\n"
            f"──────────────────────\n"
            f"{joined_rows}\n"
            f"──────────────────────\n"
            f"_State synced. Monitoring continues._"
        )

    async def shadow_report(self, text: str) -> None:
        if not self._enabled:
            return
        await self._send(text)

    async def milestone_alert(
        self,
        equity: float,
        milestone: float,
        peak: float,
        gift_emoji: str = "🎁",
        gift_desc: str = "a special reward!",
    ) -> None:
        if not self._enabled:
            return
        gain_pct = (equity - 100) / 100 * 100
        if milestone >= 1000:
            opener = f"🎉 *BOSS! WE HIT ${milestone:,.0f}!!!*\nThat's {milestone/100:.0f}x from where we started. Legendary."
        elif milestone >= 500:
            opener = f"🚀 *Boss, ${milestone:,.0f} reached!*\nWe're cooking. Compounding is real."
        elif milestone >= 200:
            opener = f"💰 *Boss, we doubled up — ${milestone:,.0f}!*\nCompounding is doing its thing."
        else:
            opener = f"✅ *Boss, ${milestone:,.0f} milestone hit!*\nSmall step, big journey."
        await self._send(
            f"{opener}\n"
            f"══════════════════════\n"
            f"💰 Equity now: *${equity:,.2f}*\n"
            f"📈 Total return: *+{gain_pct:.1f}%*\n"
            f"🏔️ Peak: *${peak:,.2f}*\n"
            f"──────────────────────\n"
            f"🎁 *Jim's reward:* {gift_emoji} {gift_desc}\n"
            f"_Well earned, Jim. Keep going._ 🧮🥷"
        )

    async def monthly_report(self, report: dict) -> None:
        if not self._enabled or report.get("trades", 0) == 0:
            return
        mode      = str(report.get("mode", "paper") or "paper")
        mode_lbl  = "🟢 LIVE" if mode == "live" else "📋 PAPER"
        days      = self._safe_float(report.get("monthly_days", 30), 30.0)
        sharpe    = self._safe_float(report.get("sharpe", 0))
        sortino   = self._safe_float(report.get("sortino", 0))
        calmar    = self._safe_float(report.get("calmar", 0))
        monthly_r = self._safe_float(report.get("monthly_return_pct", 0))
        pf        = self._safe_float(report.get("profit_factor", 0))
        next_m    = report.get("next_milestone")
        monthly_trades = self._safe_int(report.get("monthly_trades", 0))
        equity = self._safe_float(report.get("equity", 0))
        peak_equity = self._safe_float(report.get("peak_equity", 0))
        win_rate = self._safe_float(report.get("win_rate", 0))
        avg_rr = self._safe_float(report.get("avg_rr", 0))
        drawdown_pct = self._safe_float(report.get("drawdown_pct", 0))
        net_pnl = self._safe_float(report.get("net_pnl", 0))
        best_trade = self._safe_float(report.get("best_trade", 0))
        worst_trade = self._safe_float(report.get("worst_trade", 0))

        r_emoji = "📈" if monthly_r >= 0 else "📉"

        if monthly_r >= 20:
            verdict = "🔥 *Outstanding month, Boss. Absolutely elite.*"
        elif monthly_r >= 10:
            verdict = "💪 *Solid month. The fund is performing well.*"
        elif monthly_r >= 0:
            verdict = "👍 *Positive month. Slow and steady wins the race.*"
        elif monthly_r >= -5:
            verdict = "😐 *Slightly down this month. We'll make it back.*"
        else:
            verdict = "😤 *Rough month. Reviewing strategy. Won't happen again.*"

        pf_note = "exceptional" if pf >= 2.5 else "good" if pf >= 1.5 else "acceptable" if pf >= 1.0 else "needs work"
        next_line = f"\n🎯 *Next target: ${next_m:,.0f}*  — let's get there." if next_m else ""

        await self._send(
            f"📋 *MONTHLY STATEMENT — {mode_lbl}*\n"
            f"`Period: {days:.0f} days  |  {monthly_trades} trades`\n"
            f"══════════════════════\n"
            f"{verdict}\n"
            f"──────────────────────\n"
            f"💰 Equity: *${equity:,.2f}*  (peak: `${peak_equity:,.2f}`)\n"
            f"{r_emoji} Monthly Return: *{monthly_r:+.2f}%*\n"
            f"──────────────────────\n"
            f"🏆 Win Rate: `{win_rate:.1%}`\n"
            f"⚖️ Profit Factor: `{pf:.2f}` — {pf_note}\n"
            f"📐 Avg R:R: `{avg_rr:.2f}`\n"
            f"📉 Max Drawdown: `{drawdown_pct:.1f}%`\n"
            f"💵 Net P&L: `${net_pnl:+.2f}`\n"
            f"🟢 Best trade: `+${best_trade:.2f}`\n"
            f"🔴 Worst trade: `${worst_trade:.2f}`\n"
            f"──────────────────────\n"
            f"📊 Sharpe: `{sharpe:.2f}`  Sortino: `{sortino:.2f}`  Calmar: `{calmar:.2f}`"
            f"{next_line}\n"
            f"_The model never sleeps, Boss. — Jim_ 🧮🥷"
        )

    async def performance_report(
        self,
        report: dict,
        equity: float,
        mode: str,
        open_positions: list[dict] | None = None,
        floating_positions: list[dict] | None = None,
        recent_closed: list[dict] | None = None,
    ) -> None:
        """Fund manager style performance report sent periodically."""
        if not self._enabled:
            return
        n = self._safe_int(report.get("trades", 0))
        if n == 0:
            return

        win_rate = self._safe_float(report.get("win_rate", 0))
        pf = self._safe_float(report.get("profit_factor", 0))
        avg_rr = self._safe_float(report.get("avg_rr", 0))
        dd = self._safe_float(report.get("drawdown_pct", 0))
        ml_acc = self._safe_float(report.get("ml_accuracy", 0))
        ml_trend = report.get("ml_trend", "stable")
        kelly_f = self._safe_float(report.get("kelly_factor", 1.0), 1.0)
        best = self._safe_float(report.get("best_trade_usd", report.get("best_trade", 0)))
        worst = self._safe_float(report.get("worst_trade_usd", report.get("worst_trade", 0)))
        attribution = report.get("attribution", {})

        trend_emoji = {"improving": "📈", "declining": "📉", "stable": "➡️"}.get(ml_trend, "➡️")
        mode_label = "🟢 LIVE" if mode == "live" else "📋 PAPER"
        kelly_label = "🔼 Scaling Up" if kelly_f > 1.05 else "🔽 Scaling Down" if kelly_f < 0.95 else "⚖️ Neutral"
        floating_total = sum(self._safe_float(p.get("pnl_usd", 0.0)) for p in floating_positions or [])
        balance = self._safe_float(equity)
        effective_equity = balance + floating_total

        regime_lines = ""
        for regime, s in (report.get("regime_breakdown") or {}).items():
            short = {"trending_expansion": "Trend", "accumulation_compression": "Accum",
                     "distribution": "Dist", "chaos": "Chaos"}.get(regime, regime)
            regime_lines += (
                f"\n   {short}: {self._safe_float(s.get('win_rate', 0)):.0%} WR"
                f"  ({self._safe_int(s.get('total', 0))} trades)  ${self._safe_float(s.get('pnl', 0)):+.2f}"
            )

        msg = (
            f"🧮 *JIM SIMONS — FUND MANAGER REPORT*\n"
            f"`Mode: {mode_label}`\n"
            f"═══════════════════════\n"
            f"🏦 Balance: *${balance:,.2f} USDT*\n"
            f"💰 *Equity:* `${effective_equity:,.2f}`\n"
            f"📋 *Trades:* `{n}`  |  Drawdown: `{dd:.1f}%`\n"
            f"─────────────────────\n"
            f"🏆 *Win Rate:* `{win_rate:.1%}`\n"
            f"⚖️ *Profit Factor:* `{pf:.2f}`\n"
            f"📐 *Avg R:R:* `{avg_rr:.2f}`\n"
            f"🟢 *Best Trade:* `+${best:.2f}`\n"
            f"🔴 *Worst Trade:* `${worst:.2f}`\n"
            f"─────────────────────\n"
            f"🤖 *ML Accuracy:* `{ml_acc:.1%}` {trend_emoji} {ml_trend}\n"
            f"📏 *Size Multiplier:* `{kelly_f:.2f}x` — {kelly_label}\n"
        )
        if regime_lines:
            msg += f"─────────────────────\n🗂 *By Regime:*{regime_lines}\n"
        sleeve_rows = attribution.get("by_sleeve", [])[:3]
        if sleeve_rows:
            sleeve_lines = ""
            for row in sleeve_rows:
                pf_val = self._safe_float(row.get("profit_factor", 0.0))
                pf_str = "inf" if pf_val == float("inf") else f"{pf_val:.2f}"
                sleeve_lines += (
                    f"\n   `{row.get('key', '?')}`: {self._safe_float(row.get('win_rate', 0)):.0%} WR  "
                    f"PF {pf_str}  ${self._safe_float(row.get('net_pnl_usd', 0)):+.2f}  ({self._safe_int(row.get('trades', 0))})"
                )
            msg += f"─────────────────────\n🧭 *By Sleeve:*{sleeve_lines}\n"
        else:
            msg += f"─────────────────────\n🧭 *By Sleeve:* _none_\n"
        exit_rows = attribution.get("by_exit_profile", [])[:3]
        if exit_rows:
            exit_lines = ""
            for row in exit_rows:
                pf_val = self._safe_float(row.get("profit_factor", 0.0))
                pf_str = "inf" if pf_val == float("inf") else f"{pf_val:.2f}"
                exit_lines += (
                    f"\n   `{row.get('key', '?')}`: {self._safe_float(row.get('win_rate', 0)):.0%} WR  "
                    f"PF {pf_str}  ${self._safe_float(row.get('net_pnl_usd', 0)):+.2f}  ({self._safe_int(row.get('trades', 0))})"
                )
            msg += f"─────────────────────\n🎯 *By Exit:*{exit_lines}\n"
        else:
            msg += f"─────────────────────\n🎯 *By Exit:* _none_\n"
        if open_positions:
            pos_lines = ""
            for pos in open_positions[:3]:
                tp_flag = "TP1" if pos.get("tp1_hit") else "OPEN"
                pos_lines += (
                    f"\n   `{pos['symbol']}`: {str(pos['direction']).upper()}  "
                    f"entry ${self._safe_float(pos.get('entry', 0.0)):,.4f}  size ${self._safe_float(pos.get('size_usd', 0.0)):,.2f}  {tp_flag}"
                )
            msg += f"─────────────────────\n📌 *Open Positions:*{pos_lines}\n"
        else:
            msg += f"─────────────────────\n📌 *Open Positions:* _none_\n"
        if recent_closed:
            closed_lines = ""
            for row in recent_closed[:3]:
                pnl = self._safe_float(row.get("pnl_usd", 0.0))
                sign = "+" if pnl >= 0 else ""
                closed_lines += (
                    f"\n   `{row.get('symbol', '?')}`: {str(row.get('direction', '?')).upper()}  "
                    f"{sign}${pnl:.2f}  {str(row.get('reason', '')).replace('_', ' ')}"
                )
            msg += f"─────────────────────\n🧾 *Recent Executions:*{closed_lines}\n"
        else:
            msg += f"─────────────────────\n🧾 *Recent Executions:* _none_\n"

        await self._send(msg)

    async def live_transition(self, equity: float) -> None:
        """Alert sent when bot auto-switches from paper to live mode."""
        if not self._enabled:
            return
        await self._send(
            f"🚀 *Boss, I graduated. We're LIVE.*\n"
            f"══════════════════════\n"
            f"Paper trading: ✅ passed all 7 criteria.\n"
            f"Mode: Paper → *LIVE TRADING*\n"
            f"Real capital deployed: *${equity:,.2f} USDT*\n"
            f"──────────────────────\n"
            f"All risk guards are enforced.\n"
            f"🧮 *Jim Simons fully armed.* Model is live.\n"
            f"_The model never sleeps, Boss. — Jim_ 🧮🥷"
        )

    async def equity_graph_report(
        self,
        points: list[dict],
        *,
        cycle_num: int = 0,
        interval_minutes: float = 60.0,
        mode: str = "paper",
    ) -> None:
        if not self._enabled or len(points) < 2:
            return

        png = self._build_equity_graph_png(
            points,
            cycle_num=cycle_num,
            interval_minutes=interval_minutes,
            mode=mode,
        )
        latest = points[-1]
        balance = self._safe_float(latest.get("balance", 0.0))
        equity = self._safe_float(latest.get("equity", balance))
        floating = equity - balance
        peak = max(equity, max(self._safe_float(point.get("equity", balance)) for point in points))
        drawdown_pct = ((peak - equity) / peak * 100) if peak > 0 else 0.0
        now = datetime.now(UTC).strftime("%H:%M:%S UTC")
        caption = (
            f"JIM SIMONS — BALANCE / EQUITY GRAPH\n"
            f"{now}\n"
            f"Mode: {mode.upper()} | Samples: {len(points)} | Cadence: {interval_minutes:.0f}m\n"
            f"Balance: ${balance:,.2f} | Equity: ${equity:,.2f} | Floating: {floating:+.2f} | DD: {drawdown_pct:.1f}%\n"
            f"Independent PNG snapshot."
        )
        await self._send_photo(png, caption)

    async def send_raw(self, text: str) -> None:
        await self._send(text)

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    async def _send(self, text: str) -> None:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                for chunk in self._split_message(text):
                    resp = await client.post(
                        self._url,
                        json={
                            "chat_id": self._chat_id,
                            "text": chunk,
                            "parse_mode": "Markdown",
                            "disable_web_page_preview": True,
                        },
                    )
                    if resp.status_code == 200:
                        continue

                    log.warning(
                        "Telegram send failed with Markdown: %s %s",
                        resp.status_code,
                        resp.text[:200],
                    )
                    fallback = await client.post(
                        self._url,
                        json={
                            "chat_id": self._chat_id,
                            "text": chunk,
                            "disable_web_page_preview": True,
                        },
                    )
                    if fallback.status_code != 200:
                        log.warning(
                            "Telegram plain-text fallback failed: %s %s",
                            fallback.status_code,
                            fallback.text[:200],
                        )
        except Exception as exc:
            log.warning("Telegram error: %s", exc)

    async def _send_photo(self, photo_bytes: bytes, caption: str) -> None:
        if not self._enabled:
            return
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(
                    self._photo_url,
                    data={
                        "chat_id": self._chat_id,
                        "caption": caption,
                    },
                    files={
                        "photo": ("equity_graph.png", photo_bytes, "image/png"),
                    },
                )
                if resp.status_code != 200:
                    log.warning(
                        "Telegram photo send failed: %s %s",
                        resp.status_code,
                        resp.text[:200],
                    )
        except Exception as exc:
            log.warning("Telegram photo error: %s", exc)

    @staticmethod
    def _active_session_label() -> str:
        return active_market_session_label()

    def _trade_performance_metrics(
        self,
        trade_log: list,
        equity_curve: list[float],
        *,
        starting_equity: float,
        equity: float,
    ) -> dict[str, float]:
        trades = list(trade_log or [])
        n = len(trades)
        if n == 0:
            return {
                "trades": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "z_score": 0.0,
                "ghpr_pct": 0.0,
                "cagr_pct": 0.0,
                "mar": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "avg_wl": 0.0,
                "expectancy_pct": 0.0,
                "recovery_factor": 0.0,
                "max_dd_pct": 0.0,
            }

        pnl_usd = [self._safe_float(getattr(t, "pnl_usd", 0.0)) for t in trades]
        pnl_pct = [self._safe_float(getattr(t, "pnl_pct", 0.0)) for t in trades]
        wins = [p for p in pnl_usd if p > 0]
        losses = [p for p in pnl_usd if p <= 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        win_rate = len(wins) / n if n else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if wins else 0.0)
        avg_win_pct = (sum(p for p in pnl_pct if p > 0) / len(wins)) if wins else 0.0
        avg_loss_pct = (abs(sum(p for p in pnl_pct if p <= 0)) / len(losses)) if losses else 0.0
        avg_wl = (avg_win_pct / avg_loss_pct) if avg_loss_pct > 0 else (float("inf") if avg_win_pct > 0 else 0.0)
        expectancy_pct = sum(pnl_pct) / n
        max_dd_usd = self._max_drawdown_usd(equity_curve, starting_equity, equity)
        recovery_factor = (sum(pnl_usd) / max_dd_usd) if max_dd_usd > 0 else 0.0
        max_dd_pct = self._max_drawdown_pct(equity_curve, starting_equity, equity)
        ghpr_pct = self._ghpr_pct(pnl_pct)
        z_score = self._runs_z_score(trades)
        cagr_pct = self._cagr_pct(trades, starting_equity, equity)
        sharpe = self._sharpe_ratio_from_curve(equity_curve)
        sortino = self._sortino_ratio_from_curve(equity_curve)
        mar = (cagr_pct / max_dd_pct) if max_dd_pct > 0 else 0.0
        return {
            "trades": float(n),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "z_score": z_score,
            "ghpr_pct": ghpr_pct,
            "cagr_pct": cagr_pct,
            "mar": mar,
            "sharpe": sharpe,
            "sortino": sortino,
            "avg_win_pct": avg_win_pct,
            "avg_loss_pct": avg_loss_pct,
            "avg_wl": avg_wl,
            "expectancy_pct": expectancy_pct,
            "recovery_factor": recovery_factor,
            "max_dd_pct": max_dd_pct,
        }

    def _runs_z_score(self, trades: list) -> float:
        outcomes = [self._safe_float(getattr(t, "pnl_usd", 0.0)) > 0 for t in trades]
        n1 = sum(1 for outcome in outcomes if outcome)
        n2 = len(outcomes) - n1
        if n1 == 0 or n2 == 0 or len(outcomes) < 2:
            return 0.0
        runs = 1
        for idx in range(1, len(outcomes)):
            if outcomes[idx] != outcomes[idx - 1]:
                runs += 1
        expected = 1 + (2 * n1 * n2) / (n1 + n2)
        variance = (
            2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)
        ) / (((n1 + n2) ** 2) * (n1 + n2 - 1))
        if variance <= 0:
            return 0.0
        return (runs - expected) / (variance ** 0.5)

    @staticmethod
    def _ghpr_pct(pnl_pct: list[float]) -> float:
        if not pnl_pct:
            return 0.0
        growth = 1.0
        for pnl in pnl_pct:
            growth *= max(0.0, 1.0 + pnl / 100.0)
        return (growth ** (1 / len(pnl_pct)) - 1.0) * 100.0 if growth > 0 else -100.0

    @staticmethod
    def _cagr_pct(trades: list, starting_equity: float, equity: float) -> float:
        if starting_equity <= 0 or equity <= 0 or not trades:
            return 0.0
        if len(trades) < 20:
            return 0.0
        timestamps = [
            float(getattr(t, "closed_at", 0.0) or getattr(t, "opened_at", 0.0) or 0.0)
            for t in trades
        ]
        timestamps = [ts for ts in timestamps if ts > 0]
        if not timestamps:
            return 0.0
        elapsed_seconds = max(timestamps) - min(timestamps)
        if elapsed_seconds <= 0:
            return 0.0
        years = elapsed_seconds / 31557600.0
        if years < (90.0 / 365.25):
            return 0.0
        try:
            return ((equity / starting_equity) ** (1 / years) - 1.0) * 100.0
        except Exception:
            return 0.0

    @staticmethod
    def _max_drawdown_usd(equity_curve: list[float], starting_equity: float, equity: float) -> float:
        curve = [float(v) for v in equity_curve if v is not None]
        if not curve:
            peak = max(starting_equity, equity)
            return max(0.0, peak - equity)
        peak = float(starting_equity if starting_equity > 0 else curve[0])
        max_dd = 0.0
        for value in curve:
            if value > peak:
                peak = value
            max_dd = max(max_dd, peak - value)
        return max_dd

    @staticmethod
    def _max_drawdown_pct(equity_curve: list[float], starting_equity: float, equity: float) -> float:
        curve = [float(v) for v in equity_curve if v is not None]
        if not curve:
            peak = max(starting_equity, equity)
            return ((peak - equity) / peak * 100.0) if peak > 0 else 0.0
        peak = float(starting_equity if starting_equity > 0 else curve[0])
        max_dd = 0.0
        for value in curve:
            if value > peak:
                peak = value
            if peak > 0:
                max_dd = max(max_dd, (peak - value) / peak * 100.0)
        return max_dd

    @staticmethod
    def _curve_returns(equity_curve: list[float]) -> list[float]:
        curve = [float(v) for v in equity_curve if v is not None]
        return [
            (curve[i] - curve[i - 1]) / curve[i - 1]
            for i in range(1, len(curve))
            if curve[i - 1] != 0
        ]

    def _sharpe_ratio_from_curve(self, equity_curve: list[float]) -> float:
        returns = self._curve_returns(equity_curve)
        if len(returns) < 3:
            return 0.0
        mean_r = sum(returns) / len(returns)
        std_r = (sum((r - mean_r) ** 2 for r in returns) / len(returns)) ** 0.5
        if std_r == 0:
            return 0.0
        return (mean_r / std_r) * (4 * 365) ** 0.5

    def _sortino_ratio_from_curve(self, equity_curve: list[float]) -> float:
        returns = self._curve_returns(equity_curve)
        if len(returns) < 3:
            return 0.0
        mean_r = sum(returns) / len(returns)
        downside = [r for r in returns if r < 0]
        if not downside:
            return 9.99
        down_std = (sum(r ** 2 for r in downside) / len(downside)) ** 0.5
        if down_std == 0:
            return 0.0
        return (mean_r / down_std) * (4 * 365) ** 0.5

    def _build_equity_graph_png(
        self,
        points: list[dict],
        *,
        cycle_num: int,
        interval_minutes: float,
        mode: str,
    ) -> bytes:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        ordered = sorted(points, key=lambda p: self._safe_float(p.get("ts", 0.0)))
        timestamps = [
            datetime.fromtimestamp(self._safe_float(point.get("ts", 0.0)), tz=UTC)
            for point in ordered
        ]
        balances = [self._safe_float(point.get("balance", 0.0)) for point in ordered]
        equities = [
            self._safe_float(point.get("equity", balance))
            for point, balance in zip(ordered, balances)
        ]
        peaks: list[float] = []
        running_peak = 0.0
        for equity in equities:
            running_peak = max(running_peak, equity)
            peaks.append(running_peak)
        drawdowns = [
            ((peak - equity) / peak * 100) if peak > 0 else 0.0
            for peak, equity in zip(peaks, equities)
        ]

        plt.style.use("dark_background")
        fig, (ax, ax_dd) = plt.subplots(
            2,
            1,
            figsize=(10, 6.2),
            dpi=160,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")
        ax_dd.set_facecolor("#0f172a")

        ax.plot(timestamps, balances, color="#38bdf8", linewidth=2.2, label="Balance")
        ax.plot(timestamps, equities, color="#a78bfa", linewidth=2.2, label="Equity")
        ax.fill_between(timestamps, balances, equities, color="#a78bfa", alpha=0.14)
        ax.scatter([timestamps[-1]], [balances[-1]], color="#38bdf8", s=24, zorder=5)
        ax.scatter([timestamps[-1]], [equities[-1]], color="#a78bfa", s=24, zorder=5)

        latest_balance = balances[-1]
        latest_equity = equities[-1]
        floating = latest_equity - latest_balance
        latest_drawdown = drawdowns[-1] if drawdowns else 0.0
        ax.set_title("Balance vs Equity", fontsize=15, pad=12, weight="bold")
        ax.text(
            0.01,
            0.98,
            f"Mode: {mode.upper()}  |  Cycle: #{cycle_num}  |  Cadence: {interval_minutes:.0f}m",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            color="#cbd5e1",
        )
        ax.text(
            0.01,
            0.91,
            f"Latest Balance ${latest_balance:,.2f}  |  Equity ${latest_equity:,.2f}  |  Floating {floating:+.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            color="#cbd5e1",
        )
        ax_dd.axhline(0.0, color="#475569", linewidth=1.0)
        ax_dd.fill_between(timestamps, drawdowns, color="#fb7185", alpha=0.28)
        ax_dd.plot(timestamps, drawdowns, color="#fb7185", linewidth=1.8, label="Drawdown %")
        ax_dd.scatter([timestamps[-1]], [latest_drawdown], color="#fb7185", s=24, zorder=5)
        ax_dd.set_ylabel("DD %", fontsize=9, color="#cbd5e1")
        ax_dd.text(
            0.01,
            0.82,
            f"Latest DD {latest_drawdown:.1f}%",
            transform=ax_dd.transAxes,
            va="top",
            fontsize=9,
            color="#cbd5e1",
        )

        locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax_dd.xaxis.set_major_locator(locator)
        ax_dd.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.grid(True, alpha=0.18)
        ax_dd.grid(True, alpha=0.18)
        ax.legend(loc="upper left", frameon=False)
        ax_dd.legend(loc="upper left", frameon=False)
        ax.tick_params(colors="#e2e8f0")
        ax_dd.tick_params(colors="#e2e8f0")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        for spine in ax_dd.spines.values():
            spine.set_color("#334155")

        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    def _split_message(self, text: str) -> list[str]:
        if len(text) <= _MAX_MESSAGE_LEN:
            return [text]

        chunks: list[str] = []
        current = ""
        for line in text.splitlines(keepends=True):
            if len(current) + len(line) <= _MAX_MESSAGE_LEN:
                current += line
                continue
            if current:
                chunks.append(current.rstrip())
                current = ""
            while len(line) > _MAX_MESSAGE_LEN:
                chunks.append(line[:_MAX_MESSAGE_LEN].rstrip())
                line = line[_MAX_MESSAGE_LEN:]
            current = line
        if current:
            chunks.append(current.rstrip())
        return chunks or [text[:_MAX_MESSAGE_LEN]]
