"""
Shadow mode engine — runs ghost trades in parallel with live/paper trading.

On every loop tick:
  shadow.on_tick(price_map)         — check exits on open shadow trades
  shadow.on_signal(bd, setup)       — mirror a signal that the live bot acted on
  shadow.print_report()             — print performance table to console
  shadow.format_telegram_report()   — returns formatted string for Telegram
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.backtest.engine import BacktestTrade
from src.risk.risk_manager import TradeSetup
from src.scoring.scorer import SignalBreakdown

log = logging.getLogger(__name__)
console = Console()


@dataclass
class _ClosedShadowTrade:
    pnl_usd: float
    pnl_pct: float
    regime: str
    ev_predicted: float
    exit_reason: str
    # ML pre-training fields
    scores: dict = field(default_factory=dict)
    symbol: str = ""
    direction: str = ""
    sector: str = "unknown"
    entry_price: float = 0.0
    exit_price: float = 0.0
    opened_at: float = 0.0
    closed_at: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0


@dataclass
class ShadowMetrics:
    closed: int = 0
    open: int = 0
    wins: int = 0
    losses: int = 0
    winrate: float = 0.0
    profit_factor: float = 0.0
    net_pnl_pct: float = 0.0
    avg_ev: float = 0.0
    avg_real: float = 0.0
    drift: float = 0.0
    trending_pf: float = 0.0
    compression_pf: float = 0.0
    max_dd: float = 0.0
    max_consec_losses: int = 0
    trade_frequency_pct: float = 0.0


class ShadowEngine:
    """
    Mirrors every trade signal the live bot takes and simulates the outcome
    using subsequent price ticks. Does not place any real orders.
    """

    def __init__(self, cfg: dict):
        self._exit_cfg = cfg["exit"]
        self._commission_pct = cfg["backtest"]["commission_pct"] / 100
        self._slippage_pct = cfg["backtest"]["slippage_pct"] / 100
        self._initial_equity = float(cfg["backtest"]["initial_capital"])

        save_dir = Path(cfg["learning"]["save_path"]).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        self._save_path = save_dir / "shadow_state.json"

        self._open: dict[str, BacktestTrade] = {}
        self._open_meta: dict[str, dict] = {}   # symbol → {regime, ev_predicted, size_usd}
        self._closed: list[_ClosedShadowTrade] = []
        self._tick_counter: dict[str, int] = {}
        self._total_ticks: int = 0

        self._rejected: list[dict] = []  # counterfactual: signals blocked by gates
        self._equity = self._initial_equity
        self._peak_equity = self._initial_equity
        self._max_dd: float = 0.0
        self._consec_losses: int = 0
        self._max_consec_losses: int = 0

        self._load_state()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def on_signal(self, bd: SignalBreakdown, setup: TradeSetup, scores_dict: dict | None = None) -> None:
        sym = setup.symbol
        if sym in self._open:
            log.info("[SHADOW] skip %s: already open (dedup)", sym)
            return

        slip = setup.entry_price * self._slippage_pct
        entry = (
            setup.entry_price + slip
            if setup.direction == "long"
            else setup.entry_price - slip
        )

        trade = BacktestTrade(
            symbol=sym,
            direction=setup.direction,
            exit_profile=setup.exit_profile,
            entry_bar=self._tick_counter.get(sym, 0),
            entry_price=entry,
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
        )
        self._open[sym] = trade
        self._open_meta[sym] = {
            "regime": bd.regime.value if bd.regime else "unknown",
            "ev_predicted": bd.ev_result.ev_net_pct if bd.ev_result else 0.0,
            "size_usd": setup.size_usd,
            "scores": scores_dict or {},
            "symbol": sym,
            "direction": setup.direction,
            "sector": str((setup.setup_passport or {}).get("sector", "unknown")),
            "entry_price": entry,
            "opened_at": time.time(),
        }
        log.info(
            "[SHADOW] Opened  %s %s  entry=%.4f  SL=%.4f  TP1=%.4f  TP2=%.4f",
            sym, setup.direction.upper(), entry, setup.stop_loss, setup.tp1, setup.tp2,
        )
        self._save_state()

    def record_rejected(
        self, bd: SignalBreakdown, stage: str, features: dict | None = None
    ) -> None:
        """Log a signal rejected by a gate (counterfactual record for ML)."""
        self._rejected.append({
            "ts": time.time(),
            "symbol": bd.symbol,
            "direction": bd.direction,
            "stage": stage,
            "total_score": float(getattr(bd, "total_score", 0.0)),
            "regime": bd.regime.value if bd.regime else "unknown",
            "features": features or {},
        })
        if len(self._rejected) > 5000:
            self._rejected = self._rejected[-5000:]
        log.info("[SHADOW] Rejected %s %s stage=%s", bd.symbol, bd.direction, stage)
        self._save_state()

    def on_tick(self, price_map: dict[str, float]) -> None:
        self._total_ticks += 1
        closed_syms: list[str] = []

        for sym, price in price_map.items():
            self._tick_counter[sym] = self._tick_counter.get(sym, 0) + 1
            if sym not in self._open:
                continue

            trade = self._open[sym]
            bar_idx = self._tick_counter[sym]

            # Update MFE/MAE using current tick price
            r = trade.r_distance
            if r > 0:
                entry = trade.entry_price
                if trade.direction == "long":
                    fav = price - entry
                    adv = entry - price
                else:
                    fav = entry - price
                    adv = price - entry
                if fav > 0:
                    trade.mfe_r = max(trade.mfe_r, fav / r)
                if adv > 0:
                    trade.mae_r = max(trade.mae_r, adv / r)

            trade, did_close = self._check_exits(
                trade, high=price, low=price, close=price, bar_idx=bar_idx
            )
            if did_close:
                # Round-trip commission on the actual notional (entry + exit
                # legs). Historic implementation scaled with |pnl_usd| which
                # double-charged big winners and ignored break-even closes.
                commission = trade.size_usd * self._commission_pct * 2
                trade.pnl_usd -= commission

                meta = self._open_meta.get(sym, {})
                size_usd = meta.get("size_usd", 1.0) or 1.0
                pnl_pct = trade.pnl_usd / size_usd * 100

                self._closed.append(_ClosedShadowTrade(
                    pnl_usd=trade.pnl_usd,
                    pnl_pct=pnl_pct,
                    regime=meta.get("regime", "unknown"),
                    ev_predicted=meta.get("ev_predicted", 0.0),
                    exit_reason=trade.exit_reason,
                    scores=meta.get("scores", {}),
                    symbol=meta.get("symbol", sym),
                    direction=meta.get("direction", trade.direction),
                    sector=meta.get("sector", "unknown"),
                    entry_price=meta.get("entry_price", trade.entry_price),
                    exit_price=trade.exit_price,
                    opened_at=meta.get("opened_at", 0.0),
                    closed_at=time.time(),
                    mfe_r=trade.mfe_r,
                    mae_r=trade.mae_r,
                ))

                self._equity += trade.pnl_usd
                if self._equity > self._peak_equity:
                    self._peak_equity = self._equity
                dd = (self._peak_equity - self._equity) / self._peak_equity * 100
                if dd > self._max_dd:
                    self._max_dd = dd

                if trade.pnl_usd <= 0:
                    self._consec_losses += 1
                    if self._consec_losses > self._max_consec_losses:
                        self._max_consec_losses = self._consec_losses
                else:
                    self._consec_losses = 0

                closed_syms.append(sym)
                tag = "WIN " if trade.pnl_usd > 0 else "LOSS"
                log.info(
                    "[SHADOW] Closed  %s %s  pnl=$%.2f (%.2f%%)  reason=%s",
                    sym, tag, trade.pnl_usd, pnl_pct, trade.exit_reason,
                )

        for sym in closed_syms:
            del self._open[sym]
            self._open_meta.pop(sym, None)

        if closed_syms:
            self._save_state()

    def _save_state(self) -> None:
        state = {
            "closed": [asdict(t) for t in self._closed],
            "open": {
                sym: {
                    "trade": asdict(t),
                    "meta": self._open_meta.get(sym, {}),
                }
                for sym, t in self._open.items()
            },
            "equity": self._equity,
            "peak_equity": self._peak_equity,
            "max_dd": self._max_dd,
            "consec_losses": self._consec_losses,
            "max_consec_losses": self._max_consec_losses,
            "total_ticks": self._total_ticks,
            "rejected": self._rejected,
        }
        try:
            self._save_path.write_text(json.dumps(state, indent=2))
        except Exception as exc:
            log.warning("Shadow state save failed: %s", exc)

    def _load_state(self) -> None:
        if not self._save_path.exists():
            return
        try:
            state = json.loads(self._save_path.read_text())
            self._closed = [_ClosedShadowTrade(**t) for t in state.get("closed", [])]
            for sym, entry in state.get("open", {}).items():
                trade_payload = dict(entry["trade"])
                trade_payload.setdefault("exit_profile", "default")
                trade_payload.setdefault("tp1_size_pct", 0.50)
                trade_payload.setdefault("tp2_size_pct", 0.30)
                trade_payload.setdefault("trailing_atr_multiplier", 1.5)
                trade_payload.setdefault("max_hold_duration_s", 172800)
                trade_payload.setdefault("atr", 0.0)
                self._open[sym] = BacktestTrade(**trade_payload)
                self._open_meta[sym] = entry["meta"]
            self._equity = state.get("equity", self._initial_equity)
            self._peak_equity = state.get("peak_equity", self._initial_equity)
            self._max_dd = state.get("max_dd", 0.0)
            self._consec_losses = state.get("consec_losses", 0)
            self._max_consec_losses = state.get("max_consec_losses", 0)
            self._total_ticks = state.get("total_ticks", 0)
            self._rejected = state.get("rejected", [])
            log.info(
                "Shadow state loaded: %d closed, %d open trades",
                len(self._closed), len(self._open),
            )
        except Exception as exc:
            log.warning("Shadow state load failed (starting fresh): %s", exc)

    def get_ml_trade_log(self) -> list:
        """
        Returns shadow trades as TradeRecord objects for ML pre-training.
        Only includes trades that have full feature scores (set via on_signal scores_dict).
        """
        from src.learning.learner import TradeRecord

        def _score_text(scores: dict, key: str, default: str = "unknown") -> str:
            value = scores.get(key, default)
            return str(value or default)

        def _score_float(scores: dict, key: str, default: float = 0.0) -> float:
            try:
                return float(scores.get(key, default) or default)
            except (TypeError, ValueError):
                return default

        def _score_int(scores: dict, key: str, default: int = -1) -> int:
            try:
                return int(scores.get(key, default))
            except (TypeError, ValueError):
                return default

        records = []
        for t in self._closed:
            if not t.scores:
                continue
            scores = {**t.scores, "is_shadow": 1.0}
            records.append(
                TradeRecord(
                    symbol=t.symbol or "shadow",
                    direction=t.direction or "long",
                    entry_price=t.entry_price,
                    exit_price=t.exit_price,
                    pnl_usd=t.pnl_usd,
                    pnl_pct=t.pnl_pct,
                    reason=t.exit_reason,
                    # Tag shadow so ML can learn to discount simulator-distribution artifacts
                    scores=scores,
                    opened_at=t.opened_at,
                    closed_at=t.closed_at,
                    regime=t.regime or _score_text(scores, "regime", "unknown"),
                    strategy_sleeve=_score_text(scores, "strategy_sleeve", "unknown"),
                    exit_profile=_score_text(scores, "exit_profile", "unknown"),
                    dispersion_value=_score_float(scores, "dispersion_value", 0.0),
                    dispersion_state=_score_text(scores, "dispersion_state", "normal"),
                    mfe_r=t.mfe_r,
                    mae_r=t.mae_r,
                    session=_score_text(scores, "session", "unknown"),
                    hour_of_day=_score_int(scores, "hour_of_day", -1),
                    day_of_week=_score_int(scores, "day_of_week", -1),
                    asset=t.symbol.split('/')[0] if '/' in t.symbol else "unknown",
                    sector=t.sector or _score_text(scores, "sector", "unknown"),
                    market_risk_on_state=_score_text(scores, "market_risk_on_state", "unknown"),
                    market_rotation_state=_score_text(scores, "market_rotation_state", "unknown"),
                    market_btc_trend=_score_text(scores, "market_btc_trend", "unknown"),
                    market_eth_btc_trend=_score_text(scores, "market_eth_btc_trend", "unknown"),
                    market_btc_d_trend=_score_text(scores, "market_btc_d_trend", "unknown"),
                    market_total_trend=_score_text(scores, "market_total_trend", "unknown"),
                    market_context_confidence=_score_float(scores, "market_context_confidence", 0.0),
                )
            )
        return records

    def metrics(self) -> ShadowMetrics:
        trades = self._closed
        if not trades:
            return ShadowMetrics(open=len(self._open))

        wins = [t for t in trades if t.pnl_usd > 0]
        losses = [t for t in trades if t.pnl_usd <= 0]
        gross_win = sum(t.pnl_usd for t in wins)
        gross_loss = abs(sum(t.pnl_usd for t in losses))

        net_pnl_pct = (self._equity - self._initial_equity) / self._initial_equity * 100

        avg_ev = sum(t.ev_predicted for t in trades) / len(trades)
        avg_real = sum(t.pnl_pct for t in trades) / len(trades)

        trending = [t for t in trades if t.regime == "trending_expansion"]
        compression = [t for t in trades if t.regime == "accumulation_compression"]

        def _pf(subset: list[_ClosedShadowTrade]) -> float:
            if not subset:
                return 0.0
            gw = sum(t.pnl_usd for t in subset if t.pnl_usd > 0)
            gl = abs(sum(t.pnl_usd for t in subset if t.pnl_usd <= 0))
            return gw / gl if gl > 0 else float("inf")

        freq = len(trades) / self._total_ticks * 100 if self._total_ticks else 0.0

        return ShadowMetrics(
            closed=len(trades),
            open=len(self._open),
            wins=len(wins),
            losses=len(losses),
            winrate=len(wins) / len(trades),
            profit_factor=gross_win / gross_loss if gross_loss > 0 else float("inf"),
            net_pnl_pct=net_pnl_pct,
            avg_ev=avg_ev,
            avg_real=avg_real,
            drift=avg_real - avg_ev,
            trending_pf=_pf(trending),
            compression_pf=_pf(compression),
            max_dd=self._max_dd,
            max_consec_losses=self._max_consec_losses,
            trade_frequency_pct=freq,
        )

    def format_telegram_report(self) -> str:
        m = self.metrics()
        if m.closed == 0:
            return "📊 *SHADOW ENGINE* — No closed trades yet."

        pf_str = f"{m.profit_factor:.2f}" if m.profit_factor != float("inf") else "∞"
        drift_flag = "✅" if m.drift >= -0.05 else "❌"
        compression_flag = "✅" if m.compression_pf >= 1.0 else "❌"
        trending_flag = "✅" if m.trending_pf >= 1.0 else "❌"

        def _pf_str(pf: float) -> str:
            if pf == 0.0:
                return "N/A"
            return "∞" if pf == float("inf") else f"{pf:.2f}"

        def _regime_line(label: str, pf: float) -> str:
            if pf == 0.0:
                return f"{label}: N/A"
            flag = "✅" if pf >= 1.0 else "❌"
            return f"{label}: PF {_pf_str(pf)} {flag}"

        return (
            f"📊 *SHADOW PERFORMANCE*\n"
            f"{'─' * 22}\n"
            f"Trades: *{m.closed}*\n"
            f"Winrate: *{m.winrate:.0%}*\n"
            f"Profit Factor: *{pf_str}*\n"
            f"Net PnL: *{m.net_pnl_pct:+.1f}%*\n"
            f"\n"
            f"*─── EDGE VALIDATION ───*\n"
            f"Avg EV: `{m.avg_ev:+.2f}%`\n"
            f"Avg Real: `{m.avg_real:+.2f}%`\n"
            f"Drift: `{m.drift:+.2f}%` {drift_flag}\n"
            f"\n"
            f"*─── REGIME ───*\n"
            f"{_regime_line('Trending', m.trending_pf)}\n"
            f"{_regime_line('Compression', m.compression_pf)}\n"
            f"\n"
            f"*─── RISK ───*\n"
            f"Max DD: `{m.max_dd:.1f}%`\n"
            f"Consec Loss: `{m.max_consec_losses}`\n"
            f"\n"
            f"*─── QUALITY ───*\n"
            f"Trade Frequency: `{m.trade_frequency_pct:.1f}%`"
        )

    def print_report(self) -> None:
        m = self.metrics()
        if m.closed == 0:
            console.print("[dim][SHADOW] No closed trades yet.[/dim]")
            return

        pf_str = f"{m.profit_factor:.2f}" if m.profit_factor != float("inf") else "∞"
        wr_color = "green" if m.winrate >= 0.5 else "red"
        pnl_color = "green" if m.net_pnl_pct >= 0 else "red"

        tbl = Table(title="[bold cyan]Shadow Engine Report[/bold cyan]", show_header=True)
        tbl.add_column("Metric", style="cyan")
        tbl.add_column("Value", justify="right")
        tbl.add_row("Closed trades", str(m.closed))
        tbl.add_row("Open (shadow)", str(m.open))
        tbl.add_row("Wins / Losses", f"{m.wins} / {m.losses}")
        tbl.add_row("Win rate", f"[{wr_color}]{m.winrate:.1%}[/]")
        tbl.add_row("Profit factor", pf_str)
        tbl.add_row("Net PnL", f"[{pnl_color}]{m.net_pnl_pct:+.1f}%[/]")
        tbl.add_row("Avg EV / Real", f"{m.avg_ev:+.2f}% / {m.avg_real:+.2f}%")
        tbl.add_row("EV Drift", f"{m.drift:+.2f}%")
        tbl.add_row("Trending PF", f"{m.trending_pf:.2f}" if m.trending_pf else "N/A")
        tbl.add_row("Compression PF", f"{m.compression_pf:.2f}" if m.compression_pf else "N/A")
        tbl.add_row("Max Drawdown", f"{m.max_dd:.1f}%")
        tbl.add_row("Max Consec Loss", str(m.max_consec_losses))
        tbl.add_row("Trade Frequency", f"{m.trade_frequency_pct:.1f}%")
        console.print(tbl)

    # ------------------------------------------------------------------ #
    #  Exit logic (mirrors BacktestEngine._check_exits)                   #
    # ------------------------------------------------------------------ #

    def _check_exits(
        self,
        trade: BacktestTrade,
        high: float,
        low: float,
        close: float,
        bar_idx: int,
    ) -> tuple[BacktestTrade, bool]:
        ec = self._exit_cfg
        # ATR-based trail mirrors live; falls back to %-based when ATR
        # was not captured at entry (older shadow_state.json snapshots).
        atr_mult = float(trade.trailing_atr_multiplier or 0.0)
        trail_dist = float(trade.atr or 0.0) * atr_mult
        atr_mode = trail_dist > 0.0
        trail_pct = ec.get("tp2_trailing_stop_pct", 0.15)
        mult = 1 if trade.direction == "long" else -1

        # Order matches live + backtest engine:
        #   1. SL hit (always closes residual)
        #   2. Trailing stop hit (residual)
        #   3. TP1 partial (tp1_size_pct of ORIGINAL)
        #   4. TP2 partial (tp2_size_pct of ORIGINAL)
        #   5. TP3 closes runner

        # 1. SL
        if trade.is_sl_hit(low, high):
            trade.pnl_usd += mult * (trade.stop_loss - trade.entry_price) * trade.remaining_contracts
            trade.exit_price = trade.stop_loss
            trade.exit_reason = "stop_loss"
            trade.exit_bar = bar_idx
            return trade, True

        # 2. Trailing stop
        if trade.tp1_hit and trade.trailing_stop is not None:
            ts_hit = (
                (trade.direction == "long" and low <= trade.trailing_stop)
                or (trade.direction == "short" and high >= trade.trailing_stop)
            )
            if ts_hit:
                trade.pnl_usd += mult * (trade.trailing_stop - trade.entry_price) * trade.remaining_contracts
                trade.exit_price = trade.trailing_stop
                trade.exit_reason = "trailing_stop"
                trade.exit_bar = bar_idx
                return trade, True

        # 3. TP1 partial
        if not trade.tp1_hit and trade.is_tp1_hit(low, high):
            tp1_qty = trade.size_contracts * trade.tp1_size_pct
            tp1_pnl = mult * (trade.tp1 - trade.entry_price) * tp1_qty
            trade.pnl_usd += tp1_pnl
            trade.remaining_contracts -= tp1_qty
            trade.tp1_hit = True
            trade.stop_loss = trade.entry_price
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

        # 3b. Ratchet trailing (never widen)
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

        # 4. TP2 partial (closes tp2_size_pct of ORIGINAL position; runner
        #    is left to trail to TP3, matching live + backtest engine).
        tp2_already_hit = getattr(trade, "_tp2_hit", False)
        if trade.tp1_hit and not tp2_already_hit and trade.is_tp2_hit(low, high):
            tp2_qty = min(
                trade.remaining_contracts,
                trade.size_contracts * trade.tp2_size_pct,
            )
            trade.pnl_usd += mult * (trade.tp2 - trade.entry_price) * tp2_qty
            trade.remaining_contracts -= tp2_qty
            setattr(trade, "_tp2_hit", True)
            if trade.remaining_contracts <= 1e-9:
                trade.exit_price = trade.tp2
                trade.exit_reason = "tp2_full"
                trade.exit_bar = bar_idx
                return trade, True

        # 5. TP3 closes runner
        tp3_r = float(ec.get("tp3_r_multiple", 3.0))
        tp3 = trade.entry_price + mult * trade.r_distance * tp3_r
        tp3_hit = (
            (trade.direction == "long" and high >= tp3)
            or (trade.direction == "short" and low <= tp3)
        )
        if tp2_already_hit and tp3_hit and trade.remaining_contracts > 0:
            trade.pnl_usd += mult * (tp3 - trade.entry_price) * trade.remaining_contracts
            trade.exit_price = tp3
            trade.exit_reason = "tp3"
            trade.exit_bar = bar_idx
            return trade, True

        return trade, False
