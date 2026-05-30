"""
Algorithmic Fund Manager — institutional-grade position scaling and portfolio management.

Mimics how a real human fund manager thinks before sizing a position:
  1. How strong is this specific signal? (conviction)
  2. Are we on a hot/cold streak? (momentum discipline)
  3. What regime are we in? (market context)
  4. How have we performed recently overall? (macro momentum)
  5. How much risk do we already have on? (portfolio heat)
  6. How are we doing today? (daily P&L protection)
  7. What's the drawdown from peak? (capital preservation)
  8. What time is it — is the market liquid? (session awareness)
  9. Have we been profitable in this regime lately? (regime edge tracking)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.utils.atomic_write import atomic_write

from src.reporting.attribution import build_attribution_report

if TYPE_CHECKING:
    from src.scoring.scorer import SignalBreakdown
    from src.learning.learner import TradeRecord

log = logging.getLogger(__name__)

STATE_PATH = Path("models/fund_manager_state.json")

MILESTONES = [110, 120, 150, 200, 300, 500, 750, 1000, 2000, 5000, 10000, 25000, 50000, 100000]

# Virtual gifts Jim earns at equity milestones — reward his performance 🎁
MILESTONE_GIFTS: dict[int, tuple[str, str]] = {
    110:    ("🍕", "a pizza — first win deserves a treat!"),
    120:    ("🧋", "bubble tea — keep the momentum!"),
    150:    ("🍣", "sushi dinner — halfway to doubling up!"),
    200:    ("🚲", "a new bike — you doubled the account!"),
    300:    ("🎮", "a gaming setup — Jim's on fire!"),
    500:    ("✈️", "a flight ticket — 5x returns!"),
    750:    ("🏖️", "a beach holiday — serious money now!"),
    1000:   ("🏎️", "a sports car ride — four-figure club!"),
    2000:   ("🛥️", "a yacht weekend — unstoppable!"),
    5000:   ("🌍", "a world tour — elite fund manager!"),
    10000:  ("🏠", "a house deposit — legendary status!"),
    25000:  ("🚀", "a private jet trip — Jim is the GOAT!"),
    50000:  ("🏝️", "your own island — untouchable!"),
    100000: ("👑", "the crown — Jim conquered the markets!"),
}

EQUITY_TIERS = [
    (5000,  6, 10, 1.0),
    (2000,  5, 10, 1.2),
    (1000,  5, 10, 1.5),
    (500,   4, 10, 1.5),
    (200,   3,  7, 1.5),
    (0,     4,  5, 1.5),
]


@dataclass
class FundDecision:
    size_multiplier: float
    conviction_mult: float
    streak_mult: float
    regime_mult: float
    performance_mult: float
    heat_mult: float = 1.0
    daily_pnl_mult: float = 1.0
    drawdown_mult: float = 1.0
    session_mult: float = 1.0
    regime_edge_mult: float = 1.0
    notes: list[str] = field(default_factory=list)
    vetoed: bool = False
    veto_reason: str = ""


class FundManager:

    def __init__(self, cfg: dict, telegram_notifier=None, exploration_mode: bool = False):
        self._cfg = cfg
        self._telegram = telegram_notifier
        self._exploration = bool(exploration_mode)
        self._hit_milestones: set[float] = set()
        self._monthly_start_equity: float = 0.0
        self._monthly_start_ts: float = time.time()
        self._monthly_trades_start: int = 0
        # Accountability: track sizing decisions and their outcomes
        # Each entry: {symbol, scale, tier, pnl: None until closed}
        self._sizing_log: list[dict] = []
        # Performance bonus: Jim earns sizing authority by being right
        # Range: 0.0 – 0.50 (added on top of the 2.0x base cap)
        self._bonus: float = 0.0
        self._load()

    # ── Main API ──────────────────────────────────────────────────────────────

    def get_size_multiplier(
        self,
        breakdown: "SignalBreakdown",
        equity: float,
        consecutive_wins: int,
        consecutive_losses: int,
        trade_log: list["TradeRecord"],
        open_risk_pct: float = 0.0,
        daily_pnl_pct: float = 0.0,
        drawdown_pct: float = 0.0,
        open_trade_count: int = 0,
    ) -> FundDecision:
        # Exploration mode: neutral multipliers, no veto
        if self._exploration:
            return FundDecision(
                size_multiplier=1.0, conviction_mult=1.0, streak_mult=1.0,
                regime_mult=1.0, performance_mult=1.0,
                notes=["exploration mode: flat sizing"],
            )

        # ── Veto check first ─────────────────────────────────────────────────
        vetoed, veto_reason = self._veto_check(
            drawdown_pct, daily_pnl_pct, consecutive_losses, open_risk_pct
        )
        if vetoed:
            log.warning("Jim VETOED trade on %s: %s", breakdown.symbol, veto_reason)
            return FundDecision(
                size_multiplier=0.0, conviction_mult=0.0, streak_mult=0.0,
                regime_mult=0.0, performance_mult=0.0, vetoed=True,
                veto_reason=veto_reason, notes=[f"VETOED: {veto_reason}"],
            )

        conv     = self._conviction_mult(breakdown)
        stk      = self._streak_mult(consecutive_wins, consecutive_losses)
        reg      = self._regime_mult(breakdown)
        perf     = self._recent_performance_mult(trade_log)
        heat     = self._portfolio_heat_mult(open_risk_pct, open_trade_count)
        dpnl     = self._daily_pnl_mult(daily_pnl_pct)
        dd       = self._drawdown_protection_mult(drawdown_pct)
        session  = self._session_mult()
        reg_edge = self._regime_edge_mult(trade_log, breakdown)

        cap = round(2.0 + self._bonus, 2)
        combined = conv * stk * reg * perf * heat * dpnl * dd * session * reg_edge
        combined = round(max(0.40, min(cap, combined)), 3)

        notes = []
        if conv >= 1.40:
            notes.append(f"elite signal {conv:.2f}x")
        elif conv >= 1.20:
            notes.append(f"high conviction {conv:.2f}x")
        elif conv < 0.90:
            notes.append(f"weak signal {conv:.2f}x")
        if stk >= 1.15:
            notes.append(f"hot streak {stk:.2f}x")
        elif stk <= 0.85:
            notes.append(f"cold streak {stk:.2f}x")
        if reg >= 1.15:
            notes.append(f"strong regime {reg:.2f}x")
        elif reg <= 0.70:
            notes.append(f"weak regime {reg:.2f}x")
        if heat < 0.90:
            notes.append(f"portfolio heat {heat:.2f}x")
        if dpnl < 0.90:
            notes.append(f"daily P&L protection {dpnl:.2f}x")
        elif dpnl > 1.02:
            notes.append(f"good day, sizing up {dpnl:.2f}x")
        if dd < 0.90:
            notes.append(f"drawdown guard {dd:.2f}x")
        if session < 0.90:
            notes.append(f"thin market {session:.2f}x")
        elif session > 1.05:
            notes.append(f"peak session {session:.2f}x")
        if reg_edge >= 1.10:
            notes.append(f"regime edge {reg_edge:.2f}x")
        elif reg_edge <= 0.80:
            notes.append(f"regime cold {reg_edge:.2f}x")

        return FundDecision(
            size_multiplier=combined,
            conviction_mult=conv,
            streak_mult=stk,
            regime_mult=reg,
            performance_mult=perf,
            heat_mult=heat,
            daily_pnl_mult=dpnl,
            drawdown_mult=dd,
            session_mult=session,
            regime_edge_mult=reg_edge,
            notes=notes,
        )

    def update_tiers(self, equity: float) -> list[str]:
        changes = []
        for min_eq, max_trades, max_lev, risk_pct in EQUITY_TIERS:
            if equity >= min_eq:
                prev_trades = self._cfg["trading"].get("max_open_trades", 2)
                prev_lev    = self._cfg["risk"].get("max_leverage", 5)
                prev_risk   = self._cfg["risk"].get("risk_per_trade_pct", 1.5)
                if prev_trades != max_trades:
                    self._cfg["trading"]["max_open_trades"] = max_trades
                    changes.append(f"max_open_trades {prev_trades}→{max_trades}")
                if prev_lev != max_lev:
                    self._cfg["risk"]["max_leverage"] = max_lev
                    changes.append(f"max_leverage {prev_lev}x→{max_lev}x")
                if abs(prev_risk - risk_pct) > 0.01:
                    self._cfg["risk"]["risk_per_trade_pct"] = risk_pct
                    changes.append(f"risk_per_trade {prev_risk}%→{risk_pct}%")
                break
        return changes

    def check_milestones(self, equity: float) -> list[tuple[float, str, str]]:
        """Returns list of (milestone, gift_emoji, gift_desc) newly crossed."""
        new = []
        for m in MILESTONES:
            if equity >= m and m not in self._hit_milestones:
                self._hit_milestones.add(m)
                gift_emoji, gift_desc = MILESTONE_GIFTS.get(m, ("🎁", "a special reward!"))
                new.append((m, gift_emoji, gift_desc))
        if new:
            self._save()
        return new

    # ── Risk-adjusted metrics ─────────────────────────────────────────────────

    def sharpe_ratio(self, equity_curve: list[float]) -> float:
        returns = self._returns(equity_curve)
        if len(returns) < 3:
            return 0.0
        mean_r = sum(returns) / len(returns)
        std_r  = (sum((r - mean_r) ** 2 for r in returns) / len(returns)) ** 0.5
        if std_r == 0:
            return 0.0
        return round((mean_r / std_r) * (4 * 365) ** 0.5, 3)

    def sortino_ratio(self, equity_curve: list[float]) -> float:
        returns = self._returns(equity_curve)
        if len(returns) < 3:
            return 0.0
        mean_r   = sum(returns) / len(returns)
        downside = [r for r in returns if r < 0]
        if not downside:
            return 9.99
        down_std = (sum(r ** 2 for r in downside) / len(downside)) ** 0.5
        if down_std == 0:
            return 0.0
        return round((mean_r / down_std) * (4 * 365) ** 0.5, 3)

    def calmar_ratio(self, equity_curve: list[float], peak_equity: float, equity: float) -> float:
        if len(equity_curve) < 2 or peak_equity == 0:
            return 0.0
        dd = (peak_equity - equity) / peak_equity
        if dd == 0:
            return 9.99
        returns = self._returns(equity_curve)
        mean_r = sum(returns) / len(returns) if returns else 0
        return round((mean_r * 4 * 365) / dd, 3)

    def build_report(
        self,
        trade_log: list["TradeRecord"],
        equity: float,
        peak_equity: float,
        equity_curve: list[float],
        mode: str,
    ) -> dict:
        n = len(trade_log)
        if n == 0:
            return {"trades": 0, "equity": equity}

        wins   = [t for t in trade_log if t.pnl_usd > 0]
        losses = [t for t in trade_log if t.pnl_usd <= 0]
        gp     = sum(t.pnl_usd for t in wins)
        gl     = abs(sum(t.pnl_usd for t in losses)) or 0.001

        win_rate = len(wins) / n
        pf       = gp / gl
        avg_win  = (sum(t.pnl_pct for t in wins) / len(wins)) if wins else 0
        avg_loss = (abs(sum(t.pnl_pct for t in losses)) / len(losses)) if losses else 0.001
        avg_rr   = avg_win / avg_loss if avg_loss > 0 else 0
        drawdown = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0

        monthly_elapsed  = time.time() - self._monthly_start_ts
        monthly_pnl_pct  = (
            (equity - self._monthly_start_equity) / self._monthly_start_equity * 100
            if self._monthly_start_equity > 0 else 0
        )
        monthly_trades = n - self._monthly_trades_start

        best  = max(trade_log, key=lambda t: t.pnl_usd)
        worst = min(trade_log, key=lambda t: t.pnl_usd)

        return {
            "trades": n, "win_rate": win_rate, "profit_factor": pf, "avg_rr": avg_rr,
            "gross_profit": gp, "gross_loss": gl, "net_pnl": gp - gl,
            "drawdown_pct": drawdown, "equity": equity, "peak_equity": peak_equity,
            "best_trade": best.pnl_usd, "worst_trade": worst.pnl_usd,
            "sharpe": self.sharpe_ratio(equity_curve),
            "sortino": self.sortino_ratio(equity_curve),
            "calmar": self.calmar_ratio(equity_curve, peak_equity, equity),
            "monthly_return_pct": monthly_pnl_pct,
            "monthly_trades": monthly_trades,
            "monthly_days": monthly_elapsed / 86400,
            "next_milestone": next((m for m in sorted(MILESTONES) if m > equity), None),
            "mode": mode,
            "attribution": build_attribution_report(trade_log, min_trades=2),
        }

    def reset_monthly(self, equity: float, trade_count: int) -> None:
        self._monthly_start_equity = equity
        self._monthly_start_ts = time.time()
        self._monthly_trades_start = trade_count
        self._save()

    # ── Multiplier components ─────────────────────────────────────────────────

    def _conviction_mult(self, b: "SignalBreakdown") -> float:
        """
        Signal quality: score + gates + EV magnitude.

        UPGRADED: More aggressive tiers — the old system gave 1.0x to a score=63
        setup which is already a good signal. Now a score=55 with positive EV
        still gets 1.0x (baseline), and high-conviction gets up to 1.80x.

        Rationale: The EV model + regime + smart money gates already filter
        garbage. If a signal passes all gates, it DESERVES to be sized properly.
        The conviction multiplier's job is to differentiate GREAT from GOOD,
        not to handicap everything that isn't perfect.
        """
        score  = b.total_score
        gates  = int(b.regime_ok) + int(b.smart_money_ok) + int(b.ev_ok)
        ev_pct = b.ev_result.ev_net_pct if b.ev_result else 0.0
        p_win  = b.ev_result.p_win if b.ev_result else 0.52
        confidence = b.ev_result.confidence if b.ev_result else 0.4

        # Tier 1: ELITE — everything aligned, proven edge, max size
        if score >= 82 and gates == 3 and ev_pct > 0.30 and p_win >= 0.58 and confidence >= 0.7:
            return 1.80
        # Tier 2: HIGH CONVICTION — strong signal with positive EV
        elif score >= 75 and gates == 3 and ev_pct > 0.15 and p_win >= 0.52:
            return 1.50
        # Tier 3: SOLID — good score, gates passing, EV positive
        elif score >= 65 and gates == 3 and ev_pct > 0.05:
            return 1.30
        # Tier 4: ACCEPTABLE — passes threshold, at least 2 gates
        elif score >= 55 and gates >= 2 and ev_pct > 0:
            return 1.15
        # Tier 5: MARGINAL — barely qualifying
        elif score >= 45 and gates >= 2:
            return 1.00
        # Below threshold — should rarely reach here due to upstream filters
        else:
            return 0.80

    def _streak_mult(self, wins: int, losses: int) -> float:
        """
        Anti-martingale: cut size after losses, ride winning streaks HARD.

        UPGRADED: More aggressive on winning streaks. When the system is hot,
        the data says to push — this is the core of anti-martingale sizing
        (Vince 1992, "The Mathematics of Money Management").
        """
        if losses >= 3:
            return 0.60   # 3+ losses: hard cut, protect capital
        elif losses >= 2:
            return 0.75
        elif losses == 1:
            return 0.88
        elif wins >= 7:
            return 1.50   # 7+ wins: system is ON FIRE — press hard
        elif wins >= 5:
            return 1.40
        elif wins >= 4:
            return 1.30
        elif wins >= 3:
            return 1.20
        elif wins >= 2:
            return 1.12
        elif wins >= 1:
            return 1.05
        return 1.00

    def _regime_mult(self, b: "SignalBreakdown") -> float:
        """
        Regime-direction aware sizing.

        UPGRADED: Distribution regime is no longer universally penalized.
        If you're SHORT in distribution, that's a HIGH-EDGE setup (institutional
        selling pressure). Only penalize if direction conflicts with regime.
        """
        regime   = b.regime.value if b.regime else "chaos"
        sm_phase = b.smart_money.phase.value if b.smart_money else "neutral"
        direction = b.direction

        # Direction-aware regime scoring
        if regime == "trending_expansion":
            base = 1.25  # Upgraded from 1.20
        elif regime == "accumulation_compression":
            base = 1.05  # Upgraded from 1.00
        elif regime == "distribution":
            if direction == "short":
                base = 1.20  # HIGH: shorts in distribution are premium setups
            else:
                base = 0.70  # Penalize longs fighting distribution
        elif regime == "chaos":
            base = 0.50  # Reduced from 0.55 — chaos is truly dangerous
        else:
            base = 1.00

        # Smart money alignment bonus
        if sm_phase in ("trending", "accumulation") and direction == "long":
            base = min(base * 1.12, 1.40)
        elif sm_phase == "distribution" and direction == "short":
            base = min(base * 1.12, 1.40)  # NEW: SM confirms short direction
        elif sm_phase == "liquidity_sweep":
            base = min(base * 1.08, 1.35)  # Sweeps are high-conviction reversals
        elif sm_phase == "distribution" and direction == "long":
            base *= 0.85  # Fighting SM direction

        return round(base, 3)

    def _recent_performance_mult(self, trade_log: list["TradeRecord"]) -> float:
        """
        Last 15 trades profit factor — overall system momentum.

        UPGRADED: More aggressive scaling on proven performance.
        If last 15 trades show PF > 2, the system is working — SIZE UP.
        """
        if len(trade_log) < 5:
            return 1.0
        recent = trade_log[-15:]
        wins   = [t for t in recent if t.pnl_usd > 0]
        losses = [t for t in recent if t.pnl_usd <= 0]
        gp     = sum(t.pnl_usd for t in wins)
        gl     = abs(sum(t.pnl_usd for t in losses)) or 0.001
        pf     = gp / gl

        if pf >= 5.0:   return 1.45  # System is crushing it
        elif pf >= 3.5: return 1.35
        elif pf >= 2.5: return 1.25
        elif pf >= 2.0: return 1.18
        elif pf >= 1.5: return 1.10
        elif pf >= 1.2: return 1.05
        elif pf >= 1.0: return 1.00
        elif pf >= 0.8: return 0.88
        elif pf >= 0.6: return 0.75
        else:           return 0.65  # System bleeding — cut hard

    def _portfolio_heat_mult(self, open_risk_pct: float, open_trade_count: int) -> float:
        """
        Scale down as portfolio risk exposure grows.
        A real fund manager never lets one bad day wipe out the book.
        """
        if open_trade_count == 0:
            return 1.0
        if open_risk_pct >= 4.0:
            return 0.60   # Almost at max risk — minimal new exposure
        elif open_risk_pct >= 3.0:
            return 0.75
        elif open_risk_pct >= 2.0:
            return 0.85
        elif open_trade_count >= 2:
            return 0.90   # Two trades already on, be measured
        return 1.0

    def _daily_pnl_mult(self, daily_pnl_pct: float) -> float:
        """
        Protect profits: if we've had a great day, reduce size to lock in gains.
        Don't revenge trade after a bad day either.
        """
        if daily_pnl_pct >= 5.0:
            return 0.65   # Great day — protect it, reduce aggression
        elif daily_pnl_pct >= 3.0:
            return 0.75
        elif daily_pnl_pct >= 1.5:
            return 0.90
        elif daily_pnl_pct >= 0.0:
            return 1.00   # Flat to slightly up — normal
        elif daily_pnl_pct >= -2.0:
            return 0.90   # Small loss — slightly cautious
        elif daily_pnl_pct >= -4.0:
            return 0.75   # Meaningful loss — cut size
        else:
            return 0.60   # Bad day — survive, don't dig deeper

    def _drawdown_protection_mult(self, drawdown_pct: float) -> float:
        """
        Progressive protection as drawdown from peak increases.
        Preserve capital above all else.
        """
        if drawdown_pct >= 12.0:
            return 0.50
        elif drawdown_pct >= 8.0:
            return 0.65
        elif drawdown_pct >= 5.0:
            return 0.80
        elif drawdown_pct >= 3.0:
            return 0.92
        return 1.0

    def _session_mult(self) -> float:
        """
        Market liquidity varies by time of day.
        Peak sessions: EU open (07-12 UTC), US open (13-17 UTC).
        Dead zone: late US / early Asia (21-03 UTC) — thin, unpredictable.
        """
        hour = time.gmtime().tm_hour
        if 7 <= hour < 12:
            return 1.10   # EU open — trend initiation, good fills
        elif 13 <= hour < 17:
            return 1.10   # US open — high volume
        elif 12 <= hour < 13:
            return 1.05   # EU/US overlap
        elif 3 <= hour < 7:
            return 0.95   # Pre-EU — moderate
        elif 17 <= hour < 21:
            return 1.00   # US afternoon
        else:
            return 0.80   # 21-03 UTC — thin market, wide spreads

    def _regime_edge_mult(
        self, trade_log: list["TradeRecord"], b: "SignalBreakdown"
    ) -> float:
        """
        Per-regime win rate over last 10 trades in this specific regime.
        Jim sizes up in regimes where he's been right, cuts in regimes where he's wrong.
        """
        if not b.regime or len(trade_log) < 5:
            return 1.0

        regime_str = b.regime.value
        regime_trades = [
            t for t in trade_log
            if getattr(t, "regime", None) == regime_str
        ][-10:]

        if len(regime_trades) < 4:
            return 1.0   # Not enough data for this regime

        wins = sum(1 for t in regime_trades if t.pnl_usd > 0)
        wr   = wins / len(regime_trades)

        if wr >= 0.70:
            return 1.20   # Strong edge in this regime
        elif wr >= 0.60:
            return 1.10
        elif wr >= 0.50:
            return 1.00
        elif wr >= 0.40:
            return 0.85
        else:
            return 0.70   # Losing money in this regime — cut way down

    def _veto_check(
        self,
        drawdown_pct: float,
        daily_pnl_pct: float,
        consecutive_losses: int,
        open_risk_pct: float,
    ) -> tuple[bool, str]:
        """
        Jim's right to say NO. Hard stops that override all other logic.
        These conditions mean the risk/reward of opening ANY new position is negative.
        """
        if self._exploration:
            return False, ""
        paper_validation_mode = (
            self._cfg.get("trading", {}).get("mode") == "paper"
            and bool(self._cfg.get("paper_validation", {}).get("enabled", True))
        )
        if open_risk_pct >= 4.5:
            return True, f"portfolio at max capacity ({open_risk_pct:.1f}% risk on)"
        if drawdown_pct >= 10.0 and daily_pnl_pct < -2.0 and not paper_validation_mode:
            return True, f"double danger: {drawdown_pct:.1f}% drawdown + {daily_pnl_pct:.1f}% today"
        if consecutive_losses >= 2 and daily_pnl_pct < -3.0 and not paper_validation_mode:
            return True, f"{consecutive_losses} consecutive losses on a bad day ({daily_pnl_pct:.1f}%)"
        if drawdown_pct >= 13.0 and not paper_validation_mode:
            return True, f"near max drawdown ({drawdown_pct:.1f}%) — capital preservation mode"
        return False, ""

    # ── Accountability tracking ───────────────────────────────────────────────

    def record_sizing_decision(self, symbol: str, scale: float) -> None:
        """Log a sizing decision at trade open — outcome filled in on close."""
        tier = "high" if scale >= 1.3 else ("low" if scale <= 0.7 else "normal")
        self._sizing_log.append({
            "symbol": symbol, "scale": scale, "tier": tier,
            "ts": time.time(), "pnl": None,
        })
        # Keep last 200 records
        if len(self._sizing_log) > 200:
            self._sizing_log = self._sizing_log[-200:]

    def record_sizing_outcome(self, symbol: str, pnl: float) -> str | None:
        """
        Fill in P&L and update Jim's bonus.
        Returns a bonus message if the bonus changed meaningfully, else None.
        """
        for rec in reversed(self._sizing_log):
            if rec["symbol"] == symbol and rec["pnl"] is None:
                rec["pnl"] = pnl
                # Only high-conviction calls affect the bonus — Jim is judged on his best bets
                if rec["tier"] == "high":
                    prev = self._bonus
                    if pnl > 0:
                        self._bonus = min(0.50, self._bonus + 0.05)
                    else:
                        self._bonus = max(0.0, self._bonus - 0.10)
                    self._bonus = round(self._bonus, 2)
                    if self._bonus != prev:
                        direction = "📈" if self._bonus > prev else "📉"
                        msg = (
                            f"{direction} Jim's bonus {'increased' if self._bonus > prev else 'reduced'} "
                            f"to `+{self._bonus:.2f}x` (cap now `{2.0 + self._bonus:.2f}x`)"
                        )
                        log.info("Jim bonus update: %.2f → %.2f", prev, self._bonus)
                        return msg
                break
        return None

    @property
    def bonus(self) -> float:
        return self._bonus

    @property
    def size_cap(self) -> float:
        return round(2.0 + self._bonus, 2)

    def sizing_accuracy(self) -> dict:
        """
        Reports Jim's win rate by sizing tier.
        Tells us: when Jim sized up, was he right? When he pulled back, was he right?
        """
        closed = [r for r in self._sizing_log if r["pnl"] is not None]
        if not closed:
            return {}
        result = {}
        for tier in ("high", "normal", "low"):
            tier_trades = [r for r in closed if r["tier"] == tier]
            if not tier_trades:
                continue
            wins = sum(1 for r in tier_trades if r["pnl"] > 0)
            avg_pnl = sum(r["pnl"] for r in tier_trades) / len(tier_trades)
            result[tier] = {
                "trades": len(tier_trades),
                "win_rate": wins / len(tier_trades),
                "avg_pnl": round(avg_pnl, 3),
            }
        return result

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(STATE_PATH, json.dumps({
                "hit_milestones":       list(self._hit_milestones),
                "monthly_start_equity": self._monthly_start_equity,
                "monthly_start_ts":     self._monthly_start_ts,
                "monthly_trades_start": self._monthly_trades_start,
                "bonus":                self._bonus,
            }, indent=2))
        except Exception as e:
            log.warning("FundManager save failed: %s", e)

    def _load(self) -> None:
        try:
            if STATE_PATH.exists():
                s = json.loads(STATE_PATH.read_text())
                self._hit_milestones       = set(s.get("hit_milestones", []))
                self._monthly_start_equity = s.get("monthly_start_equity", 0.0)
                self._monthly_start_ts     = s.get("monthly_start_ts", time.time())
                self._monthly_trades_start = s.get("monthly_trades_start", 0)
                self._bonus                = float(s.get("bonus", 0.0))
        except Exception as e:
            log.warning("FundManager load failed: %s", e)

    @staticmethod
    def _returns(equity_curve: list[float]) -> list[float]:
        return [
            (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            for i in range(1, len(equity_curve))
            if equity_curve[i - 1] != 0
        ]
