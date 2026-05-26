"""
Rejection forensics — per-candidate decomposition of every pre-execution reject.

Zero behavioral impact: diagnostics only. Writes to data/rejections.parquet with
a flush cadence (default 25) to bound I/O. Every rejected candidate also emits
one structured log line so logs remain greppable without the parquet.

Stages recognized:
  gate_regime, gate_sm, gate_ev   — institutional gate failures (pre-score)
  score_threshold                 — total_score below regime-adjusted threshold
  ml_rejected                     — ML p_win below dynamic threshold
  regime_gate                     — progressive ML regime gate hard-lock
  risk_guard                      — RiskManager.can_open_trade() blocked
  extreme_vol                     — safety volatility halt
  fm_veto                         — FundManager veto
  setup_none                      — RiskManager.calculate_setup() returned None
  confirmation_pending            — optional, for visibility on why signals idle
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_PATH = Path("data/rejections.parquet")
DEFAULT_FLUSH_EVERY = 25

# Stage → human-readable label for log lines
_STAGE_LABELS = {
    "gate_regime":          "G-REG",
    "gate_sm":              "G-SM",
    "gate_ev":              "G-EV",
    "score_threshold":      "SCORE",
    "ml_rejected":          "ML",
    "regime_gate":          "RGATE",
    "risk_guard":           "RISK",
    "extreme_vol":          "VOL",
    "fm_veto":              "FM",
    "setup_none":           "SETUP",
    "confirmation_pending": "CONF",
}


@dataclass
class RejectionRecord:
    ts: float
    symbol: str
    stage: str
    reject_reason: str

    # Context
    setup_type: str
    strategy_sleeve: str
    setup_quality_score: float
    direction: str
    total_score: float
    regime_state: str
    smart_money_state: str
    smart_money_score: float
    btc_state: str
    btcd_state: str

    # EV decomposition
    p_win: float
    avg_win: float
    avg_loss: float
    real_rr: float
    fees: float
    slippage: float
    spread: float
    funding_cost: float
    funding_rate: float
    raw_ev: float
    adjusted_ev: float
    regime_factor: float
    bootstrap_penalty: float
    confidence: float
    confidence_penalty: float

    # Gate diagnostics
    gate_regime_ok: bool
    gate_sm_ok: bool
    gate_ev_ok: bool

    # ML / FM / threshold context
    ml_p_win: float
    ml_threshold: float
    fm_scale: float
    threshold_required: float
    trade_count: int


class RejectionLogger:
    """Structured forensic sink for every pre-execution rejection."""

    def __init__(
        self,
        cfg: dict,
        path: Path | str = DEFAULT_PATH,
        flush_every: int = DEFAULT_FLUSH_EVERY,
    ):
        rej_cfg = cfg.get("rejection_logger", {}) if isinstance(cfg, dict) else {}
        self._enabled = rej_cfg.get("enabled", True)
        self._path = Path(rej_cfg.get("path", str(path)))
        self._flush_every = int(rej_cfg.get("flush_every", flush_every))

        ev_cfg = cfg.get("ev_model", {}) if isinstance(cfg, dict) else {}
        # Raw percentages (not fractions) — matches EVModel convention on disk
        self._taker_fee_pct = float(ev_cfg.get("taker_fee_pct", 0.04))
        self._slippage_pct = float(ev_cfg.get("slippage_pct", 0.05))
        self._min_ev_pct = float(ev_cfg.get("min_ev_pct", 0.10))
        self._min_confidence = float(ev_cfg.get("min_confidence", 0.55))
        self._min_trades = int(ev_cfg.get("min_trades_for_ev", 20))

        self._buffer: list[dict] = []
        # Per-cycle tally — independent of disk buffer so flush failures don't
        # erase visibility. Reset by cycle_summary(reset=True) at the cycle
        # boundary in main.py after the Telegram report has consumed it.
        self._cycle_counts: Counter = Counter()
        self._cycle_reasons: dict[str, Counter] = {}
        if self._enabled:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def log(
        self,
        *,
        stage: str,
        reason: str,
        breakdown=None,
        symbol: str | None = None,
        threshold_required: float = 0.0,
        fm_scale: float = 0.0,
        ml_p_win: float = 0.0,
        ml_threshold: float = 0.0,
        btc_state: str = "",
        btcd_state: str = "",
        funding_rate: float = 0.0,
    ) -> None:
        if not self._enabled:
            return
        try:
            rec = self._build(
                stage=stage,
                reason=reason,
                breakdown=breakdown,
                symbol=symbol,
                threshold_required=threshold_required,
                fm_scale=fm_scale,
                ml_p_win=ml_p_win,
                ml_threshold=ml_threshold,
                btc_state=btc_state,
                btcd_state=btcd_state,
                funding_rate=funding_rate,
            )
            label = _STAGE_LABELS.get(stage, stage.upper())
            log.info(
                "REJECT[%s] %s reason=%s score=%.1f thresh=%.1f "
                "p_win=%.3f ev_net=%+.3f%% regime=%s sm=%s fm=%.2f",
                label, rec.symbol, rec.reject_reason, rec.total_score,
                rec.threshold_required, rec.p_win, rec.adjusted_ev,
                rec.regime_state or "-", rec.smart_money_state or "-",
                rec.fm_scale,
            )
            self._buffer.append(asdict(rec))
            # Per-cycle tally (separate from disk buffer)
            self._cycle_counts[stage] += 1
            stage_reasons = self._cycle_reasons.setdefault(stage, Counter())
            stage_reasons[reason] += 1
            if len(self._buffer) >= self._flush_every:
                self.flush()
        except Exception as exc:
            log.warning("RejectionLogger.log failed (%s): %s", stage, exc)

    def flush(self) -> None:
        if not self._enabled or not self._buffer:
            return
        try:
            df_new = pd.DataFrame(self._buffer)
            if self._path.exists():
                try:
                    df_old = pd.read_parquet(self._path)
                    df = pd.concat([df_old, df_new], ignore_index=True)
                except Exception as exc:
                    log.warning("RejectionLogger: could not read existing parquet (%s) — overwriting", exc)
                    df = df_new
            else:
                df = df_new
            df.to_parquet(self._path, index=False)
            self._buffer.clear()
        except Exception as exc:
            log.warning("RejectionLogger.flush failed: %s", exc)

    # ------------------------------------------------------------------ #
    #  Per-cycle summary (telemetry for Telegram cycle report)             #
    # ------------------------------------------------------------------ #

    def cycle_summary(self, top_n: int = 3, reset: bool = False) -> dict:
        """Return aggregated rejection counts since the last reset.

        Args:
            top_n:  how many stages to surface (sorted by count desc).
            reset:  if True, clear the per-cycle tally after snapshotting.

        Returns:
            {
              "total": int,                     # all rejects this window
              "stages": [
                  {"stage": str, "count": int,
                   "top_reason": str, "top_count": int}, ...
              ]
            }
            Empty stages list when nothing was logged.
        """
        total = int(sum(self._cycle_counts.values()))
        stages = []
        for stage, count in self._cycle_counts.most_common(max(1, int(top_n))):
            reasons = self._cycle_reasons.get(stage)
            if reasons:
                top_reason, top_count = reasons.most_common(1)[0]
            else:
                top_reason, top_count = "", 0
            stages.append({
                "stage": stage,
                "count": int(count),
                "top_reason": str(top_reason),
                "top_count": int(top_count),
            })
        if reset:
            self._cycle_counts.clear()
            self._cycle_reasons.clear()
        return {"total": total, "stages": stages}

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _build(
        self,
        *,
        stage,
        reason,
        breakdown,
        symbol,
        threshold_required,
        fm_scale,
        ml_p_win,
        ml_threshold,
        btc_state,
        btcd_state,
        funding_rate,
    ) -> RejectionRecord:
        b = breakdown
        ev = b.ev_result if (b is not None and b.ev_result is not None) else None
        sm = b.smart_money if (b is not None and b.smart_money is not None) else None

        if ev is not None:
            p_win = ev.p_win
            avg_win = ev.avg_win_pct
            avg_loss = ev.avg_loss_pct
            real_rr = (avg_win / avg_loss) if avg_loss > 0 else 0.0
            cost = ev.cost_pct
            raw_ev = ev.ev_gross_pct
            adj_ev = ev.ev_net_pct
            confidence = ev.confidence
            trade_count = ev.trade_count
        else:
            p_win = avg_win = avg_loss = real_rr = 0.0
            cost = raw_ev = adj_ev = confidence = 0.0
            trade_count = 0

        # Cost decomposition — EVModel stores cost as round-trip (fee+slip)*2 + funding.
        # We reverse that here for forensic visibility. Values remain in %.
        fees = self._taker_fee_pct * 2
        slippage = self._slippage_pct * 2
        funding_cost = max(0.0, cost - fees - slippage)

        # Bootstrap penalty: fraction of prior weight applied when n<min_trades.
        # Matches EVModel.compute Bayesian blend formula (cap=0.2).
        bootstrap_penalty = 0.0
        if trade_count < self._min_trades and self._min_trades > 0:
            blend = min(trade_count / self._min_trades, 0.2)
            bootstrap_penalty = round(1.0 - blend, 4)

        # Confidence penalty: how far below the min_confidence gate we sit.
        confidence_penalty = max(0.0, self._min_confidence - confidence)

        sym = symbol if symbol is not None else (b.symbol if b is not None else "")
        direction = b.direction if b is not None else ""
        regime_state = b.regime.value if (b is not None and b.regime is not None) else ""
        setup_type = ""
        strategy_sleeve = ""
        setup_quality_score = 0.0
        if b is not None:
            setup_type = str(getattr(b, "setup_type", "") or "")
            strategy_sleeve = str(getattr(b, "strategy_sleeve", "") or "")
            setup_quality_score = float(getattr(b, "setup_quality_score", 0.0) or 0.0)
            passport = getattr(b, "setup_passport", None)
            if not setup_type and passport is not None:
                setup_type = str(getattr(passport, "setup_type", "") or "")
            if not strategy_sleeve and passport is not None:
                strategy_sleeve = str(getattr(passport, "sleeve", "") or "")
            if not setup_quality_score and passport is not None:
                setup_quality_score = float(getattr(passport, "quality_score", 0.0) or 0.0)
        if not setup_type:
            setup_type = f"{regime_state or 'na'}_{direction or 'na'}"
        if not strategy_sleeve:
            strategy_sleeve = "unknown"

        return RejectionRecord(
            ts=time.time(),
            symbol=sym,
            stage=stage,
            reject_reason=reason,
            setup_type=setup_type,
            strategy_sleeve=strategy_sleeve,
            setup_quality_score=round(float(setup_quality_score), 4),
            direction=direction,
            total_score=float(b.total_score) if b is not None else 0.0,
            regime_state=regime_state,
            smart_money_state=sm.phase.value if sm is not None else "",
            smart_money_score=float(sm.score) if sm is not None else 0.0,
            btc_state=btc_state or "",
            btcd_state=btcd_state or "",
            p_win=round(float(p_win), 4),
            avg_win=round(float(avg_win), 4),
            avg_loss=round(float(avg_loss), 4),
            real_rr=round(float(real_rr), 4),
            fees=round(fees, 4),
            slippage=round(slippage, 4),
            spread=0.0,
            funding_cost=round(float(funding_cost), 6),
            funding_rate=round(float(funding_rate), 6),
            raw_ev=round(float(raw_ev), 4),
            adjusted_ev=round(float(adj_ev), 4),
            regime_factor=1.0,
            bootstrap_penalty=bootstrap_penalty,
            confidence=round(float(confidence), 4),
            confidence_penalty=round(float(confidence_penalty), 4),
            gate_regime_ok=bool(b.regime_ok) if b is not None else False,
            gate_sm_ok=bool(b.smart_money_ok) if b is not None else False,
            gate_ev_ok=bool(b.ev_ok) if b is not None else False,
            ml_p_win=round(float(ml_p_win), 4),
            ml_threshold=round(float(ml_threshold), 4),
            fm_scale=round(float(fm_scale), 4),
            threshold_required=round(float(threshold_required), 4),
            trade_count=int(trade_count),
        )
