"""
Phase 1 ML Dataset Logger.

Append-only parquet log of trade rows and near-miss no-trade rows.

Near-miss filter: only logs no-trade decisions where score >= NEAR_MISS_FLOOR
to prevent 43k-rows/day ratio imbalance from low-score rejects.

Lifecycle for trades:
  1. open_record(bd, setup, snap)  → returns record_id (str)
  2. close_record(record_id, ...)  → completes the row and flushes to parquet

For no-trade near misses:
  log_no_trade(bd, reason, snap)  → single call, filtered by NEAR_MISS_FLOOR
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.session_clock import active_market_session_key

log = logging.getLogger(__name__)

NEAR_MISS_FLOOR = 55.0      # min score to log a no-trade row
SCHEMA_VERSION  = "1.0"

_REGIME_MAP = {
    "trending_expansion": 3,
    "accumulation_compression": 2,
    "distribution": 1,
    "chaos": 0,
}


def _session_from_utc(dt_utc: datetime) -> str:
    """Canonical session bucket for a UTC timestamp.

    The whole codebase keys session attribution off
    ``session_clock.active_market_session_key`` (Asia/Makassar / WITA).
    Historically this module rolled its own UTC-hour table with
    different cutoffs, which produced the same string vocabulary but
    misaligned bucket boundaries — TradeRecord.session and the parquet
    row for the same trade could disagree by ~1-2 hours.

    Funnel everything through ``active_market_session_key`` so per-
    session attribution lines up across TradeRecord, dataset_logger,
    cohort attribution, and the Telegram cycle report.
    """
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    return active_market_session_key(dt_utc)


@dataclass
class TradeDataPoint:
    # ── Identity ──────────────────────────────────────────────────────────
    record_id: str
    schema_version: str
    record_type: str        # "trade" | "no_trade" | "shadow_trade"
    source: str             # "paper" | "shadow" | "backtest" | "live"
    decision: str           # "TRADE" | "NO_TRADE"
    near_miss: bool         # True if score >= NEAR_MISS_FLOOR but trade not taken

    # ── Core ──────────────────────────────────────────────────────────────
    symbol: str
    direction: str          # "long" | "short" | "none"
    opened_at: float
    hour_of_day: int
    day_of_week: int
    session: str            # "asia" | "london" | "ny" | "overlap_london_ny"

    # ── Signal ────────────────────────────────────────────────────────────
    total_score: float
    score_threshold: float
    passed_filter: int      # 1 if all gates passed, else 0
    regime_ok: bool
    smart_money_ok: bool
    ev_ok: bool
    no_trade_reason: str    # "" for trades
    cohort_key: str
    lifecycle_status: str
    recommendation: str

    # ── Score components ─────────────────────────────────────────────────
    score_trend_strength: float
    score_volume_confirmation: float
    score_structure_quality: float
    score_open_interest: float
    score_funding_sentiment: float
    score_order_book: float
    score_volatility: float

    # ── EV model ─────────────────────────────────────────────────────────
    ev_net_pct: float
    ev_gross_pct: float
    ev_cost_pct: float
    p_win: float
    p_loss: float
    kelly_fraction: float
    signal_confidence: float
    ev_trade_count: int
    ev_is_tradeable: bool

    # ── Regime & feature vector ───────────────────────────────────────────
    regime: str
    regime_code: int
    volatility_regime: float
    momentum_strength: float
    oi_change_rate: float
    funding_deviation: float
    volume_anomaly_score: float
    order_flow_imbalance: float
    liquidation_pressure: float
    market_structure: str
    directional_alignment: float

    # ── Smart money ───────────────────────────────────────────────────────
    sm_phase: str
    sm_score: float
    sm_direction_bias: str
    sm_aligns: bool

    # ── Spot context ──────────────────────────────────────────────────────
    spot_basis_pct: float
    spot_volume_ratio: float
    coinbase_premium_pct: float

    # ── Order flow (from MarketSnapshot) ─────────────────────────────────
    funding_rate: float
    oi_change_pct: float
    long_short_ratio: float
    taker_buy_ratio: float
    volume_24h_usdt: float
    timeframe: str = "unknown"
    asset: str = "unknown"
    signal_type: str = "unknown"
    entry_reason: str = "unknown"
    volatility_bucket: str = "unknown"
    trend_bucket: str = "unknown"
    strategy_name: str = "unknown"
    setup_type: str = "unknown"
    setup_quality_score: float = 0.0
    setup_sector: str = "unknown"
    setup_rotation_state: str = "unknown"
    setup_crowding_state: str = "unknown"
    setup_funding_state: str = "unknown"
    setup_microstructure_state: str = "unknown"
    setup_expected_path: str = "unknown"
    setup_hold_profile: str = "unknown"

    # ── Broad market context ─────────────────────────────────────────────
    market_btc_trend: str = "unknown"
    market_eth_btc_trend: str = "unknown"
    market_btc_d_trend: str = "unknown"
    market_total_trend: str = "unknown"
    market_risk_on_state: str = "unknown"
    market_rotation_state: str = "unknown"
    market_context_confidence: float = 0.0
    sector_rotation_state: str = "unknown"
    sector_rotation_rank: int = 0
    sector_rotation_relative_btc_pct: float = 0.0
    sector_rotation_breadth: float = 0.0
    sector_rotation_confidence: float = 0.0

    # ── Causal time-series diagnostics ──────────────────────────────────
    time_series_state: str = "unknown"
    time_series_ewma_vol_pct: float = 0.0
    time_series_realized_vol_pct: float = 0.0
    time_series_volatility_ratio: float = 1.0
    time_series_latest_abs_return_z: float = 0.0
    time_series_drift_t_stat: float = 0.0
    time_series_stability_score: float = 0.0
    time_series_markov_state: str = "unknown"
    time_series_markov_bull_prob: float = 0.0
    time_series_markov_bear_prob: float = 0.0
    time_series_markov_sideways_prob: float = 0.0
    time_series_markov_edge: float = 0.0
    time_series_score_mult: float = 1.0
    time_series_size_mult: float = 1.0
    time_series_reason: str = ""

    # ── Position setup (null for no_trade) ───────────────────────────────
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    r_distance: Optional[float] = None
    size_usd: Optional[float] = None
    size_contracts: Optional[float] = None
    risk_pct: Optional[float] = None
    atr: Optional[float] = None
    leverage: Optional[int] = None
    fees_estimate_usd: Optional[float] = None
    slippage_estimate_usd: Optional[float] = None
    fees_slippage_pct: Optional[float] = None

    # ── Trade dynamics (null for no_trade) ───────────────────────────────
    mfe_r: Optional[float] = None
    mae_r: Optional[float] = None
    mfe_usd: Optional[float] = None
    mae_usd: Optional[float] = None
    time_to_mfe_peak_s: Optional[float] = None
    drawdown_duration_s: Optional[float] = None
    hold_duration_s: Optional[float] = None
    tp1_hit: Optional[bool] = None

    # ── Outcome (null for no_trade) ───────────────────────────────────────
    closed_at: Optional[float] = None
    exit_price: Optional[float] = None
    exit_type: Optional[str] = None
    pnl_usd: Optional[float] = None
    pnl_pct: Optional[float] = None
    rr_achieved: Optional[float] = None
    result: Optional[int] = None        # 1 = win, 0 = loss

    # ── Backtest only ─────────────────────────────────────────────────────
    entry_bar: Optional[int] = None
    exit_bar: Optional[int] = None


# ─────────────────────────────────────────────────────────────────────────────

def _extract_signal_fields(bd, snap, score_threshold: float) -> dict:
    """Pull all signal-time fields out of a SignalBreakdown + MarketSnapshot."""
    now = time.time()
    dt  = datetime.fromtimestamp(now, tz=timezone.utc)

    ev  = bd.ev_result
    fv  = bd.feature_vector
    sm  = bd.smart_money
    sc  = bd.spot_context or (snap.spot_context if snap else None)
    passport = getattr(bd, "setup_passport", None)
    market_ctx = getattr(bd, "market_context", None) or (getattr(snap, "market_context", None) if snap else None)
    sector_rotation = getattr(bd, "sector_rotation", None) or (getattr(snap, "sector_rotation", None) if snap else None)
    ts_diag = getattr(bd, "time_series", None) or (getattr(snap, "time_series", None) if snap else None)

    regime_str  = bd.regime.value if bd.regime else "chaos"
    regime_code = _REGIME_MAP.get(regime_str, 0)

    return dict(
        # Core
        symbol=bd.symbol,
        direction=bd.direction,
        opened_at=now,
        hour_of_day=dt.hour,
        day_of_week=dt.weekday(),
        session=_session_from_utc(dt),

        # Signal
        total_score=bd.total_score,
        score_threshold=score_threshold,
        passed_filter=1 if bd.all_gates_passed else 0,
        regime_ok=bd.regime_ok,
        smart_money_ok=bd.smart_money_ok,
        ev_ok=bd.ev_ok,
        cohort_key=str(getattr(bd, "cohort_key", "") or ""),
        lifecycle_status=str(getattr(bd, "lifecycle_status", "RESEARCH") or "RESEARCH"),
        recommendation=str(getattr(bd, "recommendation", "PAPER_ONLY") or "PAPER_ONLY"),

        # Score components
        score_trend_strength=bd.trend_strength,
        score_volume_confirmation=bd.volume_confirmation,
        score_structure_quality=bd.structure_quality,
        score_open_interest=bd.open_interest,
        score_funding_sentiment=bd.funding_sentiment,
        score_order_book=bd.order_book,
        score_volatility=bd.volatility,

        # EV model
        ev_net_pct      =ev.ev_net_pct       if ev else 0.0,
        ev_gross_pct    =ev.ev_gross_pct      if ev else 0.0,
        ev_cost_pct     =ev.cost_pct          if ev else 0.0,
        p_win           =ev.p_win             if ev else 0.52,
        p_loss          =ev.p_loss            if ev else 0.48,
        kelly_fraction  =ev.kelly_fraction    if ev else 0.0,
        signal_confidence=ev.confidence       if ev else 0.0,
        ev_trade_count  =ev.trade_count       if ev else 0,
        ev_is_tradeable =ev.is_tradeable      if ev else False,

        # Regime & feature vector
        regime=regime_str,
        regime_code=regime_code,
        volatility_regime  =fv.volatility_regime    if fv else 0.0,
        momentum_strength  =fv.momentum_strength     if fv else 0.0,
        oi_change_rate     =fv.oi_change_rate        if fv else 0.0,
        funding_deviation  =fv.funding_deviation     if fv else 0.0,
        volume_anomaly_score=fv.volume_anomaly_score if fv else 0.0,
        order_flow_imbalance=fv.order_flow_imbalance if fv else 0.0,
        liquidation_pressure=fv.liquidation_pressure if fv else 0.0,
        market_structure   =fv.market_structure      if fv else "none",
        directional_alignment=fv.directional_alignment(bd.direction) if fv else 0.0,

        # Smart money
        sm_phase        =sm.phase.value           if sm else "neutral",
        sm_score        =sm.score                 if sm else 50.0,
        sm_direction_bias=sm.direction_bias       if sm else "neutral",
        sm_aligns       =sm.aligns_with(bd.direction) if sm else False,

        # Spot context
        spot_basis_pct      =sc.basis_pct            if sc else 0.0,
        spot_volume_ratio   =sc.spot_volume_ratio     if sc else 1.0,
        coinbase_premium_pct=sc.coinbase_premium_pct  if sc else 0.0,

        # Order flow from MarketSnapshot
        funding_rate    =snap.funding_rate   if snap else 0.0,
        oi_change_pct   =snap.oi_change_pct  if snap else 0.0,
        long_short_ratio=snap.ls_ratio       if snap else 1.0,
        taker_buy_ratio =snap.taker_buy_ratio if snap else 0.5,
        volume_24h_usdt =snap.volume_24h_usdt if snap else 0.0,
        timeframe="1h",
        asset=bd.symbol.split("/")[0],
        signal_type=f"{regime_str}|{sm.phase.value if sm else 'neutral'}",
        entry_reason=str(getattr(bd, "strategy_reason", "unknown") or "unknown"),
        volatility_bucket=("low" if bd.volatility <= 20 else "medium" if bd.volatility <= 40 else "high" if bd.volatility <= 70 else "extreme"),
        trend_bucket=("weak" if bd.trend_strength < 40 else "moderate" if bd.trend_strength < 65 else "strong" if bd.trend_strength < 85 else "very_strong"),
        strategy_name=str(getattr(bd, "strategy_sleeve", "unknown") or "unknown"),
        setup_type=str(getattr(bd, "setup_type", "unknown") or "unknown"),
        setup_quality_score=float(getattr(bd, "setup_quality_score", 0.0) or 0.0),
        setup_sector=str(getattr(passport, "sector", "unknown") if passport else "unknown"),
        setup_rotation_state=str(getattr(passport, "rotation_state", "unknown") if passport else "unknown"),
        setup_crowding_state=str(getattr(passport, "crowding_state", "unknown") if passport else "unknown"),
        setup_funding_state=str(getattr(passport, "funding_state", "unknown") if passport else "unknown"),
        setup_microstructure_state=str(getattr(passport, "microstructure_state", "unknown") if passport else "unknown"),
        setup_expected_path=str(getattr(passport, "expected_path", "unknown") if passport else "unknown"),
        setup_hold_profile=str(getattr(passport, "hold_profile", "unknown") if passport else "unknown"),
        market_btc_trend=str(getattr(market_ctx, "btc_trend", "unknown") if market_ctx else "unknown"),
        market_eth_btc_trend=str(getattr(market_ctx, "eth_btc_trend", "unknown") if market_ctx else "unknown"),
        market_btc_d_trend=str(getattr(market_ctx, "btc_d_trend", "unknown") if market_ctx else "unknown"),
        market_total_trend=str(getattr(market_ctx, "total_trend", "unknown") if market_ctx else "unknown"),
        market_risk_on_state=str(getattr(market_ctx, "risk_on_state", "unknown") if market_ctx else "unknown"),
        market_rotation_state=str(getattr(market_ctx, "rotation_state", "unknown") if market_ctx else "unknown"),
        market_context_confidence=float(getattr(market_ctx, "confidence", 0.0) if market_ctx else 0.0),
        sector_rotation_state=str(getattr(sector_rotation, "state", "unknown") if sector_rotation else "unknown"),
        sector_rotation_rank=int(getattr(sector_rotation, "rank", 0) if sector_rotation else 0),
        sector_rotation_relative_btc_pct=float(getattr(sector_rotation, "relative_btc_pct", 0.0) if sector_rotation else 0.0),
        sector_rotation_breadth=float(getattr(sector_rotation, "breadth", 0.0) if sector_rotation else 0.0),
        sector_rotation_confidence=float(getattr(sector_rotation, "confidence", 0.0) if sector_rotation else 0.0),
        time_series_state=str(getattr(ts_diag, "state", "unknown") if ts_diag else "unknown"),
        time_series_ewma_vol_pct=float(getattr(ts_diag, "ewma_vol_pct", 0.0) if ts_diag else 0.0),
        time_series_realized_vol_pct=float(getattr(ts_diag, "realized_vol_pct", 0.0) if ts_diag else 0.0),
        time_series_volatility_ratio=float(getattr(ts_diag, "volatility_ratio", 1.0) if ts_diag else 1.0),
        time_series_latest_abs_return_z=float(getattr(ts_diag, "latest_abs_return_z", 0.0) if ts_diag else 0.0),
        time_series_drift_t_stat=float(getattr(ts_diag, "drift_t_stat", 0.0) if ts_diag else 0.0),
        time_series_stability_score=float(getattr(ts_diag, "stability_score", 0.0) if ts_diag else 0.0),
        time_series_markov_state=str(getattr(ts_diag, "markov_state", "unknown") if ts_diag else "unknown"),
        time_series_markov_bull_prob=float(getattr(ts_diag, "markov_bull_prob", 0.0) if ts_diag else 0.0),
        time_series_markov_bear_prob=float(getattr(ts_diag, "markov_bear_prob", 0.0) if ts_diag else 0.0),
        time_series_markov_sideways_prob=float(getattr(ts_diag, "markov_sideways_prob", 0.0) if ts_diag else 0.0),
        time_series_markov_edge=float(getattr(ts_diag, "markov_edge", 0.0) if ts_diag else 0.0),
        time_series_score_mult=float(getattr(bd, "time_series_score_mult", 1.0) or 1.0),
        time_series_size_mult=float(getattr(bd, "time_series_size_mult", 1.0) or 1.0),
        time_series_reason=str(getattr(bd, "time_series_reason", "") or ""),
    )


class DatasetLogger:
    def __init__(self, cfg: dict, source: str = "paper"):
        self._source = source
        path_str = cfg["learning"].get(
            "trade_log_path", "data/trades/trade_log_futures.parquet"
        )
        self._path = Path(path_str)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        trading_cfg = cfg.get("trading", {})
        self._score_threshold = trading_cfg.get("min_score_threshold", 65)
        self._ev_threshold    = cfg.get("ev_model", {}).get("min_ev_pct", 0.05)
        ev_cfg = cfg.get("ev_model", {})
        self._fee_pct = float(ev_cfg.get("taker_fee_pct", 0.04)) / 100.0
        self._slippage_pct = float(ev_cfg.get("slippage_pct", 0.05)) / 100.0

        self._pending: dict[str, TradeDataPoint] = {}   # record_id → open record

    # ── Public API ────────────────────────────────────────────────────────

    def open_record(self, bd, setup, snap=None, source: str | None = None) -> str:
        """Call immediately after a trade is confirmed open. Returns record_id."""
        record_id = str(uuid.uuid4())
        fields = _extract_signal_fields(bd, snap, self._score_threshold)
        dp = TradeDataPoint(
            record_id=record_id,
            schema_version=SCHEMA_VERSION,
            record_type="trade",
            source=source or self._source,
            decision="TRADE",
            near_miss=False,
            no_trade_reason="",
            **fields,
        )
        dp.entry_price     = setup.entry_price
        dp.stop_loss       = setup.stop_loss
        dp.tp1             = setup.tp1
        dp.tp2             = setup.tp2
        dp.r_distance      = setup.r_distance
        dp.size_usd        = setup.size_usd
        dp.size_contracts  = setup.size_contracts
        dp.risk_pct        = setup.risk_pct
        dp.atr             = setup.atr
        dp.leverage        = setup.leverage
        dp.fees_estimate_usd = round(setup.size_usd * self._fee_pct * 2, 4)
        dp.slippage_estimate_usd = round(setup.size_usd * self._slippage_pct * 2, 4)
        dp.fees_slippage_pct = round((self._fee_pct + self._slippage_pct) * 2 * 100, 4)
        self._pending[record_id] = dp
        return record_id

    def close_record(
        self,
        record_id: str,
        closed_at: float,
        exit_price: float,
        exit_type: str,
        pnl_usd: float,
        pnl_pct: float,
        mfe_r: float,
        mae_r: float,
        tp1_hit: bool,
        time_to_mfe_peak_s: float | None = None,
        drawdown_duration_s: float | None = None,
    ) -> None:
        """Call when a trade closes. Completes the row and writes to parquet."""
        dp = self._pending.pop(record_id, None)
        if dp is None:
            log.warning("DatasetLogger.close_record: unknown record_id %s", record_id)
            return

        dp.closed_at           = closed_at
        dp.exit_price          = exit_price
        dp.exit_type           = exit_type
        dp.pnl_usd             = pnl_usd
        dp.pnl_pct             = pnl_pct
        dp.mfe_r               = mfe_r
        dp.mae_r               = mae_r
        dp.tp1_hit             = tp1_hit
        dp.time_to_mfe_peak_s  = time_to_mfe_peak_s
        dp.drawdown_duration_s = drawdown_duration_s
        dp.hold_duration_s     = closed_at - dp.opened_at
        dp.result              = 1 if pnl_usd > 0 else 0

        # rr_achieved = pnl_usd / 1R_in_usd
        if dp.r_distance and dp.size_contracts and dp.r_distance > 0:
            r_usd = dp.r_distance * dp.size_contracts
            dp.rr_achieved = pnl_usd / r_usd if r_usd != 0 else None
            dp.mfe_usd     = mfe_r * r_usd
            dp.mae_usd     = mae_r * r_usd

        self._flush(dp)

    def log_no_trade(
        self, bd, reason: str, snap=None, source: str | None = None
    ) -> None:
        """Log a near-miss. Silently ignored if score < NEAR_MISS_FLOOR."""
        if bd.total_score < NEAR_MISS_FLOOR:
            return
        record_id = str(uuid.uuid4())
        fields = _extract_signal_fields(bd, snap, self._score_threshold)
        dp = TradeDataPoint(
            record_id=record_id,
            schema_version=SCHEMA_VERSION,
            record_type="no_trade",
            source=source or self._source,
            decision="NO_TRADE",
            near_miss=True,
            no_trade_reason=reason,
            **fields,
        )
        self._flush(dp)

    # ── Internal ──────────────────────────────────────────────────────────

    def _flush(self, dp: TradeDataPoint) -> None:
        try:
            import pandas as pd
            row = pd.DataFrame([asdict(dp)])
            if self._path.exists():
                existing = pd.read_parquet(self._path)
                for col in row.columns:
                    if col not in existing.columns:
                        existing[col] = None
                for col in existing.columns:
                    if col not in row.columns:
                        row[col] = None
                row = row[existing.columns]         # align column order
                out = pd.concat([existing, row], ignore_index=True)
            else:
                out = row
            out.to_parquet(self._path, index=False)
            log.debug("DatasetLogger flushed %s record %s", dp.record_type, dp.record_id[:8])
        except Exception as exc:
            log.warning("DatasetLogger flush failed: %s", exc)
