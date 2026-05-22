"""
Pydantic v2 schema for ``config/config.yaml``.

Strictness policy
-----------------
* Risk-critical sections (``exchange``, ``trading``, ``risk``, ``exit``,
  ``kelly``, ``ev_model``, ``safety``) are typed with bounded ranges and
  cross-field consistency rules. Each rule catches a specific class of
  operator typo that would otherwise trade with the wrong knob.
* Additive / fast-iterating sections (``indicators``, ``strategy``,
  ``lifecycle``, ``edge_policy``, ``paper_validation``, ``smart_money``,
  ``regime``, ``execution``, ``derivatives``, ``shadow``, ``timeframes``,
  ``filters``, ``telegram``, ``learning``, ``backtest``, ``logging``,
  ``order_book``, ``structure``, ``scoring``, ``exit_profiles``,
  ``correlation_filter``-children, ``ml``) are passed through verbatim
  via ``extra="allow"``. Promote individual sections to typed models in
  follow-up PRs once the keys stabilise.

The cross-section validator at the bottom of ``NinjaConfig`` enforces:
* ``trading.mode == "live"`` is incompatible with ``exchange.testnet == True``.
* ``ev_model.max_kelly_pct <= kelly.max_kelly_pct`` (Kelly is the absolute hard cap).
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
#  Risk-critical sub-models                                                   
# ---------------------------------------------------------------------------


class ExchangeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = "binance_futures"
    testnet: bool = True
    api_key: str = ""
    api_secret: str = ""
    recv_window: int = Field(default=10000, gt=0, le=60000)
    fapi_base_url: str | None = None


class _RegimeThresholds(BaseModel):
    model_config = ConfigDict(extra="allow")
    trending_expansion: float = Field(ge=0, le=100)
    accumulation_compression: float = Field(ge=0, le=100)
    distribution: float = Field(ge=0, le=100)


class TradingConfig(BaseModel):
    """Risk-critical: every key here gates whether a trade is admitted."""

    model_config = ConfigDict(extra="allow")
    mode: Literal["paper", "live", "backtest"]
    base_currency: str = "USDT"
    paper_starting_equity: float = Field(gt=0, le=1_000_000)
    min_score_threshold: float = Field(ge=0, le=100)
    max_open_trades: int = Field(ge=0, le=20)
    top_pairs_to_trade: int = Field(ge=0, le=20)
    scan_pool_size: int = Field(ge=1, le=200)
    scan_interval_seconds: int = Field(ge=5, le=3600)
    candle_refresh_seconds: int = Field(ge=5, le=3600)
    signal_confirmation_scans: int = Field(ge=0, le=10)
    signal_max_age_seconds: int = Field(ge=10, le=86400)
    regime_thresholds: _RegimeThresholds | None = None


class _CorrelationFilter(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = False
    lookback_bars: int = Field(default=48, ge=10, le=500)
    corr_max: float = Field(default=0.85, ge=0.0, le=1.0)
    min_overlap_bars: int = Field(default=24, ge=5, le=500)


class RiskConfig(BaseModel):
    """Risk-critical: per-trade caps, leverage, drawdown brake."""

    model_config = ConfigDict(extra="allow")
    risk_per_trade_pct: float = Field(gt=0, le=10.0)
    max_risk_per_trade_pct: float = Field(gt=0, le=10.0)
    daily_loss_cap_pct: float = Field(ge=0, le=100.0)
    weekly_loss_cap_pct: float = Field(ge=0, le=100.0)
    max_drawdown_pct: float = Field(gt=0, le=100.0)
    min_rr_ratio: float = Field(gt=0, le=10.0)
    # Binance USDT-perp ceiling is 125x; we enforce that as the absolute upper
    # bound. Operator can still set lower per-symbol caps elsewhere.
    max_leverage: int = Field(gt=0, le=125)
    default_leverage: int = Field(gt=0, le=125)
    stop_loss_atr_multiplier: float = Field(gt=0, le=10.0)
    min_notional_usd: float = Field(gt=0, le=10_000)
    min_risk_usd: float = Field(gt=0, le=10_000)
    max_symbol_risk_pct: float = Field(gt=0, le=20.0)
    max_direction_risk_pct: float = Field(gt=0, le=50.0)
    correlation_filter: _CorrelationFilter | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> "RiskConfig":
        if self.max_risk_per_trade_pct < self.risk_per_trade_pct:
            raise ValueError(
                f"max_risk_per_trade_pct ({self.max_risk_per_trade_pct}) must be "
                f">= risk_per_trade_pct ({self.risk_per_trade_pct})"
            )
        if self.default_leverage > self.max_leverage:
            raise ValueError(
                f"default_leverage ({self.default_leverage}) must be <= "
                f"max_leverage ({self.max_leverage})"
            )
        if self.max_symbol_risk_pct < self.max_risk_per_trade_pct:
            raise ValueError(
                f"max_symbol_risk_pct ({self.max_symbol_risk_pct}) must be >= "
                f"max_risk_per_trade_pct ({self.max_risk_per_trade_pct}); "
                f"otherwise the single-symbol cap silently blocks every entry"
            )
        return self


class ExitConfig(BaseModel):
    """Risk-critical: TP / trail sizing must sum to ~1.0; tp2 must beat tp1."""

    model_config = ConfigDict(extra="allow")
    tp1_r_multiple: float = Field(gt=0, le=20.0)
    tp2_r_multiple: float = Field(gt=0, le=20.0)
    tp1_size_pct: float = Field(ge=0, le=1.0)
    tp2_size_pct: float = Field(ge=0, le=1.0)
    trail_size_pct: float = Field(ge=0, le=1.0)
    breakeven_trigger_r: float = Field(ge=0, le=10.0)
    trailing_atr_multiplier: float = Field(gt=0, le=10.0)
    # 30 day ceiling — anything higher is almost certainly a typo
    # (`max_hold_duration_s: 17_280_000` instead of `1_728_000`).
    max_hold_duration_s: int = Field(gt=0, le=2_592_000)

    @model_validator(mode="after")
    def _check_consistency(self) -> "ExitConfig":
        if self.tp2_r_multiple <= self.tp1_r_multiple:
            raise ValueError(
                f"tp2_r_multiple ({self.tp2_r_multiple}) must be > "
                f"tp1_r_multiple ({self.tp1_r_multiple})"
            )
        size_sum = self.tp1_size_pct + self.tp2_size_pct + self.trail_size_pct
        if abs(size_sum - 1.0) > 0.01:
            raise ValueError(
                f"tp1_size_pct + tp2_size_pct + trail_size_pct must sum to "
                f"~1.0, got {size_sum:.4f}"
            )
        return self


class KellyConfig(BaseModel):
    """Risk-critical: max_kelly_pct is the absolute hard cap on position risk."""

    model_config = ConfigDict(extra="allow")
    fraction: float = Field(gt=0, le=1.0)
    max_kelly_pct: float = Field(gt=0, le=10.0)
    min_kelly_pct: float = Field(ge=0, le=10.0)
    volatility_scale: bool = True
    vol_target_atr_pct: float = Field(gt=0, le=10.0)

    @model_validator(mode="after")
    def _check_consistency(self) -> "KellyConfig":
        if self.min_kelly_pct > self.max_kelly_pct:
            raise ValueError(
                f"min_kelly_pct ({self.min_kelly_pct}) must be <= "
                f"max_kelly_pct ({self.max_kelly_pct})"
            )
        return self


class EvModelConfig(BaseModel):
    """Risk-critical: probability floors + EV threshold."""

    model_config = ConfigDict(extra="allow")
    taker_fee_pct: float = Field(ge=0, le=1.0)
    maker_fee_pct: float = Field(ge=0, le=1.0)
    slippage_pct: float = Field(ge=0, le=5.0)
    min_ev_pct: float = Field(ge=-100, le=100)
    min_p_win: float = Field(ge=0, le=1.0)
    min_confidence: float = Field(ge=0, le=1.0)
    min_trades_for_ev: int = Field(ge=0, le=10_000)
    prior_win_rate: float = Field(ge=0, le=1.0)
    prior_avg_win_pct: float = Field(ge=0, le=100)
    prior_avg_loss_pct: float = Field(ge=0, le=100)
    max_kelly_pct: float = Field(gt=0, le=10.0)


class SafetyConfig(BaseModel):
    """Risk-critical: kill switches + heartbeat + session sizing."""

    model_config = ConfigDict(extra="allow")
    pause_on_extreme_volatility: bool = True
    extreme_vol_atr_multiplier: float = Field(gt=0, le=10.0)
    max_consecutive_losses: int = Field(ge=0, le=50)
    api_retry_attempts: int = Field(ge=0, le=10)
    api_retry_delay_seconds: int = Field(ge=0, le=60)
    heartbeat_interval_seconds: int = Field(ge=5, le=600)
    auto_live_on_readiness: bool = False
    min_live_balance_usd: float = Field(ge=0, le=1_000_000)
    performance_report_interval_minutes: float = Field(ge=0, le=1440)
    heartbeat_fail_kill_switch_count: int = Field(ge=0, le=100)
    runtime_error_kill_switch_count: int = Field(ge=0, le=100)
    equity_mismatch_tolerance_pct: float = Field(ge=0, le=100)
    session_size_multipliers: dict[str, float] | None = None

    @model_validator(mode="after")
    def _check_session_multipliers(self) -> "SafetyConfig":
        if self.session_size_multipliers is None:
            return self
        for key, val in self.session_size_multipliers.items():
            if not (0.0 < float(val) <= 5.0):
                raise ValueError(
                    f"session_size_multipliers.{key} = {val} must be in (0, 5]"
                )
        return self


# ---------------------------------------------------------------------------
#  Top-level config                                                           
# ---------------------------------------------------------------------------


class NinjaConfig(BaseModel):
    """Top-level config schema. Strict on risk-critical sections; lax on others."""

    model_config = ConfigDict(extra="allow")  # let unknown sections through
    exchange: ExchangeConfig
    trading: TradingConfig
    risk: RiskConfig
    exit: ExitConfig
    kelly: KellyConfig
    ev_model: EvModelConfig
    safety: SafetyConfig

    @model_validator(mode="after")
    def _check_cross_section(self) -> "NinjaConfig":
        if self.trading.mode == "live" and self.exchange.testnet:
            raise ValueError(
                "trading.mode='live' is incompatible with exchange.testnet=true; "
                "set testnet=false in config or use --no-testnet"
            )
        if self.ev_model.max_kelly_pct > self.kelly.max_kelly_pct:
            raise ValueError(
                f"ev_model.max_kelly_pct ({self.ev_model.max_kelly_pct}) must be "
                f"<= kelly.max_kelly_pct ({self.kelly.max_kelly_pct}); "
                f"Kelly is the absolute hard cap"
            )
        return self
