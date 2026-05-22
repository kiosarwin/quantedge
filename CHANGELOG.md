# Ninja Trader — Changelog

> Session handoff: state terbaru ada di `session.md`; jangan re-derive konteks lama dari changelog ini.

---

## [Unreleased] — 2026-05-22 — Futures-only + Aggressive-edge upgrade layer

### Removed (spot pipeline retired)
- Bot is now Binance Futures only.  Spot edition was a parallel pipeline
  with its own scanner, scorer, risk manager, executor, trade manager and
  config; it has been removed wholesale to eliminate drift risk.
- `src/spot_main.py`, `src/scoring/spot_scorer.py`,
  `src/risk/spot_risk_manager.py`, `src/execution/spot_executor.py`,
  `src/execution/spot_trade_manager.py`, `src/scanner/spot_scanner.py`,
  `src/data/spot_client.py`, `src/data/spot_market_data.py`,
  `src/backtest/spot_backtester.py`, `config/config_spot.yaml`,
  `SPOT_GUIDE.md`.
- `src/analysis/spot_context.py` is **kept** — it is a futures-side feature
  that pulls spot reference price/volume to detect basis premium and
  Coinbase divergence; it has no dependency on the spot trading pipeline.

### Added (futures upgrade layer)
- `src/risk/session_modulator.py`: applies a session-aware size multiplier
  (Asia/London/NY/overlap) on top of FundManager + Kelly.  Defaults derived
  from rolling Sharpe-by-session studies on 2024–2025 BTC/ETH perpetuals;
  overrides via `safety.session_size_multipliers`.
- `src/risk/correlation_filter.py`: blocks opening trades that compound
  existing exposure (same direction × high positive correlation, or opposite
  direction × high negative correlation) with any open trade.  Hedging bets
  pass freely.  Default cap: 0.85 absolute correlation over 48 1H bars.
- `src/execution/trade_manager.py`: adverse-cut early exit.  Trades that
  bleed past `mae_r_threshold` (0.65R) within the first 2 hours and have
  not shown a meaningful favourable excursion (`mfe_r_ceiling` 0.20R) are
  cut at current price instead of waiting for full -1R SL.  Converts ~1R
  losers into ~0.6R losers without touching winners.

### Changed
- `src/models/edge_detector.py`: now reads `edge_detector.mode`
  (`observer`|`soft_gate`|`active_gate`).  `soft_gate` applies size_mult
  but never blocks; `active_gate` also blocks.  Default flipped from
  `observer` to `soft_gate` so realised edge starts shaping notional.
- `src/models/cohort_policy.py`: `CohortDecision.size_mult` now reflects
  attribution health bounded `[0.6, 1.25]`; main.py already wires this.
- `config/config.yaml`: added `risk.correlation_filter`,
  `safety.session_size_multipliers`, `exit.early_cut`; flipped
  `edge_detector.mode` to `soft_gate`.

### Tests
- `tests/test_futures_upgrades.py`: 13 new cases covering session modulator,
  correlation filter (blocks compounding, allows hedge), edge detector
  mode mapping, cohort size_mult attribution, and adverse-cut behaviour.

### Verification
- 87/87 tests pass (74 prior + 13 new).
- NinjaTrader instantiates cleanly with all new modules wired.

---

## [Unreleased] — 2026-05-19 — Strategy Stack Alignment

### Changed
- `STRATEGY_STACK.md`: clarified that `neutral` is residual routing, not alpha; added Binance Futures operational rule table and runtime mapping.
- `README.md`: documented that `trend_following` and `reversal` are the core alpha sleeves, while `neutral` is fallback only.
- `QUANT_OPERATING_MODEL.md`: noted that `neutral` is excluded from default live admission allowlists and added expectancy-per-sleeve as a metric.
- `config/config.yaml`: removed `neutral` from the default paper strategy sleeve allowlist.

### Code
- `src/models/strategy_router.py`: aligned sleeve routing with regime/flow/funding/OI/participation filters.
- `src/scoring/scorer.py`: kept alpha sleeves anchored to institutional score while making `neutral` conservative.
- `src/main.py`, `src/models/cohort_policy.py`: default admission path now blocks `neutral` by allowlist.

### Tests
- `tests/test_strategy_router.py`
- `tests/test_paper_validation.py`
- `tests/test_cohort_policy.py`

### Why
- The bot was still able to classify and carry `neutral` through some internal paths. This update makes the docs and runtime policy explicit: `neutral` is not alpha.

## [Unreleased] — 2026-04-23 — Setup-Specific Pwin + Rejection Forensics

### Added
- `src/analysis/rejection_logger.py`: `RejectionRecord` + buffered parquet sink (`data/rejections.parquet`, 35 cols). Hooks in `src/main.py` at 8 reject sites. Structured `REJECT[STAGE]` log lines. Diagnostics only — zero behavioral change.
- `src/models/pwin_engine.py`: structural (feature-conditional) prior over (regime, SM phase, SM alignment, structure-quality bucket, volatility bucket, direction, BTC alignment) + hierarchical Bayesian shrinkage (bucket→regime→global→prior, k=10/20/40).

### Changed
- `src/models/ev_model.py`: `compute()` accepts optional `pwin_ctx`; delegates p_win / avg_win / avg_loss to `PwinEngine` when supplied.
- `src/scoring/scorer.py`: reordered — `ts/sq/vs` computed before EV; builds `PwinContext` and passes to EV.

### Why
- Forensics proved EV deadlock was portfolio-global p_win (variance=0.0, all candidates at 0.348). Post-deploy: variance 0.0 → 0.14, 11 distinct values, EV range −2.58% to −0.23%.

### Unchanged
- Thresholds, bootstrap floors, FM scaling, SM enforcement, slippage, ML gate.

---

## [Unreleased] — 2026-04-22 — Progressive Regime Gate

### Changed
- `src/models/ml_engine.py`: added `regime_gate(regime) → (allow, scale, reason)`. Progressive thresholds replace the binary 5-trade / 38% hard lock:
  - `n < 10` → exploration, no lock, position scale `0.5x`
  - `10 ≤ n < 20` → soft penalty `0.25x` if winrate < 38%, no lock
  - `n ≥ 20` → hard lock if winrate < 38%
- `src/main.py`: call `regime_gate()` instead of `regime_confidence()` for the gate decision; regime scale applied to `total_scale` *after* the 0.40 floor so soft-penalty can shrink sizing below it.

### Why
- Prior gate locked `trending_expansion` at 14/18 sub-38% trades, freezing entries while still in the data-collection phase (<20 trades).
- Edge-only preserved: gate never forces or amplifies — only shrinks size or skips.

### Unchanged
- ML model, scoring weights, entry thresholds, 38% winrate rule, EV gate logic.

---

## [1.0.0] — 2026-04-19 — First Stable Paper Trading Release

### Core Bot
- Main scan loop with configurable YAML config (`config/config.yaml`)
- Paper / live mode switch with hard safety guards — live mode rejected if testnet flag is set, paper mode forces testnet unconditionally
- Binance Futures testnet integration via CCXT
- Rich console logging + rotating file log (`logs/futures_trader.log`)

### Signal Scoring Engine (`src/scoring/scorer.py`)
- 7-signal weighted scoring pipeline:
  - Trend strength, volume confirmation, structure quality, open interest, funding sentiment, order book imbalance, volatility
- Four-state regime classifier gate — blocks trades in `chaos` regime
- Smart money phase detector gate — requires directional bias
- EV model gate — blocks trades where net EV < threshold after fees + slippage + funding
- Near-miss floor at score 55.0 — pairs below this are not evaluated further

### Risk & Sizing
- Fractional Kelly Criterion sizer (`src/risk/kelly_sizer.py`) — quarter-Kelly by default, with ATR-based volatility scalar (0.5–1.5×)
- Risk manager with per-trade max risk %, stop-loss and TP1/TP2 calculation (`src/risk/risk_manager.py`)
- Algorithmic Fund Manager (`src/models/fund_manager.py`) — 9-factor position scaling:
  1. Signal conviction
  2. Hot/cold streak momentum
  3. Market regime
  4. Macro momentum (rolling performance)
  5. Portfolio heat (total open exposure)
  6. Daily P&L protection
  7. Peak drawdown preservation
  8. Session liquidity (Asia / London / Overlap / NY)
  9. Per-regime edge tracking
- Equity tier scaling — max concurrent trades and leverage scale with account size
- Milestone reward system (Jim earns gifts at $110, $120, $150 … $100,000)

### ML Pipeline (Phase-Gated)
- **Phase 1** (0–19 closed trades): Bayesian priors only, no ML adjustment
- **Phase 2** (20+ closed trades): XGBoost EV model + weight adjustment enabled (`src/models/ml_engine.py`)
- **Phase 3** (50+ closed trades): Per-signal decay enabled
- Self-learning weight adjuster (`src/learning/learner.py`) — nudges signal weights toward patterns correlated with wins, re-normalises after each adjustment
- XGBoost trade predictor — 21 features including signal scores, regime code, smart money phase, EV model outputs, session hour

### Dataset Logger (`src/data/dataset_logger.py`)
- Append-only Parquet log at `data/trades/trade_log_futures.parquet`
- Schema v1.0 — 85 columns per row
- Two-phase trade lifecycle: `open_record` → `close_record`
- Near-miss no-trade rows logged for pairs scoring ≥ 55.0
- Captures: MFE/MAE (R-multiples + USD), hold duration, drawdown duration, exit type, RR achieved

### Execution & Trade Management
- Executor with paper simulation — fill at market price, no real orders on testnet
- Trade manager with TP1 / TP2 / trailing stop / stop-loss exit logic
- Open trade state persisted to `data/state.json` — survives restarts

### Shadow Engine (`src/backtest/shadow_engine.py`)
- Runs ghost trades in parallel with every live signal — tracks shadow win rate, profit factor, and per-regime performance without risking capital

### Watchdog (`watchdog.py`)
- Monitors bot process every 30s
- Auto-restarts on: process death, hang (no log activity > 300s), memory > threshold
- Log diagnosis on every restart — classifies crash as `external` (Binance/network) or `bot_bug`
- Telegram alerts include last 8 relevant log lines + verdict
- 60s grace period after fresh start before hang detection activates
- Max 10 restarts per hour before pausing and alerting for manual intervention

### Notifications (`src/notifications/telegram.py`)
- Telegram alerts for: trade opened/closed, milestone reached, watchdog events, shadow reports, live-readiness assessment

### Spot Context (`src/analysis/spot_context.py`)
- Spot/futures basis, Coinbase premium, spot volume ratio
- HTTP 400 handled gracefully for futures-only symbols (returns 0.0 instead of crashing)

### Backtest Infrastructure
- Full backtest engine (`src/backtest/engine.py`) with realistic fee + slippage simulation
- Binance FAPI data loader with pagination and disk cache (`src/backtest/data_loader.py`)
- Shadow backtester, reporter, learner and trade manager for offline strategy validation

---

## Live Status — 2026-04-20

- **Mode**: Paper trading, Binance Futures testnet
- **Equity**: $523.03 (started $500)
- **Closed trades**: 6 — net +$10.35 USD (3W / 3L)
- **Open trades**: 3 (PIEVERSE long, ZEC short, HYPE short)
- **Dataset rows**: 1,939 (6 closed trades, 1,933 near-misses)
- **Phase**: 1 of 3 — 14 more closed trades needed to unlock ML
- **Watchdog**: Healthy, ~9h uptime, 291MB mem
- **Known issue**: Hang storm on 2026-04-19 ~02:00 UTC — suspected Binance API call with no timeout; not yet fixed
