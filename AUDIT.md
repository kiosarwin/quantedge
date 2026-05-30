# QuantEdge Comprehensive Security & Architecture Audit

## Executive Summary

QuantEdge is a well-architected crypto futures trading engine with 11 institutional layers. The codebase demonstrates strong domain knowledge and thoughtful risk management. However, several critical and high-severity issues were identified across security, execution reliability, and ML model validity that should be addressed before live deployment with meaningful capital.

**Key statistics:** 84 source files, ~25,000 LOC, 39 test files (~9,919 LOC), 3 config files.

**Overall assessment:** The system is production-ready for paper trading and small live accounts. For institutional-scale deployment, the issues below must be addressed.

---

## 1. Code Quality Audit

### MEDIUM - God Object: src/main.py (3,651 lines, 77 methods)

`NinjaTrader` class is a monolithic god object handling scanning, scoring, trade execution, position rotation, paper validation, ML gates, loss streak guards, EV probes, and Telegram reporting all in one class.

**Recommendation:** Extract into smaller service classes (ScanCycle, AdmissionPolicy, PositionRotator, PaperValidationPolicy, ReportingService).

### LOW - Global mutable state in src/data/client.py

Module-level `_ls_cache` and `_taker_cache` dicts use global mutable state for TTL caching. Thread-safe in asyncio but makes testing harder.

**Code reference:** `_ls_cache: dict[str, tuple[float, float]] = {}` and `_taker_cache`.

### MEDIUM - Inconsistent error handling patterns

Some methods use bare `except Exception` (e.g., `src/models/thompson_bandit.py:_save`, `src/risk/vol_targeting.py:_save`), others raise properly. Mix of return-None vs raise-exception for failures.

### LOW - Limited type hints on complex return types

Many methods return `dict` or `tuple` without TypedDict or NamedTuple definitions. The code uses dataclasses well for domain objects but not for intermediate results.

### LOW - Magic numbers in entry_strategies.py

Confidence thresholds (0.45, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10) are hardcoded rather than configurable. RSI ranges (45-78 for long, 22-55 for short) are hardcoded.

### MEDIUM - Dead/disabled code paths

- `enable_short_setups: false` in config but the entire `short_strategies.py` module is loaded and ready.
- `crypto_news_interval_minutes: 0` disables the crypto news feature but the import and module still load (with a fallback try/except).

---

## 2. Trading Strategy Audit

### HIGH - Entry strategies lack confirmation across timeframes (entry_strategies.py)

The `detect_entry()` function uses only the primary timeframe DataFrame. It does not check higher-timeframe alignment for MOMENTUM_BREAKOUT, VWAP_RECLAIM, or LIQUIDITY_SWEEP strategies.

The separate MTF and VWAP pullback detectors DO check multiple timeframes, but the main `entry_strategies.py` detector does not.

**Risk:** Entering momentum breakouts against the 4H trend leads to false breakouts.

### MEDIUM - VWAP calculation is an approximation (entry_strategies.py:_vwap)

Uses a rolling 20-period typical-price*volume / volume approximation rather than true session VWAP anchored to session open. For 1H candles this gives a 20-hour rolling VWAP, not the standard daily-session VWAP used in traditional markets.

**Impact:** Acceptable for crypto (no fixed session) but should be documented as "rolling VWAP" not "session VWAP".

### MEDIUM - Funding rate extreme thresholds may be stale (entry_strategies.py)

`extreme_long_threshold = 0.0005` (0.05% per 8h) and `extreme_short_threshold = -0.0003`. These were reasonable in 2023-2024 but Binance funding has compressed in 2025-2026. Should be dynamically computed or percentile-based.

### LOW - OI divergence uses single-snapshot OI change

`oi_change_pct` is a single point-in-time comparison. A trend of OI changes over multiple snapshots would be more robust.

### HIGH - Smart money detection has potential false positives (smart_money.py)

The DISTRIBUTION detection fires on `funding_rate > funding_extreme * 1.5 AND ls_crowded_long` without requiring OI confirmation. This can false-fire during healthy bull trends where funding naturally stays positive.

The ACCUMULATION detection requires `price_is_flat` (< 0.5% change over 5 bars) which may miss gradual accumulation during slow grinds.

### MEDIUM - Liquidity sweep reversal lookback is configurable but default is short (24 bars)

On 1H timeframe, 24 bars = 1 day. Liquidity pools form over days/weeks. A sweep of a 24-hour high/low is less significant than a multi-day level.

**Recommendation:** Consider longer default or weight by level age.

### LOW - Entry signal confidence scoring uses additive components that can exceed 1.0

All strategies cap at `min(1.0, confidence)` but the additive nature means very different market conditions can produce identical confidence scores (0.90 from volume+RSI vs 0.90 from EMA+body).

---

## 3. Risk Management Audit

### CRITICAL - Weekly loss cap disabled (weekly_loss_cap_pct: 0.0)

The config sets `weekly_loss_cap_pct: 0.0` which disables the weekly loss limit. Combined with `cooldown_after_loss_streak: 0` and `max_consecutive_losses: 0`, the bot has NO streak-based circuit breaker in live mode.

Only the daily cap (5%) and max drawdown (20%) provide protection.

**Risk:** In a prolonged adverse regime, the bot can lose 5%/day for 4 consecutive days before the 20% drawdown kills it.

### HIGH - Kelly fraction at 30% is aggressive (kelly_sizer.py)

Standard institutional practice is quarter-Kelly (25%) or less. The code uses `self._kelly_quarter = 0.30` which is actually 30%-Kelly, not quarter-Kelly despite the variable name.

Combined with the `confidence_scaling` that can push to 1.4x, effective Kelly can reach 42%.

**Risk:** Kelly criterion assumes accurate P(win) and payoff estimates. With limited crypto sample sizes, over-betting is likely.

### HIGH - Drawdown recovery ramp-up assumes time-based recovery (risk_manager.py:PortfolioState)

Recovery stages advance by elapsed time (1 hour per stage) regardless of whether the market has actually stabilized. A 3-hour recovery from a 15% drawdown may be far too fast if the adverse regime persists.

**Recommendation:** Tie recovery to both time AND equity reclaiming a percentage of the drawdown.

### MEDIUM - VaR/CVaR calculation uses too few samples

`RiskAnalytics` uses trade returns (max 100 in deque) for VaR. With 2 trades/day, this is 50 days of data. VaR at 95% confidence with 100 samples has wide confidence intervals. The 99% VaR with `idx = int(100 * 0.01) = 1` uses only the single worst observation.

### MEDIUM - Position rotation can churn in adverse conditions

The rotation system allows 3 rotations/day. If all trades are losing, the bot may continuously rotate into new losing positions, crystallizing unrealized losses. The `max_victim_adverse_r: 0.90` prevents rotating deep losers, but trades at -0.5R to -0.89R can be rotated.

### LOW - Correlation filter uses Pearson on returns which assumes linearity

Crypto correlations are regime-dependent (correlations spike to 1.0 in crashes). A tail-dependence measure or rolling-window minimum correlation would be more conservative.

---

## 4. Execution Audit

### HIGH - No fill confirmation loop for live orders (executor.py)

After `create_order()`, the executor returns the order dict immediately without waiting for fill confirmation. If the market order partially fills or is rejected after initial acceptance, the system proceeds with the assumed full fill.

**Recommendation:** Add a fill confirmation poll (fetch_order status check) with timeout for live mode.

### MEDIUM - Race condition in move_stop_loss (executor.py:move_stop_loss)

Between canceling the old SL and placing the new SL, there is a window where no exchange-level stop exists. If the bot crashes during this window AND the market moves adversely, the position has no exchange protection.

The code acknowledges this ("relying on local stop") but the local stop only works if the bot is running.

### MEDIUM - Paper mode global counter is not thread-safe (executor.py)

`_PAPER_ORDER_ID` uses a global int with `global` keyword. In asyncio this is safe, but if the code is ever used with threading (e.g., for parallel backtests), it would race.

### HIGH - Trade state persistence is not atomic (trade_manager.py:_save_state)

`STATE_PATH.write_text(json.dumps(payload))` is not atomic. If the process is killed during write, the state file may be corrupted/truncated.

**Recommendation:** Write to a temp file, then atomic rename (`os.replace`).

### MEDIUM - No reconciliation of exchange vs local state on startup

When restoring from `data/open_trades.json`, the system trusts the local state. If exchange-side orders were filled or positions liquidated while the bot was down, local state will be stale.

There is a `_last_reconcile_ts` field suggesting reconciliation exists somewhere in main.py but it should run on every startup.

---

## 5. Model/ML Audit

### HIGH - Online logistic regression has only 10 features and no feature engineering (online_logistic.py)

The model uses raw score components (score, structure, trend, volume, funding, OI, volatility, momentum, direction) plus a bias. There are no interaction features, no lagged features, and no cross-sectional features.

With such simple features, the model cannot capture regime-conditional effects (e.g., "high volume is bullish in trends but bearish in distribution").

**Risk:** The model may learn spurious correlations from small samples and degrade performance.

### HIGH - Thompson Bandit prior is uniform Beta(2,2) for all contexts (thompson_bandit.py)

All (context, arm) pairs start with the same prior. This means a brand-new regime+direction combination needs at least 10-20 trades to differentiate from random.

In crypto where regimes shift every few days, the bandit may never accumulate enough samples in any single context to be useful.

**Empirical evidence:** The bandit's `size_mult` range is [0.6, 1.4] which is a +/-40% swing on position size based on potentially unstable estimates.

### MEDIUM - Alpha decay tracker has revival mechanism (alpha_decay.py)

After `revival_after_s` (default 86400 = 1 day), a "dead" pair is automatically revived. If the pair's edge truly decayed (e.g., listing changes, liquidity migration), automatic revival will lead to repeated losses.

### MEDIUM - Online logistic learning rate (0.05) may be too aggressive for concept drift

With SGD at lr=0.05 on binary outcomes, a few consecutive wins/losses can dramatically shift the model weights. The L2 regularization (0.01) is weak relative to the learning rate.

The `is_drifting()` check (loss > 0.85) only detects when the model is already consistently wrong.

### LOW - Regime transition matrix loses context across restarts

Unlike Thompson Bandit and Online Logistic which persist to JSON, the regime transition matrix observation state is not clearly persisted (no STATE_PATH visible in the code pattern).

### MEDIUM - EV model Bayesian shrinkage is asymmetric by design but trusts losses quickly

`loss_shrink_factor: 0.4` means when realised P(win) < prior, the effective prior weight drops from 6 to 2.4. This means 3-4 losses can overcome the prior, potentially blocking strategies after a normal variance drawdown.

---

## 6. Backtest Engine Audit

### HIGH - Backtest does not model funding costs (backtest/engine.py)

The engine applies commission and slippage but does NOT deduct funding rate costs. For trades held across 8h boundaries (common with `max_hold_duration_s` up to 345600), funding can be 0.01-0.05% per period.

Over a year of backtesting, this omission can overstate returns by 5-15% depending on direction bias.

### MEDIUM - No realistic order book simulation

The backtest uses `bar_close * slippage_pct` for slippage estimation. In reality, slippage depends on order size relative to book depth, which varies enormously across pairs. A $50 position on BTCUSDT has negligible slippage; a $50 position on a microcap may have 0.5%+ slippage.

### MEDIUM - Look-ahead in higher-timeframe alignment

`df_high_slice = df_high.loc[df_high.index < bar_ts]` correctly prevents look-ahead, but the relationship between 1H and 4H bars depends on how the data is aligned. If the 4H bar closes at the same timestamp as a 1H bar, the 4H bar's close is known only at that 4H boundary.

This is handled correctly by the `<` comparison but should be validated with actual data.

### LOW - Single-symbol backtesting only

The engine tests one symbol at a time without portfolio-level constraints. In live trading, `max_open_trades=2` means symbols compete for slots. The backtest cannot replicate this.

### LOW - No survivorship bias handling

Symbols that were delisted or had liquidity issues during the backtest period are not excluded or flagged.

### MEDIUM - MFE/MAE tracking uses bar high/low which has intra-bar uncertainty

Recording `trade.mfe_r = max(mfe_r, fav / r_distance)` using bar high assumes the trade experienced the full bar high. In reality, the entry and exit within a bar are path-dependent.

---

## 7. Security Audit

### CRITICAL - API keys can be set directly in config.yaml (config.yaml:api_key, api_secret)

While the README recommends using .env, the config has `api_key: ""` and `api_secret: ""` fields. Users may accidentally commit keys in config.yaml.

The .gitignore should explicitly exclude config files with keys, or the config should not have these fields at all.

### HIGH - No encryption at rest for state files

`models/bandit_state.json`, `models/online_logistic_state.json`, `models/vol_targeter.json`, `data/open_trades.json` all store in plaintext.

While they do not contain API keys, they contain trade intelligence (entry prices, position sizes, strategy parameters) that could be exploited.

### MEDIUM - Pickle deserialization in backtest/data_loader.py

`pd.read_pickle(cache_file)` is used for caching OHLCV data. Pickle deserialization is a known RCE vector if the cache files are tampered with.

**Recommendation:** Use parquet or feather format for data caching.

### LOW - Telegram token stored in environment without rotation

Standard practice, but no mechanism for token rotation or detection of compromised tokens.

### LOW - No input validation on config.yaml values

While `src/config/schema.py` exists, numeric values from config are used directly with `float()` and `int()` without range validation. Negative risk percentages or zero equity values could cause division-by-zero.

---

## 8. Infrastructure Audit

### HIGH - No graceful shutdown signal handling (src/main.py)

No SIGINT/SIGTERM handlers found. The main loop uses `try/finally` with `_shutdown()` but process signals are not explicitly caught.

If the process is killed by systemd or a watchdog, open positions may not be cleanly recorded. `close_positions_on_shutdown: false` means positions survive restarts, which is correct, but state must be persisted first.

### MEDIUM - Circuit breaker uses time-based reset only

`_circuit_breaker_until` resets by wall-clock time. If the underlying issue (e.g., exchange maintenance) persists beyond the exponential backoff window, the bot will immediately retry and fail again.

Should incorporate health checks before resuming.

### MEDIUM - State file corruption recovery

`_load_state()` in trade_manager.py has a bare `except Exception` that logs a warning and continues with empty state. If state is corrupted, all open positions are silently lost from the bot's perspective (though they still exist on the exchange).

No backup/rotation of state files.

### HIGH - No watchdog or liveness probe

The safety config mentions `heartbeat_interval_seconds: 30` but this is an internal metric, not an external health check.

If the main loop hangs (e.g., deadlocked await), there is no external mechanism to detect and restart.

**Recommendation:** Write a heartbeat file with timestamp; external watchdog (systemd WatchdogSec, or a cron health check) can detect stale heartbeats.

### LOW - Log rotation is configured but no log level filtering per module

All modules log at the same level. In production, exchange client debug logs may be noisy while risk manager warnings are critical.

---

## 9. Test Coverage Assessment

### HIGH - Critical paths untested

- No test for `Executor.open_position()` in live mode (only paper mode exercised implicitly).
- No test for `TradeManager._load_state()` with corrupted JSON.
- No test for `RiskManager.calculate_setup()` with edge cases (zero equity, negative ATR, extreme leverage).
- No integration test for the full scan-score-execute pipeline.

### MEDIUM - Test count vs coverage gap

39 test files (~9,919 lines) for 84 source files (~25,000 lines). Ratio is reasonable but several critical modules lack dedicated tests:

- `src/data/client.py` - no test (exchange integration)
- `src/backtest/engine.py` - only `test_backtest_atr_trailing.py` tests one aspect
- `src/models/ev_model.py` - only `test_ev_model_fixes.py` (specific fixes, not comprehensive)
- `src/execution/executor.py` - no dedicated test file
- `src/main.py` - only `test_main_closed_trade_metadata.py` and `test_main_reporting.py`

### MEDIUM - conftest.py is minimal

Only adds the project root to sys.path. No shared fixtures for common test data (OHLCV DataFrames, config dicts, mock exchange clients). Each test file likely recreates these independently, leading to maintenance burden.

### LOW - No property-based testing

Trading systems benefit enormously from property-based testing (hypothesis library) to find edge cases in indicator calculations and risk math. Example: `atr()` should always return positive values; `calculate_setup()` should never produce negative position sizes.

### LOW - No performance/stress tests

No test for how the system performs under rapid price changes, exchange timeouts, or concurrent operations.

---

## Summary Table

| Domain | Critical | High | Medium | Low |
|--------|----------|------|--------|-----|
| Code Quality | 0 | 0 | 3 | 3 |
| Trading Strategies | 0 | 2 | 3 | 2 |
| Risk Management | 1 | 2 | 2 | 1 |
| Execution | 0 | 2 | 3 | 0 |
| ML/Models | 0 | 2 | 3 | 1 |
| Backtest Engine | 0 | 1 | 3 | 2 |
| Security | 1 | 1 | 1 | 2 |
| Infrastructure | 0 | 2 | 2 | 1 |
| Test Coverage | 0 | 1 | 2 | 2 |
| **Total** | **2** | **13** | **22** | **14** |

---

## Priority Recommendations

### Immediate (before live deployment with >$1000):

1. Fix atomic state persistence (write-then-rename pattern)
2. Add fill confirmation loop for live orders
3. Enable weekly loss cap and consecutive loss protection for live mode
4. Add SIGINT/SIGTERM graceful shutdown handlers
5. Remove api_key/api_secret fields from config.yaml template
6. Replace pickle caching with parquet

### Short-term (before scaling capital):

1. Refactor main.py god object into smaller services
2. Add funding cost to backtest engine
3. Add external watchdog/liveness probe
4. Implement reconciliation with exchange on startup
5. Rename `_kelly_quarter = 0.30` or reduce to actual 0.25
6. Add feature interactions to online logistic model

### Medium-term (ongoing improvement):

1. Add property-based tests for indicator and risk math
2. Replace hardcoded strategy thresholds with config
3. Add integration tests for full pipeline
4. Implement dynamic funding rate thresholds
5. Add regime-conditional correlation estimation
6. Extend backtest to multi-symbol portfolio simulation
