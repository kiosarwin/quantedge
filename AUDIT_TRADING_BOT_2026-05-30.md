# Trading Bot Strategy Audit - 2026-05-30

Scope: futures bot in the current `ninja_trader` worktree. This audit used the current files, logs, local state, and test results as evidence.

## Verification Performed

- Read strategy docs: `STRATEGY_STACK.md`, `QUANT_OPERATING_MODEL.md`, `README.md`.
- Audited runtime path: `src/main.py`, `src/models/strategy_router.py`, `src/risk/risk_manager.py`, `src/execution/trade_manager.py`, `src/execution/executor.py`, `src/scanner/scanner.py`, `src/config/schema.py`.
- Read active config: `config/config.yaml`.
- Read local trade/rejection artifacts: `data/trades/trade_log_futures.parquet`, `data/rejections.parquet`, `data/open_trades.json`, `data/state.json`.
- Read runtime logs: `logs/ninja_trader.log`, `logs/futures_trader.log`.
- Ran test suite: `venv/bin/pytest -q` -> `394 passed in 357.09s`.

## Findings

### High - Current runtime scanner is not producing a tradeable universe

Evidence:

- `logs/ninja_trader.log` repeatedly shows:
  - `Scanner ticker fetch failed: 'str' object has no attribute 'keys'`
  - `No pairs found - sleeping`
  - heartbeat remains `equity=$100.00`, `open=0`.
- `src/scanner/scanner.py:60-64` catches any ticker-fetch exception and returns an empty list.
- `src/scanner/scanner.py:70` assumes `tickers` is a mapping and iterates `tickers.items()`.

Impact:

The strategy stack cannot be forward-validated while scanner fetch returns this shape/error. The bot is alive but not evaluating pairs, so paper/live performance evidence is currently blocked.

Recommendation:

Instrument `BinanceFuturesClient.fetch_tickers()` to log the returned type/shape before scanner consumption, add a defensive normalization branch for unexpected ccxt responses, and add a regression test where `fetch_tickers()` returns a bad non-dict shape.

### High - Restored open trades undercount sector exposure after restart

Evidence:

- `src/execution/trade_manager.py:663-668` restores open trades into risk state with `symbol`, `direction`, and `risk_pct`, but does not pass `sector`.
- `src/risk/risk_manager.py:398-402` enforces `max_sector_risk_pct` from `state.sector_risk_pct`.
- `src/execution/trade_manager.py:240-245` does pass sector for newly opened trades, so the gap is specifically restart restore.

Impact:

After restart, sector exposure cap can be undercounted until positions close. This weakens the stated sector guard in `STRATEGY_STACK.md:19` and `config/config.yaml:193`.

Recommendation:

Pass `sector=str((trade.setup.setup_passport or {}).get("sector", "unknown"))` in `_load_state()` restore and add a restart-state test asserting `sector_risk_pct` is rebuilt.

### Medium - Strategy documentation is stale relative to active config

Evidence:

- `STRATEGY_STACK.md:5` says the latest operational truth is `session.md`, but `session.md` is absent in the current worktree.
- `STRATEGY_STACK.md:30` says paper baseline is `$1000`; `config/config.yaml:18` is `paper_starting_equity: 100`.
- `STRATEGY_STACK.md:32` says paper validation floor can go to `35`; `config/config.yaml:82-83` sets relaxation `22.0` and floor `28.0`.

Impact:

Operators auditing live/paper posture from docs will read wrong capital and admission thresholds. For a trading bot, stale operational docs are a risk control issue, not just documentation polish.

Recommendation:

Either restore/update `session.md` or remove it as the authority. Update `STRATEGY_STACK.md` to match `config/config.yaml` or explicitly mark its values as historical examples.

### Medium - `max_sector_risk_pct` is risk-critical but not typed in config schema

Evidence:

- `config/config.yaml:193` defines `max_sector_risk_pct: 2.5`.
- `src/risk/risk_manager.py:398-402` enforces it at runtime.
- `src/config/schema.py:78-98` types `max_symbol_risk_pct` and `max_direction_risk_pct`, but not `max_sector_risk_pct`.

Impact:

A typo or out-of-range sector risk cap can pass validation even though this value gates real exposure.

Recommendation:

Add `max_sector_risk_pct: float = Field(gt=0, le=50.0)` to `RiskConfig`, plus a test mirroring the existing symbol/direction risk tests.

### Medium - Strategy config contains a non-effective `trend_long_only` knob

Evidence:

- `config/config.yaml:362` declares `trend_long_only: false`.
- `src/models/strategy_router.py:44` hardcodes `self._trend_long_only = False` instead of reading config.
- `tests/test_strategy_router.py:11` sets `trend_long_only: True`, while `tests/test_strategy_router.py:108-116` still expects short trend-following to be allowed.

Impact:

This may be intentional policy, but the config key is misleading: changing it will not disable short trend-following. In a live trading context, non-effective risk/strategy knobs create operator false confidence.

Recommendation:

Remove the config key if two-sided trend following is mandatory, or restore config-driven behavior and update tests to match the intended policy.

### Medium - Positive-EV paper probe can bypass paper scope by design

Evidence:

- `src/main.py:1095-1144` allows positive-EV probes for configured short liquidity-sweep conditions, including `neutral` sleeve from `config/config.yaml:150-152`.
- `src/main.py:1762-1769` lets positive-EV probes bypass `_paper_trade_scope_check()`.

Impact:

The bypass is bounded by direction/regime/sm/score/EV checks, and it is paper-only. Still, it weakens the clean separation in docs that `neutral` is not alpha. This is acceptable only if the dataset/reporting clearly labels those records as probes.

Recommendation:

Persist a dedicated `paper_positive_ev_probe` flag/reason in setup passport and dataset rows, then report probe performance separately from normal sleeve performance.

## Performance Evidence

Local `data/trades/trade_log_futures.parquet` currently has:

- `16` rows.
- `16` are `record_type=no_trade`, `decision=NO_TRADE`.
- `0` closed trade rows.
- All 16 local rows are `paper_scope_blocked`, `reversal`, `sweep_reversal`, `long`, `asia`, `trending_expansion`, `liquidity_sweep`.

Conclusion: the local parquet is not sufficient evidence of positive or negative live strategy expectancy. It mostly proves that the current admission gates are logging near-miss/no-trade records.

## Current Audit Verdict

The code has broad unit coverage and the admission/risk architecture is materially better than a simple signal bot: regime, smart-money, EV observation, cohort/lifecycle, AdaptiveBrain, fund manager, risk caps, correlation, spread/session guards, and trade lifecycle management are all present.

The current blocking issue is operational: scanner runtime failure prevents forward validation. The main risk-control defects found are restart sector-risk undercounting and stale/non-effective operator-facing configuration/docs.

