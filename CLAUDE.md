
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Session handoff: read `session.md` first. It is the canonical current state for repo, VM, and bot runtime.

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Project: Ninja Trader

Automated Binance Futures (USDT-margined perpetual) trading bot in Python. A spot edition lives in `src/spot_main.py`. `watchdog.py` runs alongside to auto-restart on crashes or hangs.

### Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Credentials go in `.env` (never `config.yaml`):
```
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
TELEGRAM_TOKEN=...
TELEGRAM_CHAT_ID=...
```

### Running

```bash
# Paper trading (default, safe)
python -m src

# Live trading
python -m src --mode live --no-testnet

# Spot edition
python -m src.spot_main

# Backtest (default: BTC/ETH/SOL/BNB)
python -m src.backtest.run_backtest
python -m src.backtest.run_backtest --symbols BTCUSDT ETHUSDT --start 2023-01-01 --end 2024-01-01 --capital 5000

# Custom config
python -m src --config /path/to/config.yaml
```

There is no test suite — validate changes by running in paper mode.

### Architecture

Single async loop (`NinjaTrader._loop`) on a configurable scan interval:

```
Scanner → MarketDataService → Scorer (7 signals + 3 gates) → RiskManager
→ Executor → TradeManager (TP1/TP2/trail/SL/timeout)
→ Learner → MLPredictor (XGBoost, gates entries after 20+ trades)
→ FundManager (size multiplier) + ShadowEngine (ghost trades for ML data)
```

### Key files

| Path | Purpose |
|---|---|
| `config/config.yaml` | All tunable parameters (mode, risk, scoring weights, etc.) |
| `src/analysis/` | Indicators, regime, smart money, structure (BOS/sweeps), EV model |
| `src/models/ev_model.py` | Probabilistic EV: Bayesian p_win, fees, slippage |
| `src/models/regime_classifier.py` | Regime: trending_expansion / accumulation_compression / distribution / chaos |
| `data/state.json` | Live dashboard state written each tick |
| `models/learning_state_futures.json` | Persisted scorer weights |
| `data/trades/` | Parquet trade log (ML training + weight learning) |

### Domain invariants

- **Paper mode always forces `testnet: true`** — the config loader enforces this.
- **Live mode rejects `testnet: true`** — `RuntimeError` at startup.
- **Signal confirmation**: signal must appear above threshold for `signal_confirmation_scans` consecutive cycles before opening.
- **Three gates must all pass**: regime, smart money, EV — checked before ML gate.
- **ML gate**: blocks entries below dynamic p_win threshold and per-regime win-rate floor (38%) once ≥20 trades are logged.
- **Auto-live**: if `safety.auto_live_on_readiness: true`, bot self-transitions from paper to live after all `LiveReadiness` criteria pass.
