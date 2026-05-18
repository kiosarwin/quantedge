# Ninja Trader — Handbook

## What Is This

An automated crypto futures trading bot running on Binance USDT-margined perpetual futures. It scans pairs, scores signals using institutional-grade indicators, manages positions with dynamic exits, and learns from its own trade history to improve over time.

Named after Jim Simons — the bot's internal performance persona is called **Jim**.

---

## Architecture Overview

```
Scanner → Scorer → Risk / EV Gate → Executor → TradeManager
                                                     ↓
                                              DatasetLogger
                                              Learner (weights)
                                              ShadowEngine
```

| Module | File | Role |
|---|---|---|
| Scanner | `scanner/scanner.py` | Filters Binance Futures pairs by volume, blacklist |
| Market Data | `data/market_data.py` | Fetches OHLCV, funding rate, OI, L/S ratio, order book |
| Scorer | `scoring/scorer.py` | Builds feature vector, classifies regime, scores signal |
| Risk Manager | `risk/risk_manager.py` | Equity tracking, drawdown guards, position sizing |
| Kelly Sizer | `risk/kelly_sizer.py` | Fractional Kelly position sizing |
| Executor | `execution/executor.py` | Places entry, SL, TP orders on Binance |
| Trade Manager | `execution/trade_manager.py` | Monitors open trades, manages exits, MFE/MAE tracking |
| Learner | `learning/learner.py` | Records trades, adjusts signal weights over time |
| EV Model | `models/ev_model.py` | Probabilistic expected value gate |
| ML Engine | `models/ml_engine.py` | Live readiness checker, fund manager reporting |
| Fund Manager | `models/fund_manager.py` | Sharpe, Calmar, profit factor reporting |
| Shadow Engine | `backtest/shadow_engine.py` | Ghost trades running in parallel to build ML data faster |
| Dataset Logger | `data/dataset_logger.py` | Writes 85-column parquet dataset for future ML training |
| Telegram | `notifications/telegram.py` | Alerts for trades, heartbeat, fund manager reports |

---

## Signal Scoring Pipeline

Every 60 seconds the bot scores all eligible pairs. A trade is only taken when all gates pass.

### Step 1 — Feature Vector
Built from multi-timeframe data (1h primary, 4h higher, 15m lower, 5m entry):
- ATR, ADX, EMA stack (21/55/200)
- RSI, volume ratio, taker buy ratio
- Open interest change rate
- Funding rate deviation
- Order book imbalance
- Liquidation pressure
- BTC correlation

### Step 2 — Regime Classification (4 states)
| Regime | Condition | Score threshold |
|---|---|---|
| `trending_expansion` | ADX > 25, expanding range | 55 |
| `accumulation_compression` | Low ATR, OI rising | 65 |
| `distribution` | Bearish structure, OI falling | blocked |
| `chaos` | Extreme vol or no structure | blocked |

### Step 3 — Smart Money Gate
Detects institutional accumulation/distribution via OI + price + volume patterns. Trade blocked if no directional bias or score < 60.

### Step 4 — EV Model Gate
Calculates expected value after taker fees (0.04%) and slippage (0.10%). Trade blocked if net EV < 0.05%. Uses Bayesian priors until 20 closed trades, then switches to empirical win rate.

### Step 5 — Weighted Score
| Component | Weight |
|---|---|
| Trend strength | 20 |
| Structure quality | 20 |
| Volume confirmation | 15 |
| Open interest | 15 |
| Order book | 10 |
| Funding sentiment | 10 |
| Volatility | 10 |

Weights are dynamically adjusted by the Learner after each closed trade (adjustment rate: 0.02 per trade).

**Minimum score to trade: 60** (regime-dependent, see Step 2).

### Step 6 — Signal Confirmation
A signal must appear on **2 consecutive scans** (2 × 60s = 2 minutes) before an order is placed. Expires if not confirmed within 5 minutes.

---

## Position Management

### Entry
- Limit orders preferred (offset 0.05% from mid)
- Times out after 60s → cancelled
- Scaled into 2 levels: 60% first, 40% second

### Risk per trade
- Default: 1.5% of equity ($7.50 at $500)
- Hard cap: 2.0% ($10.00)
- Position sized via fractional Kelly (quarter-Kelly, capped at 3%)
- Leverage: 5x default, up to 10x
- Min notional: $5 (Binance minimum)

### Exit levels
| Level | Trigger | Action |
|---|---|---|
| SL | 1.5× ATR from entry | Close 100% |
| TP1 | 1.5R | Close 50%, move SL to breakeven, start trailing |
| TP2 | 2.0R | Close 30% of remainder, continue trailing |
| Trailing stop | ATR × 1.5 ratchet | Close remaining after TP1 |
| Max hold | 48 hours | Close all |

### MFE / MAE Tracking
Every open trade tracks in real time:
- **MFE (Maximum Favorable Excursion)** in R — how far price moved in your favor at peak
- **MAE (Maximum Adverse Excursion)** in R — how far price moved against you at worst
- `time_to_mfe_peak_s` — seconds from entry to peak
- `drawdown_duration_s` — total seconds spent underwater

---

## Risk Guards

| Guard | Threshold | Action |
|---|---|---|
| Daily loss cap | -5% equity ($25) | Stop trading for the day |
| Max drawdown | -15% equity ($75) | Stop trading |
| Consecutive losses | 3 in a row | Pause |
| Extreme volatility | ATR × 3.0 | Pause |
| Min R:R ratio | 2.0 | Block trade |
| Max open trades | 2 | Block new entries |

---

## Learning System

### Phase 1 — Priors only (0–19 closed trades)
- EV model uses conservative Bayesian priors (52% win rate)
- Signal weights updated after each trade but with small adjustments

### Phase 2 — EV model active (20+ closed trades) ✅ Jim is here
- Empirical win rate replaces priors
- Kelly sizing adjusts based on actual outcome distribution
- Signal weights actively shift toward components that predicted winning trades

### Phase 3 — Per-signal decay (50+ closed trades, future)
- Individual signal components weighted by their own historical predictiveness

State persists across restarts in `models/learning_state_futures.json`.

---

## ML Dataset (Parquet)

Every signal evaluation writes a row to `data/trades/trade_log_futures.parquet`.

**85 columns** covering:
- Signal scores and EV model output at decision time
- Regime, feature vector, smart money, spot context
- Trade dynamics: MFE, MAE, hold duration, time to peak
- Outcome: PnL, R:R achieved, exit type, win/loss

**No-trade near-misses** (score ≥ 55 but trade not taken) are also logged — enabling the model to learn what it rejected and whether that was correct.

This dataset is the foundation for Phase 3 ML training.

---

## Shadow Engine

Runs parallel "ghost trades" at a lower score threshold (no capital at risk). Accelerates ML data collection. State persists to `models/shadow_state.json` across restarts.

---

## Deployment

**Infrastructure:** GCP e2-micro, `systemd` service `ninja-watchdog`

**Modes:**
- `paper` — virtual $500 equity, real Binance testnet API, no real orders
- `live` — real capital, requires mainnet API keys, `testnet: false`
- `backtest` — historical OHLCV replay

**Safeguard:** `paper` mode forces `testnet: true` in code regardless of config. `live` mode refuses to start if `testnet: true`.

**Auto-live transition:** When 7 readiness criteria are met (win rate, drawdown, trade count, etc.), the bot can auto-switch from paper to live. Requires mainnet API keys pre-loaded in `.env`.

### Environment variables (`.env`)
```
BINANCE_API_KEY=
BINANCE_API_SECRET=
TELEGRAM_TOKEN=
TELEGRAM_CHAT_ID=
```

### Run
```bash
python -m src                          # default config
python -m src --mode paper             # force paper mode
python -m src --config path/to/config.yaml
```

---

## Key Files

```
src/
  config.yaml                    — all settings
  main.py                        — bot orchestrator
  data/
    client.py                    — Binance API client
    market_data.py               — OHLCV + derivatives data
    dataset_logger.py            — parquet ML dataset writer
  analysis/
    indicators.py                — ATR, ADX, EMA, RSI, volume
    structure.py                 — BOS, liquidity sweeps, swing points
    smart_money.py               — OI-based institutional detection
    feature_engine.py            — unified feature vector
    regime.py                    — regime classification
    spot_context.py              — spot/futures basis, Coinbase premium
  scoring/scorer.py              — full signal pipeline
  risk/
    risk_manager.py              — equity, drawdown, position limits
    kelly_sizer.py               — Kelly position sizing
  execution/
    executor.py                  — order placement
    trade_manager.py             — open trade lifecycle + MFE/MAE
  learning/learner.py            — weight adjustment, trade log
  models/
    ev_model.py                  — expected value calculation
    ml_engine.py                 — live readiness, ML prediction
    fund_manager.py              — performance metrics
  backtest/
    engine.py                    — historical backtester
    shadow_engine.py             — parallel ghost trading
    reporter.py                  — backtest results tables
  notifications/telegram.py      — trade alerts, heartbeat

models/
  learning_state_futures.json    — learner weights + trade log (persisted)
  shadow_state.json              — shadow engine state (persisted)
data/trades/
  trade_log_futures.parquet      — ML training dataset
logs/
  futures_trader.log             — rotating log file
```

---

## Telegram Reports

| Message | Trigger |
|---|---|
| Trade opened | On entry fill |
| Trade closed | On exit with PnL, exit reason, MFE/MAE |
| Heartbeat | Every 60 minutes — equity, drawdown, daily PnL, open positions |
| Fund manager report | Every 10 closed trades — Sharpe, win rate, profit factor, Jim's bonus |
| Live readiness | When all 7 criteria pass |

---

## Tuning Cheat Sheet

| Goal | Config key | Location |
|---|---|---|
| Trade more often | Lower `min_score_threshold` | `trading` |
| Trade less in choppy markets | Raise `accumulation_compression` threshold | `trading.regime_thresholds` |
| Risk more per trade | Raise `risk_per_trade_pct` | `risk` |
| Take profit earlier | Lower `tp1_r_multiple` | `exit` |
| Tighter trailing stop | Lower `trailing_atr_multiplier` | `exit` |
| Require stronger trend | Raise `adx_trending_threshold` | `regime` |
| Filter lower volume pairs | Raise `min_24h_volume_usdt` | `filters` |
