# Ninja Trader — How-to Guide

Ninja Trader is an automated Binance Futures trading bot written in Python.
It scans USDT-margined perpetual pairs, scores them across multiple signals,
manages risk automatically, and learns from its own trade history.

> Session handoff: refer to `session.md` for the current repo/VM/bot state before repeating setup or debugging steps.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [API Key Setup](#3-api-key-setup)
4. [Understanding the Config](#4-understanding-the-config)
5. [Running the Bot](#5-running-the-bot)
6. [Running a Backtest](#6-running-a-backtest)
7. [How Signals & Scoring Work](#7-how-signals--scoring-work)
8. [Risk Management Rules](#8-risk-management-rules)
9. [Trade Lifecycle](#9-trade-lifecycle)
10. [Self-Learning Weights](#10-self-learning-weights)
11. [Logs & Data Files](#11-logs--data-files)
12. [Tuning Guide](#12-tuning-guide)
13. [Safety Checklist](#13-safety-checklist)

---

## 1. Prerequisites

| Requirement | Minimum version |
|---|---|
| Python | 3.10+ |
| pip | 23+ |
| Binance account | Futures enabled |

---

## 2. Installation

```bash
# Clone / enter the project folder
cd ninja_trader

# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 3. API Key Setup

### Step 1 — Create a Binance Futures API key

1. Log in to Binance → **Profile** → **API Management**
2. Create a new API key
3. Enable **Futures trading** permission
4. Disable **Withdrawals** (never needed by the bot)
5. Whitelist your server IP for extra safety

### Step 2 — Set keys in `.env`

Create a file called `.env` in the `ninja_trader/` folder:

```bash
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

The bot reads these at startup. **Never put keys directly in `config.yaml`.**

### Testnet (recommended first)

Binance Testnet gives you fake USDT to trade with real market conditions.

1. Go to [testnet.binancefuture.com](https://testnet.binancefuture.com) and create keys
2. Put those keys in `.env`
3. Keep `exchange.testnet: true` in `config.yaml`

---

## 4. Understanding the Config

All settings live in `config/config.yaml`. Key sections:

### `exchange`
```yaml
exchange:
  testnet: true        # true = testnet, false = live trading
  api_key: ""          # leave blank — set via .env
  api_secret: ""
```

### `trading`
```yaml
trading:
  mode: paper          # paper | live
                       # paper = simulates fills, no real orders sent
  min_score_threshold: 42   # only enter if signal score >= configured threshold
  max_open_trades: 2        # maximum simultaneous positions (paper override widens further)
  top_pairs_to_trade: 2     # pick top 2 scored pairs each cycle
  scan_interval_seconds: 60 # how often to re-scan the market
```

### `risk`
```yaml
risk:
  risk_per_trade_pct: 1.5    # risk 1.5% of account per trade
  max_risk_per_trade_pct: 2.5
  daily_loss_cap_pct: 5.0    # stop trading if down 5% on the day
  max_drawdown_pct: 20.0     # stop if portfolio drops 20% from peak
  min_rr_ratio: 2.0          # skip trade if reward < 2x risk
  default_leverage: 6        # 6x leverage on every position
  max_leverage: 10
```

### `exit`
```yaml
exit:
  tp1_r_multiple: 1.0        # take first profit at 1R gain
  tp2_r_multiple: 2.0        # take second profit at 2R gain
  tp1_size_pct: 0.50         # sell 50% of position at TP1
  tp2_size_pct: 0.30         # sell 30% of position at TP2
  tp2_trailing_stop_pct: 0.15  # trail remaining 20% by 15%
  max_hold_duration_s: 3600  # force-close after 1 hour
```

### `scoring.weights`
```yaml
scoring:
  weights:
    trend_strength: 20       # EMA alignment + RSI
    volume_confirmation: 15  # volume spike + buy/sell pressure
    structure_quality: 20    # BOS + liquidity sweeps
    open_interest: 15        # OI change direction
    funding_sentiment: 10    # funding rate (contrarian)
    order_book: 10           # bid/ask imbalance + walls
    volatility: 10           # ATR % of price (moderate = good)
```

Weights are automatically adjusted over time by the learning module.

### `backtest`
```yaml
backtest:
  start_date: "2023-01-01"
  end_date: "2024-01-01"
  initial_capital: 10000
  commission_pct: 0.04       # Binance taker fee
  slippage_pct: 0.05
```

---

## 5. Running the Bot

### Paper mode (no real money — default)

```bash
cd ninja_trader
python -m src
```

### Paper mode on testnet with explicit flags

```bash
python -m src --mode paper --testnet
```

### Live trading (real money — use with caution)

```bash
# 1. Set testnet: false in config.yaml first
# 2. Use real Binance API keys in .env
python -m src --mode live --no-testnet
```

### Custom config file

```bash
python -m src --config /path/to/my_config.yaml
```

### Stop the bot

Press `Ctrl+C`. The bot will close all open positions before exiting.

---

## 6. Running a Backtest

The backtester runs a walk-forward simulation over historical data.

### Default (2023, BTC/ETH/SOL/BNB, $10k)

```bash
cd ninja_trader
python -m src.backtest.run_backtest
```

### Custom date range and symbols

```bash
python -m src.backtest.run_backtest \
  --start 2022-01-01 \
  --end 2023-01-01 \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --capital 5000
```

### What the output shows

```
BTC/USDT:USDT — Trade Log
┌──┬──────┬─────────┬─────────┬──────────┬─────────┬──────────────┐
│# │ Dir  │  Entry  │  Exit   │    SL    │  TP1   │   Reason     │
└──┴──────┴─────────┴─────────┴──────────┴─────────┴──────────────┘

Backtest Summary
Symbol         Trades   Win%    PF   Total PnL   Return%   Max DD%
BTC/USDT:USDT    12    58.3%  1.82   +$428.11    +4.3%      6.2%
```

**Columns explained:**

| Column | Meaning |
|---|---|
| Win% | Percentage of trades that were profitable |
| PF (Profit Factor) | Gross wins ÷ gross losses. >1.5 is good |
| Total PnL | Net profit/loss in USDT |
| Return% | % return on starting capital |
| Max DD% | Largest peak-to-trough equity drop |

### Note on data

When Binance is unreachable (e.g. ISP restrictions), the backtest automatically
uses synthetic data generated with Geometric Brownian Motion — realistic price
behaviour with volatility clustering. Results are indicative, not exact.

---

## 7. How Signals & Scoring Work

Every scan cycle the bot scores each pair 0–100. Only pairs above
`min_score_threshold` (current paper baseline 42) are traded.

### Signal pipeline

```
1h candles ──► Trend strength    ─┐
4h candles ──► Structure quality  ├─► Weighted sum ──► Total score (0-100)
Live OB    ──► Order book         │
Funding    ──► Sentiment          │
OI data    ──► Open interest      │
Volume     ──► Vol confirmation  ─┘
```

### Signal descriptions

**Trend Strength (weight 20)**
Uses EMA 21/55/200 alignment and RSI.
- EMA 21 > EMA 55 > EMA 200 + price above all = strong long signal
- RSI 60–80 on a long = momentum confirmation

**Volume Confirmation (weight 15)**
- Detects volume spikes (>2x 20-bar average)
- Measures buy/sell pressure via candle body direction
- High buy volume on longs, high sell volume on shorts = confirmation

**Structure Quality (weight 20)**
- Detects Break-of-Structure (BOS): price closes above last swing high (bullish) or below last swing low (bearish)
- Detects Liquidity Sweeps: wick through a swing level followed by reversal
- BOS + sweep in same direction = highest score

**Open Interest (weight 15)**
- Rising OI + price trending = conviction
- Falling OI = weakening trend

**Funding Sentiment (weight 10)**
- Contrarian signal: very high positive funding → crowded longs → short bias
- Very negative funding → crowded shorts → long bias
- Near-zero funding = neutral filter state, not a sleeve by itself

**Order Book (weight 10)**
- Bid/ask volume ratio above 1.5 threshold = bullish pressure
- Large bid walls (>5x average level size) = support
- Opposite for short setups

**Volatility (weight 10)**
- ATR as % of price
- Ideal range: 0.5%–3%
- Too low = no opportunity; too high = dangerous = lower score

---

## 8. Risk Management Rules

The risk manager blocks new trades when any of these are true:

| Guard | Default trigger |
|---|---|
| Max open trades | 2 positions |
| Daily loss cap | −5% of account equity |
| Max drawdown | −20% from equity peak |
| Consecutive losses | 5 losses in a row |

### Position sizing formula

```
Risk per trade = Account equity × 1.5%
Position size  = Risk / Distance to stop loss
```

Example:
- Account: $70
- Risk per trade: $1.05
- Stop loss distance: 2% below entry
- Position size: $1.05 / 0.02 = $52.50 notional

### Leverage

Default leverage is 6x. The bot sets leverage automatically via API before
opening each position. Cap is 10x regardless of config.

---

## 9. Trade Lifecycle

```
Entry signal detected
        │
        ▼
Risk checks pass?  ──No──► Skip
        │Yes
        ▼
Open market order (entry)
Set stop-loss order
Set TP1 order (50% size)
Set TP2 order (30% size)
        │
        ▼ (polling every 0.5s)
        │
    TP1 hit? ──Yes──► Sell 50%
        │             Move SL to breakeven
        │             Start 15% trailing stop
        │
    Trailing stop hit? ──Yes──► Close remaining
        │
    TP2 hit? ──Yes──► Close remaining
        │
    SL hit? ──Yes──► Close all (loss)
        │
    1h timeout? ──Yes──► Force close
```

---

## 10. Self-Learning Weights

After every 20 closed trades, the learner analyses which signals
predicted wins vs losses and nudges weights accordingly.

`AdaptiveBrain` is already active as a sizing and pair-health overlay. The passive `MLPredictor` interface remains available for compatibility, but it is not the main decision brain.

**Example:**
- If `structure_quality` score was consistently higher on winning trades → its weight increases slightly
- If `open_interest` showed no difference between wins/losses → its weight stays flat

Learning state is saved to `models/learning_state.json`. Delete this file
to reset weights back to the defaults in `config.yaml`.

To see current weights:

```python
import json
state = json.load(open("models/learning_state.json"))
print(state["weights"])
```

---

## 11. Logs & Data Files

| Path | Contents |
|---|---|
| `logs/ninja_trader.log` | Full bot activity log |
| `logs/trades.log` | Trade-specific events |
| `models/learning_state.json` | Learned signal weights + trade history |
| `data/trades/` | Trade records directory |
| `data/cache/` | Cached backtest OHLCV data (pickle) |

### Log levels

Change `logging.level` in `config.yaml`:
- `DEBUG` — every tick, indicator value, order attempt
- `INFO` — trade opens/closes, heartbeats, scan summaries (default)
- `WARNING` — only problems

---

## 12. Tuning Guide

### Be more selective (fewer but higher-quality trades)

```yaml
trading:
  min_score_threshold: 82   # raise from 75
  top_pairs_to_trade: 1     # only the best pair
```

### Be more aggressive (more trades)

```yaml
trading:
  min_score_threshold: 65
  top_pairs_to_trade: 3
risk:
  risk_per_trade_pct: 2.0
```

### Tighter risk management

```yaml
risk:
  daily_loss_cap_pct: 3.0
  max_drawdown_pct: 7.0
  max_consecutive_losses: 3
exit:
  tp2_trailing_stop_pct: 0.10   # tighter trail
```

### Increase structure sensitivity

```yaml
structure:
  swing_lookback: 5             # detect shorter swings
  bos_confirmation_candles: 1   # faster BOS confirmation
  liquidity_sweep_pct: 0.2
```

### Favour trend-following over structure

```yaml
scoring:
  weights:
    trend_strength: 35
    structure_quality: 10
    volume_confirmation: 20
    open_interest: 15
    funding_sentiment: 10
    order_book: 5
    volatility: 5
```

---

## 13. Safety Checklist

Before going live, verify all of these:

- [ ] Tested in **paper mode** for at least 1 week
- [ ] Run a backtest — profit factor > 1.3 and max drawdown < 15%
- [ ] API key has **no withdrawal permission**
- [ ] API key IP-whitelisted to your server
- [ ] `exchange.testnet: false` set **only** when ready
- [ ] `daily_loss_cap_pct` set to a value you're comfortable losing in a day
- [ ] Starting capital is money you can afford to lose
- [ ] Monitor the bot for the first 48 hours of live trading
- [ ] `.env` file is in `.gitignore` (never commit API keys)

---

## Quick Reference

```bash
# Paper trading (default safe mode)
python -m src

# Live trading
python -m src --mode live --no-testnet

# Backtest 2023 full year
python -m src.backtest.run_backtest

# Backtest custom period
python -m src.backtest.run_backtest --start 2022-01-01 --end 2023-01-01 --capital 5000

# Backtest specific pairs
python -m src.backtest.run_backtest --symbols BTCUSDT ETHUSDT

# Custom config
python -m src --config /path/to/config.yaml
```
