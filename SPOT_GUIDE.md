# Ninja Smart Trader — Spot Edition
## Complete Setup & Strategy Guide

> Session handoff: sebelum setup ulang atau debug, baca `session.md` untuk state terbaru repo/VM/bot.

> **A precision Binance Spot swing trading bot — long only, no leverage,
> designed for consistent capital growth.**

---

## Table of Contents

1. [Philosophy & Design](#1-philosophy--design)
2. [Prerequisites & Installation](#2-prerequisites--installation)
3. [API Key Setup](#3-api-key-setup)
4. [Configuration Reference](#4-configuration-reference)
5. [How the Strategy Works](#5-how-the-strategy-works)
6. [Scoring System Explained](#6-scoring-system-explained)
7. [Entry Strategies](#7-entry-strategies)
8. [Risk Management Rules](#8-risk-management-rules)
9. [Running Paper Trading](#9-running-paper-trading)
10. [Running Live Trading](#10-running-live-trading)
11. [Running a Backtest](#11-running-a-backtest)
12. [Adaptive Learning](#12-adaptive-learning)
13. [Safety Systems](#13-safety-systems)
14. [Reading the Logs](#14-reading-the-logs)
15. [Tuning Guide](#15-tuning-guide)

---

## 1. Philosophy & Design

This is a **precision Spot swing trading system** built on these principles:

| Principle | Implementation |
|-----------|---------------|
| Wait patiently | Score threshold of 75/100 — only top setups trade |
| No overtrading | Max 3 concurrent positions |
| Trend following | No trades in sideways / ranging markets |
| Structure-based entries | Three distinct high-probability patterns |
| BTC correlation | Altcoin entries halted when BTC is dumping |
| Capital protection | ATR-based stops, daily loss cap, drawdown guard |
| Consistent growth | Partial TPs (10/18/25%) lock in profit progressively |

**Long only. No leverage. No shorting. Spot markets only.**

---

## 2. Prerequisites & Installation

**Requirements:**

| Dependency | Minimum |
|-----------|---------|
| Python | 3.10+ |
| pip | 23+ |
| Binance account | Spot API enabled |

```bash
# Navigate to the project
cd ninja_trader

# Create virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 3. API Key Setup

### Step 1 — Create Binance Spot API Key

1. Log in to Binance → **Profile** → **API Management**
2. **Create API key** (label it `ninja_spot_trader`)
3. **Enable:** Spot & Margin Trading ✓
4. **Disable:** Futures, Withdrawals ✗ (never needed)
5. Restrict to your IP address for security

### Step 2 — Set Environment Variables

Create a `.env` file in the `ninja_trader/` folder:

```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

> **Never commit `.env` to git. It's in `.gitignore` by default.**

### Step 3 — Verify (paper mode only)

```bash
python -m src.spot_main --mode paper
```

You should see `Connected to Binance Spot` in the logs.

---

## 4. Configuration Reference

All settings live in `config/config.yaml`.

### Capital

```yaml
capital:
  total_usdt: 1000.0       # Starting balance for paper mode
  min_position_pct: 10     # Minimum 10% per trade
  max_position_pct: 25     # Maximum 25% per trade
```

The bot dynamically sizes between 10–25% based on setup quality (R:R ratio).

### Scoring Threshold

```yaml
trading:
  min_score_threshold: 75   # Skip any setup scoring below 75/100
  max_open_trades: 3        # Never hold more than 3 coins at once
```

Raising this to 80 makes the bot more selective (fewer, higher-quality trades).

### Exit Strategy

```yaml
exit:
  tp1_pct: 0.10            # Sell 40% at +10%
  tp1_size_pct: 0.40
  tp2_pct: 0.18            # Sell 35% at +18%
  tp2_size_pct: 0.35
  tp3_pct: 0.25            # Sell remaining at +25%
  tp3_size_pct: 0.25
  trailing_stop_pct: 0.05  # 5% trailing after TP1 hit
  max_hold_hours: 120      # Force close after 5 days
```

After TP1 hits, the stop-loss moves to breakeven (free trade), and a 5%
trailing stop starts following the price up.

### BTC Safety Gate

```yaml
btc:
  safety_enabled: true
  dump_threshold_4h_pct: -3.0   # BTC -3% on 4H candle = HALT
  recovery_threshold_pct: 1.5   # BTC must recover +1.5% to resume
  below_ema_halt: true          # Also halt if BTC is below EMA50
```

This is the most critical filter. When BTC dumps, altcoins follow.
The gate prevents new entries during BTC weakness and waits for recovery
before allowing new trades.

---

## 5. How the Strategy Works

### Main Loop (every 5 minutes)

```
1. Fetch BTC 4H candles → evaluate BTC Guard
   └─ If HALTED → skip new entries (monitor open trades only)

2. Scan all Binance Spot USDT pairs
   └─ Filter: volume > $5M, active, not stablecoin

3. Fetch OHLCV (4H, 1H, 15m) for top 30 pairs

4. Score each pair (0–100):
   ├─ Regime check (4H ADX + ATR)
   │   └─ SIDEWAYS / HIGH_VOL → score = 0 → skip
   ├─ Trend strength (EMA50/200 alignment + RSI)
   ├─ Volume expansion (spike vs average + buy pressure)
   ├─ Structure quality (BOS + liquidity sweep)
   ├─ Momentum alignment (RSI + ADX)
   ├─ BTC correlation (BTC trend + return correlation)
   └─ Volatility condition (ATR in ideal range)

5. Sort by score → pick top candidates with score ≥ 75

6. Run entry strategy detector on 15m timeframe:
   ├─ Breakout + Retest
   ├─ Pullback to EMA50
   └─ Liquidity Sweep

7. Only enter if entry strategy confirms (confidence > 0.5)

8. Open position: market buy → track in memory

9. Monitor every tick:
   ├─ Price hits SL → sell all (stop-loss)
   ├─ Price +10% → sell 40%, move SL to entry (breakeven)
   ├─ Price +18% → sell 35%, continue trailing
   ├─ Price +25% → sell remaining 25%
   └─ Trailing stop 5% from peak → sell remainder
```

### Market Regime Classification

| Regime | ADX | Condition | Trading |
|--------|-----|-----------|---------|
| TRENDING | > 25 | Price > EMA200 + EMAs aligned | ✅ Allowed |
| SIDEWAYS | < 25 | No clear direction | ❌ No trade |
| HIGH_VOLATILITY | any | ATR > 2.5× average | ⚠️ Skipped |

---

## 6. Scoring System Explained

Each component scores 0–100. Final score is the weighted average.

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| **Trend Strength** | 25 | EMA50/200 alignment + RSI momentum |
| **Volume Expansion** | 20 | Current volume vs 20-bar average + buy pressure |
| **Structure Quality** | 20 | Break of Structure (BOS) + liquidity sweep patterns |
| **Momentum Alignment** | 15 | RSI 50-65 zone (not overbought) + ADX > 25 |
| **BTC Correlation** | 10 | BTC above EMA50 + positive return correlation with alt |
| **Volatility Condition** | 10 | ATR% in ideal range (0.5%–3%) |

**ONLY TRADE if total score ≥ 75.**

### Score interpretation

| Score | Meaning |
|-------|---------|
| 90–100 | Premium setup — exceptional alignment |
| 75–89 | Quality setup — enter with full sizing |
| 60–74 | Marginal — skip (below threshold) |
| < 60 | No trade |

---

## 7. Entry Strategies

### Strategy 1: Breakout + Retest ⭐

**Best for:** Confirmed uptrends with clean breakouts

1. A swing high (resistance) is broken with strong volume
2. Price pulls back to retest the broken level (now support)
3. A bullish rejection candle forms at the support zone
4. Volume is above average (confirming buyers defending the level)

```
Resistance level: ████████████────────────────
                  ─────────────────█───────────  ← breakout candle
                  ──────────────────────█──────  ← retest (pullback)
                  ───────────────────────█─────  ← rejection (ENTRY)
Stop-loss: below the support level
```

### Strategy 2: Pullback to EMA50 ⭐

**Best for:** Strong trends with healthy corrections

1. Price is in a confirmed uptrend (above EMA200, EMA50 > EMA200)
2. Price retraces to the EMA50 zone (within 0.5 ATR)
3. A bullish bounce candle closes above EMA50
4. Volume at or above average

```
                  ┌─────── price
EMA50 ──────────┐ │    ┌── bounce (ENTRY)
                └─┘    │
EMA200 ────────────────┘
Stop-loss: below EMA50 – 2×ATR
```

### Strategy 3: Liquidity Sweep ⭐⭐ (Highest R:R)

**Best for:** Market structure traps / stop hunts

1. Price makes a fake breakdown below a swing low (hunts stops)
2. Strong recovery closes back above the swept level
3. The recovery candle is large and bullish (volume spike required)
4. Enter on the strong close — stops are very tight (just below the wick)

```
Swing low: ─────────────────████─────────────
                              │
Sweep wick: ─────────────────┘█─────────────  ← fake breakdown
Recovery:   ──────────────────████───────────  ← ENTRY (strong close above)
Stop-loss: below the wick low
```

---

## 8. Risk Management Rules

### Position Sizing

Position size scales with setup quality:

```
size = equity × (10% to 25%)
      — minimum 10% when R:R is at threshold
      — maximum 25% when R:R is significantly above threshold
```

### Stop-Loss Calculation

```
Stop-loss = entry_price - ATR(14) × 2.0

Example:
  Entry: $100.00
  ATR:   $2.50
  SL:    $100.00 - $2.50 × 2 = $95.00 (5% risk)
```

### Daily Guards

| Guard | Default | Action |
|-------|---------|--------|
| Daily loss cap | -5% | No new entries for rest of day |
| Max drawdown | -12% from peak | Bot pauses all entries |
| Consecutive losses | 4 in a row | Pause and review |

### Partial TP Logic

```
Entry: $100.00 (100 units bought, cost $10,000)
─────────────────────────────────────────────
TP1: $110.00 → sell 40 units ($4,400) → SL moves to $100 (breakeven)
                                         Trailing stop starts at $104.50
TP2: $118.00 → sell 21 units ($2,478)
TP3: $125.00 → sell remaining 39 units ($4,875)

Total realized: ~$1,753 profit on $10,000 invested = +17.5%
If trailing stop hits at $115 after TP2: still captures $11.5% on remaining
```

---

## 9. Running Paper Trading

```bash
cd ninja_trader
source venv/bin/activate

# Paper mode (default) — real data, simulated orders
python -m src.spot_main --mode paper

# Or explicitly:
python -m src.spot_main --config config/config.yaml --mode paper
```

Paper mode uses real Binance market data but never places real orders.
All fills are simulated at current market price.

**Recommended:** Run paper mode for at least 2–4 weeks before going live.

---

## 10. Running Live Trading

> ⚠️ **Warning:** Live mode uses real money. Start with small capital.
> Test paper mode first. Understand all risks.

```bash
# 1. Set your API keys in .env
# 2. Set testnet: false in config/config.yaml
# 3. Start with small capital (e.g., $200–500)

python -m src.spot_main --mode live
```

**Safety checklist before going live:**
- [ ] Paper traded for 2+ weeks
- [ ] API key has NO withdrawal permissions
- [ ] API key restricted to your IP
- [ ] `max_open_trades: 3` (start conservative)
- [ ] `total_usdt` set to your actual balance
- [ ] BTC guard enabled (`safety_enabled: true`)
- [ ] Verified config `min_score_threshold: 75`

---

## 11. Running a Backtest

```bash
# Default backtest (BTC, ETH, SOL, BNB, AVAX — 2024)
python -m src.backtest.spot_backtester

# Custom symbols and date range
python -m src.backtest.spot_backtester \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --capital 10000

# Single symbol deep dive
python -m src.backtest.spot_backtester --symbols SOLUSDT --start 2023-01-01
```

### Backtester Output

```
─────────────────────────────────────────
  Spot Backtest Summary
┌────────┬────────┬──────┬──────┬───────┐
│ Symbol │ Trades │ Win% │  PF  │ Return│
├────────┼────────┼──────┼──────┼───────┤
│ BTC    │  42    │ 61.9%│ 2.14 │ +38.2%│
│ ETH    │  38    │ 57.9%│ 1.87 │ +29.6%│
│ SOL    │  51    │ 64.7%│ 2.41 │ +67.3%│
└────────┴────────┴──────┴──────┴───────┘
```

**What to look for:**
- Win rate > 50%
- Profit factor > 1.5
- Max drawdown < 15%
- At least 30+ trades per symbol for statistical significance

---

## 12. Adaptive Learning

The bot tracks every closed trade and uses the results to nudge scoring weights
toward signals that actually correlated with profitable trades.

After 20+ closed trades, the learner:
1. Compares average signal scores for wins vs losses
2. Increases weight for signals that were higher on winning trades
3. Decreases weight for signals that were misleading
4. Re-normalises weights to maintain the 100-point total

**Learning state** is saved to `models/learning_state.json`.

To reset learning:
```bash
rm models/learning_state.json
```

---

## 13. Safety Systems

### BTC Guard (most important)

When BTC drops > 3% on a single 4H candle OR trades below its EMA50:
- All new altcoin entries are blocked
- Open positions continue to be monitored
- The gate reopens only when BTC recovers +1.5% from its low AND is back above EMA50

### Consecutive Loss Pause

After 4 straight losses, no new entries are placed.
The bot continues monitoring open trades.
You must review and restart to resume.

### Daily Loss Cap

If the portfolio loses more than 5% in a single day, no new entries for the rest of that day.

### Extreme Volatility Skip

If a coin's ATR is more than 2.5× its average, it's skipped (even if score is high).
Extreme volatility = unpredictable = skip.

### Regime Filter

No trades in sideways markets (ADX < 25).
This alone eliminates a large % of losing trades.

---

## 14. Reading the Logs

```
INFO  ♥ Heartbeat  equity=$1,234.56  daily_pnl=+2.1%  drawdown=0.3%  open=2
INFO  BTC [SAFE]  price=67234.50  ema50=65100.00  4h_chg=+0.87%
INFO  SpotScanner: 47 candidates (from 312 total tickers)
INFO  Fetching data for 30 pairs...

  Top Spot Signals
┌──────────┬───────┬──────────┬───────┬────────┬────────┐
│ Symbol   │ Score │ Regime   │ Trend │ Volume │ Struct │
├──────────┼───────┼──────────┼───────┼────────┼────────┤
│ SOLUSDT  │ 83.2  │ trending │   91  │   78   │   72   │
│ AVAXUSDT │ 79.4  │ trending │   82  │   71   │   68   │
└──────────┴───────┴──────────┴───────┴────────┴────────┘

INFO  → ENTRY SOLUSDT  score=83  strategy=breakout_retest  conf=0.72
INFO  OPENED SOLUSDT  price=145.230  qty=8.610 ($1250)  sl=136.500  tp1=159.750  strategy=breakout_retest

INFO  TP1 hit SOLUSDT @ 159.750  sold=3.444  pnl=+$46.32  new_sl=145.230 (breakeven)
INFO  TP2 hit SOLUSDT @ 171.370  sold=2.682  pnl=+$69.85
INFO  ✅ CLOSED SOLUSDT @ 176.200  total_pnl=[+$143.20] (11.5%)  reason=tp3
```

---

## 15. Tuning Guide

### Bot is trading too frequently
```yaml
trading:
  min_score_threshold: 80   # raise from 75

entry:
  min_volume_ratio: 2.0     # require stronger volume confirmation
```

### Bot is not finding trades
```yaml
trading:
  min_score_threshold: 70   # lower slightly (be careful)
  scan_interval_seconds: 180  # scan more frequently

btc:
  below_ema_halt: false     # disable EMA50 halt (more permissive)
```

### Stops are too tight (stopped out too often)
```yaml
indicators:
  atr_sl_multiplier: 2.5    # widen from 2.0
```

### Stops are too wide (losses are large)
```yaml
indicators:
  atr_sl_multiplier: 1.5    # tighten from 2.0
risk:
  min_rr_ratio: 2.5         # require better R:R
```

### Backtesting a different period
```yaml
backtest:
  start_date: "2023-01-01"
  end_date: "2023-12-31"
```

---

*Built with precision. Trade with discipline. Protect capital first.*
