# Ninja Trader — Futures Edition Runbook

## Overview

Institutional-grade Binance USDT-margined Perpetual Futures bot.
Paper / live modes. Triple-gate trade filter: regime + smart money + EV.

---

## Quick Start

```bash
cd /home/arwin/ninja_trader
source venv/bin/activate
python -m src.main
```

---

## Prerequisites

### VPN (Required — Binance blocked in Indonesia)

```bash
# Start WARP VPN
warp-cli --accept-tos connect
warp-cli status          # should say "Connected"

# Enable on boot (already set)
sudo systemctl status warp-svc
```

If WARP is disconnected, restart it:
```bash
warp-cli --accept-tos connect
```

### Python Environment

```bash
cd /home/arwin/ninja_trader
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install httpx          # Telegram notifications
```

---

## Configuration

All settings in `config/config.yaml`.

### Key Settings

| Setting | Default | Description |
|---|---|---|
| `trading.mode` | `paper` | `paper` / `live` / `backtest` |
| `trading.paper_starting_equity` | `10000` | Virtual USDT for paper mode |
| `trading.min_score_threshold` | `75` | Minimum score to open trade |
| `trading.max_open_trades` | `3` | Max concurrent positions |
| `risk.default_leverage` | `5` | Leverage for new positions |
| `risk.risk_per_trade_pct` | `1.5` | % equity risked per trade |
| `risk.daily_loss_cap_pct` | `5.0` | Circuit breaker: daily loss % |
| `safety.max_consecutive_losses` | `3` | Circuit breaker: losing streak |

### API Keys (Live Trading Only)

Set in `.env` or directly in config:
```bash
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
```

### Telegram Alerts

Already configured in `config/config.yaml`:
```yaml
telegram:
  token: "YOUR_TELEGRAM_TOKEN"
  chat_id: YOUR_TELEGRAM_CHAT_ID
  heartbeat_interval_minutes: 60
```

---

## Trading Logic

### Trade Lifecycle

```
Scan symbols (every 60s)
  → Score each symbol (0–100)
  → Triple-gate filter:
      [1] Regime gate   — must be TRENDING_EXPANSION or ACCUMULATION_COMPRESSION
      [2] Smart money   — must not be CHAOS; NEUTRAL passes through
      [3] EV gate       — net EV > 0 after fees (bypassed for first 20 trades)
  → Score must be ≥ 75
  → Open position with Kelly-sized risk
  → Manage exits: TP1 (50%) → TP2 (30%) → trailing stop (20%)
  → Log to parquet, update ML weights
```

### Score Components (sum = 100 pts)

| Component | Weight | Signal |
|---|---|---|
| Trend Strength | 20 | EMA alignment + ADX |
| Structure Quality | 20 | BOS/CHoCH, swing levels |
| Volume Confirmation | 15 | Volume spike vs 20-bar avg |
| Open Interest | 15 | OI change vs 15-min baseline |
| Funding Sentiment | 10 | Funding rate extremes |
| Order Book | 10 | Bid/ask imbalance |
| Volatility | 10 | ATR regime |

### Score Penalties

- Regime = CHAOS: score × 0.0 (blocked)
- Smart money fails: score × 0.6
- EV negative: score × 0.7

### Gate Behavior

- **Regime**: only TRENDING_EXPANSION and ACCUMULATION_COMPRESSION allow trades
- **Smart money NEUTRAL**: passes through (no info = don't block)
- **EV bootstrap**: first 20 trades bypass EV gate (insufficient history)

---

## Exit Strategy

| Exit | Trigger | Size |
|---|---|---|
| TP1 | +1.0R | 50% |
| TP2 | +2.0R | 30% |
| Trailing stop | After TP1 hit | 20% (1.5 ATR trail) |
| Breakeven | After TP1 | Move SL to entry |
| Max hold | 60 minutes | Force close remaining |

---

## Monitoring

### Log Files

```bash
# Live log (tail)
tail -f logs/futures_trader.log

# Trade log
tail -f logs/futures_trades.log

# Filter for trades only
grep "TRADE\|OPEN\|CLOSE\|BLOCKED\|Score" logs/futures_trader.log
```

### Key Log Messages

| Message | Meaning |
|---|---|
| `[BTC/USDT] BLOCKED regime=HIGH_VOLATILITY_CHAOS` | Regime gate rejected |
| `[BTC/USDT] BLOCKED smart_money phase=CHAOS` | SM gate rejected |
| `[BTC/USDT] BLOCKED EV=...` | EV gate rejected (EV negative) |
| `Score=XX.X / top pair` | Symbol scored, checking threshold |
| `TRADE OPENED` | Position entered |
| `TRADE CLOSED` | Position exited |
| `Circuit breaker triggered` | Bot halted — check reason |

### Telegram Alerts

Bot sends:
- Startup notification
- Every trade open (with full setup details)
- Every trade close (with P&L)
- Circuit breaker triggered
- Hourly heartbeat (equity, drawdown, top candidates)

---

## Troubleshooting

### Bot shows $0 equity

Cause: Either `trading.mode` is not `paper`, or `paper_starting_equity` is missing.

Fix:
```yaml
trading:
  mode: paper
  paper_starting_equity: 10000
```

### No trades opening (scores 60–70, below 75)

Normal when market is ranging or low-volatility. Check:
```bash
grep "BLOCKED\|Score" logs/futures_trader.log | tail -50
```

To lower threshold temporarily for testing (not recommended for live):
```yaml
trading:
  min_score_threshold: 60
```

### Binance connection refused / SSL error

WARP VPN is disconnected. Reconnect:
```bash
warp-cli --accept-tos connect
warp-cli status
```

### L/S ratio or taker_buy stuck at 1.00 / 0.50

These are fallback values returned when the Binance endpoint fails.
Check debug logs:
```bash
grep "L/S ratio\|Taker buy" logs/futures_trader.log
```

### Circuit breaker triggered

Bot halted due to daily loss cap (5%) or consecutive losses (3).
Check the reason in log/Telegram. Resume next day or restart:
```bash
python -m src.main
```

### Regime always HIGH_VOLATILITY_CHAOS

Extreme market conditions (news event, crash). Wait for market to stabilize.
The `atr_extreme_vol_multiplier: 2.5` threshold determines this.

---

## Paper → Live Transition

1. Get Binance Futures API key (enable futures, disable withdrawals)
2. Set API keys in `.env`:
   ```bash
   BINANCE_API_KEY=xxx
   BINANCE_API_SECRET=xxx
   ```
3. Change config:
   ```yaml
   trading:
     mode: live
   exchange:
     testnet: false
   ```
4. Start with minimum size: `risk_per_trade_pct: 0.5`
5. Monitor first 5 trades manually before stepping away

---

## File Structure

```
ninja_trader/
├── config/
│   └── config.yaml          # All configuration
├── src/
│   ├── main.py              # Bot entry point and main loop
│   ├── data/
│   │   ├── client.py        # Binance FAPI wrapper (ccxt)
│   │   └── market_data.py   # Snapshot aggregator + OI rolling baseline
│   ├── scoring/
│   │   └── scorer.py        # Signal scoring engine (7 components)
│   ├── analysis/
│   │   ├── indicators.py    # EMA, ADX, ATR, RSI
│   │   ├── structure.py     # BOS/CHoCH, swing levels
│   │   ├── order_book.py    # Bid/ask imbalance
│   │   ├── sentiment.py     # Funding rate, OI scoring
│   │   ├── smart_money.py   # Smart money phase detection
│   │   └── feature_engine.py # Feature vector F(t)
│   ├── models/
│   │   ├── ev_model.py      # Probabilistic EV calculation
│   │   └── regime_classifier.py # 4-state regime (trending/sideways/chaos)
│   ├── risk/
│   │   ├── risk_manager.py  # Position sizing, circuit breakers
│   │   └── kelly_sizer.py   # Fractional Kelly criterion
│   ├── execution/
│   │   └── executor.py      # Order placement, TP/SL management
│   ├── learning/
│   │   └── learner.py       # Trade log, adaptive weight updates
│   └── notifications/
│       └── telegram.py      # Telegram Bot API alerts
├── logs/                    # Runtime logs
├── data/trades/             # Trade history (parquet)
├── models/                  # Learning state (JSON)
└── runbook.md               # This file
```

---

## Emergency Stop

Kill the bot immediately:
```bash
# Find PID
ps aux | grep "src.main"

# Kill
kill <PID>
```

Or press `Ctrl+C` in the terminal running the bot.

---

## Maintenance

### Weekly

- Review trade log: `data/trades/trade_log_futures.parquet`
- Check if EV model confidence is building (needs 20+ trades)
- Review adaptive weights in `models/learning_state_futures.json`

### Monthly

- Rotate logs if > 10 MB: `logs/` directory
- Review blacklisted pairs in config vs current market
- Adjust `min_score_threshold` based on win rate

---

## Key Numbers

| Metric | Value |
|---|---|
| Min score to trade | 75 / 100 |
| Max concurrent trades | 3 |
| Default leverage | 5× |
| Risk per trade | 1.5% equity |
| Daily loss cap | 5% equity |
| Max drawdown cap | 10% equity |
| EV bootstrap trades | 20 |
| Taker fee | 0.04% |
| TP1 / TP2 / Trail | 50% / 30% / 20% |
