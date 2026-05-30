# ⚡ QuantEdge — Institutional-Grade Crypto Futures Trading Engine

> A professional quantitative trading system for Binance USDT-margined perpetual futures.
> Built with 11 institutional-grade layers, adaptive ML, and production risk management.

---

## 🎯 What Is This?

QuantEdge is a **fully automated crypto futures trading engine** that operates like a mini quant fund. It's not a simple signal bot — it's a multi-layered decision system where every trade must pass through **8-10 institutional gates** before execution.

**Think of it as hiring a full quant team** — from CRO to portfolio manager to execution desk — all running 24/7 on autopilot.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    QUANTEDGE ENGINE                      │
├─────────────────────────────────────────────────────────┤
│  Scanner → Market Data → Scorer → Strategy Router       │
│     ↓          ↓           ↓           ↓                │
│  Passport → Admission → Cohort Policy → Lifecycle       │
│     ↓          ↓           ↓           ↓                │
│  Adaptive Brain → Risk Manager → Execution              │
│     ↓          ↓           ↓           ↓                │
│  Trade Manager → Reporting → Telegram Notifications      │
└─────────────────────────────────────────────────────────┘
```

### 11 Institutional Layers

| Layer | Role | What It Does |
|-------|------|-------------|
| **Scanner** | Universe Selection | Discovers liquid pairs via composite liquidity scoring |
| **Market Data** | Data Engineering | Fetches OHLCV, order book, funding, OI, market context |
| **Scorer** | Alpha Research | Multi-factor scoring with z-score normalization |
| **Strategy Router** | Strategy Allocation | Routes to trend/reversal/breakout sleeves |
| **Setup Passport** | Trade Contract | Creates thesis contract for each trade |
| **Admission Policy** | Compliance | Paper/live gate with regime and direction filters |
| **Cohort Policy** | Performance Tracking | Monitors health by sleeve × regime × session |
| **Adaptive Brain** | AI Orchestrator | Thompson bandit + online ML + regime HMM + alpha decay |
| **Risk Manager** | CRO / Risk | VaR/CVaR, dynamic drawdown, Kelly sizing, exposure caps |
| **Execution** | Trading Desk | Order placement, SL/TP, trailing stops, partial exits |
| **Trade Manager** | Position Management | Momentum stall detection, time-decay trailing, exit profiling |

---

## 🧠 Key Features

### Professional Risk Management
- **VaR / CVaR** — Historical Value-at-Risk and Conditional VaR from equity curve
- **Dynamic Drawdown Management** — Graduated size reduction (warning: 50%, critical: 25%)
- **Recovery Periods** — 3-hour ramp-up after drawdown (60% → 70% → 85% → 100%)
- **Kelly Criterion** — Fractional Kelly sizing with volatility adjustment
- **Portfolio Heat Tracking** — Real-time monitoring of total open risk
- **Circuit Breakers** — Exponential backoff on repeated errors

### Adaptive Intelligence
- **Thompson Sampling Bandit** — Adaptive sleeve allocation per market context
- **Online Bayesian Logistic** — Multi-feature P(win) prediction
- **Regime Transition Matrix** — Forward-looking regime stability
- **Alpha Decay Tracker** — Per-pair edge monitoring
- **Volatility Targeter** — Portfolio-level vol control (Moreira & Muir 2017)

### Smart Execution
- **Momentum Stall Detection** — Auto-exit if trade stagnates without progress
- **Volatility-Adjusted Trailing** — Tightens when momentum dies
- **Time-Decay Exit Intelligence** — Professional exit profiling
- **Position Rotation** — Replace weak trades with stronger candidates
- **Session-Aware Sizing** — Weekend reduction, transition tightening

### Multi-Factor Scoring
- **Cross-Sectional Z-Score** — Normalize signals relative to recent distribution
- **Adaptive Thresholds** — Regime-aware score thresholds
- **Smart Money Detection** — Institutional order flow analysis
- **Sector Rotation** — Cross-sector money flow tracking
- **Market Context** — BTC trend, ETH/BTC, risk-on/risk-off awareness

---

## 📊 What You Get

### Telegram Reports
- Real-time trade alerts with full institutional trade cards
- Hourly heartbeat with equity, drawdown, open positions
- Daily bootstrap audit with statistical significance testing
- Risk-adjusted metrics (Sharpe, Sortino, Calmar)
- Sector rotation and market context updates

### Backtesting & Shadow Mode
- Full backtest engine with historical data
- Shadow mode: simulate trades in parallel without risk
- Dataset logger for ML training
- Attribution reporting by sleeve, regime, sector, direction

### Production-Grade Infrastructure
- Graceful error recovery with circuit breakers
- State persistence across restarts
- Rotating log files
- Workspace auto-cleanup
- Watchdog monitoring

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your Binance API keys (optional for paper mode)
```

### 3. Run in Paper Mode

```bash
python -m src --config config/config.yaml
```

### 4. Run in Live Mode

```bash
# Set mode: live in config/config.yaml
# Ensure BINANCE_API_KEY and BINANCE_API_SECRET are set in .env
python -m src --config config/config.yaml --mode live
```

---

## ⚙️ Configuration

All configuration is in `config/config.yaml`. Key sections:

```yaml
exchange:
  name: binance_futures
  testnet: false
  # fapi_base_url: "https://fapi1.binance.com"  # Use if fapi.binance.com is blocked

trading:
  mode: paper                    # paper | live | backtest
  min_score_threshold: 42        # Minimum score to consider a trade
  max_open_trades: 2             # Maximum concurrent positions
  scan_interval_seconds: 60      # How often to scan for opportunities

risk:
  risk_per_trade_pct: 1.5        # Risk per trade as % of equity
  max_risk_per_trade_pct: 2.5    # Hard cap per trade
  max_drawdown_pct: 15.0         # Kill switch threshold
  daily_loss_cap_pct: 5.0        # Daily loss limit
  default_leverage: 5            # Default leverage
  max_leverage: 10               # Maximum leverage allowed

exit:
  tp1_r_multiple: 1.5            # First take-profit at 1.5R
  tp2_r_multiple: 2.5            # Second take-profit at 2.5R
  breakeven_trigger_r: 1.0       # Move SL to breakeven at 1R
  trailing_atr_multiplier: 1.5   # ATR-based trailing stop
```

---

## 📁 Project Structure

```
quantedge/
├── config/
│   └── config.yaml              # Main configuration
├── src/
│   ├── main.py                  # Entry point & main loop
│   ├── analysis/                # Market analysis modules
│   │   ├── indicators.py        # Technical indicators
│   │   ├── structure.py         # Market structure analysis
│   │   ├── smart_money.py       # Institutional flow detection
│   │   ├── order_book.py        # Order book analysis
│   │   └── ...
│   ├── models/                  # ML & decision models
│   │   ├── adaptive_brain.py    # AI orchestrator
│   │   ├── strategy_router.py   # Strategy allocation
│   │   ├── ev_model.py          # Expected value model
│   │   ├── regime_classifier.py # Market regime detection
│   │   └── ...
│   ├── risk/                    # Risk management
│   │   ├── risk_manager.py      # CRO — VaR, drawdown, sizing
│   │   ├── kelly_sizer.py       # Kelly criterion sizing
│   │   ├── correlation_filter.py # Exposure correlation
│   │   └── session_modulator.py # Session-aware sizing
│   ├── scoring/                 # Signal scoring
│   │   └── scorer.py            # Multi-factor scoring engine
│   ├── execution/               # Trade execution
│   │   ├── executor.py          # Order placement
│   │   └── trade_manager.py     # Position lifecycle management
│   ├── data/                    # Market data
│   │   ├── client.py            # Binance API client
│   │   └── market_data.py       # Data service
│   ├── scanner/                 # Universe selection
│   │   └── scanner.py           # Pair scanner
│   ├── reporting/               # Analytics & reporting
│   │   ├── attribution.py       # Trade attribution
│   │   └── trading_brain.py     # Performance journal
│   ├── notifications/           # Alerts
│   │   └── telegram.py          # Telegram notifier
│   └── learning/                # ML training
│       └── learner.py           # Trade record learning
├── tests/                       # 394 comprehensive tests
├── .env.example                 # Environment template
└── requirements.txt             # Python dependencies
```

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -q

# Run specific test
python -m pytest tests/test_risk_manager.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

---

## 📈 Performance Metrics

The system tracks and reports institutional-grade metrics:

- **Sharpe Ratio** — Risk-adjusted return (annualized)
- **Sortino Ratio** — Downside deviation-adjusted return
- **Calmar Ratio** — Return / max drawdown
- **VaR / CVaR** — Value-at-Risk and Conditional VaR
- **Profit Factor** — Gross profit / gross loss
- **Win Rate** — Percentage of winning trades
- **Kelly Criterion** — Theoretical optimal bet fraction
- **Recovery Factor** — Net profit / max drawdown

---

## 🔒 Safety Features

- **Paper Mode** — Test with virtual money before going live
- **Kill Switch** — Automatic shutdown on repeated failures
- **Drawdown Limits** — Daily, weekly, and max drawdown caps
- **Consecutive Loss Protection** — Cooldown after loss streaks
- **Exposure Caps** — Per-symbol, per-direction, per-sector limits
- **Circuit Breakers** — Exponential backoff on system errors
- **State Persistence** — Survives restarts without losing positions

---

## 🎓 How It Works (Simplified)

1. **Scanner** finds liquid futures pairs on Binance
2. **Market Data** fetches candles, order book, funding, OI
3. **Scorer** computes a multi-factor score (0-100) with z-score normalization
4. **Strategy Router** assigns the best strategy sleeve (trend/reversal/breakout)
5. **Setup Passport** creates a trade thesis contract
6. **Admission Policy** checks regime, direction, and sleeve eligibility
7. **Cohort Policy** verifies historical health of this pattern
8. **Adaptive Brain** applies ML-based sizing adjustments
9. **Risk Manager** calculates position size, stops, and targets
10. **Execution** places the order with proper SL/TP
11. **Trade Manager** monitors and exits based on momentum and time

**Every trade must pass ALL layers.** A good signal alone is not enough.

---

## 🛠️ Tech Stack

- **Python 3.12+**
- **CCXT** — Exchange connectivity (Binance)
- **Pandas / NumPy** — Data processing
- **SciPy / scikit-learn** — ML models
- **aiohttp** — Async HTTP
- **Rich** — Terminal UI
- **Telegram Bot API** — Notifications

---

## 📝 License

MIT License — See [LICENSE](LICENSE) for details.

---

## ⚠️ Disclaimer

This software is for **educational and research purposes**. Trading cryptocurrency futures involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk. The authors are not responsible for any financial losses incurred from using this software.

Always start in **paper mode** and thoroughly test before using real money.

---

## 🤝 Support

For questions, issues, or feature requests, please open an issue on GitHub.

---

**Built with ❤️ for quantitative traders who demand institutional-grade tools.**
