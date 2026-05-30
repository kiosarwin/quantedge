# QuantEdge — Technical Handbook

## What Is This

An automated crypto futures trading engine running on Binance USDT-margined perpetual futures. It scans pairs, scores signals using institutional-grade indicators, manages positions with dynamic exits, and learns from its own trade history to improve over time.

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
| Trade Manager | `execution/trade_manager.py` | Monitors open positions, manages trailing/TP/SL exits |
| Learner | `learning/learner.py` | Stores closed trades, computes running statistics |
| Shadow Engine | `backtest/shadow_engine.py` | Simulates alternative exit strategies |
| Telegram | `notifications/telegram.py` | Sends trade notifications and reports |
| Fund Manager | `models/fund_manager.py` | Institutional position scaling and portfolio management |
| Adaptive Brain | `models/adaptive_brain.py` | AI orchestrator with Thompson bandit + online ML |
| Strategy Router | `models/strategy_router.py` | Routes signals to strategy sleeves |
| Cohort Policy | `models/cohort_policy.py` | Tracks health of trade cohorts |
| Risk Analytics | `risk/risk_manager.py` | VaR/CVaR, Sharpe, Sortino, Calmar metrics |

---

## Key Concepts

### Score
A 0-100 value computed from trend, volume, structure, OI, funding, order book, and volatility. Cross-sectionally normalized via z-score.

### Regime
Four-state classification: `trending_expansion`, `accumulation_compression`, `distribution`, `chaos`. Chaos blocks all trades.

### Smart Money Phase
Detects institutional activity: `accumulation`, `trending`, `distribution`, `liquidity_sweep`, `chaos`, `neutral`.

### Expected Value (EV)
Probability-weighted outcome: `EV = P(win) × avg_win - P(loss) × avg_loss`. Must be positive for trade admission.

### Kelly Criterion
Optimal bet fraction: `Kelly = (p × b - q) / b`. Used at 25-30% fraction (quarter to third Kelly).

### Setup Passport
A contract created at entry that defines the expected trade path, invalidation conditions, and quality metrics.

### Cohort
A group of trades sharing: `strategy_sleeve × market_regime × sm_phase × session × direction`.

---

## Trade Lifecycle

```
1. Scanner finds eligible pairs
2. Market Data fetches snapshots
3. Scorer computes score + regime + smart money + EV
4. Strategy Router assigns sleeve (trend/reversal/breakout)
5. Setup Passport creates trade thesis
6. Admission Policy checks gates
7. Cohort Policy verifies historical health
8. Adaptive Brain applies ML sizing
9. Risk Manager calculates stops/targets/size
10. Execution places orders
11. Trade Manager monitors and exits
```

---

## Risk Management

### Position Sizing
- Fractional Kelly (25-30% of optimal)
- Volatility-adjusted (scale down in high vol)
- Confidence-based (scale up for high-conviction setups)
- Drawdown-aware (reduce during drawdown periods)

### Exposure Limits
- Per-symbol risk cap
- Per-direction risk cap
- Per-sector risk cap
- Total portfolio heat monitoring

### Kill Switches
- Max drawdown exceeded
- Daily loss cap hit
- Weekly loss cap hit
- Consecutive loss limit
- Runtime error burst
- Heartbeat failure

---

## Configuration

All parameters are in `config/config.yaml`. See `SETUP.md` for detailed configuration guide.

---

## Testing

```bash
python -m pytest tests/ -q  # Run all 394 tests
```

---

## Monitoring

### Terminal
Rich-formatted output with color-coded log levels.

### Telegram
- Trade alerts with full trade cards
- Hourly heartbeat with equity/drawdown
- Daily performance reports
- Error notifications

### Log Files
`logs/futures_trader.log` with rotation (10MB, 5 backups).

---

## Key Files

| File | Purpose |
|---|---|
| `config/config.yaml` | All trading parameters |
| `src/main.py` | Entry point and main loop |
| `src/risk/risk_manager.py` | Risk management (CRO) |
| `src/scoring/scorer.py` | Signal scoring engine |
| `src/models/adaptive_brain.py` | AI orchestrator |
| `src/execution/trade_manager.py` | Position management |
| `src/reporting/attribution.py` | Performance attribution |
| `data/open_trades.json` | Persisted open positions |
| `data/learner_state.json` | Learning state |
| `data/fund_manager_state.json` | Fund manager state |

---

## Glossary

| Term | Definition |
|---|---|
| **R-multiple** | Profit/loss expressed in units of initial risk (1R = 1× stop distance) |
| **MFE** | Maximum Favorable Excursion — best price reached during trade |
| **MAE** | Maximum Adverse Excursion — worst price reached during trade |
| **VaR** | Value-at-Risk — maximum expected loss at given confidence level |
| **CVaR** | Conditional VaR — expected loss beyond VaR (tail risk) |
| **Sharpe Ratio** | Risk-adjusted return (excess return / volatility) |
| **Sortino Ratio** | Risk-adjusted return (excess return / downside volatility) |
| **Calmar Ratio** | Return / max drawdown |
| **Kelly Fraction** | Optimal bet size as fraction of bankroll |
| **PF** | Profit Factor — gross profit / gross loss |
| **ATR** | Average True Range — volatility measure |
| **OI** | Open Interest — total outstanding contracts |
| **BOS** | Break of Structure — market structure change |
| **CHoCH** | Change of Character — trend reversal signal |

---

For setup instructions, see [SETUP.md](SETUP.md).
For the full strategy architecture, see [STRATEGY_STACK.md](STRATEGY_STACK.md).
