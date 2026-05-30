# QuantEdge — Setup Guide

## Prerequisites

- Python 3.12 or higher
- pip (Python package manager)
- Binance account (for live trading)
- Telegram bot (optional, for notifications)

## Step 1: Clone & Install

```bash
git clone <your-repo-url> quantedge
cd quantedge
pip install -r requirements.txt
```

## Step 2: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

### For Paper Mode (No API Keys Required)
Leave `BINANCE_API_KEY` and `BINANCE_API_SECRET` empty. The bot will use public market data.

### For Live Mode
1. Go to [Binance API Management](https://www.binance.com/en/my/settings/api-management)
2. Create a new API key
3. Enable Futures trading permissions
4. Copy the API Key and Secret to `.env`:

```env
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

### For Telegram Notifications (Optional)
1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Create a new bot with `/newbot`
3. Copy the bot token to `.env`
4. Message [@userinfobot](https://t.me/userinfobot) to get your chat ID
5. Copy the chat ID to `.env`

## Step 3: Configure Trading Parameters

Edit `config/config.yaml`:

```yaml
trading:
  mode: paper                    # Start with paper mode!
  min_score_threshold: 42        # Minimum score (0-100) to consider
  max_open_trades: 2             # Max concurrent positions
  scan_interval_seconds: 60      # Scan frequency

risk:
  risk_per_trade_pct: 1.5        # Risk per trade (% of equity)
  max_drawdown_pct: 15.0         # Kill switch threshold
  default_leverage: 5            # Default leverage (1-125)
```

## Step 4: Run the Bot

### Paper Mode (Recommended First)
```bash
python -m src --config config/config.yaml
```

### Live Mode
```bash
# First, update config/config.yaml:
# trading:
#   mode: live

python -m src --config config/config.yaml --mode live
```

### Backtest Mode
```bash
python -m src --config config/config.yaml --mode backtest
```

## Step 5: Monitor

### Terminal Output
The bot logs to terminal with Rich formatting. Key info:
- Heartbeat updates every 30 seconds
- Trade signals and scores
- Risk metrics (VaR, Sharpe, drawdown)
- Position updates

### Telegram Notifications
If configured, you'll receive:
- Trade open/close alerts
- Hourly heartbeat reports
- Daily performance summaries
- Error alerts

### Log Files
Logs are saved to `logs/futures_trader.log` with rotation.

## Running in Background

### Using tmux (Recommended)
```bash
tmux new-session -d -s quantedge "python -m src --config config/config.yaml"
tmux attach -t quantedge  # Reattach to monitor
```

### Using nohup
```bash
nohup python -m src --config config/config.yaml > /dev/null 2>&1 &
```

## Troubleshooting

### "Connection timeout to Binance"
If `fapi.binance.com` is blocked in your region:
```yaml
exchange:
  fapi_base_url: "https://fapi1.binance.com"
```

### "No pairs found"
- Check internet connection
- Verify Binance API is accessible
- Try increasing `min_24h_volume_usdt` in config

### "Kill switch activated"
The bot stopped due to repeated errors. Check:
1. API key permissions
2. Account balance
3. Network connectivity
4. Log files for specific errors

### Reset Paper State
To start fresh in paper mode:
```yaml
trading:
  paper_starting_equity: 1000
  paper_state_reset_token: "reset_$(date +%Y%m%d)"
```

## Production Deployment

### On a VPS (Recommended)
1. Use Ubuntu 22.04+ or similar
2. Install Python 3.12+
3. Use tmux or systemd for process management
4. Set up log rotation
5. Monitor with Telegram alerts

### Systemd Service (Optional)
Create `/etc/systemd/system/quantedge.service`:
```ini
[Unit]
Description=QuantEdge Trading Engine
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/quantedge
ExecStart=/usr/bin/python3 -m src --config config/config.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable quantedge
sudo systemctl start quantedge
sudo journalctl -u quantedge -f  # Monitor logs
```

## Security Best Practices

1. **Never commit `.env`** — It's in `.gitignore` by default
2. **Use IP whitelisting** — Restrict API key to your server IP
3. **Disable withdrawal permissions** — API key only needs trading permissions
4. **Start with paper mode** — Test thoroughly before live
5. **Set conservative limits** — Start with low leverage and small position sizes
6. **Monitor actively** — Check Telegram alerts and log files regularly

## Next Steps

1. Run in paper mode for at least 1-2 weeks
2. Monitor the Telegram reports for performance
3. Adjust scoring thresholds based on results
4. Gradually increase position sizes
5. Consider moving to live mode only after consistent paper profits

---

**Remember:** Past performance does not guarantee future results. Always use proper risk management and never invest more than you can afford to lose.
