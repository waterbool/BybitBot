# Bybit Trading Bot - Web UI Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Web UI
```bash
python web_ui.py
```

### 3. Access Dashboard
Open your browser and navigate to:
```
http://localhost:5000
```

## Features

### 📊 Dashboard
- Real-time trading statistics
- Performance charts
- Recent trade history
- Win rate and PnL tracking

### ⚙️ Configuration
- Edit all bot settings through web interface
- API key management
- Strategy parameters
- Risk management settings
- Changes saved to `config.yaml`

### 📈 Backtest
- Run backtests with custom parameters
- Visual chart with indicators
- Detailed performance metrics
- Trade-by-trade analysis

### 💹 Live Trading
- Start/Stop bot with one click
- Real-time position monitoring
- Live trading statistics
- Dry-run mode support

### 📝 Logs
- Real-time log streaming
- Color-coded log levels
- Auto-scrolling display

## Safety Features

- **Dry Run Mode**: Test strategies without real orders
- **Testnet Support**: Practice on Bybit testnet
- **Risk Limits**: Max trades per day and daily loss limits
- **Configuration Backup**: Auto-backup before saving changes

## Usage Tips

1. **First Time Setup**:
   - Go to Configuration tab
   - Enter your API keys
   - Enable "Testnet" and "Dry Run" for safety
   - Save configuration

2. **Testing Strategy**:
   - Go to Backtest tab
   - Set number of days to test
   - Click "Run Backtest"
   - Review results and charts

3. **Live Trading**:
   - Review configuration carefully
   - Start with Dry Run mode enabled
   - Monitor logs for any issues
   - Click "Start Trading" when ready

## Important Notes

⚠️ **Always test on Testnet first!**
⚠️ **Use Dry Run mode before live trading!**
⚠️ **Review risk limits before starting!**

## Troubleshooting

**Can't access UI?**
- Check if web_ui.py is running
- Try http://127.0.0.1:5000 instead

**Bot won't start?**
- Check API keys in configuration
- Verify Testnet/Mainnet setting matches your keys
- Review logs for error messages

**No data in charts?**
- Make sure bot has been running
- Check if trades have been executed
- Verify symbol and timeframe settings
