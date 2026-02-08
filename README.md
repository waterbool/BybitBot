# Bybit ETHUSDT Algorithmic Trading Bot

This is a Python-based algorithmic trading bot for Bybit Futures (USDT Perpetual), specifically designed for `ETHUSDT`. It uses the `pybit` library for API interaction and `pandas`/`ta` for technical analysis.

## Features

- **Strategy**: EMA Crossover (Fast/Slow) with ATR-based Dynamic Stop Loss and Take Profit.
- **Risk Management**: Fixed USDT position sizing (not margin, but total contract value).
- **Mode**: Bybit Unified Trading Account (Linear Category).
- **Environment**: Supports both Testnet (Demo) and Mainnet.

## Project Structure

- `main.py`: Entry point. Runs the trading loop.
- `config.yaml`: Configuration (API keys, trading parameters).
- `bybit_client.py`: Wrapper for Bybit API.
- `indicators.py`: TA calculations (EMA, ATR).
- `strategy.py`: Signal generation logic.
- `risk.py`: Position sizing and precision handling.
- `requirements.txt`: Python dependencies.

## Installation

1. **Install Python 3.10+**.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Open `config.yaml`.
2. **Add your API Keys**:
   - Go to [Bybit Testnet](https://testnet.bybit.com) (for demo) or Bybit Mainnet.
   - Create API Keys with "Read-Write" permissions and "Contract - Orders/Positions" enabled.
   - Initial setting is `testnet: true`. Change to `false` for real money.
3. **Adjust Settings**:
   - `max_leverage`: Default 10.
   - `fixed_usdt_size`: Default 2.0 (The bot will buy ~2 USDT worth of ETH).
   - `dry_run`: Set to `true` to Simulate orders (Logs only). Set to `false` for Real trading.

4. **Risk Management**:
   - `max_trades_per_day`: Limit daily number of trades.
   - `max_daily_loss_usdt`: Stop trading if daily loss exceeds this amount.

## Safe Mode & Testing Workflow

1. **Start with Testnet + Dry Run**:
   - Set `api: testnet: true`
   - Set `trading: dry_run: true`
   - Run the bot. It will fetch data and Log "SIMULATED ORDER" without sending requests.

2. **Testnet Real Execution**:
   - Set `trading: dry_run: false`
   - Verify orders appear in Bybit Testnet.

3. **Live Trading**:
   - Set `api: testnet: false` (Update API keys to Mainnet keys)
   - Set `trading: dry_run: false`


## Usage

Run the bot from the terminal:

```bash
python main.py
```

The bot will:
1. Connect to Bybit.
2. Set your leverage.
3. Fetch candle data every minute.
4. Check for EMA crossovers.
5. If a signal occurs and no position is open, it will place a Market Order with attached TP/SL.

## Strategy Details

- **Entry**: 
  - BUY if EMA(Fast) crosses *above* EMA(Slow).
  - SELL if EMA(Fast) crosses *below* EMA(Slow).
- **Exit**: 
  - Take Profit: `Entry +/- (ATR * Multiplier * RewardRatio)`
  - Stop Loss: `Entry +/- (ATR * Multiplier)`

## Disclaimer

This software is for educational purposes. Trading futures involves high risk. Use Testnet first!
