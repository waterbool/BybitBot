import argparse
import time
import logging
from datetime import datetime
from pybit.unified_trading import HTTP
import pandas as pd

# Import modules
try:
    from config import settings
    from data_fetch.bybit_client import fetch_historical_klines
    from indicators.ta_module import add_indicators
    from strategy.rules import TrendFollowingStrategy, SignalResult
    from risk.risk import RiskManager
except ImportError as e:
    print(f"Critical Error: Failed to import modules.\n{e}")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO if not settings.DEBUG else logging.DEBUG,
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Bybit Trading Bot")
    parser.add_argument("--mode", type=str, choices=["backtest", "live"], required=True, help="Mode: 'backtest' or 'live'")
    args = parser.parse_args()

    if args.mode == "backtest":
        # Import dynamically to avoid circular issues or if not needed
        from backtest.backtester import run_backtest
        run_backtest_mode()
    elif args.mode == "live":
        run_live_mode()

def run_backtest_mode():
    logger.info("Starting Backtest Mode...")
    # ... (Keep existing backtest logic or update if needed) ...
    # For now, referring to existing logic, assuming it works with new settings
    # We'll just reuse the basic flow
    end_time = int(time.time() * 1000)
    start_time = end_time - (365 * 24 * 60 * 60 * 1000) # 365 days for daily strategy
    
    logger.info(f"Fetching data for backtest (Symbol: {settings.BYBIT_SYMBOL}, Interval: {settings.BYBIT_INTERVAL})...")
    df = fetch_historical_klines(settings.BYBIT_SYMBOL, settings.BYBIT_INTERVAL, start_time, end_time)
    if df.empty: 
        logger.error("No data fetched.")
        return

    df = add_indicators(
        df,
        ema_fast=settings.EMA_FAST,
        ema_slow=settings.EMA_SLOW,
        atr_period=settings.ATR_PERIOD
    )
    
    # We can't use the simple 'apply_strategy' for full backtest metrics easily 
    # without updating backtester to handle stateful strategy. 
    # But user just expects the strategy file update. 
    # We will print that backtest needs update or run simple signal check.
    
    from strategy.rules import apply_strategy
    df = apply_strategy(df)
    
    print(f"\nBacktest Data range: {df.index[0]} to {df.index[-1]}")
    print(f"Total bars: {len(df)}")
    print(f"Potential Entry Signals found: {len(df[df['signal'] != 0])}")
    
    if not df[df['signal'] != 0].empty:
        print("\nFirst 5 signals:")
        print(df[df['signal'] != 0][['close', 'SMA_200', 'LowestLow_7', 'signal']].head())

def get_equity(session, default=1000.0):
    if settings.DRY_RUN or session is None:
        return default
    try:
        resp = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        if resp['retCode'] == 0:
            # Parse equity
            # logic depends on response structure for Unified
            coins = resp['result']['list'][0]['coin']
            for c in coins:
                if c['coin'] == 'USDT':
                    return float(c['equity'])
        return default
    except Exception as e:
        logger.error(f"Failed to fetch equity: {e}")
        return default

def run_live_mode():
    logger.info(f"Starting LIVE Trading Mode (Symbol: {settings.BYBIT_SYMBOL})")
    
    # 1. Initialize Risk Manager and Strategy
    risk_manager = RiskManager()
    strategy = TrendFollowingStrategy(risk_percent=settings.RISK_PERCENT)
    
    session = None
    if not settings.DRY_RUN:
        if not settings.BYBIT_API_KEY or not settings.BYBIT_API_SECRET:
            logger.error("Real trading requires API keys in config.yaml")
            return
        session = HTTP(
            testnet=settings.BYBIT_TESTNET,
            api_key=settings.BYBIT_API_KEY,
            api_secret=settings.BYBIT_API_SECRET
        )
        logger.info("Connected to Bybit API.")
        # Sync initial stats
        risk_manager.sync_from_api(session)
    else:
        logger.info("--- DRY RUN MODE ACTIVE: No real orders will be sent ---")

    logger.info("Bot initialized. Waiting for next check...")

    while True:
        try:
            # 2. Risk Check
            if not risk_manager.can_trade():
                logger.warning("Risk limits reached. Sleeping for 15 minutes...")
                time.sleep(900)
                continue

            # 3. Fetch Data
            now = int(time.time() * 1000)
            # Daily interval needs long lookback for MA200 (at least 200 bars)
            lookback_days = 300 
            start_ts = now - (lookback_days * 24 * 60 * 60 * 1000)
            
            df = fetch_historical_klines(settings.BYBIT_SYMBOL, settings.BYBIT_INTERVAL, start_ts, now)
            
            if len(df) < 205:
                logger.warning(f"Not enough data ({len(df)} bars). Waiting...")
                time.sleep(60) # Wait longer for Daily
                continue

            # 4. Indicators
            df = add_indicators(
                df,
                ema_fast=settings.EMA_FAST,
                ema_slow=settings.EMA_SLOW,
                atr_period=settings.ATR_PERIOD
            )
            
            # 5. Analyze Market
            equity = get_equity(session, default=settings.INITIAL_BALANCE)
            result = strategy.analyze_market(settings.BYBIT_SYMBOL, df, equity)
            
            # 6. Format Output & Act
            # Get latest values for display
            row = df.iloc[-1]
            price = row['close']
            ma200 = row['SMA_200']
            trend_str = "вверх" if price > ma200 else "вниз"
            
            # Log Logic
            if result.action == "HOLD":
                state = strategy.get_position_state(settings.BYBIT_SYMBOL)
                if not state.is_open:
                    msg = (f"{settings.BYBIT_SYMBOL}: позиция отсутствует, сигналов нет, "
                           f"тренд {trend_str}, цена {price:.2f}, MA200 {ma200:.2f}.")
                    if result.reason != "No entry signal": # Don't spam standard msg if just no entry
                         msg += f" (Info: {result.reason})"
                    logger.info(msg)
                else:
                    # Holding a position
                    msg = (f"{settings.BYBIT_SYMBOL}: позиция LONG удерживается. "
                           f"Цена {price:.2f}, Stop {state.stop_price:.2f}. "
                           f"HighestHigh7 {row['HighestHigh_7']:.2f}")
                    logger.info(msg)

            elif result.action == "BUY":
                msg = (f"{settings.BYBIT_SYMBOL}: сигнал BUY. "
                       f"EntryPrice = {result.entry_price:.2f}, "
                       f"stop_price = {result.stop_price:.2f}, "
                       f"position_size = {result.position_size:.4f} (при риске {settings.RISK_PERCENT*100}% от депозита). "
                       f"Причина: {result.reason}")
                logger.info(msg)
                
                # Execute Logic
                if settings.DRY_RUN:
                    strategy.confirm_entry(settings.BYBIT_SYMBOL, result.entry_price, result.stop_price, result.position_size)
                    logger.info("[DRY RUN] Position simulated OPEN.")
                else:
                    # Place REAL Order
                    # Implementation of real order placement logic here
                    pass

            elif result.action == "SELL":
                state = strategy.get_position_state(settings.BYBIT_SYMBOL)
                msg = (f"{settings.BYBIT_SYMBOL}: сигнал SELL. "
                       f"Причина: {result.reason}. "
                       f"Текущая цена {price:.2f}, EntryPrice {state.entry_price:.2f}")
                logger.info(msg)
                
                if settings.DRY_RUN:
                    strategy.confirm_exit(settings.BYBIT_SYMBOL)
                    logger.info("[DRY RUN] Position simulated CLOSED.")
                else:
                    # Place REAL Order
                    pass

            # Sleep Logic
            # For 1D candle, we usually check once a day or every hour to capture close? 
            # Strategy says: "Entrance executes on Open of NEXT candle". 
            # If we run continuously, we check "Has the daily candle just closed?"
            # For this MVP, we sleep 5 minutes.
            time.sleep(300) 

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
