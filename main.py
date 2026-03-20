import argparse
import time
import logging
from pybit.unified_trading import HTTP

# Import modules
try:
    from config import settings
    from bot_controller import BotController
    from backtest.backtester import run_backtest
    from risk.risk import RiskManager
    from single_symbol_pipeline import (
        fetch_price_frame_in_window,
        fetch_signal_frame_for_lookback,
    )
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
        run_backtest_mode()
    elif args.mode == "live":
        run_live_mode()
def run_backtest_mode():
    logger.info("Starting Backtest Mode...")
    logger.info("CLI backtest uses the same single-symbol baseline strategy path as CLI live mode.")
    end_time = int(time.time() * 1000)
    start_time = end_time - (365 * 24 * 60 * 60 * 1000)

    logger.info(f"Fetching data for backtest (Symbol: {settings.BYBIT_SYMBOL}, Interval: {settings.BYBIT_INTERVAL})...")
    df, _interval_minutes = fetch_price_frame_in_window(
        symbol=settings.BYBIT_SYMBOL,
        interval=settings.BYBIT_INTERVAL,
        start_ts=start_time,
        end_ts=end_time,
        category=settings.BYBIT_CATEGORY,
        now_ms=end_time,
    )
    if df.empty:
        logger.error("No data fetched.")
        return

    metrics, trades_df, equity_df, _monthly_stats = run_backtest(
        df,
        initial_balance=settings.INITIAL_BALANCE,
    )

    start_dt = df["datetime"].iloc[0]
    end_dt = df["datetime"].iloc[-1]
    print(f"\nBacktest range: {start_dt} -> {end_dt}")
    print(f"Bars: {len(df)}")
    print(f"Trades: {metrics['total_trades']}")
    print(f"Net PnL: {metrics['net_pnl']:.4f}")
    print(f"Final balance: {metrics['final_balance']:.4f}")
    print(f"Max drawdown: {metrics['max_drawdown']:.4f}")
    print(f"Profit factor: {metrics['profit_factor']}")

    if trades_df is not None and not trades_df.empty:
        print("\nLast 5 trades:")
        print(trades_df[['entry_time', 'exit_time', 'side', 'reason', 'pnl']].tail())
    elif equity_df is not None and not equity_df.empty:
        print("\nNo trades were executed on the fetched sample.")


def run_live_mode():
    logger.info(f"Starting LIVE Trading Mode (Single-Symbol Baseline, Symbol: {settings.BYBIT_SYMBOL})")
    if getattr(settings, "LIVE_SELECTOR_ENABLED", False):
        logger.info("LIVE_SELECTOR is enabled in config, but main.py runs the single-symbol baseline path. Use web_ui.py for selector mode.")

    controller = BotController()
    controller.risk_manager = RiskManager()

    if not settings.DRY_RUN:
        if not settings.BYBIT_API_KEY or not settings.BYBIT_API_SECRET:
            logger.error("Real trading requires API keys in config.yaml")
            return
        controller.session = HTTP(
            testnet=settings.BYBIT_TESTNET,
            api_key=settings.BYBIT_API_KEY,
            api_secret=settings.BYBIT_API_SECRET
        )
        logger.info("Connected to Bybit API.")
        controller.risk_manager.sync_from_api(controller.session)
    else:
        logger.info("--- DRY RUN MODE ACTIVE: No real orders will be sent ---")

    logger.info("Bot initialized. Waiting for next check...")

    while True:
        try:
            if not controller.risk_manager.can_trade():
                logger.warning("Risk limits reached. Sleeping for 15 minutes...")
                time.sleep(900)
                continue

            df, interval_mins = fetch_signal_frame_for_lookback(
                symbol=settings.BYBIT_SYMBOL,
                interval=settings.BYBIT_INTERVAL,
                category=settings.BYBIT_CATEGORY,
            )
            if df.empty:
                logger.warning("No closed candles available for live loop. Waiting...")
                time.sleep(10)
                continue

            current_signal_row = df.iloc[-1]
            signal = int(current_signal_row['signal'])
            close_price = float(current_signal_row['close'])

            if controller.current_position:
                controller._manage_open_position(current_signal_row, interval_mins)
            elif signal != 0:
                controller._execute_trade(
                    signal,
                    close_price,
                    current_signal_row,
                    symbol=settings.BYBIT_SYMBOL,
                    strategy_name="single_symbol_strategy",
                    execution_mode="live" if not settings.DRY_RUN else "paper",
                    base_interval=settings.BYBIT_INTERVAL,
                )
            else:
                logger.info("%s: no signal on latest closed candle.", settings.BYBIT_SYMBOL)

            if controller.session:
                controller.risk_manager.sync_from_api(controller.session)

            time.sleep(10)

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
