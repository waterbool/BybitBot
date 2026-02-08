"""
Bot Controller - Manages bot lifecycle and state for web UI
"""
import threading
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
from queue import Queue
import pandas as pd

from config import settings
from data_fetch.bybit_client import fetch_historical_klines
from indicators.ta_module import add_indicators
from strategy.rules import apply_strategy
from risk.risk import RiskManager
from pybit.unified_trading import HTTP

logger = logging.getLogger(__name__)

def _interval_to_minutes(interval_value: str) -> int:
    """Convert Bybit interval string to minutes (e.g. '1', '60', 'D', '1D')."""
    s = str(interval_value).strip().upper()
    if s.isdigit():
        return int(s)
    if s.endswith('D'):
        days = s[:-1]
        if days == "":
            return 1440
        if days.isdigit():
            return int(days) * 1440
    raise ValueError(f"Unsupported interval format: {interval_value}")


class BotController:
    """Controls bot execution and provides status for web UI"""
    
    def __init__(self):
        self.status = "idle"  # idle, running, stopped, error
        self.thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        self.risk_manager: Optional[RiskManager] = None
        self.session: Optional[HTTP] = None
        
        # Statistics
        self.trades_history: List[Dict] = []
        self.current_position: Optional[Dict] = None
        self.stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "daily_pnl": 0.0,
            "daily_trades": 0
        }
        
        # Log queue for web UI
        self.log_queue = Queue(maxsize=1000)
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging handler to capture logs for web UI"""
        class QueueHandler(logging.Handler):
            def __init__(self, log_queue):
                super().__init__()
                self.log_queue = log_queue
                
            def emit(self, record):
                try:
                    msg = self.format(record)
                    if not self.log_queue.full():
                        self.log_queue.put({
                            'timestamp': datetime.now().isoformat(),
                            'level': record.levelname,
                            'message': msg
                        })
                except Exception:
                    pass
        
        handler = QueueHandler(self.log_queue)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(handler)
    
    def start_trading(self) -> Dict:
        """Start live trading in a separate thread"""
        if self.status == "running":
            return {"success": False, "message": "Bot is already running"}
        
        try:
            self.stop_flag.clear()
            self.status = "running"
            self.thread = threading.Thread(target=self._trading_loop, daemon=True)
            self.thread.start()
            
            logger.info("Bot started successfully")
            return {"success": True, "message": "Bot started successfully"}
        except Exception as e:
            self.status = "error"
            logger.error(f"Failed to start bot: {e}")
            return {"success": False, "message": f"Failed to start: {str(e)}"}
    
    def stop_trading(self) -> Dict:
        """Stop live trading"""
        if self.status != "running":
            return {"success": False, "message": "Bot is not running"}
        
        try:
            self.stop_flag.set()
            self.status = "stopped"
            logger.info("Bot stop requested")
            return {"success": True, "message": "Bot stopped"}
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
            return {"success": False, "message": f"Error stopping: {str(e)}"}
    
    def get_status(self) -> Dict:
        """Get current bot status and statistics"""
        return {
            "status": self.status,
            "stats": self.stats,
            "current_position": self.current_position,
            "config": {
                "symbol": settings.BYBIT_SYMBOL,
                "dry_run": settings.DRY_RUN,
                "testnet": settings.BYBIT_TESTNET,
                "timeframe": settings.BYBIT_INTERVAL
            }
        }
    
    def get_recent_logs(self, count: int = 100) -> List[Dict]:
        """Get recent logs from queue"""
        logs = []
        temp_logs = []
        
        # Drain queue
        while not self.log_queue.empty() and len(temp_logs) < count:
            temp_logs.append(self.log_queue.get())
        
        # Put back and return
        for log in temp_logs:
            if not self.log_queue.full():
                self.log_queue.put(log)
        
        return temp_logs[-count:]
    
    def _trading_loop(self):
        """Main trading loop (runs in separate thread)"""
        try:
            # Initialize
            self.risk_manager = RiskManager()
            
            if not settings.DRY_RUN:
                if not settings.BYBIT_API_KEY or not settings.BYBIT_API_SECRET:
                    logger.error("Real trading requires API keys")
                    self.status = "error"
                    return
                
                self.session = HTTP(
                    testnet=settings.BYBIT_TESTNET,
                    api_key=settings.BYBIT_API_KEY,
                    api_secret=settings.BYBIT_API_SECRET
                )
                logger.info("Connected to Bybit API")
                self.risk_manager.sync_from_api(self.session)
            else:
                logger.info("DRY RUN MODE: No real orders will be placed")
            
            # Trading loop
            while not self.stop_flag.is_set():
                try:
                    # Risk check
                    if not self.risk_manager.can_trade():
                        logger.warning("Risk limits reached. Waiting...")
                        time.sleep(60)
                        continue
                    
                    # Fetch data
                    now = int(time.time() * 1000)
                    try:
                        interval_mins = _interval_to_minutes(settings.BYBIT_INTERVAL)
                    except ValueError as e:
                        logger.error(str(e))
                        time.sleep(60)
                        continue
                    lookback_mins = 200 * interval_mins
                    start_ts = now - (lookback_mins * 60 * 1000)
                    
                    df = fetch_historical_klines(settings.BYBIT_SYMBOL, settings.BYBIT_INTERVAL, start_ts, now)
                    
                    if len(df) < 50:
                        logger.warning("Not enough data. Waiting...")
                        time.sleep(10)
                        continue
                    
                    # Add indicators and apply strategy
                    df = add_indicators(
                        df,
                        ema_fast=settings.EMA_FAST,
                        ema_slow=settings.EMA_SLOW,
                        atr_period=settings.ATR_PERIOD
                    )
                    df = apply_strategy(df)
                    
                    # Check signal
                    current_signal_row = df.iloc[-1]
                    signal = current_signal_row['signal']
                    close_price = current_signal_row['close']
                    
                    if signal != 0:
                        self._execute_trade(signal, close_price, current_signal_row)
                    
                    # Sleep
                    time.sleep(10)
                    
                    # Sync stats
                    if not settings.DRY_RUN and self.session:
                        self.risk_manager.sync_from_api(self.session)
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    time.sleep(5)
            
            logger.info("Trading loop stopped")
            
        except Exception as e:
            logger.error(f"Fatal error in trading loop: {e}")
            self.status = "error"
    
    def _execute_trade(self, signal: int, close_price: float, signal_row: pd.Series):
        """Execute a trade based on signal"""
        side = "Buy" if signal == 1 else "Sell"
        qty = round(settings.FIXED_USDT_SIZE / close_price, 3)
        
        # Calculate SL/TP
        atr_col = f'ATR_{settings.ATR_PERIOD}'
        atr = signal_row.get(atr_col)
        if atr is None or pd.isna(atr):
            atr = 10
        sl_dist = atr * settings.SL_ATR_MULT
        tp_dist = atr * settings.RISK_REWARD_RATIO
        
        if side == "Buy":
            sl_price = round(close_price - sl_dist, 2)
            tp_price = round(close_price + tp_dist, 2)
        else:
            sl_price = round(close_price + sl_dist, 2)
            tp_price = round(close_price - tp_dist, 2)
        
        logger.info(f"SIGNAL: {side} @ {close_price} | SL: {sl_price} | TP: {tp_price}")
        
        if settings.DRY_RUN:
            logger.info(f"[DRY RUN] Order: Side={side}, Qty={qty}, Entry={close_price}, SL={sl_price}, TP={tp_price}")
            
            # Record trade
            self.trades_history.append({
                'timestamp': datetime.now().isoformat(),
                'side': side,
                'entry_price': close_price,
                'qty': qty,
                'sl': sl_price,
                'tp': tp_price,
                'status': 'simulated'
            })
            self.stats['total_trades'] += 1
            self.stats['daily_trades'] += 1
        else:
            # Execute real order
            try:
                logger.info(f"Placing REAL {side} Order...")
                resp = self.session.place_order(
                    category="linear",
                    symbol=settings.BYBIT_SYMBOL,
                    side=side,
                    orderType="Market",
                    qty=str(qty),
                    stopLoss=str(sl_price),
                    takeProfit=str(tp_price),
                    timeInForce="GTC"
                )
                logger.info(f"Order Placed: {resp}")
                
                # Record trade
                self.trades_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'side': side,
                    'entry_price': close_price,
                    'qty': qty,
                    'sl': sl_price,
                    'tp': tp_price,
                    'status': 'executed',
                    'order_id': resp.get('result', {}).get('orderId')
                })
                self.stats['total_trades'] += 1
                self.stats['daily_trades'] += 1
                
                time.sleep(60)  # Avoid double entry
                
            except Exception as e:
                logger.error(f"Order execution failed: {e}")
