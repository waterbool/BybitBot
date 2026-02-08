import logging
from datetime import datetime
from pybit.unified_trading import HTTP
from config import settings

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self):
        self.max_trades = settings.MAX_TRADES_PER_DAY
        self.max_loss = settings.MAX_DAILY_LOSS_USDT
        
        # State
        self.current_date = datetime.utcnow().date()
        self.daily_trades = 0
        self.daily_pnl = 0.0
        
    def _check_reset_daily(self):
        """Reset stats if the day has changed (UTC)."""
        now_date = datetime.utcnow().date()
        if now_date > self.current_date:
            logger.info(f"New day detected ({now_date}). Resetting risk stats.")
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.current_date = now_date

    def can_trade(self) -> bool:
        """Check if trading is allowed based on risk limits."""
        self._check_reset_daily()
        
        if self.daily_trades >= self.max_trades:
            logger.warning(f"Risk Limit Reached: Max trades per day ({self.max_trades}) hit.")
            return False
            
        if self.daily_pnl <= -self.max_loss:
            logger.warning(f"Risk Limit Reached: Max daily loss ({self.max_loss}) hit. Current PnL: {self.daily_pnl}")
            return False
            
        return True

    def update_stats(self, pnl: float):
        """Update daily stats with a closed trade PnL (Manual/Simulated)."""
        self._check_reset_daily()
        self.daily_trades += 1
        self.daily_pnl += pnl
        logger.info(f"Risk Update: Daily Trades: {self.daily_trades}/{self.max_trades}, Daily PnL: {self.daily_pnl:.2f}/{self.max_loss}")

    def sync_from_api(self, session: HTTP):
        """
        Sync daily stats from Bybit API (Real Execution).
        Fetches closed PnL for the current day.
        """
        if settings.DRY_RUN:
            return

        self._check_reset_daily()
        try:
            # Get start of the day in ms
            start_of_day = datetime.combine(datetime.utcnow().date(), datetime.min.time())
            start_ts = int(start_of_day.timestamp() * 1000)
            
            # Fetch Closed PnL
            response = session.get_closed_pnl(category="linear", startTime=start_ts, limit=100)
            if response['retCode'] == 0:
                deals = response['result']['list']
                
                # Calculate total PnL and trades for the day
                # Note: 'list' contains individual closed positions.
                total_pnl = sum(float(d['closedPnl']) for d in deals)
                total_trades = len(deals)
                
                self.daily_pnl = total_pnl
                self.daily_trades = total_trades
                
                logger.info(f"Synced Risk Stats from API: Trades={self.daily_trades}, PnL={self.daily_pnl:.2f}")
            else:
                logger.error(f"Failed to sync risk stats: {response['retMsg']}")
                
        except Exception as e:
            logger.error(f"Error syncing risk stats from API: {e}")


    def get_stats(self):
        return {
            "date": self.current_date,
            "trades": self.daily_trades,
            "pnl": self.daily_pnl
        }
