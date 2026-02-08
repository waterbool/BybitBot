from typing import Tuple, Optional, Dict
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class Strategy:
    def __init__(self, ema_fast: int, ema_slow: int, sl_atr_multiplier: float, risk_reward_ratio: float):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.sl_atr_multiplier = sl_atr_multiplier
        self.risk_reward_ratio = risk_reward_ratio

    def check_signal(self, df: pd.DataFrame) -> Optional[str]:
        """
        Analyze the DataFrame to generate buy/sell signals.
        Strategy: EMA Crossover.
        Buy if EMA_fast crosses above EMA_slow.
        Sell if EMA_fast crosses below EMA_slow.
        We check the last completed candle (index -2) and the one before (index -3) to detect crossover.
        (Index -1 is the current forming candle).
        """
        if len(df) < 5:
            return None

        # Get relevant columns
        try:
            curr_fast = df[f'EMA_{self.ema_fast}'].iloc[-1]
            curr_slow = df[f'EMA_{self.ema_slow}'].iloc[-1]
            prev_fast = df[f'EMA_{self.ema_fast}'].iloc[-2]
            prev_slow = df[f'EMA_{self.ema_slow}'].iloc[-2]
        except KeyError as e:
            logger.error(f"Missing indicator columns in DataFrame: {e}")
            return None

        # Check for Crossover
        # Bullish Crossover: Prev Fast <= Prev Slow AND Curr Fast > Curr Slow
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            return 'Buy'
        
        # Bearish Crossover: Prev Fast >= Prev Slow AND Curr Fast < Curr Slow
        if prev_fast >= prev_slow and curr_fast < curr_slow:
            return 'Sell'
            
        return None

    def calculate_tp_sl(self, entry_price: float, side: str, atr: float, instrument_info: Dict) -> Tuple[Optional[str], Optional[str]]:
        """
        Calculate dynamic Take Profit and Stop Loss based on ATR.
        """
        # Get formatting precision
        from risk import price_to_precision
        
        sl_dist = atr * self.sl_atr_multiplier
        tp_dist = sl_dist * self.risk_reward_ratio
        
        if side == 'Buy':
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        elif side == 'Sell':
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist
        else:
            return None, None
            
        # Format prices
        # Ensure SL/TP are valid (positive)
        if sl_price <= 0 or tp_price <= 0:
            return None, None
            
        sl_str = price_to_precision(sl_price, instrument_info)
        tp_str = price_to_precision(tp_price, instrument_info)
        
        return tp_str, sl_str
