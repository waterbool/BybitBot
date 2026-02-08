import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Dict
from config import settings

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class PositionState:
    is_open: bool = False
    entry_price: float = 0.0
    stop_price: float = 0.0
    position_size: float = 0.0
    symbol: str = ""

@dataclass
class SignalResult:
    action: str # BUY, SELL, HOLD
    reason: str
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    position_size: Optional[float] = None

class TrendFollowingStrategy:
    def __init__(self, risk_percent: float = 0.01):
        self.risk_percent = risk_percent
        self.positions: Dict[str, PositionState] = {}

    def get_position_state(self, symbol: str) -> PositionState:
        if symbol not in self.positions:
            self.positions[symbol] = PositionState(symbol=symbol)
        return self.positions[symbol]

    def analyze_market(self, symbol: str, df: pd.DataFrame, current_equity: float) -> SignalResult:
        """
        Analyzes the market data and returns a trading signal.
        Should be called on DAILY candles (1D).
        """
        if df.empty:
            return SignalResult("HOLD", "No data")

        # Get latest closed candle (or current developing if accepted, but strategy usually on Close)
        # Strategy says: "Entry is performed on the OPEN of the NEXT candle, 
        # based on conditions of the CURRENT CLOSED candle."
        # So we look at the last completed row (iloc[-1] if we assume df contains completed candles, 
        # or iloc[-2] if df includes a developing candle).
        # Assuming df passed here includes the MOST RECENT COMPLETED CANDLE at iloc[-1].
        
        row = df.iloc[-1]
        
        # Extract Indicators
        close = row.get('close')
        ma200 = row.get('SMA_200')
        atr_col = f'ATR_{settings.ATR_PERIOD}'
        atr = row.get(atr_col)
        if pd.isna(atr):
            atr = row.get('ATR_20')
        highest_high_7 = row.get('HighestHigh_7')
        lowest_low_7 = row.get('LowestLow_7')
        
        # Check if indicators are valid
        if pd.isna([ma200, atr, highest_high_7, lowest_low_7]).any():
             return SignalResult("HOLD", "Indicators not ready (NaN)")

        state = self.get_position_state(symbol)

        # --- UPDATE INDICATORS LOGIC ---
        # (Implicitly done by receiving fresh df with indicators calculated)

        # --- LOGIC IF NO POSITION ---
        if not state.is_open:
            # Filter Trend (Long Only): Close > MA200
            trend_up = close > ma200
            
            # Entry Condition: Close < LowestLow(7)
            # wait, Close < LowestLow(7)? 
            # LowestLow(7) is MIN(Low) of last 7 days.
            # If Close < LowestLow(7), it means we made a new 7-day low.
            # Strategy text: "Close_today < LowestLow(7) — price closed below minimum of last 7 days 
            # (false breakout down against uptrend)."
            pullback_condition = close < lowest_low_7
            
            if trend_up and pullback_condition:
                # GENERATE BUY SIGNAL
                # Calculate Risk
                entry_price = close # Actually we enter on Open next, but we estimate with Close or next Open. 
                # Strategy says "Entry executes on Open of next candle". 
                # We return signal now, bot executes next. 
                # We use 'close' as proxy for 'EntryPrice' calculation or wait for execution price.
                # However, StopPrice is defined as EntryPrice - 2*ATR.
                # Let's use Close as estimated EntryPrice for calculation.
                
                stop_price = entry_price - (2 * atr)
                risk_amount = current_equity * self.risk_percent
                stop_distance = entry_price - stop_price
                
                if stop_distance <= 0:
                     return SignalResult("HOLD", "Invalid stop distance")

                position_size = risk_amount / stop_distance
                
                # We don't update state yet, effectively. The BOT logic should call 'on_order_filled'.
                # But for this function to be pure logic, we return the proposal.
                
                reason = f"Trend UP (Close {close:.2f} > MA200 {ma200:.2f}) AND Pullback (Close < LL7 {lowest_low_7:.2f})"
                return SignalResult(
                    action="BUY",
                    reason=reason,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    position_size=position_size
                )
            
            else:
                 return SignalResult("HOLD", "No entry signal", entry_price=None)

        # --- LOGIC IF POSITION OPEN ---
        else:
            # Check Exit Conditions
            # 1. Take Profit: Close > HighestHigh(7)
            impulse_exit = close > highest_high_7
            
            # 2. Stop Loss: Price <= StopPrice
            # We check if Low of candle hit stop? Or Close? 
            # Strategy: "If Price <= stop_price, generate SELL".
            # Usually intraday check, but here on daily close check? 
            # "If Price <= stop_price" implies any price. 
            # If we only have daily bars, we check Low <= StopPrice.
            low = row.get('low')
            stop_hit = low <= state.stop_price
            
            if impulse_exit:
                return SignalResult("SELL", f"Impulse Exit: Close {close:.2f} > HH7 {highest_high_7:.2f}")
            elif stop_hit:
                 return SignalResult("SELL", f"Stop Loss Hit: Low {low:.2f} <= Stop {state.stop_price:.2f}")
            
            return SignalResult("HOLD", "Position Open, no exit")

    def confirm_entry(self, symbol: str, entry_price: float, stop_price: float, size: float):
        """Call this when the bot actually executes the BUY."""
        s = self.get_position_state(symbol)
        s.is_open = True
        s.entry_price = entry_price
        s.stop_price = stop_price
        s.position_size = size
        logger.info(f"Strategy State Updated: {symbol} LONG @ {entry_price}, Stop {stop_price}")

    def confirm_exit(self, symbol: str):
        """Call this when the bot executes SELL."""
        s = self.get_position_state(symbol)
        s.is_open = False
        s.entry_price = 0.0
        s.stop_price = 0.0
        s.position_size = 0.0
        logger.info(f"Strategy State Updated: {symbol} CLOSED")

# Global instance for compatibility if needed, or Main instantiates it.
# To keep main.py causing less errors immediately, we can expose a helper.
strategy_instance = TrendFollowingStrategy()

def apply_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy wrapper to maintain compatibility with existing main.py structure 
    if strictly needed, but the new strategy is Stateful and works per-step.
    Vectorized backtesting this specific logic is harder with state (trailing stops/fixed stops).
    
    For now, we will NOT implement full vectorized apply_strategy for this specific logic 
    unless requested for backtest. The user asked to "Add this to the strategy file".
    
    We can add a 'signal' column for ENTRY conditions at least.
    """
    # Just mark potential entries for visualization/debug
    df['signal'] = 0
    # Vectorized logic for ENTRY only (Stateful exit is hard to vectorise easily without loop)
    # Trend
    trend = df['close'] > df['SMA_200']
    # Pullback
    pullback = df['close'] < df['LowestLow_7']
    
    entries = trend & pullback
    df.loc[entries, 'signal'] = 1 
    
    return df
