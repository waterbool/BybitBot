import pandas as pd
from typing import Dict, Union, Optional
from config import settings
import logging

logger = logging.getLogger(__name__)

def analyze_market(df: pd.DataFrame) -> Dict[str, Union[str, float, None]]:
    """
    Analyzes the market data and generates a trading signal.
    
    Args:
        df: DataFrame with indicators.
        
    Returns:
        dict: {
            "signal": "long" | "short" | "flat",
            "sl": float | None,
            "tp": float | None,
            "reason": str
        }
    """
    signal_result = {
        "signal": "flat",
        "sl": None,
        "tp": None,
        "reason": ""
    }
    
    if df.empty or len(df) < settings.VOLUME_LOOKBACK + 1:
        return signal_result
        
    # Get latest completed candle data (assuming the strategy runs on closed candles)
    # If using current developing candle, be careful of repainting. 
    # Usually "breakout" strategies check if the *last closed* candle broke out or current price breaks.
    # The prompt says: "Trade 1-minute candles ... Signal on breakout of local high/low on background of elevated volume and aligned trend."
    # Let's assess the latest CLOSED candle for trend and volume, and potential breakout.
    # OR, assess current price vs levels.
    # Standard practice: Analysis on Closed Candle.
    
    current = df.iloc[-1]
    prev = df.iloc[-2] # Previous closed candle
    
    # 0. Time Filter (New)
    # Check if we are inside trading hours
    timestamp = current.get('timestamp')
    if timestamp:
        try:
            current_hour = pd.to_datetime(timestamp, unit='ms').hour
            if not (settings.TRADING_START_HOUR <= current_hour < settings.TRADING_END_HOUR):
                signal_result["reason"] = f"Outside Trading Hours ({current_hour}:00)"
                return signal_result
        except Exception as e:
            logger.warning(f"Time filter failed: {e}")

    # 1. Trend Filter
    ema_fast = current[f'EMA_{settings.EMA_FAST}']
    ema_slow = current[f'EMA_{settings.EMA_SLOW}']
    close = current['close']
    
    is_uptrend = (close > ema_fast) and (close > ema_slow) and (ema_fast > ema_slow)
    is_downtrend = (close < ema_fast) and (close < ema_slow) and (ema_fast < ema_slow)
    
    if not is_uptrend and not is_downtrend:
        signal_result["reason"] = "No Trend"
        return signal_result

    # 2. Volume Filter
    # Calc average volume manually if not in DF, or assume caller added it.
    # Let's compure it here on the fly for safety using pandas rolling.
    # Optimization: Caller should ideally add this. But we can do it quickly here.
    # Note: df['volume'] is required.
    
    # We need Avg Volume of PREVIOUS N candles to compare with CURRENT volume
    # "Current volume > multiplier * average volume of last N"
    # If we are looking at the *last closed* candle as the signal candle:
    avg_vol = df['volume'].rolling(window=settings.VOLUME_LOOKBACK).mean().iloc[-2] # Avg of N candles BEFORE the current one? Or including?
    # Usually "Average volume of last N".
    # Let's take rolling mean of last N at current index.
    avg_vol_current = df['volume'].rolling(window=settings.VOLUME_LOOKBACK).mean().iloc[-1]
    
    # But strictly: "Volume of current candle > X * Avg Volume".
    # If current is "Latest Closed", then yes.
    current_vol = current['volume']
    
    # Avoid lookahead bias/self-inclusion if strictly "past N". 
    # But usually simple rolling mean is fine.
    # Let's use the rolling mean value at the current bar.
    
    is_volume_high = current_vol > (settings.VOLUME_MULTIPLIER * avg_vol_current)
    
    if not is_volume_high:
        signal_result["reason"] = "Low Volume"
        return signal_result
        
    # 3. Levels (Breakout)
    # Find local High/Low over last N candles (excluding current signal candle)
    # We look at `settings.LEVELS_LOOKBACK` candles before the current one.
    # Note: LEVELS_LOOKBACK might be missing in updated settings.py, need to ensure compatibility.
    levels_lookback = getattr(settings, 'LEVELS_LOOKBACK', 10)
    
    start_idx = -1 - levels_lookback
    if abs(start_idx) > len(df):
        start_idx = 0
    
    # Slice for previous candles
    prev_candles = df.iloc[start_idx:-1]
    
    local_high = prev_candles['high'].max()
    local_low = prev_candles['low'].min()
    
    # Signal Logic
    breakout_long = (current['close'] > local_high)
    breakout_short = (current['close'] < local_low)
    
    # 4. Volatility (ATR)
    atr = current.get(f'ATR_{settings.ATR_PERIOD}', 0)
    if atr == 0:
        signal_result["reason"] = "ATR Missing/Zero"
        return signal_result
        
    # Final Decision
    if is_uptrend and breakout_long:
        signal_result["signal"] = "long"
        signal_result["reason"] = f"Breakout High {local_high:.2f} + Vol Spike + Uptrend"
        
        # Dynamic SL/TP
        # SL = Price - mult * ATR
        sl_dist = settings.SL_ATR_MULT * atr
        signal_result["sl"] = close - sl_dist
        
        # TP = Price + (mult * ATR * Ratio) OR just TP_ATR_MULT (if we had it, but user used Ratio now)
        # Ratio is 1:1.5 usually.
        # User config has RISK_REWARD_RATIO.
        # TP Distance = SL Distance * Ratio
        tp_dist = sl_dist * settings.RISK_REWARD_RATIO
        signal_result["tp"] = close + tp_dist
        
    elif is_downtrend and breakout_short:
        signal_result["signal"] = "short"
        signal_result["reason"] = f"Breakout Low {local_low:.2f} + Vol Spike + Downtrend"
        
        # Dynamic SL/TP for Short
        # SL = Price + mult * ATR
        sl_dist = settings.SL_ATR_MULT * atr
        signal_result["sl"] = close + sl_dist
        
        # TP = Price - (SL Dist * Ratio)
        tp_dist = sl_dist * settings.RISK_REWARD_RATIO
        signal_result["tp"] = close - tp_dist
        
    else:
        signal_result["reason"] = f"Trend/Vol/Time ok, No Breakout (H:{local_high:.2f}/L:{local_low:.2f} vs C:{close:.2f})"
 
    return signal_result
