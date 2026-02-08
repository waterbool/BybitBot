import pandas as pd
import ta

def klines_to_df(klines_list: list) -> pd.DataFrame:
    """
    Convert Bybit raw kline list to Pandas DataFrame.
    Bybit raw list: [startTime, open, high, low, close, volume, turnover]
    Data is usually returned in reverse chronological order (newest first).
    We need it in chronological order (oldest first) for TA calculations.
    """
    if not klines_list:
        return pd.DataFrame()
    
    # Define columns
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
    
    # Create DataFrame
    df = pd.DataFrame(klines_list, columns=cols)
    
    # Convert types
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = pd.to_numeric(df[col])
        
    # Sort by timestamp ascending (oldest first)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

def add_indicators(df: pd.DataFrame, 
                   ema_fast: int = 9, 
                   ema_slow: int = 21, 
                   atr_period: int = 14) -> pd.DataFrame:
    """
    Add technical indicators to the DataFrame.
    - EMA Fast
    - EMA Slow
    - ATR
    - Volume SMA (optional, using EMA fast period for now)
    """
    if df.empty:
        return df
        
    # EMA
    df[f'EMA_{ema_fast}'] = ta.trend.EMAIndicator(close=df['close'], window=ema_fast).ema_indicator()
    df[f'EMA_{ema_slow}'] = ta.trend.EMAIndicator(close=df['close'], window=ema_slow).ema_indicator()
    
    # ATR
    # ta library ATR requires high, low, close
    df['ATR'] = ta.volatility.AverageTrueRange(
        high=df['high'], 
        low=df['low'], 
        close=df['close'], 
        window=atr_period
    ).average_true_range()
    
    # Volume SMA (Average Volume)
    df['Volume_SMA'] = df['volume'].rolling(window=20).mean() # Fixed 20 or configurable? Using 20 as standard.
    
    return df
