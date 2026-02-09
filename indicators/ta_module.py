
import pandas as pd
import numpy as np

def add_indicators(df: pd.DataFrame, 
                   ema_fast: int = 9, 
                   ema_slow: int = 50, 
                   rsi_period: int = 14, 
                   stoch_k: int = 14, 
                   stoch_d: int = 3, 
                   atr_period: int = 14) -> pd.DataFrame:
    """
    Adds technical indicators to the DataFrame.
    
    Args:
        df: DataFrame with columns 'open', 'high', 'low', 'close', 'volume'.
        ema_fast: Period for fast EMA.
        ema_slow: Period for slow EMA.
        rsi_period: Period for RSI.
        stoch_k: Period for Stochastic %K.
        stoch_d: Period for Stochastic %D.
        atr_period: Period for ATR.
        
    Returns:
        New DataFrame with added indicator columns.
    """
    # Avoid modifying the original DataFrame inplace
    df = df.copy()
    
    # 1. EMA
    df[f'EMA_{ema_fast}'] = _calculate_ema(df['close'], window=ema_fast)
    df[f'EMA_{ema_slow}'] = _calculate_ema(df['close'], window=ema_slow)
    df['EMA_200'] = _calculate_ema(df['close'], window=200)
    
    # 2. RSI
    df[f'RSI_{rsi_period}'] = _calculate_rsi(df['close'], window=rsi_period)
    
    # 3. Stochastic Oscillator
    stoch_k_series, stoch_d_series = _calculate_stochastic(
        df['high'], df['low'], df['close'], k_window=stoch_k, d_window=stoch_d
    )
    df['Stoch_k'] = stoch_k_series
    df['Stoch_d'] = stoch_d_series
    
    # 4. ATR
    df[f'ATR_{atr_period}'] = _calculate_atr(
        df['high'], df['low'], df['close'], window=atr_period
    )
    if atr_period != 14:
        df['ATR_14'] = _calculate_atr(
            df['high'], df['low'], df['close'], window=14
        )
    else:
        df['ATR_14'] = df[f'ATR_{atr_period}']

    # 4.1 ADX (for regime detection)
    df['ADX_14'] = _calculate_adx(df['high'], df['low'], df['close'], window=14)
    
    # 5. Volume SMA
    # Use 20 as default or pass it? For now, hardcode 20 or add arg if we want to be strict.
    # The requirement is just to add it. Let's add 'Volume_SMA_20'
    df['Volume_SMA_20'] = df['volume'].rolling(window=20).mean()

    # 6. SMA (Simple Moving Average) - Strategy requires MA200
    # We allow flexible period, but default can be passed or we just add the one we need.
    # To keep it generic, let's add specific ones or generic arg.
    # The signature didn't change, so let's stick to what we have or add generic helpers.
    # We will add MA200 specifically or just generic SMA support.
    # Let's add 'SMA_200' as standard for this strategy.
    df['SMA_200'] = _calculate_sma(df['close'], window=200)

    # 7. Highest High / Lowest Low (for Donchian/Breakout logic)
    # Strategy needs HighestHigh(7) and LowestLow(7)
    df['HighestHigh_7'] = _calculate_rolling_max(df['high'], window=7)
    df['LowestLow_7'] = _calculate_rolling_min(df['low'], window=7)
    
    # Clean up initial NaN values if desired, or leave them. 
    # For this requirement, we'll leave them as they validly represent 
    # the lack of data for initial periods.
    
    return df

def _calculate_ema(series: pd.Series, window: int) -> pd.Series:
    """Calculates Exponential Moving Average."""
    return series.ewm(span=window, adjust=False).mean()

def _calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """Calculates Simple Moving Average."""
    return series.rolling(window=window).mean()

def _calculate_rolling_max(series: pd.Series, window: int) -> pd.Series:
    """Calculates Rolling Maximum (Highest High). Excludes current candle usually? 
    Standard Donchian is often N previous candles. 
    However, pandas rolling includes current row by default.
    The strategy says: "HighestHigh(7): maximum price of last 7 COMPLETED daily candles".
    When running on 'close_today', we usually look at PREVIOUS values.
    But here we are just adding the indicator column. The shifting logic (to start from previous) 
    should happen in the strategy check or we should shift here.
    Common practice: Standard indicator includes current. Strategy Logic does .shift(1) to get 'previous completed'.
    We will just compute standard rolling max here.
    """
    return series.rolling(window=window).max()

def _calculate_rolling_min(series: pd.Series, window: int) -> pd.Series:
    """Calculates Rolling Minimum (Lowest Low)."""
    return series.rolling(window=window).min()

def _calculate_rsi(series: pd.Series, window: int) -> pd.Series:
    """Calculates Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    # Avoid division by zero
    rs = gain / loss.replace(0, np.nan) 
    rsi = 100 - (100 / (1 + rs))
    
    # Create simple moving average for the first simplified calculation if needed,
    # but Wilder's smoothing is often approximated by standard EWM or SMA.
    # Here we use simple rolling mean for simplicity and speed.
    # For more accuracy closer to TA-Lib, one might use EWM for gain/loss.
    
    return rsi.fillna(0) # Handle NaN at start

def _calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                          k_window: int, d_window: int) -> tuple[pd.Series, pd.Series]:
    """Calculates Stochastic Oscillator (%K and %D)."""
    low_min = low.rolling(window=k_window).min()
    high_max = high.rolling(window=k_window).max()
    
    k_percent = 100 * ((close - low_min) / (high_max - low_min))
    d_percent = k_percent.rolling(window=d_window).mean()
    
    return k_percent, d_percent

def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Calculates Average True Range."""
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    
    return atr

def _calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculates Average Directional Index (ADX)."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr.replace(0, np.nan))

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.rolling(window=window).mean()
    return adx

if __name__ == "__main__":
    # Create valid dummy data for testing
    dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
    data = {
        'timestamp': dates,
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(200, 210, 100),
        'low': np.random.uniform(90, 100, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    }
    df_test = pd.DataFrame(data)
    
    # Ensure high >= low etc. for sanity (optional for rough test)
    df_test['high'] = df_test[['open', 'close', 'high']].max(axis=1)
    df_test['low'] = df_test[['open', 'close', 'low']].min(axis=1)

    print("Original DataFrame Head:")
    print(df_test.head())
    print("\nAdding Indicators...")
    
    df_with_indicators = add_indicators(df_test)
    
    print("\nDataFrame with Indicators Head:")
    print(df_with_indicators.head(20)) # Print more to see indicators that need history
    print("\nColumns:", df_with_indicators.columns.tolist())
