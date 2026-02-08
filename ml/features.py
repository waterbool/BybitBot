import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build feature matrix from OHLCV + indicators.
    Uses only current and past data (no future leak).
    """
    close = df['close']

    ema_fast = df.get('EMA_9', df.get('EMA_fast'))
    ema_slow = df.get('EMA_21', df.get('EMA_slow'))
    atr = df.get('ATR', df.filter(like='ATR_').iloc[:, 0] if df.filter(like='ATR_').shape[1] > 0 else None)

    if ema_fast is None:
        ema_fast = close.ewm(span=9, adjust=False).mean()
    if ema_slow is None:
        ema_slow = close.ewm(span=21, adjust=False).mean()
    if atr is None:
        # Simple ATR fallback
        high = df['high']
        low = df['low']
        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()

    # Price distance to EMAs
    dist_fast = (close - ema_fast) / close
    dist_slow = (close - ema_slow) / close

    # Log returns
    log_return_1 = np.log(close / close.shift(1))
    log_return_3 = np.log(close / close.shift(3))
    log_return_12 = np.log(close / close.shift(12))

    # Rolling std(20)
    rolling_std_20 = close.rolling(window=20).std()

    # Bollinger Bands width (20, 2)
    bb_mid = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid

    # Volume z-score(50)
    volume = df['volume']
    vol_mean = volume.rolling(window=50).mean()
    vol_std = volume.rolling(window=50).std()
    vol_z = (volume - vol_mean) / vol_std

    features = pd.DataFrame({
        'ema_fast': ema_fast,
        'ema_slow': ema_slow,
        'dist_fast': dist_fast,
        'dist_slow': dist_slow,
        'atr': atr,
        'log_return_1': log_return_1,
        'log_return_3': log_return_3,
        'log_return_12': log_return_12,
        'rolling_std_20': rolling_std_20,
        'bb_width': bb_width,
        'vol_z': vol_z,
    }, index=df.index)

    return features.to_numpy()
