import os
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Dict
from config import settings
from ml.features import build_features
from ml.model import load_model, predict_proba

# Configure logging
logger = logging.getLogger(__name__)

_ML_MODEL = None
_ML_MODEL_PATH = None


def _get_ml_model():
    global _ML_MODEL, _ML_MODEL_PATH
    model_path = getattr(settings, 'ML_MODEL_PATH', os.path.join(settings.BASE_DIR, 'models', 'ethusdt_5m_lgbm.pkl'))
    if _ML_MODEL is not None and _ML_MODEL_PATH == model_path:
        return _ML_MODEL
    if not os.path.exists(model_path):
        logger.warning(f"ML model not found at {model_path}")
        return None
    try:
        _ML_MODEL = load_model(model_path)
        _ML_MODEL_PATH = model_path
        return _ML_MODEL
    except Exception as e:
        logger.error(f"Failed to load ML model: {e}")
        return None


def _apply_ml_filter(df: pd.DataFrame, signal: int, ml_enabled: Optional[bool] = None) -> int:
    effective_ml_enabled = getattr(settings, 'ML_ENABLED', False) if ml_enabled is None else bool(ml_enabled)
    if signal == 0 or not effective_ml_enabled:
        return signal

    model = _get_ml_model()
    if model is None:
        logger.info("ML filter: model not available -> reject signal")
        return 0

    try:
        features = build_features(df)
        if features.size == 0:
            logger.info("ML filter: no features -> reject signal")
            return 0
        last = features[-1]
        if np.isnan(last).any():
            logger.info("ML filter: NaN features -> reject signal")
            return 0

        p_up, p_flat, p_down = predict_proba(model, last)
        logger.info(
            f"ML probs: up={p_up:.3f} flat={p_flat:.3f} down={p_down:.3f} | base_signal={signal}"
        )

        if getattr(settings, 'ML_FLAT_FILTER', True) and p_flat > 0.55:
            logger.info("ML filter: flat>0.55 -> reject signal")
            return 0

        if signal == 1:
            if p_up > settings.ML_MIN_PROB and p_down < 0.25:
                logger.info("ML filter: BUY allowed")
                return 1
            logger.info("ML filter: BUY rejected")
            return 0
        if signal == -1:
            if p_down > settings.ML_MIN_PROB and p_up < 0.25:
                logger.info("ML filter: SELL allowed")
                return -1
            logger.info("ML filter: SELL rejected")
            return 0
    except Exception as e:
        logger.error(f"ML filter failed: {e}")
        return 0

    return 0


def _apply_market_filters(df: pd.DataFrame, signal: int) -> int:
    """
    Apply market regime filters before ML:
    - Volatility filter via ATR(14) percent
    - Higher timeframe trend filter via EMA200
    """
    if signal == 0:
        return 0

    row = df.iloc[-1]
    close = row.get('close')
    if close is None or pd.isna(close) or close == 0:
        logger.info("Market filter: invalid close -> reject signal")
        return 0

    atr = row.get('ATR_14')
    if atr is None or pd.isna(atr):
        atr = row.get(f'ATR_{settings.ATR_PERIOD}')
    if atr is None or pd.isna(atr):
        logger.info("Market filter: ATR missing -> reject signal")
        return 0

    atr_percent = float(atr) / float(close)
    if atr_percent < settings.MIN_ATR_THRESHOLD:
        logger.info(
            f"Market filter: ATR% {atr_percent:.6f} < {settings.MIN_ATR_THRESHOLD:.6f} -> reject signal"
        )
        return 0

    ema200 = row.get('EMA_200')
    if ema200 is None or pd.isna(ema200):
        logger.info("Market filter: EMA200 missing -> reject signal")
        return 0

    if signal == 1 and close <= ema200:
        logger.info("Market filter: BUY requires close > EMA200 -> reject signal")
        return 0
    if signal == -1 and close >= ema200:
        logger.info("Market filter: SELL requires close < EMA200 -> reject signal")
        return 0

    return signal


def _apply_impulse_filter(df: pd.DataFrame, signal: int) -> int:
    if signal == 0:
        return 0
    if len(df) < 2:
        return 0
    close = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2]
    if pd.isna(close) or pd.isna(prev_close) or prev_close == 0:
        logger.info("Impulse filter: invalid close -> reject signal")
        return 0
    ret1 = abs((float(close) / float(prev_close)) - 1.0)
    if ret1 <= settings.IMPULSE_THRESHOLD:
        logger.info(f"Impulse filter: |return(1)| {ret1:.6f} <= {settings.IMPULSE_THRESHOLD:.6f} -> reject signal")
        return 0
    return signal


def _apply_cooldown_filter(df: pd.DataFrame, signal: int) -> int:
    if signal == 0 or settings.COOLDOWN_CANDLES <= 0:
        return signal
    if 'signal' not in df.columns or len(df) < 2:
        return signal

    prev_signals = df['signal'].iloc[:-1]
    non_zero = prev_signals[prev_signals != 0]
    if non_zero.empty:
        return signal

    last_idx = non_zero.index[-1]
    try:
        last_pos = df.index.get_loc(last_idx)
    except Exception:
        return signal

    candles_since = (len(df) - 1) - int(last_pos)
    last_side = int(non_zero.loc[last_idx])

    if signal == 1 and last_side == -1 and candles_since <= settings.COOLDOWN_CANDLES:
        logger.info(f"Cooldown filter: {candles_since} candles since SELL <= {settings.COOLDOWN_CANDLES} -> reject signal")
        return 0
    if signal == -1 and last_side == 1 and candles_since <= settings.COOLDOWN_CANDLES:
        logger.info(f"Cooldown filter: {candles_since} candles since BUY <= {settings.COOLDOWN_CANDLES} -> reject signal")
        return 0

    return signal

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
        highest_high_7 = df['high'].shift(1).rolling(window=7).max().iloc[-1]
        lowest_low_7 = df['low'].shift(1).rolling(window=7).min().iloc[-1]
        
        # Check if indicators are valid
        if pd.isna([ma200, atr, highest_high_7, lowest_low_7]).any():
             return SignalResult("HOLD", "Indicators not ready (NaN)")

        state = self.get_position_state(symbol)

        # --- MARKET REGIME FILTERS (pre-entry) ---
        atr14 = row.get('ATR_14')
        if atr14 is None or pd.isna(atr14):
            atr14 = atr
        if atr14 is None or pd.isna(atr14) or close == 0:
            return SignalResult("HOLD", "ATR missing/invalid")
        atr_percent = float(atr14) / float(close)
        if atr_percent < settings.MIN_ATR_THRESHOLD:
            return SignalResult("HOLD", f"Low volatility (ATR% {atr_percent:.6f})")

        ema200 = row.get('EMA_200')
        if ema200 is None or pd.isna(ema200):
            return SignalResult("HOLD", "EMA200 missing")

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
            
            if trend_up and close > ema200 and pullback_condition:
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

def apply_strategy(df: pd.DataFrame, ml_enabled: Optional[bool] = None) -> pd.DataFrame:
    """
    EMA crossover strategy signal generator.
    Uses ML probabilities (if present) as a filter on the latest candle.
    """
    df['signal'] = 0

    ema_fast_col = f'EMA_{settings.EMA_FAST}'
    ema_slow_col = f'EMA_{settings.EMA_SLOW}'

    if ema_fast_col not in df.columns or ema_slow_col not in df.columns:
        return df

    fast = df[ema_fast_col]
    slow = df[ema_slow_col]
    prev_fast = fast.shift(1)
    prev_slow = slow.shift(1)

    bull = (prev_fast <= prev_slow) & (fast > slow)
    bear = (prev_fast >= prev_slow) & (fast < slow)

    df.loc[bull, 'signal'] = 1
    df.loc[bear, 'signal'] = -1

    # ML filter (latest candle only)
    last_idx = df.index[-1]
    base_signal = int(df.at[last_idx, 'signal'])
    if base_signal == 1:
        logger.info("Strategy signal: BUY (EMA crossover)")
    elif base_signal == -1:
        logger.info("Strategy signal: SELL (EMA crossover)")
    else:
        logger.info("Strategy signal: NONE")

    impulse_signal = _apply_impulse_filter(df, base_signal)
    cooldown_signal = _apply_cooldown_filter(df, impulse_signal)
    market_signal = _apply_market_filters(df, cooldown_signal)
    df.at[last_idx, 'signal'] = _apply_ml_filter(df, market_signal, ml_enabled=ml_enabled)

    return df


def apply_mean_reversion_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean reversion strategy signal generator.
    LONG:
      - close < EMA(50)
      - RSI(14) < 30
      - close < lower Bollinger(20, 2)
      - ATR(14)/close > 0.0015
    SHORT: зеркально
    """
    df['signal'] = 0

    def _ensure_rsi_14(series: pd.Series) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(0)

    def _ensure_atr_14(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=14).mean()

    ema_period = int(getattr(settings, "MEAN_REV_EMA_PERIOD", 50))
    rsi_period = int(getattr(settings, "MEAN_REV_RSI_PERIOD", 14))
    bb_period = int(getattr(settings, "MEAN_REV_BB_PERIOD", 20))
    bb_std = float(getattr(settings, "MEAN_REV_BB_STD", 2.0))
    atr_period = int(getattr(settings, "MEAN_REV_ATR_PERIOD", 14))
    min_atr_pct = float(getattr(settings, "MEAN_REV_MIN_ATR_PCT", 0.0015))
    rsi_long = float(getattr(settings, "MEAN_REV_RSI_LONG", 30.0))
    rsi_short = float(getattr(settings, "MEAN_REV_RSI_SHORT", 70.0))

    ema_col = f'EMA_{ema_period}'
    rsi_col = f'RSI_{rsi_period}'
    bb_lower_col = f'BB_LOWER_{bb_period}'
    bb_upper_col = f'BB_UPPER_{bb_period}'
    atr_col = f'ATR_{atr_period}'

    if ema_col not in df.columns:
        df[ema_col] = df['close'].ewm(span=ema_period, adjust=False).mean()
    if rsi_col not in df.columns:
        if rsi_period == 14:
            df[rsi_col] = _ensure_rsi_14(df['close'])
        else:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss.replace(0, np.nan)
            df[rsi_col] = (100 - (100 / (1 + rs))).fillna(0)
    if bb_lower_col not in df.columns or bb_upper_col not in df.columns:
        bb_mid = df['close'].rolling(window=bb_period).mean()
        bb_std_dev = df['close'].rolling(window=bb_period).std()
        df[f'BB_MID_{bb_period}'] = bb_mid
        df[bb_upper_col] = bb_mid + (bb_std * bb_std_dev)
        df[bb_lower_col] = bb_mid - (bb_std * bb_std_dev)
    if atr_col not in df.columns:
        if atr_period == 14:
            df[atr_col] = _ensure_atr_14(df['high'], df['low'], df['close'])
        else:
            prev_close = df['close'].shift(1)
            tr1 = df['high'] - df['low']
            tr2 = (df['high'] - prev_close).abs()
            tr3 = (df['low'] - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df[atr_col] = tr.rolling(window=atr_period).mean()

    close = df['close']
    ema_series = df[ema_col]
    rsi_series = df[rsi_col]
    bb_lower = df[bb_lower_col]
    bb_upper = df[bb_upper_col]
    atr_series = df[atr_col]

    atr_percent = atr_series / close.replace(0, np.nan)

    long_cond = (
        (close < ema_series) &
        (rsi_series < rsi_long) &
        (close < bb_lower) &
        (atr_percent > min_atr_pct)
    )
    short_cond = (
        (close > ema_series) &
        (rsi_series > rsi_short) &
        (close > bb_upper) &
        (atr_percent > min_atr_pct)
    )

    df.loc[long_cond, 'signal'] = 1
    df.loc[short_cond, 'signal'] = -1

    last_idx = df.index[-1]
    base_signal = int(df.at[last_idx, 'signal'])
    if base_signal == 1:
        logger.info("Mean reversion signal: BUY")
    elif base_signal == -1:
        logger.info("Mean reversion signal: SELL")
    else:
        logger.info("Mean reversion signal: NONE")

    return df


def apply_volatility_compression_breakout_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volatility compression breakout strategy signal generator.
    LONG:
      - ATR(14) < atr_mean_50
      - bb_width in lowest 10% over last 50 candles
      - close > EMA(50)
      - close > previous_close
    SHORT: зеркально
    """
    df['signal'] = 0

    ema_period = int(getattr(settings, "VOL_COMP_EMA_PERIOD", 50))
    atr_period = int(getattr(settings, "VOL_COMP_ATR_PERIOD", 14))
    atr_ma_period = int(getattr(settings, "VOL_COMP_ATR_MA_PERIOD", 50))
    bb_period = int(getattr(settings, "VOL_COMP_BB_PERIOD", 20))
    bb_std = float(getattr(settings, "VOL_COMP_BB_STD", 2.0))
    bb_width_mult = float(getattr(settings, "VOL_COMP_BB_WIDTH_MULT", 1.10))

    ema_col = f'EMA_{ema_period}'
    atr_col = f'ATR_{atr_period}'
    bb_upper_col = f'BB_UPPER_{bb_period}'
    bb_lower_col = f'BB_LOWER_{bb_period}'

    if ema_col not in df.columns:
        df[ema_col] = df['close'].ewm(span=ema_period, adjust=False).mean()
    if atr_col not in df.columns:
        prev_close = df['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df[atr_col] = tr.rolling(window=atr_period).mean()
    if bb_lower_col not in df.columns or bb_upper_col not in df.columns:
        bb_mid = df['close'].rolling(window=bb_period).mean()
        bb_std_dev = df['close'].rolling(window=bb_period).std()
        df[f'BB_MID_{bb_period}'] = bb_mid
        df[bb_upper_col] = bb_mid + (bb_std * bb_std_dev)
        df[bb_lower_col] = bb_mid - (bb_std * bb_std_dev)

    close = df['close']
    ema_series = df[ema_col]
    atr_series = df[atr_col]
    bb_upper = df[bb_upper_col]
    bb_lower = df[bb_lower_col]

    atr_mean = atr_series.rolling(window=atr_ma_period).mean()
    bb_width = (bb_upper - bb_lower) / close.replace(0, np.nan)
    min_bb_width = bb_width.rolling(window=atr_ma_period).min()
    bb_width_threshold = min_bb_width * bb_width_mult

    prev_close = close.shift(1)

    long_cond = (
        (atr_series < atr_mean) &
        (bb_width <= bb_width_threshold) &
        (close > ema_series) &
        (close > prev_close)
    )
    short_cond = (
        (atr_series < atr_mean) &
        (bb_width <= bb_width_threshold) &
        (close < ema_series) &
        (close < prev_close)
    )

    df.loc[long_cond, 'signal'] = 1
    df.loc[short_cond, 'signal'] = -1

    last_idx = df.index[-1]
    base_signal = int(df.at[last_idx, 'signal'])
    if base_signal == 1:
        logger.info("Volatility compression signal: BUY")
    elif base_signal == -1:
        logger.info("Volatility compression signal: SELL")
    else:
        logger.info("Volatility compression signal: NONE")

    return df


def apply_mtf_trend_pullback_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-timeframe trend pullback strategy signal generator.
    Expects a 'bias' column in df:
      1 = LONG bias (1H close > EMA200)
     -1 = SHORT bias (1H close < EMA200)
      0 = no bias / insufficient data

    15m LONG:
      - bias = LONG
      - close touches or below EMA(50)
      - RSI(14) < 45
      - current close > previous close
    SHORT: зеркально
    """
    df['signal'] = 0

    if 'bias' not in df.columns:
        logger.info("MTF pullback: bias missing -> no signals")
        return df

    ema_period = int(getattr(settings, "MTF_PULLBACK_EMA_PERIOD", 50))
    rsi_period = int(getattr(settings, "MTF_PULLBACK_RSI_PERIOD", 14))
    rsi_long = float(getattr(settings, "MTF_PULLBACK_RSI_LONG", 45.0))
    rsi_short = float(getattr(settings, "MTF_PULLBACK_RSI_SHORT", 55.0))

    ema_col = f'EMA_{ema_period}'
    rsi_col = f'RSI_{rsi_period}'

    if ema_col not in df.columns:
        df[ema_col] = df['close'].ewm(span=ema_period, adjust=False).mean()
    if rsi_col not in df.columns:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        df[rsi_col] = rsi.fillna(0)

    close = df['close']
    ema_series = df[ema_col]
    rsi_series = df[rsi_col]
    prev_close = close.shift(1)
    bias = df['bias']

    long_cond = (
        (bias == 1) &
        (close <= ema_series) &
        (rsi_series < rsi_long) &
        (close > prev_close)
    )
    short_cond = (
        (bias == -1) &
        (close >= ema_series) &
        (rsi_series > rsi_short) &
        (close < prev_close)
    )

    df.loc[long_cond, 'signal'] = 1
    df.loc[short_cond, 'signal'] = -1

    last_idx = df.index[-1]
    base_signal = int(df.at[last_idx, 'signal'])
    if base_signal == 1:
        logger.info("MTF pullback signal: BUY")
    elif base_signal == -1:
        logger.info("MTF pullback signal: SELL")
    else:
        logger.info("MTF pullback signal: NONE")

    return df


def apply_funding_extreme_reversal_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Funding extreme reversal strategy signal generator.
    LONG:
      - funding_rate < -0.00007
      - open_interest current >= previous
      - close > previous_close
    SHORT:
      - funding_rate > 0.00007
      - open_interest current >= previous
      - close < previous_close
    """
    df['signal'] = 0

    if 'funding_rate' not in df.columns or 'open_interest' not in df.columns:
        logger.info("Funding extreme: missing funding_rate or open_interest -> no signals")
        return df

    close = df['close']
    prev_close = close.shift(1)
    oi = df['open_interest']
    prev_oi = oi.shift(1)
    fr = df['funding_rate']
    long_threshold = float(getattr(settings, "FUNDING_EXTREME_LONG_THRESHOLD", -0.00007))
    short_threshold = float(getattr(settings, "FUNDING_EXTREME_SHORT_THRESHOLD", 0.00007))
    oi_min_change = float(getattr(settings, "FUNDING_EXTREME_MIN_OI_CHANGE", 0.0))
    price_move_min = float(getattr(settings, "FUNDING_EXTREME_MIN_PRICE_MOVE", 0.0))
    oi_change = (oi - prev_oi) / prev_oi.replace(0, np.nan)
    price_change = (close / prev_close.replace(0, np.nan)) - 1.0

    long_cond = (
        (fr < long_threshold) &
        (oi_change >= oi_min_change) &
        (price_change >= price_move_min)
    )
    short_cond = (
        (fr > short_threshold) &
        (oi_change >= oi_min_change) &
        (price_change <= -price_move_min)
    )

    df.loc[long_cond, 'signal'] = 1
    df.loc[short_cond, 'signal'] = -1

    last_idx = df.index[-1]
    base_signal = int(df.at[last_idx, 'signal'])
    if base_signal == 1:
        logger.info("Funding extreme signal: BUY")
    elif base_signal == -1:
        logger.info("Funding extreme signal: SELL")
    else:
        logger.info("Funding extreme signal: NONE")

    return df
