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


def _apply_ml_filter(df: pd.DataFrame, signal: int) -> int:
    if signal == 0 or not getattr(settings, 'ML_ENABLED', False):
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

    df.at[last_idx, 'signal'] = _apply_ml_filter(df, base_signal)

    return df
