import os
from typing import Tuple, Optional, Dict
import pandas as pd
import logging
import numpy as np
from config import settings
from ml.features import build_features
from ml.model import load_model, predict_proba

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


def _ml_allows(df: pd.DataFrame, base_signal: Optional[str]) -> Optional[str]:
    if base_signal is None or not getattr(settings, 'ML_ENABLED', False):
        return base_signal

    model = _get_ml_model()
    if model is None:
        logger.info("ML filter: model not available -> reject signal")
        return None

    try:
        features = build_features(df)
        if features.size == 0:
            logger.info("ML filter: no features -> reject signal")
            return None
        last = features[-1]
        if np.isnan(last).any():
            logger.info("ML filter: NaN features -> reject signal")
            return None

        p_up, p_flat, p_down = predict_proba(model, last)
        logger.info(
            f"ML probs: up={p_up:.3f} flat={p_flat:.3f} down={p_down:.3f} | base_signal={base_signal}"
        )

        if getattr(settings, 'ML_FLAT_FILTER', True) and p_flat > 0.55:
            logger.info("ML filter: flat>0.55 -> reject signal")
            return None

        if base_signal == 'Buy':
            if p_up > settings.ML_MIN_PROB and p_down < 0.25:
                logger.info("ML filter: BUY allowed")
                return 'Buy'
            logger.info("ML filter: BUY rejected")
            return None
        if base_signal == 'Sell':
            if p_down > settings.ML_MIN_PROB and p_up < 0.25:
                logger.info("ML filter: SELL allowed")
                return 'Sell'
            logger.info("ML filter: SELL rejected")
            return None
    except Exception as e:
        logger.error(f"ML filter failed: {e}")
        return None

    return None


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
            return _ml_allows(df, 'Buy')
        
        # Bearish Crossover: Prev Fast >= Prev Slow AND Curr Fast < Curr Slow
        if prev_fast >= prev_slow and curr_fast < curr_slow:
            return _ml_allows(df, 'Sell')
            
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
