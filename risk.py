import math
from decimal import Decimal, ROUND_FLOOR
from typing import Dict
from config import settings

def calculate_qty(price: float) -> float:
    """
    Calculates the quantity of assets to buy/sell based on a fixed USDT size.
    
    Args:
        price: Current price of the asset.
        
    Returns:
        float: Quantity to trade, rounded to appropriate precision (assuming 0.01 for ETH for now, 
               but ideally should be dynamic. For ETHUSDT usually 2 decimals or 0.01 min step).
    """
    if price <= 0:
        return 0.0
        
    raw_qty = settings.FIXED_USDT_SIZE / price
    
    # Precision handling.
    # Bybit ETHUSDT min qty is typically 0.01 or similar.
    # We will round to 3 decimal places to be safe for now, or 2.
    # Let's assume 2 decimal places for ETH (0.01).
    qty = math.floor(raw_qty * 100) / 100
    
    return qty

def price_to_precision(price: float, instrument_info: Dict) -> str:
    """
    Format price to the instrument's tick size.
    Falls back to 2 decimal places if tick size is unavailable.
    """
    tick = None
    if isinstance(instrument_info, dict):
        price_filter = instrument_info.get('priceFilter', {})
        tick = price_filter.get('tickSize') or price_filter.get('minPrice')
    if not tick:
        return f"{price:.2f}"

    tick_dec = Decimal(str(tick))
    price_dec = Decimal(str(price))
    steps = (price_dec / tick_dec).to_integral_value(rounding=ROUND_FLOOR)
    rounded = steps * tick_dec
    return format(rounded, 'f')
