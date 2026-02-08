import pandas as pd
import numpy as np
from strategy.strategy import analyze_market
from config import settings

def test_strategy_logic():
    print("Testing Strategy Logic...")
    
    # 1. Create Mock Data
    dates = pd.date_range(start='2024-01-01', periods=50, freq='1min')
    data = {
        'open': np.full(50, 100.0),
        'high': np.full(50, 105.0),
        'low': np.full(50, 95.0),
        'close': np.full(50, 100.0),
        'volume': np.full(50, 1000.0),
        f'EMA_{settings.EMA_FAST}': np.full(50, 100.0),
        f'EMA_{settings.EMA_SLOW}': np.full(50, 100.0),
        f'ATR_{settings.ATR_PERIOD}': np.full(50, 2.0),
        'ATR_14': np.full(50, 2.0),
        'EMA_200': np.full(50, 100.0)
    }
    df = pd.DataFrame(data, index=dates)
    
    # Test 1: Flat market (No signal)
    print("\nTest 1: Flat market")
    res = analyze_market(df)
    print(f"Result: {res['signal']} (Reason: {res['reason']})")
    assert res['signal'] == 'flat'
    
    # Test 2: Long Signal Setup
    # - Uptrend: Close > Fast > Slow
    # - Volume: High
    # - Breakout: Close > Local High
    
    # Modify last few rows
    # Uptrend
    df.loc[df.index[-1], 'close'] = 110.0
    df.loc[df.index[-1], f'EMA_{settings.EMA_FAST}'] = 105.0
    df.loc[df.index[-1], f'EMA_{settings.EMA_SLOW}'] = 102.0
    
    # Volume Spike
    # Avg volume is 1000. Multiplier 1.2 -> Need > 1200
    df.loc[df.index[-1], 'volume'] = 1500.0
    
    # Breakout
    # Max High of prev candles is 105.0. Close is 110.0. -> Breakout
    
    print("\nTest 2: Long Signal Setup")
    print(f"Close: {df.iloc[-1]['close']}, EMA_Fast: {df.iloc[-1][f'EMA_{settings.EMA_FAST}']}, EMA_Slow: {df.iloc[-1][f'EMA_{settings.EMA_SLOW}'] }")
    print(f"Volume: {df.iloc[-1]['volume']}, Avg Vol: {df['volume'].rolling(20).mean().iloc[-1]}")
    
    res = analyze_market(df)
    print(f"Result: {res['signal']} (Reason: {res['reason']})")
    print(f"SL: {res['sl']}, TP: {res['tp']}")
    
    assert res['signal'] == 'long'
    assert res['sl'] is not None
    assert res['tp'] is not None
    
    # Verify SL/TP math
    # SL = Price - mult * ATR = 110 - 2.0 * 2.0 = 106.0 (if settings.SL_ATR_MULT is 2.0)
    # TP = Price + SL_dist * 1.5 = 110 + 4.0 * 1.5 = 116.0
    
    expected_sl = 110.0 - (settings.SL_ATR_MULT * 2.0)
    print(f"Expected SL: {expected_sl}")
    # Allow float precision diff
    assert abs(res['sl'] - expected_sl) < 0.01

    print("\nSUCCESS: All tests passed.")

if __name__ == "__main__":
    test_strategy_logic()
