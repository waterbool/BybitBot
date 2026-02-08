import pandas as pd
import numpy as np

def run_backtest(df: pd.DataFrame, initial_balance: float, bet_size: float, expiry_minutes: int = 1, payout_rate: float = 0.80) -> dict:
    """
    Runs a backtest on the provided DataFrame with 'signal' column.
    
    Args:
        df: DataFrame with 'close' prices and 'signal' column (1, -1, 0).
        initial_balance: Starting balance for the account.
        bet_size: Amount risked per trade.
        expiry_minutes: Duration of the option in minutes (candles).
        payout_rate: Profit rate for winning trades (e.g., 0.80 for 80%).
        
    Returns:
        dict: A dictionary containing backtest metrics.
    """
    df = df.copy()
    balance = initial_balance
    trades = []
    
    # Ensure signal column exists
    if 'signal' not in df.columns:
        raise ValueError("DataFrame must contain 'signal' column")

    # Iterate through the DataFrame to find signals
    # Using iterrows is generally slow, but necessary for sequential balance updates if bet size is dynamic (compounding).
    # Since bet_size is fixed here, we could vectorize, but requirement asks for clean logic without global vars, simple loop is fine.
    
    # We need to access future prices, so index access is better.
    # Assuming standard RangeIndex or careful usage of iloc.
    
    n_rows = len(df)
    
    for i in range(n_rows):
        signal = df.iloc[i]['signal']
        
        if signal == 0:
            continue
            
        # Check if we have enough data for expiry
        expiry_index = i + expiry_minutes
        if expiry_index >= n_rows:
            break
            
        entry_price = df.iloc[i]['close']
        expiry_price = df.iloc[expiry_index]['close']
        entry_time = df.index[i]
        expiry_time = df.index[expiry_index]
        
        # Determine Outcome
        win = False
        if signal == 1: # CALL
            if expiry_price > entry_price:
                win = True
        elif signal == -1: # PUT
            if expiry_price < entry_price:
                win = True
        else:
            continue # Should be 0, handled above
            
        # Calculate PnL
        if win:
            pnl = bet_size * payout_rate
        else:
            pnl = -bet_size
            
        balance += pnl
        
        trades.append({
            'entry_time': entry_time,
            'expiry_time': expiry_time,
            'signal': signal,
            'entry_price': entry_price,
            'expiry_price': expiry_price,
            'result': 'WIN' if win else 'LOSS',
            'pnl': pnl,
            'balance': balance
        })
        
    # Create DataFrame from trades for easy analysis
    trades_df = pd.DataFrame(trades)
    
    # Calculate Metrics
    total_trades = len(trades)
    if total_trades > 0:
        win_rate = len(trades_df[trades_df['result'] == 'WIN']) / total_trades * 100
        
        # Max Consecutive Wins/Losses
        # Create a boolean series: True for Win, False for Loss
        results_bool = trades_df['result'] == 'WIN'
        # Group by value change to identify streaks
        streaks = results_bool.ne(results_bool.shift()).cumsum()
        streak_counts = results_bool.groupby(streaks).agg(['value_counts']).reset_index(drop=True)
        
        # This is a bit complex to get cleanly with pandas in one line, let's just loop locally for simplicity and correctness
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0
        
        for res in trades_df['result']:
            if res == 'WIN':
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
        
        # Drawdown
        # High Water Mark of Balance
        trades_df['peak_balance'] = trades_df['balance'].cummax()
        trades_df['drawdown'] = trades_df['balance'] - trades_df['peak_balance']
        max_drawdown = trades_df['drawdown'].min() # Negative value
        
    else:
        win_rate = 0
        max_win_streak = 0
        max_loss_streak = 0
        max_drawdown = 0
        
    metrics = {
        'initial_balance': initial_balance,
        'final_balance': balance,
        'total_trades': total_trades,
        'win_rate_percent': win_rate,
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak,
        'max_drawdown': max_drawdown,
        'pnl_total': balance - initial_balance
    }
    
    return metrics, trades_df

if __name__ == "__main__":
    # Test script
    print("Running Backtester Test...")
    
    # Create Random Data
    np.random.seed(123)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1min')
    prices = 100 + np.cumsum(np.random.randn(200)) # Random walk
    signals = np.random.choice([0, 0, 0, 0, 1, -1], size=200) # Mostly 0s
    
    df = pd.DataFrame({
        'close': prices,
        'signal': signals
    }, index=dates)
    
    init_bal = 1000
    bet = 10
    
    stats, trade_log = run_backtest(df, init_bal, bet)
    
    print("\nBacktest Results:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")
            
    print("\nLast 5 Trades:")
    if not trade_log.empty:
        print(trade_log[['entry_time', 'signal', 'result', 'pnl', 'balance']].tail())
    else:
        print("No trades executed.")
