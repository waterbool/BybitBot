import pandas as pd
import numpy as np

from config import settings
from indicators.ta_module import add_indicators
from strategy.rules import apply_strategy


def _calc_tp_sl(entry_price: float, side: int, atr: float) -> tuple[float, float]:
    sl_dist = atr * settings.SL_ATR_MULT
    tp_dist = atr * settings.RISK_REWARD_RATIO
    if side == 1:
        sl = entry_price - sl_dist
        tp = entry_price + tp_dist
    else:
        sl = entry_price + sl_dist
        tp = entry_price - tp_dist
    return tp, sl


def run_backtest(
    df: pd.DataFrame,
    initial_balance: float,
    taker_fee_rate: float = 0.00055,
    slippage_rate: float = 0.0002,
    funding_rate_per_bar: float = 0.0,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Backtest using the same EMA crossover strategy as live, with optional ML filter.
    Pipeline per step: indicators -> ML -> decision -> trade.
    """
    df = df.copy()

    # Indicators (same as live)
    df = add_indicators(
        df,
        ema_fast=settings.EMA_FAST,
        ema_slow=settings.EMA_SLOW,
        atr_period=settings.ATR_PERIOD,
    )

    balance = initial_balance
    equity_curve = []
    trades = []

    position = None  # dict with side, entry_price, qty, tp, sl, entry_time

    n_rows = len(df)
    for i in range(n_rows):
        # Slice to current row to avoid future leakage
        current_df = df.iloc[: i + 1]
        row = current_df.iloc[-1]
        ts = row.name

        # Update equity curve (mark-to-market)
        if position is not None:
            if position['side'] == 1:
                unreal = (row['close'] - position['entry_price']) * position['qty']
            else:
                unreal = (position['entry_price'] - row['close']) * position['qty']
            equity_curve.append({'timestamp': ts, 'equity': balance + unreal})
        else:
            equity_curve.append({'timestamp': ts, 'equity': balance})

        # Apply funding if position open
        if position is not None and funding_rate_per_bar != 0.0:
            notional = row['close'] * position['qty']
            funding = notional * funding_rate_per_bar * (1 if position['side'] == -1 else -1)
            balance += funding

        # Check exit if position is open
        if position is not None:
            high = row['high']
            low = row['low']
            exit_price = None
            exit_reason = None

            if position['side'] == 1:
                if low <= position['sl']:
                    exit_price = position['sl'] * (1 - slippage_rate)
                    exit_reason = 'SL'
                elif high >= position['tp']:
                    exit_price = position['tp'] * (1 - slippage_rate)
                    exit_reason = 'TP'
            else:
                if high >= position['sl']:
                    exit_price = position['sl'] * (1 + slippage_rate)
                    exit_reason = 'SL'
                elif low <= position['tp']:
                    exit_price = position['tp'] * (1 + slippage_rate)
                    exit_reason = 'TP'

            if exit_price is not None:
                notional_entry = position['entry_price'] * position['qty']
                notional_exit = exit_price * position['qty']
                fee = (notional_entry + notional_exit) * taker_fee_rate
                if position['side'] == 1:
                    pnl = (exit_price - position['entry_price']) * position['qty'] - fee
                else:
                    pnl = (position['entry_price'] - exit_price) * position['qty'] - fee
                balance += pnl
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': ts,
                    'side': 'BUY' if position['side'] == 1 else 'SELL',
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'qty': position['qty'],
                    'pnl': pnl,
                    'result': 'WIN' if pnl > 0 else 'LOSS',
                    'reason': exit_reason,
                    'balance': balance,
                })
                position = None

        # If flat, evaluate new signal
        if position is None:
            signal_df = apply_strategy(current_df.copy())
            signal = int(signal_df.iloc[-1]['signal'])
            if signal != 0:
                atr_col = f'ATR_{settings.ATR_PERIOD}'
                atr = row.get(atr_col)
                if atr is None or pd.isna(atr) or atr == 0:
                    continue

                # Entry with slippage
                if signal == 1:
                    entry_price = row['close'] * (1 + slippage_rate)
                else:
                    entry_price = row['close'] * (1 - slippage_rate)

                qty = round(settings.FIXED_USDT_SIZE / entry_price, 6)
                tp, sl = _calc_tp_sl(entry_price, signal, atr)

                position = {
                    'side': signal,
                    'entry_price': entry_price,
                    'qty': qty,
                    'tp': tp,
                    'sl': sl,
                    'entry_time': ts,
                }

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)

    if not equity_df.empty:
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = equity_df['equity'] - equity_df['peak']
        max_drawdown = float(equity_df['drawdown'].min())
    else:
        max_drawdown = 0.0

    if not trades_df.empty:
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = -trades_df[trades_df['pnl'] < 0]['pnl'].sum()
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        total_trades = len(trades_df)
    else:
        profit_factor = 0.0
        total_trades = 0

    # Monthly stats
    if not trades_df.empty:
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        monthly = trades_df.groupby(trades_df['exit_time'].dt.to_period('M')).agg(
            trades=('pnl', 'count'),
            pnl=('pnl', 'sum'),
            win_rate=('pnl', lambda x: (x > 0).mean() * 100.0),
        ).reset_index()
        monthly['month'] = monthly['exit_time'].astype(str)
        monthly_stats = monthly[['month', 'trades', 'pnl', 'win_rate']]
    else:
        monthly_stats = pd.DataFrame(columns=['month', 'trades', 'pnl', 'win_rate'])

    metrics = {
        'initial_balance': initial_balance,
        'final_balance': balance,
        'total_trades': total_trades,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'pnl_total': balance - initial_balance,
    }

    return metrics, trades_df, equity_df, monthly_stats

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
    
    stats, trade_log, equity_df, monthly_stats = run_backtest(df, init_bal)
    
    print("\nBacktest Results:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")
            
    print("\nLast 5 Trades:")
    if not trade_log.empty:
        print(trade_log[['entry_time', 'side', 'result', 'pnl', 'balance']].tail())
    else:
        print("No trades executed.")
