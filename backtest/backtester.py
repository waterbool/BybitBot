import pandas as pd
import numpy as np

from config import settings
from indicators.ta_module import add_indicators
from ml.model import add_ml_probabilities
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
    taker_fee_rate: float | None = None,
    maker_fee_rate: float | None = None,
    use_taker: bool | None = None,
    slippage_bps: float | None = None,
    slippage_rate: float | None = None,
    execution_delay_candles: int | None = None,
    funding_rate_per_bar: float = 0.0,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Backtest using the same EMA crossover strategy as live, with optional ML filter.
    Pipeline per step: indicators -> ML -> decision -> trade.
    """
    df = df.copy()

    if taker_fee_rate is None:
        taker_fee_rate = float(getattr(settings, "BACKTEST_TAKER_FEE", 0.0006))
    if maker_fee_rate is None:
        maker_fee_rate = float(getattr(settings, "BACKTEST_MAKER_FEE", 0.0002))
    if use_taker is None:
        use_taker = bool(getattr(settings, "BACKTEST_USE_TAKER", True))
    if execution_delay_candles is None:
        execution_delay_candles = int(getattr(settings, "BACKTEST_EXECUTION_DELAY_CANDLES", 1))
    if slippage_rate is None:
        if slippage_bps is None:
            slippage_bps = float(getattr(settings, "BACKTEST_SLIPPAGE_BPS", 2))
        slippage_rate = float(slippage_bps) / 10000.0

    fee_rate = taker_fee_rate if use_taker else maker_fee_rate
    partial_tp_enabled = bool(getattr(settings, "PARTIAL_TP_ENABLED", True))
    partial_tp_atr_mult = float(getattr(settings, "PARTIAL_TP_ATR_MULT", 1.0))
    partial_tp_fraction = float(getattr(settings, "PARTIAL_TP_FRACTION", 0.5))
    be_enabled = bool(getattr(settings, "BE_ENABLED", True))
    be_buffer_bps = float(getattr(settings, "BE_BUFFER_BPS", 2))
    trailing_enabled = bool(getattr(settings, "TRAILING_ENABLED", True))
    trail_atr_mult = float(getattr(settings, "TRAIL_ATR_MULT", 1.5))
    trail_activate_atr = float(getattr(settings, "TRAIL_ACTIVATE_ATR", 1.0))
    time_stop_enabled = bool(getattr(settings, "TIME_STOP_ENABLED", True))
    time_stop_candles = int(getattr(settings, "TIME_STOP_CANDLES", 24))
    prefer_worst_case = bool(getattr(settings, "PREFER_WORST_CASE", True))

    # Indicators (same as live)
    df = add_indicators(
        df,
        ema_fast=settings.EMA_FAST,
        ema_slow=settings.EMA_SLOW,
        atr_period=settings.ATR_PERIOD,
    )
    if getattr(settings, "ML_ENABLED", False):
        df = add_ml_probabilities(df)

    balance = initial_balance
    equity_curve = []
    trades = []
    total_fees = 0.0
    total_slippage = 0.0
    gross_pnl_total = 0.0
    position_sizes = []

    position = None  # dict with side, entry_price, qty, tp, sl, entry_time, entry_fee_remaining, entry_raw_price, entry_slip_remaining
    pending_entry = None  # dict with side, execute_idx, atr
    pending_exit = None  # dict with execute_idx
    pending_time_exit = None  # dict with execute_idx
    position_id = 0

    def _calc_be_stop(entry_price: float, side: int) -> float:
        buffer_rate = (be_buffer_bps / 10000.0) + (fee_rate * 2.0) + slippage_rate
        buffer = entry_price * buffer_rate
        if side == 1:
            return entry_price + buffer
        return entry_price - buffer

    def _close_position(exit_raw_price: float, exit_price: float, exit_reason: str, qty_close: float | None = None):
        nonlocal balance, total_fees, total_slippage, gross_pnl_total, position

        if position is None:
            return

        qty_before = position['qty']
        if qty_close is None:
            qty_close = qty_before
        qty_close = min(qty_close, qty_before)
        if qty_close <= 0:
            return

        ratio = qty_close / qty_before
        entry_fee_portion = position.get('entry_fee_remaining', 0.0) * ratio
        entry_slip_portion = position.get('entry_slip_remaining', 0.0) * ratio

        notional_exit = exit_price * qty_close
        exit_fee = notional_exit * fee_rate
        exit_slip = abs(exit_price - exit_raw_price) * qty_close
        entry_raw = position.get('entry_raw_price', position['entry_price'])

        if position['side'] == 1:
            gross_pnl = (exit_raw_price - entry_raw) * qty_close
            pnl_exec = (exit_price - position['entry_price']) * qty_close
        else:
            gross_pnl = (entry_raw - exit_raw_price) * qty_close
            pnl_exec = (position['entry_price'] - exit_price) * qty_close

        pnl = pnl_exec - entry_fee_portion - exit_fee
        balance += pnl_exec - exit_fee
        total_fees += entry_fee_portion + exit_fee
        total_slippage += entry_slip_portion + exit_slip
        gross_pnl_total += gross_pnl

        is_partial = qty_close < qty_before
        is_final = qty_close >= qty_before
        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': exit_time_ts,
            'side': 'BUY' if position['side'] == 1 else 'SELL',
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'entry_fee': entry_fee_portion,
            'exit_fee': exit_fee,
            'slippage_cost': entry_slip_portion + exit_slip,
            'gross_pnl': gross_pnl,
            'qty': qty_close,
            'pnl': pnl,
            'result': 'WIN' if pnl > 0 else 'LOSS',
            'reason': exit_reason,
            'position_id': position.get('id'),
            'confidence': position.get('confidence'),
            'size_mult': position.get('size_mult'),
            'position_size_usdt': position.get('position_size_usdt'),
            'is_partial': is_partial,
            'is_final': is_final,
            'final_reason': exit_reason if is_final else None,
            'balance': balance,
        })

        # Reduce remaining position
        position['qty'] -= qty_close
        position['entry_fee_remaining'] -= entry_fee_portion
        position['entry_slip_remaining'] -= entry_slip_portion
        if position['qty'] <= 0:
            position = None

    n_rows = len(df)
    for i in range(n_rows):
        # Slice to current row to avoid future leakage
        current_df = df.iloc[: i + 1]
        row = current_df.iloc[-1]
        ts = row.name
        exit_time_ts = ts

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
            # Update trailing stop before evaluating exits
            atr_col = f'ATR_{settings.ATR_PERIOD}'
            atr_now = row.get(atr_col)
            if trailing_enabled and atr_now is not None and not pd.isna(atr_now) and atr_now > 0:
                if not position.get('trailing_active', False):
                    atr_entry = position.get('atr_entry', atr_now)
                    if position['side'] == 1:
                        if high >= position['entry_price'] + trail_activate_atr * atr_entry:
                            position['trailing_active'] = True
                    else:
                        if low <= position['entry_price'] - trail_activate_atr * atr_entry:
                            position['trailing_active'] = True
                if position.get('trailing_active', False):
                    if position['side'] == 1:
                        new_sl = high - trail_atr_mult * atr_now
                        if new_sl > position['sl']:
                            position['sl'] = new_sl
                    else:
                        new_sl = low + trail_atr_mult * atr_now
                        if new_sl < position['sl']:
                            position['sl'] = new_sl

            def _apply_exit(raw_price: float, reason: str):
                if position['side'] == 1:
                    exec_price = raw_price * (1 - slippage_rate)
                else:
                    exec_price = raw_price * (1 + slippage_rate)
                _close_position(raw_price, exec_price, reason)

            def _apply_tp1():
                tp1_price = position.get('tp1')
                if tp1_price is None:
                    return
                if position['side'] == 1:
                    exec_price = tp1_price * (1 - slippage_rate)
                else:
                    exec_price = tp1_price * (1 + slippage_rate)
                fraction = max(0.0, min(1.0, partial_tp_fraction))
                qty_close = position['qty'] * fraction
                if qty_close <= 0:
                    return
                _close_position(tp1_price, exec_price, 'TP1', qty_close=qty_close)
                if position is not None:
                    position['tp1_hit'] = True
                    if be_enabled:
                        be_stop = _calc_be_stop(position['entry_price'], position['side'])
                        if position['side'] == 1:
                            position['sl'] = max(position['sl'], be_stop)
                        else:
                            position['sl'] = min(position['sl'], be_stop)

            hit_tp1 = False
            if partial_tp_enabled and not position.get('tp1_hit', False) and position.get('tp1') is not None:
                hit_tp1 = (high >= position['tp1']) if position['side'] == 1 else (low <= position['tp1'])

            if position['side'] == 1:
                hit_sl = low <= position['sl']
                hit_tp = high >= position['tp']
            else:
                hit_sl = high >= position['sl']
                hit_tp = low <= position['tp']

            if prefer_worst_case:
                if hit_sl:
                    reason = 'TRAIL' if position.get('trailing_active', False) else 'SL'
                    _apply_exit(position['sl'], reason)
                elif hit_tp1 and hit_tp:
                    _apply_tp1()
                elif hit_tp1:
                    _apply_tp1()
                elif hit_tp:
                    _apply_exit(position['tp'], 'TP')
            else:
                if hit_tp:
                    _apply_exit(position['tp'], 'TP')
                elif hit_tp1:
                    _apply_tp1()
                elif hit_sl:
                    reason = 'TRAIL' if position.get('trailing_active', False) else 'SL'
                    _apply_exit(position['sl'], reason)

            if position is None:
                pending_exit = None
                pending_time_exit = None

            # Time stop scheduling (next candle)
            if position is not None and time_stop_enabled and time_stop_candles > 0 and pending_time_exit is None:
                if (i - position['entry_idx']) >= time_stop_candles:
                    exec_idx = i + 1
                    if exec_idx < n_rows:
                        pending_time_exit = {'execute_idx': exec_idx}

            # Execute pending time stop
            if position is not None and pending_time_exit is not None and i >= pending_time_exit['execute_idx']:
                raw_exit_price = row.get('open', row.get('close'))
                if position['side'] == 1:
                    exit_price = raw_exit_price * (1 - slippage_rate)
                else:
                    exit_price = raw_exit_price * (1 + slippage_rate)
                _close_position(raw_exit_price, exit_price, 'TIME')
                pending_time_exit = None
                pending_exit = None

        # If position open, consider signal-based exit with delay (not for TP/SL)
        if position is not None and pending_exit is None and pending_time_exit is None:
            signal_df = apply_strategy(current_df.copy())
            signal = int(signal_df.iloc[-1]['signal'])
            if signal != 0 and signal != position['side']:
                exec_idx = i + execution_delay_candles
                if exec_idx < n_rows:
                    pending_exit = {'execute_idx': exec_idx}

        # Execute pending exit on scheduled candle (open price)
        if position is not None and pending_exit is not None and i >= pending_exit['execute_idx']:
            raw_exit_price = row.get('open', row['close'])
            if position['side'] == 1:
                exit_price = raw_exit_price * (1 - slippage_rate)
            else:
                exit_price = raw_exit_price * (1 + slippage_rate)

            _close_position(raw_exit_price, exit_price, 'Signal')
            if position is None:
                pending_exit = None

        # Execute pending entry on scheduled candle (open price)
        if position is None and pending_entry is not None and i >= pending_entry['execute_idx']:
            raw_entry_price = row.get('open', row['close'])
            side = pending_entry['side']
            if side == 1:
                entry_price = raw_entry_price * (1 + slippage_rate)
            else:
                entry_price = raw_entry_price * (1 - slippage_rate)
            base_usdt = float(settings.FIXED_USDT_SIZE)
            size_mult = 1.0
            confidence = 0.0
            if getattr(settings, "POSITION_SIZING_ENABLED", False):
                min_mult = float(getattr(settings, "POSITION_SIZING_MIN_MULT", 0.5))
                max_mult = float(getattr(settings, "POSITION_SIZING_MAX_MULT", 1.5))
                prob = pending_entry.get('prob')
                if prob is None or pd.isna(prob):
                    confidence = 0.0
                else:
                    confidence = abs(float(prob) - 0.5) * 2.0
                size_mult = min_mult + confidence * (max_mult - min_mult)
                size_mult = max(min_mult, min(max_mult, size_mult))
            position_size_usdt = base_usdt * size_mult
            qty = round(position_size_usdt / entry_price, 6)
            tp, sl = _calc_tp_sl(entry_price, side, pending_entry['atr'])
            entry_fee = (entry_price * qty) * fee_rate
            entry_slip = abs(entry_price - raw_entry_price) * qty
            balance -= entry_fee

            tp1 = None
            if partial_tp_enabled and partial_tp_atr_mult > 0:
                tp1_dist = pending_entry['atr'] * partial_tp_atr_mult
                if side == 1:
                    tp1 = entry_price + tp1_dist
                else:
                    tp1 = entry_price - tp1_dist

            position_id += 1
            position = {
                'id': position_id,
                'side': side,
                'entry_price': entry_price,
                'entry_raw_price': raw_entry_price,
                'entry_fee_remaining': entry_fee,
                'entry_slip_remaining': entry_slip,
                'qty': qty,
                'position_size_usdt': position_size_usdt,
                'size_mult': size_mult,
                'confidence': confidence,
                'tp': tp,
                'sl': sl,
                'tp1': tp1,
                'tp1_hit': False,
                'trailing_active': False,
                'atr_entry': pending_entry['atr'],
                'entry_time': ts,
                'entry_idx': i,
            }
            position_sizes.append(position_size_usdt)
            pending_entry = None
            pending_exit = None
            pending_time_exit = None

        # If flat and no pending entry, evaluate new signal
        if position is None and pending_entry is None:
            signal_df = apply_strategy(current_df.copy())
            signal = int(signal_df.iloc[-1]['signal'])
            if signal != 0:
                atr_col = f'ATR_{settings.ATR_PERIOD}'
                atr = row.get(atr_col)
                if atr is None or pd.isna(atr) or atr == 0:
                    continue
                prob = None
                if getattr(settings, "ML_ENABLED", False):
                    if signal == 1:
                        prob = row.get('p_up')
                    elif signal == -1:
                        prob = row.get('p_down')

                exec_idx = i + execution_delay_candles
                if exec_idx >= n_rows:
                    continue
                pending_entry = {
                    'side': signal,
                    'execute_idx': exec_idx,
                    'atr': atr,
                    'prob': prob,
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
        partial_exits_count = int((trades_df['reason'] == 'TP1').sum())
        final_exits = trades_df[trades_df.get('is_final', False)]
        total_positions = len(final_exits)
        if total_positions > 0:
            trailing_exit_ratio = float((final_exits['final_reason'] == 'TRAIL').mean())
            time_stop_exit_ratio = float((final_exits['final_reason'] == 'TIME').mean())
        else:
            trailing_exit_ratio = 0.0
            time_stop_exit_ratio = 0.0
        if not final_exits.empty and 'confidence' in final_exits.columns:
            bins = [0.0, 0.3, 0.6, 1.0]
            labels = ["0-0.3", "0.3-0.6", "0.6-1.0"]
            conf = final_exits['confidence'].fillna(0.0).clip(0.0, 1.0)
            bucket = pd.cut(conf, bins=bins, labels=labels, include_lowest=True)
            pnl_by_conf = final_exits.groupby(bucket)['pnl'].sum().reindex(labels).fillna(0.0).to_dict()
        else:
            pnl_by_conf = {"0-0.3": 0.0, "0.3-0.6": 0.0, "0.6-1.0": 0.0}
    else:
        profit_factor = 0.0
        total_trades = 0
        partial_exits_count = 0
        trailing_exit_ratio = 0.0
        time_stop_exit_ratio = 0.0
        pnl_by_conf = {"0-0.3": 0.0, "0.3-0.6": 0.0, "0.6-1.0": 0.0}

    avg_position_size = float(np.mean(position_sizes)) if position_sizes else 0.0

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
        'gross_pnl': gross_pnl_total,
        'net_pnl': balance - initial_balance,
        'total_fees': total_fees,
        'total_slippage_cost_estimate': total_slippage,
        'partial_exits_count': partial_exits_count,
        'trailing_exit_ratio': trailing_exit_ratio,
        'time_stop_exit_ratio': time_stop_exit_ratio,
        'avg_position_size': avg_position_size,
        'pnl_by_confidence_bins': pnl_by_conf,
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
