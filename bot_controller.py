"""
Bot Controller - Manages bot lifecycle and state for web UI
"""
import threading
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from queue import Queue
import pandas as pd

from config import settings
from live.edge_snapshot import load_edge_snapshot, snapshot_is_fresh, snapshot_age_minutes
from live.scanner import scan_live_candidates
from live.trade_gate import evaluate_trade_gate
from risk.risk import RiskManager
from pybit.unified_trading import HTTP
from single_symbol_pipeline import (
    fetch_indicator_frame_for_lookback,
    fetch_signal_frame_for_lookback,
    interval_to_minutes as _shared_interval_to_minutes,
    required_history_bars,
)

logger = logging.getLogger(__name__)

def _interval_to_minutes(interval_value: str) -> int:
    """Compatibility wrapper around the shared baseline interval parser."""
    return _shared_interval_to_minutes(interval_value)


class BotController:
    """Controls bot execution and provides status for web UI"""
    
    def __init__(self):
        self.status = "idle"  # idle, running, stopped, error
        self.thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        self.risk_manager: Optional[RiskManager] = None
        self.session: Optional[HTTP] = None
        
        # Statistics
        self.trades_history: List[Dict] = []
        self.current_position: Optional[Dict] = None
        self.stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "daily_pnl": 0.0,
            "daily_trades": 0
        }
        self.selector_state = {
            "enabled": bool(getattr(settings, "LIVE_SELECTOR_ENABLED", False)),
            "execution_mode": getattr(settings, "LIVE_SELECTOR_EXECUTION_MODE", "paper"),
            "snapshot_path": getattr(settings, "LIVE_SELECTOR_EDGE_SNAPSHOT_PATH", ""),
            "snapshot_loaded": False,
            "snapshot_age_minutes": None,
            "last_scan_at": None,
            "last_candidate": None,
            "last_gate_decision": None,
            "last_skip_reason": None,
        }
        
        # Log queue for web UI
        self.log_queue = Queue(maxsize=1000)
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging handler to capture logs for web UI"""
        class QueueHandler(logging.Handler):
            def __init__(self, log_queue):
                super().__init__()
                self.log_queue = log_queue
                
            def emit(self, record):
                try:
                    msg = self.format(record)
                    if not self.log_queue.full():
                        self.log_queue.put({
                            'timestamp': datetime.now().isoformat(),
                            'level': record.levelname,
                            'message': msg
                        })
                except Exception:
                    pass
        
        handler = QueueHandler(self.log_queue)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(handler)

    def _assumed_fee_slippage(self) -> tuple[float, float]:
        use_taker = bool(getattr(settings, "BACKTEST_USE_TAKER", True))
        taker_fee = float(getattr(settings, "BACKTEST_TAKER_FEE", 0.0006))
        maker_fee = float(getattr(settings, "BACKTEST_MAKER_FEE", 0.0002))
        slippage_bps = float(getattr(settings, "BACKTEST_SLIPPAGE_BPS", 2))
        fee_rate = taker_fee if use_taker else maker_fee
        slippage_rate = slippage_bps / 10000.0
        return fee_rate, slippage_rate

    def _calc_be_stop(self, entry_price: float, side: int) -> float:
        fee_rate, slippage_rate = self._assumed_fee_slippage()
        buffer_bps = float(getattr(settings, "BE_BUFFER_BPS", 2))
        buffer_rate = (buffer_bps / 10000.0) + (fee_rate * 2.0) + slippage_rate
        buffer = entry_price * buffer_rate
        if side == 1:
            return entry_price + buffer
        return entry_price - buffer

    def _should_place_real_orders(self, execution_mode: Optional[str] = None) -> bool:
        if execution_mode is not None:
            mode = str(execution_mode).strip().lower()
            return mode == "live"
        if getattr(settings, "LIVE_SELECTOR_ENABLED", False):
            mode = str(self._selector_execution_mode()).strip().lower()
            return mode == "live"
        return not settings.DRY_RUN

    def _extract_signal_timestamp_ms(self, signal_row: pd.Series) -> Optional[int]:
        candidates = [
            signal_row.get("timestamp_ms"),
            signal_row.get("timestamp"),
            getattr(signal_row, "name", None),
        ]
        for value in candidates:
            if value is None or pd.isna(value):
                continue
            if isinstance(value, (int, float)):
                return int(value)
            try:
                return int(pd.Timestamp(value).timestamp() * 1000)
            except Exception:
                continue
        return None

    def _update_stop_loss(self, new_sl: float, symbol: Optional[str] = None):
        symbol = symbol or (self.current_position or {}).get("symbol") or settings.BYBIT_SYMBOL
        execution_mode = (self.current_position or {}).get("execution_mode")
        if not self._should_place_real_orders(execution_mode):
            logger.info(f"[DRY RUN] Update stop loss for {symbol} -> {new_sl}")
            return
        if not self.session:
            return
        try:
            resp = self.session.set_trading_stop(
                category=settings.BYBIT_CATEGORY,
                symbol=symbol,
                stopLoss=str(new_sl),
            )
            logger.info(f"Stop loss updated: {resp}")
        except Exception as e:
            logger.error(f"Failed to update stop loss: {e}")

    def _close_position_market(self, qty: float, reason: str, price_hint: float, is_partial: bool, symbol: Optional[str] = None):
        if qty <= 0:
            return
        symbol = symbol or (self.current_position or {}).get("symbol") or settings.BYBIT_SYMBOL
        side = "Sell" if self.current_position and self.current_position.get('side') == 1 else "Buy"
        execution_mode = (self.current_position or {}).get("execution_mode")
        if not self._should_place_real_orders(execution_mode):
            logger.info(f"[DRY RUN] Close {symbol} {qty} ({reason}) @ {price_hint}")
        else:
            if not self.session:
                return
            try:
                resp = self.session.place_order(
                    category=settings.BYBIT_CATEGORY,
                    symbol=symbol,
                    side=side,
                    orderType="Market",
                    qty=str(qty),
                    reduceOnly=True,
                    timeInForce="GTC",
                )
                logger.info(f"Close order placed: {resp}")
            except Exception as e:
                logger.error(f"Close order failed: {e}")
                return

        if not self.current_position:
            return

        entry_price = float(self.current_position.get('entry_price', price_hint) or price_hint)
        direction = 1 if self.current_position.get('side') == 1 else -1
        fee_rate, _ = self._assumed_fee_slippage()
        gross_pnl = (float(price_hint) - entry_price) * float(qty) * direction
        fees_paid = (entry_price + float(price_hint)) * float(qty) * fee_rate
        realized_pnl = gross_pnl - fees_paid
        self.current_position['realized_pnl'] = float(self.current_position.get('realized_pnl', 0.0) or 0.0) + realized_pnl
        self.stats['total_pnl'] += realized_pnl
        self.stats['daily_pnl'] += realized_pnl

        self.trades_history.append({
            'timestamp': datetime.now().isoformat(),
            'event_type': 'exit',
            'symbol': symbol,
            'strategy_name': self.current_position.get('strategy_name') if self.current_position else None,
            'execution_mode': self.current_position.get('execution_mode') if self.current_position else None,
            'side': side,
            'entry_price': self.current_position.get('entry_price'),
            'exit_price': price_hint,
            'qty': qty,
            'reason': reason,
            'is_partial': is_partial,
            'realized_pnl': realized_pnl,
        })

        self.current_position['qty_remaining'] -= qty
        if self.current_position['qty_remaining'] <= 0:
            final_pnl = float(self.current_position.get('realized_pnl', 0.0) or 0.0)
            if final_pnl >= 0:
                self.stats['winning_trades'] += 1
            else:
                self.stats['losing_trades'] += 1
            self.current_position = None

    def _manage_open_position(self, row: pd.Series, interval_mins: int):
        if not self.current_position:
            return

        pos = self.current_position
        high = row.get('high')
        low = row.get('low')
        if high is None or low is None:
            return

        prefer_worst_case = bool(getattr(settings, "PREFER_WORST_CASE", True))
        partial_tp_enabled = bool(getattr(settings, "PARTIAL_TP_ENABLED", True))
        be_enabled = bool(getattr(settings, "BE_ENABLED", True))
        trailing_enabled = bool(getattr(settings, "TRAILING_ENABLED", True))
        trail_activate_atr = float(getattr(settings, "TRAIL_ACTIVATE_ATR", 1.0))
        trail_atr_mult = float(getattr(settings, "TRAIL_ATR_MULT", 1.5))
        time_stop_enabled = bool(getattr(settings, "TIME_STOP_ENABLED", True))
        time_stop_candles = int(getattr(settings, "TIME_STOP_CANDLES", 24))

        atr_col = f'ATR_{settings.ATR_PERIOD}'
        atr_now = row.get(atr_col)

        prev_sl = pos['sl']
        if trailing_enabled and atr_now is not None and not pd.isna(atr_now) and atr_now > 0:
            if not pos.get('trailing_active', False):
                atr_entry = pos.get('atr_entry', atr_now)
                if pos['side'] == 1:
                    if high >= pos['entry_price'] + trail_activate_atr * atr_entry:
                        pos['trailing_active'] = True
                else:
                    if low <= pos['entry_price'] - trail_activate_atr * atr_entry:
                        pos['trailing_active'] = True
            if pos.get('trailing_active', False):
                if pos['side'] == 1:
                    new_sl = high - trail_atr_mult * atr_now
                    if new_sl > pos['sl']:
                        pos['sl'] = new_sl
                else:
                    new_sl = low + trail_atr_mult * atr_now
                    if new_sl < pos['sl']:
                        pos['sl'] = new_sl

        if pos['sl'] != prev_sl:
            self._update_stop_loss(pos['sl'], symbol=pos.get('symbol'))

        hit_tp1 = False
        if partial_tp_enabled and not pos.get('tp1_hit', False) and pos.get('tp1') is not None:
            hit_tp1 = (high >= pos['tp1']) if pos['side'] == 1 else (low <= pos['tp1'])

        if pos['side'] == 1:
            hit_sl = low <= pos['sl']
            hit_tp = high >= pos['tp']
        else:
            hit_sl = high >= pos['sl']
            hit_tp = low <= pos['tp']

        def _apply_tp1():
            fraction = max(0.0, min(1.0, float(getattr(settings, "PARTIAL_TP_FRACTION", 0.5))))
            qty_close = pos['qty_remaining'] * fraction
            if qty_close <= 0:
                return
            self._close_position_market(qty_close, "TP1", pos['tp1'], is_partial=True, symbol=pos.get('symbol'))
            if self.current_position:
                self.current_position['tp1_hit'] = True
                if be_enabled:
                    be_stop = self._calc_be_stop(self.current_position['entry_price'], self.current_position['side'])
                    if self.current_position['side'] == 1:
                        new_sl = max(self.current_position['sl'], be_stop)
                    else:
                        new_sl = min(self.current_position['sl'], be_stop)
                    if new_sl != self.current_position['sl']:
                        self.current_position['sl'] = new_sl
                        self._update_stop_loss(new_sl, symbol=self.current_position.get('symbol'))

        def _apply_full_exit(reason: str, price_hint: float):
            self._close_position_market(pos['qty_remaining'], reason, price_hint, is_partial=False, symbol=pos.get('symbol'))

        if prefer_worst_case:
            if hit_sl:
                reason = "TRAIL" if pos.get('trailing_active', False) else "SL"
                _apply_full_exit(reason, pos['sl'])
            elif hit_tp1 and hit_tp:
                _apply_tp1()
            elif hit_tp1:
                _apply_tp1()
            elif hit_tp:
                _apply_full_exit("TP", pos['tp'])
        else:
            if hit_tp:
                _apply_full_exit("TP", pos['tp'])
            elif hit_tp1:
                _apply_tp1()
            elif hit_sl:
                reason = "TRAIL" if pos.get('trailing_active', False) else "SL"
                _apply_full_exit(reason, pos['sl'])

        if not self.current_position:
            return

        if time_stop_enabled and time_stop_candles > 0:
            entry_ts = self.current_position.get('entry_ts')
            current_ts = row.get('timestamp')
            if entry_ts is not None and current_ts is not None:
                bars_elapsed = int((int(current_ts) - int(entry_ts)) / (interval_mins * 60 * 1000))
                if not self.current_position.get('time_stop_pending', False) and bars_elapsed >= time_stop_candles:
                    self.current_position['time_stop_pending'] = True
                    self.current_position['time_stop_trigger_ts'] = int(current_ts)
                if self.current_position.get('time_stop_pending', False):
                    trigger_ts = self.current_position.get('time_stop_trigger_ts')
                    if trigger_ts is not None and int(current_ts) > int(trigger_ts):
                        exit_price = row.get('open', row.get('close'))
                        self._close_position_market(
                            self.current_position['qty_remaining'],
                            "TIME",
                            exit_price,
                            is_partial=False,
                            symbol=self.current_position.get('symbol'),
                        )
    
    def start_trading(self) -> Dict:
        """Start live trading in a separate thread"""
        if self.status == "running":
            return {"success": False, "message": "Bot is already running"}
        
        try:
            self.stop_flag.clear()
            self.status = "running"
            self.thread = threading.Thread(target=self._trading_loop, daemon=True)
            self.thread.start()
            
            logger.info("Bot started successfully")
            return {"success": True, "message": "Bot started successfully"}
        except Exception as e:
            self.status = "error"
            logger.error(f"Failed to start bot: {e}")
            return {"success": False, "message": f"Failed to start: {str(e)}"}
    
    def stop_trading(self) -> Dict:
        """Stop live trading"""
        if self.status != "running":
            return {"success": False, "message": "Bot is not running"}
        
        try:
            self.stop_flag.set()
            self.status = "stopped"
            logger.info("Bot stop requested")
            return {"success": True, "message": "Bot stopped"}
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
            return {"success": False, "message": f"Error stopping: {str(e)}"}
    
    def get_status(self) -> Dict:
        """Get current bot status and statistics"""
        return {
            "status": self.status,
            "stats": self.stats,
            "current_position": self.current_position,
            "config": {
                "symbol": settings.BYBIT_SYMBOL,
                "symbols": getattr(settings, "LIVE_SELECTOR_SYMBOLS", [settings.BYBIT_SYMBOL]),
                "dry_run": settings.DRY_RUN,
                "testnet": settings.BYBIT_TESTNET,
                "timeframe": settings.BYBIT_INTERVAL,
                "live_selector_enabled": getattr(settings, "LIVE_SELECTOR_ENABLED", False),
                "live_selector_mode": getattr(settings, "LIVE_SELECTOR_EXECUTION_MODE", "paper"),
            },
            "selector_state": self.selector_state,
        }
    
    def get_recent_logs(self, count: int = 100) -> List[Dict]:
        """Get recent logs from queue"""
        logs = []
        temp_logs = []
        
        # Drain queue
        while not self.log_queue.empty() and len(temp_logs) < count:
            temp_logs.append(self.log_queue.get())
        
        # Put back and return
        for log in temp_logs:
            if not self.log_queue.full():
                self.log_queue.put(log)
        
        return temp_logs[-count:]

    def _load_live_selector_snapshot(self) -> Optional[Dict]:
        snapshot = load_edge_snapshot(settings.LIVE_SELECTOR_EDGE_SNAPSHOT_PATH)
        self.selector_state["snapshot_loaded"] = snapshot is not None
        self.selector_state["snapshot_age_minutes"] = snapshot_age_minutes(snapshot) if snapshot else None
        return snapshot

    def _fetch_latest_closed_row(self, symbol: str, interval: str, lookback_bars: int = 250) -> tuple[Optional[pd.Series], int]:
        df, interval_mins = fetch_indicator_frame_for_lookback(
            symbol=symbol,
            interval=interval,
            lookback_bars=lookback_bars,
            category=settings.BYBIT_CATEGORY,
        )
        if df.empty:
            return None, interval_mins
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        return df.iloc[-1], interval_mins

    def _selector_execution_mode(self) -> str:
        mode = str(getattr(settings, "LIVE_SELECTOR_EXECUTION_MODE", "paper")).strip().lower()
        if mode not in {"paper", "live"}:
            return "paper"
        return mode

    def _execute_candidate(self, candidate):
        signal_row = pd.Series(candidate.signal_row)
        if "timestamp_ms" not in signal_row:
            try:
                signal_row["timestamp_ms"] = int(pd.Timestamp(candidate.signal_time).timestamp() * 1000)
            except Exception:
                signal_row["timestamp_ms"] = None
        self._execute_trade(
            signal=candidate.side,
            close_price=candidate.close_price,
            signal_row=signal_row,
            symbol=candidate.symbol,
            strategy_name=candidate.strategy_name,
            execution_mode=self._selector_execution_mode(),
            base_interval=candidate.base_interval,
            selector_metrics={
                "signal_score": candidate.signal_score,
                "edge_score": candidate.edge_score,
                "selection_score": candidate.selection_score,
                "signal_components": candidate.signal_components,
                "edge_components": candidate.edge_components,
            },
        )

    def _run_live_selector_cycle(self):
        snapshot = self._load_live_selector_snapshot()
        if settings.LIVE_SELECTOR_REQUIRE_EDGE_SNAPSHOT and not snapshot:
            self.selector_state["last_skip_reason"] = "missing_edge_snapshot"
            logger.warning("Live selector skipped: edge snapshot is missing")
            return
        if snapshot and not snapshot_is_fresh(snapshot, settings.LIVE_SELECTOR_EDGE_MAX_AGE_MINUTES):
            self.selector_state["last_skip_reason"] = "stale_edge_snapshot"
            logger.warning("Live selector skipped: edge snapshot is stale")
            return

        if self.current_position:
            symbol = self.current_position.get("symbol", settings.BYBIT_SYMBOL)
            interval = self.current_position.get("base_interval", settings.LIVE_SELECTOR_BASE_INTERVAL)
            row, interval_mins = self._fetch_latest_closed_row(symbol, interval)
            if row is not None:
                self._manage_open_position(row, interval_mins)
            else:
                logger.warning(f"Unable to fetch latest closed candle for open position {symbol}")
            return

        candidates = scan_live_candidates(
            symbols=settings.LIVE_SELECTOR_SYMBOLS,
            edge_snapshot=snapshot,
            strategy_names=settings.LIVE_SELECTOR_STRATEGIES,
        )
        self.selector_state["last_scan_at"] = datetime.utcnow().isoformat()
        if not candidates:
            self.selector_state["last_candidate"] = None
            self.selector_state["last_gate_decision"] = None
            self.selector_state["last_skip_reason"] = "no_candidates"
            logger.info("Live selector: no current candidates")
            return

        candidate = candidates[0]
        self.selector_state["last_candidate"] = candidate.to_dict()

        decision = evaluate_trade_gate(
            candidate=candidate,
            current_position=self.current_position,
            trades_history=self.trades_history,
            edge_snapshot=snapshot,
        )
        self.selector_state["last_gate_decision"] = {
            "allowed": decision.allowed,
            "reasons": decision.reasons,
            "metrics": decision.metrics,
        }
        if not decision.allowed:
            self.selector_state["last_skip_reason"] = ",".join(decision.reasons)
            logger.info(
                "Live selector rejected %s %s: %s",
                candidate.symbol,
                candidate.strategy_name,
                ", ".join(decision.reasons),
            )
            return

        self.selector_state["last_skip_reason"] = None
        logger.info(
            "Live selector accepted %s %s score=%.4f edge=%.4f selection=%.4f",
            candidate.symbol,
            candidate.strategy_name,
            candidate.signal_score,
            candidate.edge_score,
            candidate.selection_score,
        )
        self._execute_candidate(candidate)
    
    def _trading_loop(self):
        """Main trading loop (runs in separate thread)"""
        try:
            # Initialize
            self.risk_manager = RiskManager()

            live_selector_enabled = bool(getattr(settings, "LIVE_SELECTOR_ENABLED", False))
            allow_real_execution = self._should_place_real_orders()
            if live_selector_enabled and int(getattr(settings, "LIVE_SELECTOR_MAX_POSITIONS_TOTAL", 1)) > 1:
                logger.warning("LIVE_SELECTOR_MAX_POSITIONS_TOTAL > 1 is configured, but controller currently supports only one open position.")

            if allow_real_execution:
                if not settings.BYBIT_API_KEY or not settings.BYBIT_API_SECRET:
                    logger.error("Real trading requires API keys")
                    self.status = "error"
                    return

                self.session = HTTP(
                    testnet=settings.BYBIT_TESTNET,
                    api_key=settings.BYBIT_API_KEY,
                    api_secret=settings.BYBIT_API_SECRET
                )
                logger.info("Connected to Bybit API")
                self.risk_manager.sync_from_api(self.session)
            else:
                if live_selector_enabled:
                    logger.info("LIVE SELECTOR PAPER MODE: No real orders will be placed")
                else:
                    logger.info("DRY RUN MODE: No real orders will be placed")
            
            # Trading loop
            while not self.stop_flag.is_set():
                try:
                    # Risk check
                    if not self.risk_manager.can_trade():
                        logger.warning("Risk limits reached. Waiting...")
                        time.sleep(60)
                        continue
                    
                    if getattr(settings, "LIVE_SELECTOR_ENABLED", False):
                        self._run_live_selector_cycle()
                        time.sleep(max(5, int(getattr(settings, "LIVE_SELECTOR_SCAN_INTERVAL_SECONDS", 30))))
                    else:
                        try:
                            interval_mins = _interval_to_minutes(settings.BYBIT_INTERVAL)
                        except ValueError as e:
                            logger.error(str(e))
                            time.sleep(60)
                            continue
                        min_bars = required_history_bars()

                        df, interval_mins = fetch_signal_frame_for_lookback(
                            symbol=settings.BYBIT_SYMBOL,
                            interval=settings.BYBIT_INTERVAL,
                            lookback_bars=min_bars,
                            category=settings.BYBIT_CATEGORY,
                        )
                        if df.empty or len(df) < min_bars:
                            logger.warning(f"Not enough data ({len(df)}/{min_bars} bars). Waiting...")
                            time.sleep(10)
                            continue

                        # Check signal
                        current_signal_row = df.iloc[-1]
                        signal = current_signal_row['signal']
                        close_price = current_signal_row['close']
                        if self.current_position:
                            self._manage_open_position(current_signal_row, interval_mins)
                        elif signal != 0:
                            self._execute_trade(signal, close_price, current_signal_row)
                        
                        # Sleep
                        time.sleep(10)
                    
                    # Sync stats
                    if allow_real_execution and self.session:
                        self.risk_manager.sync_from_api(self.session)
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    time.sleep(5)
            
            logger.info("Trading loop stopped")
            
        except Exception as e:
            logger.error(f"Fatal error in trading loop: {e}")
            self.status = "error"
    
    def _execute_trade(
        self,
        signal: int,
        close_price: float,
        signal_row: pd.Series,
        symbol: Optional[str] = None,
        strategy_name: Optional[str] = None,
        execution_mode: Optional[str] = None,
        base_interval: Optional[str] = None,
        selector_metrics: Optional[Dict] = None,
    ):
        """Execute a trade based on signal"""
        if close_price <= 0:
            logger.error("Invalid close price for trade execution: %s", close_price)
            return

        target_symbol = symbol or settings.BYBIT_SYMBOL
        strategy_name = strategy_name or "single_symbol_strategy"
        base_interval = base_interval or settings.BYBIT_INTERVAL
        real_orders = self._should_place_real_orders(execution_mode)
        normalized_execution_mode = "live" if real_orders else "paper"
        side = "Buy" if signal == 1 else "Sell"
        base_usdt = float(settings.FIXED_USDT_SIZE)
        size_mult = 1.0
        confidence = 0.0
        if getattr(settings, "POSITION_SIZING_ENABLED", False):
            min_mult = float(getattr(settings, "POSITION_SIZING_MIN_MULT", 0.5))
            max_mult = float(getattr(settings, "POSITION_SIZING_MAX_MULT", 1.5))
            prob = None
            if signal == 1:
                prob = signal_row.get('p_up')
            elif signal == -1:
                prob = signal_row.get('p_down')
            if prob is None or pd.isna(prob):
                confidence = 0.0
            else:
                confidence = abs(float(prob) - 0.5) * 2.0
            size_mult = min_mult + confidence * (max_mult - min_mult)
            size_mult = max(min_mult, min(max_mult, size_mult))
        position_size_usdt = base_usdt * size_mult
        qty = round(position_size_usdt / close_price, 3)
        if qty <= 0:
            logger.warning("Calculated zero quantity for %s %s @ %s", target_symbol, strategy_name, close_price)
            return
        
        # Calculate SL/TP
        atr_col = f'ATR_{settings.ATR_PERIOD}'
        atr = signal_row.get(atr_col)
        if atr is None or pd.isna(atr):
            atr = 10
        sl_dist = atr * settings.SL_ATR_MULT
        tp_dist = atr * settings.RISK_REWARD_RATIO
        
        if side == "Buy":
            sl_price = round(close_price - sl_dist, 2)
            tp_price = round(close_price + tp_dist, 2)
        else:
            sl_price = round(close_price + sl_dist, 2)
            tp_price = round(close_price - tp_dist, 2)

        tp1_price = None
        if getattr(settings, "PARTIAL_TP_ENABLED", True):
            tp1_dist = atr * float(getattr(settings, "PARTIAL_TP_ATR_MULT", 1.0))
            if side == "Buy":
                tp1_price = round(close_price + tp1_dist, 2)
            else:
                tp1_price = round(close_price - tp1_dist, 2)

        signal_ts_ms = self._extract_signal_timestamp_ms(signal_row)

        logger.info(
            "SIGNAL: %s %s (%s) @ %s | SL: %s | TP: %s | mode=%s",
            side,
            target_symbol,
            strategy_name,
            close_price,
            sl_price,
            tp_price,
            normalized_execution_mode,
        )

        trade_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_type': 'entry',
            'symbol': target_symbol,
            'strategy_name': strategy_name,
            'execution_mode': normalized_execution_mode,
            'side': side,
            'entry_price': close_price,
            'qty': qty,
            'sl': sl_price,
            'tp': tp_price,
            'tp1': tp1_price,
            'confidence': confidence,
            'size_mult': size_mult,
            'position_size_usdt': position_size_usdt,
            'base_interval': base_interval,
            'selector_metrics': selector_metrics,
        }

        self.current_position = {
            'symbol': target_symbol,
            'strategy_name': strategy_name,
            'execution_mode': normalized_execution_mode,
            'base_interval': base_interval,
            'side': signal,
            'entry_price': close_price,
            'qty': qty,
            'qty_remaining': qty,
            'sl': sl_price,
            'tp': tp_price,
            'tp1': tp1_price,
            'confidence': confidence,
            'size_mult': size_mult,
            'position_size_usdt': position_size_usdt,
            'tp1_hit': False,
            'trailing_active': False,
            'atr_entry': atr,
            'entry_ts': signal_ts_ms,
            'selector_metrics': selector_metrics,
            'realized_pnl': 0.0,
            'time_stop_pending': False,
            'time_stop_trigger_ts': None,
        }

        if not real_orders:
            logger.info(
                "[PAPER] Order: Symbol=%s, Side=%s, Qty=%s, Entry=%s, SL=%s, TP=%s",
                target_symbol,
                side,
                qty,
                close_price,
                sl_price,
                tp_price,
            )
            trade_record['status'] = 'simulated'
            self.trades_history.append(trade_record)
            self.stats['total_trades'] += 1
            self.stats['daily_trades'] += 1
        else:
            # Execute real order
            try:
                if not self.session:
                    logger.error("Real execution requested but Bybit session is not initialized")
                    self.current_position = None
                    return
                logger.info(f"Placing REAL {side} Order for {target_symbol}...")
                resp = self.session.place_order(
                    category=settings.BYBIT_CATEGORY,
                    symbol=target_symbol,
                    side=side,
                    orderType="Market",
                    qty=str(qty),
                    stopLoss=str(sl_price),
                    takeProfit=str(tp_price),
                    timeInForce="GTC"
                )
                logger.info(f"Order Placed: {resp}")

                trade_record['status'] = 'executed'
                trade_record['order_id'] = resp.get('result', {}).get('orderId')
                self.current_position['order_id'] = trade_record['order_id']
                self.trades_history.append(trade_record)
                self.stats['total_trades'] += 1
                self.stats['daily_trades'] += 1
                
                time.sleep(60)  # Avoid double entry
                
            except Exception as e:
                self.current_position = None
                logger.error(f"Order execution failed: {e}")
