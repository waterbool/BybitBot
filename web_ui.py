"""
Web UI for Bybit Trading Bot
Flask-based web interface with REST API and WebSocket support
"""
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import yaml
import logging
import threading
import time
from pathlib import Path
from datetime import datetime

from bot_controller import BotController
from config import settings
from backtest.backtester import run_backtest
from data_fetch.bybit_client import fetch_historical_klines
from indicators.ta_module import add_indicators
from strategy.rules import apply_strategy
from ml.model import add_ml_probabilities

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='static')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize bot controller
bot_controller = BotController()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============= Web Routes =============

@app.route('/')
def index():
    """Serve main UI"""
    return send_from_directory('static', 'index.html')


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    try:
        config_path = Path(settings.BASE_DIR) / 'config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return jsonify({"success": True, "config": config})
    except Exception as e:
        logger.error(f"Error reading config: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration"""
    try:
        new_config = request.json
        config_path = Path(settings.BASE_DIR) / 'config.yaml'
        
        # Backup current config
        backup_path = config_path.with_suffix('.yaml.backup')
        with open(config_path, 'r') as f:
            backup_content = f.read()
        with open(backup_path, 'w') as f:
            f.write(backup_content)
        
        # Write new config
        with open(config_path, 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False)
        
        logger.info("Configuration updated successfully")
        return jsonify({"success": True, "message": "Configuration updated. Restart bot to apply changes."})
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/backtest', methods=['POST'])
def run_backtest_api():
    """Run backtest with parameters"""
    try:
        params = request.json
        
        # Get parameters
        days = params.get('days', 7)
        initial_balance = params.get('initial_balance', 1000.0)
        fixed_size = params.get('fixed_size', 2.0)
        payout_rate = params.get('win_rate', 0.85)
        expiry_minutes = params.get('expiry_minutes', 1)
        
        # Fetch data
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        logger.info(f"Running backtest for {days} days...")
        df = fetch_historical_klines(settings.BYBIT_SYMBOL, settings.BYBIT_INTERVAL, start_time, end_time)
        
        if df.empty:
            return jsonify({"success": False, "error": "No data fetched"}), 400
        
        # Add indicators and strategy
        df = add_indicators(
            df,
            ema_fast=settings.EMA_FAST,
            ema_slow=settings.EMA_SLOW,
            atr_period=settings.ATR_PERIOD
        )
        if getattr(settings, 'ML_ENABLED', False):
            df = add_ml_probabilities(df)
        df = apply_strategy(df)
        
        # Run backtest (strategy + ML + TP/SL)
        taker_fee_rate = float(params.get('taker_fee_rate', 0.00055))
        slippage_rate = float(params.get('slippage_rate', 0.0002))
        funding_rate_per_bar = float(params.get('funding_rate_per_bar', 0.0))

        metrics, trades_df, equity_df, monthly_stats = run_backtest(
            df,
            initial_balance,
            taker_fee_rate=taker_fee_rate,
            slippage_rate=slippage_rate,
            funding_rate_per_bar=funding_rate_per_bar,
        )
        
        # Prepare chart data
        chart_data = {
            'timestamps': df['timestamp'].tail(100).tolist(),
            'close': df['close'].tail(100).tolist(),
            'ema_fast': df[f'EMA_{settings.EMA_FAST}'].tail(100).tolist() if f'EMA_{settings.EMA_FAST}' in df.columns else [],
            'ema_slow': df[f'EMA_{settings.EMA_SLOW}'].tail(100).tolist() if f'EMA_{settings.EMA_SLOW}' in df.columns else [],
            'signals': df['signal'].tail(100).tolist()
        }
        
        # Prepare trades data
        trades_list = []
        if trades_df is not None and not trades_df.empty:
            trades_list = trades_df.tail(50).to_dict('records')

        equity_curve = []
        if equity_df is not None and not equity_df.empty:
            equity_curve = equity_df.tail(1000).to_dict('records')

        monthly = []
        if monthly_stats is not None and not monthly_stats.empty:
            monthly = monthly_stats.to_dict('records')
        
        logger.info(f"Backtest completed: {metrics}")
        
        return jsonify({
            "success": True,
            "metrics": metrics,
            "chart_data": chart_data,
            "trades": trades_list,
            "equity_curve": equity_curve,
            "monthly_stats": monthly
        })
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/trading/start', methods=['POST'])
def start_trading():
    """Start live trading"""
    result = bot_controller.start_trading()
    return jsonify(result)


@app.route('/api/trading/stop', methods=['POST'])
def stop_trading():
    """Stop live trading"""
    result = bot_controller.stop_trading()
    return jsonify(result)


@app.route('/api/trading/status', methods=['GET'])
def get_status():
    """Get bot status and statistics"""
    status = bot_controller.get_status()
    return jsonify(status)


@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get recent logs"""
    count = request.args.get('count', 100, type=int)
    logs = bot_controller.get_recent_logs(count)
    return jsonify({"success": True, "logs": logs})


@app.route('/api/trades/history', methods=['GET'])
def get_trade_history():
    """Get trade history"""
    limit = request.args.get('limit', 50, type=int)
    trades = bot_controller.trades_history[-limit:]
    return jsonify({"success": True, "trades": trades})


# ============= WebSocket Events =============

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info("Client connected to WebSocket")
    emit('connected', {'message': 'Connected to bot'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected from WebSocket")


@socketio.on('subscribe_logs')
def handle_subscribe_logs():
    """Subscribe to real-time logs"""
    logger.info("Client subscribed to logs")
    
    def send_logs():
        """Send logs to client periodically"""
        while True:
            try:
                logs = bot_controller.get_recent_logs(10)
                if logs:
                    socketio.emit('logs_update', {'logs': logs})
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error sending logs: {e}")
                break
    
    # Start log streaming in background
    thread = threading.Thread(target=send_logs, daemon=True)
    thread.start()


@socketio.on('subscribe_status')
def handle_subscribe_status():
    """Subscribe to real-time status updates"""
    logger.info("Client subscribed to status")
    
    def send_status():
        """Send status to client periodically"""
        while True:
            try:
                status = bot_controller.get_status()
                socketio.emit('status_update', status)
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error sending status: {e}")
                break
    
    # Start status streaming in background
    thread = threading.Thread(target=send_status, daemon=True)
    thread.start()


# ============= Main =============

def main():
    """Start web server"""
    logger.info("=" * 60)
    logger.info("Bybit Trading Bot - Web UI")
    logger.info("=" * 60)
    logger.info(f"Server starting on http://localhost:5001")
    logger.info(f"Symbol: {settings.BYBIT_SYMBOL}")
    logger.info(f"Dry Run: {settings.DRY_RUN}")
    logger.info(f"Testnet: {settings.BYBIT_TESTNET}")
    logger.info("=" * 60)
    
    # Run server
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
