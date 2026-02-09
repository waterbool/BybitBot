import os
import yaml
import logging
from pathlib import Path

# --- Basic Project Settings ---
# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Load Config from YAML ---
CONFIG_PATH = BASE_DIR / "config.yaml"

def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# --- Bybit Config ---
BYBIT_API_KEY = config.get("api", {}).get("api_key")
BYBIT_API_SECRET = config.get("api", {}).get("api_secret")
BYBIT_TESTNET = config.get("api", {}).get("testnet", True)

BYBIT_SYMBOL = config.get("trading", {}).get("symbol", "ETHUSDT")
BYBIT_CATEGORY = config.get("trading", {}).get("category", "linear")
# Strategy requires 5m by default, can be overridden in config
BYBIT_INTERVAL = str(config.get("strategy", {}).get("timeframe", "5"))

# --- Trading Strategy Config ---
INITIAL_BALANCE = 1000.0  # Optional simulation start balance
EMA_FAST = config.get("strategy", {}).get("ema_fast", 9)
EMA_SLOW = config.get("strategy", {}).get("ema_slow", 21)

# Updated for Trend Strategy: ATR(20)
ATR_PERIOD = config.get("strategy", {}).get("atr_period", 20) 

RISK_REWARD_RATIO = config.get("strategy", {}).get("risk_reward_ratio", 1.5)
SL_ATR_MULTIPLIER = config.get("strategy", {}).get("sl_atr_multiplier", 2.0)
VOLUME_MULTIPLIER = float(config.get("strategy", {}).get("volume_multiplier", 1.2))
VOLUME_MA_PERIOD = int(config.get("strategy", {}).get("volume_ma_period", 20)) 
LEVELS_LOOKBACK = config.get("strategy", {}).get("levels_lookback", 10)
MIN_ATR_THRESHOLD = float(config.get("strategy", {}).get("min_atr_threshold", 0.0015))
IMPULSE_THRESHOLD = float(config.get("strategy", {}).get("impulse_threshold", 0.002))
COOLDOWN_CANDLES = int(config.get("strategy", {}).get("cooldown_candles", 4))
risk_mgmt_cfg = config.get("risk_management", {})
PARTIAL_TP_ENABLED = bool(risk_mgmt_cfg.get("partial_tp_enabled", True))
PARTIAL_TP_ATR_MULT = float(
    risk_mgmt_cfg.get("partial_tp_atr_mult", config.get("strategy", {}).get("tp1_atr_mult", 1.0))
)
PARTIAL_TP_FRACTION = float(
    risk_mgmt_cfg.get("partial_tp_fraction", config.get("strategy", {}).get("tp1_partial_pct", 0.5))
)
BE_ENABLED = bool(risk_mgmt_cfg.get("be_enabled", True))
BE_BUFFER_BPS = float(risk_mgmt_cfg.get("be_buffer_bps", 2))
TRAILING_ENABLED = bool(risk_mgmt_cfg.get("trailing_enabled", True))
TRAIL_ATR_MULT = float(
    risk_mgmt_cfg.get("trail_atr_mult", config.get("strategy", {}).get("trail_atr_mult", 1.5))
)
TRAIL_ACTIVATE_ATR = float(
    risk_mgmt_cfg.get("trail_activation_atr", config.get("strategy", {}).get("trail_activate_atr", 1.0))
)
TIME_STOP_ENABLED = bool(risk_mgmt_cfg.get("time_stop_enabled", True))
TIME_STOP_CANDLES = int(
    risk_mgmt_cfg.get("time_stop_candles", config.get("strategy", {}).get("time_stop_candles", 24))
)
PREFER_WORST_CASE = bool(risk_mgmt_cfg.get("prefer_worst_case", True))

# Backward-compatible aliases
TP1_ATR_MULT = PARTIAL_TP_ATR_MULT
TP1_PARTIAL_PCT = PARTIAL_TP_FRACTION

# --- ML Filter Settings ---
ml_cfg = config.get("ml", {})
ML_ENABLED = bool(ml_cfg.get("enabled", True))
ML_HORIZON = int(ml_cfg.get("horizon_candles", 12))
ML_MIN_PROB = float(ml_cfg.get("probability_threshold", 0.60))
ML_FLAT_FILTER = bool(ml_cfg.get("flat_filter", True))
ML_MODEL_PATH = str(ml_cfg.get("model_path", BASE_DIR / "models" / "ethusdt_5m_lgbm.pkl"))

# --- Position Sizing ---
position_sizing_cfg = config.get("position_sizing", {})
POSITION_SIZING_ENABLED = bool(position_sizing_cfg.get("enabled", False))
POSITION_SIZING_MIN_MULT = float(position_sizing_cfg.get("min_size_mult", 0.5))
POSITION_SIZING_MAX_MULT = float(position_sizing_cfg.get("max_size_mult", 1.5))

# --- Market Regime ---
regime_cfg = config.get("regime", {})
REGIME_ENABLED = bool(regime_cfg.get("enabled", True))
ATR_REGIME_THRESHOLD = float(regime_cfg.get("atr_regime_threshold", 0.0018))
ADX_THRESHOLD = float(regime_cfg.get("adx_threshold", 20))

# Backward compatibility (if old strategy.* ML keys exist)
ML_FLAT_THRESHOLD = float(config.get("strategy", {}).get("ml_flat_threshold", 0.002))
ML_MIN_TRAIN_SAMPLES = int(config.get("strategy", {}).get("ml_min_train_samples", 300))
ML_TRAIN_WINDOW = int(config.get("strategy", {}).get("ml_train_window", 2000))

# Aliases for compatibility
VOLUME_LOOKBACK = VOLUME_MA_PERIOD
SL_ATR_MULT = SL_ATR_MULTIPLIER

# New Strategy Settings
SMA_TREND_PERIOD = 200
DONCHIAN_PERIOD = 7
RISK_PERCENT = config.get("risk", {}).get("risk_percent", 0.01) # 1% default

# Trading hours (store as simple integers for start/end hour)
TRADING_START_HOUR = int(config.get('strategy', {}).get('trading_start_hour', 0))
TRADING_END_HOUR = int(config.get('strategy', {}).get('trading_end_hour', 24))

FIXED_USDT_SIZE = config.get("trading", {}).get("fixed_usdt_size", 2.0)
DRY_RUN = config.get("trading", {}).get("dry_run", True)

# --- Risk Management ---
MAX_TRADES_PER_DAY = config.get("risk", {}).get("max_trades_per_day", 10)
MAX_DAILY_LOSS_USDT = config.get("risk", {}).get("max_daily_loss_usdt", 10.0)

# --- Backtest Settings ---
backtest_cfg = config.get("backtest", {})
BACKTEST_TAKER_FEE = float(backtest_cfg.get("taker_fee", 0.0006))
BACKTEST_MAKER_FEE = float(backtest_cfg.get("maker_fee", 0.0002))
BACKTEST_USE_TAKER = bool(backtest_cfg.get("use_taker", True))
BACKTEST_SLIPPAGE_BPS = float(backtest_cfg.get("slippage_bps", 2))
BACKTEST_EXECUTION_DELAY_CANDLES = int(backtest_cfg.get("execution_delay_candles", 1))


# --- Binarium Config (Legacy/Hybrid) ---
BINARIUM_EMAIL = os.getenv("BINARIUM_EMAIL", None)
BINARIUM_PASSWORD = os.getenv("BINARIUM_PASSWORD", None)
BINARIUM_DEFAULT_ASSET_NAME = "CRYPTO IDX" 

# --- System / Logging ---
DEBUG = True
LOGS_DIR = BASE_DIR / "logs"

# Ensure logs directory exists
try:
    LOGS_DIR.mkdir(exist_ok=True)
except Exception as e:
    print(f"Warning: Could not create logs directory at {LOGS_DIR}: {e}")

# --- Validation and Warnings ---
if DEBUG:
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
else:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

if MAX_DAILY_LOSS_USDT <= 0:
    logging.warning("⚠️  MAX_DAILY_LOSS_USDT is set to <= 0. Check if this is intended.")

logging.info(f"Settings loaded. Symbol: {BYBIT_SYMBOL}, Dry Run: {DRY_RUN}")
