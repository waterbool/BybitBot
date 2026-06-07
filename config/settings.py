import os
import yaml
import logging
from pathlib import Path

# --- Basic Project Settings ---
# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env from project root if present (no hard dependency on python-dotenv)
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# --- Load Config from YAML ---
CONFIG_PATH = Path(os.getenv("BYBIT_BOT_CONFIG_PATH", BASE_DIR / "config.yaml"))

def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

config = load_config()


def _ensure_symbol_list(value, fallback: str) -> list[str]:
    if isinstance(value, list):
        cleaned = [str(item).strip().upper() for item in value if str(item).strip()]
        return cleaned or [fallback]
    if isinstance(value, str):
        cleaned = [part.strip().upper() for part in value.split(",") if part.strip()]
        return cleaned or [fallback]
    return [fallback]


def _ensure_string_list(value, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned or list(fallback)
    if isinstance(value, str):
        cleaned = [part.strip() for part in value.split(",") if part.strip()]
        return cleaned or list(fallback)
    return list(fallback)


def _env_str(name: str, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return int(default)
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return float(default)
    return float(value)

# --- Bybit Config ---
BYBIT_API_KEY = _env_str("BYBIT_API_KEY", config.get("api", {}).get("api_key"))
BYBIT_API_SECRET = _env_str("BYBIT_API_SECRET", config.get("api", {}).get("api_secret"))
BYBIT_TESTNET = _env_bool("BYBIT_TESTNET", config.get("api", {}).get("testnet", True))
BYBIT_DEMO = _env_bool("BYBIT_DEMO", config.get("api", {}).get("demo", False))

BYBIT_SYMBOL = config.get("trading", {}).get("symbol", "ETHUSDT")
BYBIT_CATEGORY = config.get("trading", {}).get("category", "linear")
# Strategy requires 5m by default, can be overridden in config
BYBIT_INTERVAL = str(config.get("strategy", {}).get("timeframe", "5"))

# --- Trading Strategy Config ---
INITIAL_BALANCE = 1000.0  # Optional simulation start balance
EMA_FAST = config.get("strategy", {}).get("ema_fast", 9)
EMA_SLOW = config.get("strategy", {}).get("ema_slow", 21)

# Updated for Trend Strategy: ATR(20)
ATR_PERIOD = _env_int("ATR_PERIOD", config.get("strategy", {}).get("atr_period", 20))

RISK_REWARD_RATIO = _env_float("RISK_REWARD_RATIO", config.get("strategy", {}).get("risk_reward_ratio", 1.5))
SL_ATR_MULTIPLIER = _env_float(
    "SL_ATR_MULT",
    _env_float("SL_ATR_MULTIPLIER", config.get("strategy", {}).get("sl_atr_multiplier", 2.0)),
)
VOLUME_MULTIPLIER = float(config.get("strategy", {}).get("volume_multiplier", 1.2))
VOLUME_MA_PERIOD = int(config.get("strategy", {}).get("volume_ma_period", 20)) 
LEVELS_LOOKBACK = config.get("strategy", {}).get("levels_lookback", 10)
MIN_ATR_THRESHOLD = _env_float("MIN_ATR_THRESHOLD", config.get("strategy", {}).get("min_atr_threshold", 0.0015))
IMPULSE_THRESHOLD = _env_float("IMPULSE_THRESHOLD", config.get("strategy", {}).get("impulse_threshold", 0.002))
COOLDOWN_CANDLES = int(config.get("strategy", {}).get("cooldown_candles", 4))
risk_mgmt_cfg = config.get("risk_management", {})
PARTIAL_TP_ENABLED = _env_bool("PARTIAL_TP_ENABLED", bool(risk_mgmt_cfg.get("partial_tp_enabled", True)))
PARTIAL_TP_ATR_MULT = _env_float(
    "PARTIAL_TP_ATR_MULT",
    risk_mgmt_cfg.get("partial_tp_atr_mult", config.get("strategy", {}).get("tp1_atr_mult", 1.0)),
)
PARTIAL_TP_FRACTION = _env_float(
    "PARTIAL_TP_FRACTION",
    risk_mgmt_cfg.get("partial_tp_fraction", config.get("strategy", {}).get("tp1_partial_pct", 0.5)),
)
BE_ENABLED = _env_bool("BE_ENABLED", bool(risk_mgmt_cfg.get("be_enabled", True)))
BE_BUFFER_BPS = _env_float("BE_BUFFER_BPS", risk_mgmt_cfg.get("be_buffer_bps", 2))
TRAILING_ENABLED = _env_bool("TRAILING_ENABLED", bool(risk_mgmt_cfg.get("trailing_enabled", True)))
TRAIL_ATR_MULT = _env_float(
    "TRAIL_ATR_MULT",
    risk_mgmt_cfg.get("trail_atr_mult", config.get("strategy", {}).get("trail_atr_mult", 1.5)),
)
TRAIL_ACTIVATE_ATR = _env_float(
    "TRAIL_ACTIVATE_ATR",
    risk_mgmt_cfg.get("trail_activation_atr", config.get("strategy", {}).get("trail_activate_atr", 1.0)),
)
TIME_STOP_ENABLED = _env_bool("TIME_STOP_ENABLED", bool(risk_mgmt_cfg.get("time_stop_enabled", True)))
TIME_STOP_CANDLES = _env_int(
    "TIME_STOP_CANDLES",
    risk_mgmt_cfg.get("time_stop_candles", config.get("strategy", {}).get("time_stop_candles", 24)),
)
PREFER_WORST_CASE = _env_bool("PREFER_WORST_CASE", bool(risk_mgmt_cfg.get("prefer_worst_case", True)))

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
RISK_PERCENT = _env_float("RISK_PERCENT", config.get("risk", {}).get("risk_percent", 0.01))

# Risk-based position sizing
# When enabled: position_size = (balance * RISK_PERCENT) / sl_distance
# Capped at balance * MAX_POSITION_PCT
RISK_BASED_SIZING  = _env_bool("RISK_BASED_SIZING",  config.get("risk", {}).get("risk_based_sizing", False))
MAX_POSITION_PCT   = _env_float("MAX_POSITION_PCT",   config.get("risk", {}).get("max_position_pct", 0.10))
PAPER_BALANCE_USDT = _env_float("PAPER_BALANCE_USDT", config.get("risk", {}).get("paper_balance_usdt", 50000.0))

# Trading hours (store as simple integers for start/end hour)
TRADING_START_HOUR = _env_int("TRADING_START_HOUR", config.get('strategy', {}).get('trading_start_hour', 0))
TRADING_END_HOUR = _env_int("TRADING_END_HOUR", config.get('strategy', {}).get('trading_end_hour', 24))

FIXED_USDT_SIZE = _env_float("FIXED_USDT_SIZE", config.get("trading", {}).get("fixed_usdt_size", 2.0))
DRY_RUN = _env_bool("DRY_RUN", config.get("trading", {}).get("dry_run", True))

# --- Strategy Variant Overrides ---
_strategy_cfg = config.get("strategy", {})

FUNDING_EXTREME_LONG_THRESHOLD  = _env_float("FUNDING_EXTREME_LONG_THRESHOLD",  _strategy_cfg.get("funding_extreme_long_threshold",  -0.0003))
FUNDING_EXTREME_SHORT_THRESHOLD = _env_float("FUNDING_EXTREME_SHORT_THRESHOLD", _strategy_cfg.get("funding_extreme_short_threshold",  0.0003))
FUNDING_EXTREME_MIN_OI_CHANGE   = _env_float("FUNDING_EXTREME_MIN_OI_CHANGE",   0.0)
FUNDING_EXTREME_MIN_PRICE_MOVE  = _env_float("FUNDING_EXTREME_MIN_PRICE_MOVE",  0.0)
FUNDING_EXTREME_RSI_MAX_LONG    = _env_float("FUNDING_EXTREME_RSI_MAX_LONG",    _strategy_cfg.get("funding_extreme_rsi_max_long",  45.0))
FUNDING_EXTREME_RSI_MIN_SHORT   = _env_float("FUNDING_EXTREME_RSI_MIN_SHORT",   _strategy_cfg.get("funding_extreme_rsi_min_short", 55.0))
FUNDING_EXTREME_COOLDOWN_BARS   = _env_int(  "FUNDING_EXTREME_COOLDOWN_BARS",   _strategy_cfg.get("funding_extreme_cooldown_bars",  16))

MEAN_REV_EMA_PERIOD = _env_int("MEAN_REV_EMA_PERIOD", 50)
MEAN_REV_RSI_PERIOD = _env_int("MEAN_REV_RSI_PERIOD", 14)
MEAN_REV_BB_PERIOD = _env_int("MEAN_REV_BB_PERIOD", 20)
MEAN_REV_BB_STD = _env_float("MEAN_REV_BB_STD", 2.0)
MEAN_REV_ATR_PERIOD = _env_int("MEAN_REV_ATR_PERIOD", 14)
MEAN_REV_MIN_ATR_PCT = _env_float("MEAN_REV_MIN_ATR_PCT", 0.0015)
MEAN_REV_RSI_LONG = _env_float("MEAN_REV_RSI_LONG", 30.0)
MEAN_REV_RSI_SHORT = _env_float("MEAN_REV_RSI_SHORT", 70.0)

VOL_COMP_EMA_PERIOD = _env_int("VOL_COMP_EMA_PERIOD", 50)
VOL_COMP_ATR_PERIOD = _env_int("VOL_COMP_ATR_PERIOD", 14)
VOL_COMP_ATR_MA_PERIOD = _env_int("VOL_COMP_ATR_MA_PERIOD", 50)
VOL_COMP_BB_PERIOD = _env_int("VOL_COMP_BB_PERIOD", 20)
VOL_COMP_BB_STD = _env_float("VOL_COMP_BB_STD", 2.0)
VOL_COMP_BB_WIDTH_MULT = _env_float("VOL_COMP_BB_WIDTH_MULT", 1.10)

MTF_PULLBACK_EMA_PERIOD    = _env_int(  "MTF_PULLBACK_EMA_PERIOD",    _strategy_cfg.get("mtf_pullback_ema_period",    50))
MTF_PULLBACK_RSI_PERIOD    = _env_int(  "MTF_PULLBACK_RSI_PERIOD",    14)
MTF_PULLBACK_RSI_LONG      = _env_float("MTF_PULLBACK_RSI_LONG",      _strategy_cfg.get("mtf_pullback_rsi_long",      40.0))
MTF_PULLBACK_RSI_SHORT     = _env_float("MTF_PULLBACK_RSI_SHORT",     _strategy_cfg.get("mtf_pullback_rsi_short",     60.0))
MTF_PULLBACK_MIN_DEPTH     = _env_float("MTF_PULLBACK_MIN_DEPTH",     _strategy_cfg.get("mtf_pullback_min_depth",     0.003))
MTF_PULLBACK_VOL_MULT      = _env_float("MTF_PULLBACK_VOL_MULT",      _strategy_cfg.get("mtf_pullback_vol_mult",      1.0))
MTF_PULLBACK_COOLDOWN_BARS = _env_int(  "MTF_PULLBACK_COOLDOWN_BARS", _strategy_cfg.get("mtf_pullback_cooldown_bars", 8))

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

# --- Live Multi-Symbol Selector ---
live_selector_cfg = config.get("live_selector", {})
LIVE_SELECTOR_ENABLED = _env_bool("LIVE_SELECTOR_ENABLED", bool(live_selector_cfg.get("enabled", False)))
LIVE_SELECTOR_SYMBOLS = _ensure_symbol_list(_env_str("LIVE_SELECTOR_SYMBOLS", live_selector_cfg.get("symbols")), BYBIT_SYMBOL)
LIVE_SELECTOR_STRATEGIES = _ensure_string_list(
    _env_str("LIVE_SELECTOR_STRATEGIES", live_selector_cfg.get("strategies")),
    ["ema_crossover_baseline", "funding_extreme_reversal"],
)
LIVE_SELECTOR_EXECUTION_MODE = str(_env_str("LIVE_SELECTOR_EXECUTION_MODE", live_selector_cfg.get("execution_mode", "paper"))).strip().lower()
if LIVE_SELECTOR_EXECUTION_MODE not in {"paper", "live"}:
    LIVE_SELECTOR_EXECUTION_MODE = "paper"
LIVE_SELECTOR_EDGE_SNAPSHOT_PATH = str(_env_str(
    "LIVE_SELECTOR_EDGE_SNAPSHOT_PATH",
    live_selector_cfg.get("edge_snapshot_path", BASE_DIR / "reports" / "live_edge_snapshot.json"),
))
LIVE_SELECTOR_EDGE_LOOKBACK_DAYS = _env_int("LIVE_SELECTOR_EDGE_LOOKBACK_DAYS", live_selector_cfg.get("edge_lookback_days", 90))
LIVE_SELECTOR_EDGE_MAX_AGE_MINUTES = _env_int("LIVE_SELECTOR_EDGE_MAX_AGE_MINUTES", live_selector_cfg.get("edge_max_age_minutes", 24 * 60))
LIVE_SELECTOR_REQUIRE_EDGE_SNAPSHOT = _env_bool("LIVE_SELECTOR_REQUIRE_EDGE_SNAPSHOT", bool(live_selector_cfg.get("require_edge_snapshot", True)))
LIVE_SELECTOR_USE_ML = _env_bool("LIVE_SELECTOR_USE_ML", bool(live_selector_cfg.get("use_ml", False)))
LIVE_SELECTOR_SIGNAL_WEIGHT = _env_float("LIVE_SELECTOR_SIGNAL_WEIGHT", live_selector_cfg.get("signal_weight", 0.4))
LIVE_SELECTOR_EDGE_WEIGHT = _env_float("LIVE_SELECTOR_EDGE_WEIGHT", live_selector_cfg.get("edge_weight", 0.6))
LIVE_SELECTOR_MIN_SIGNAL_SCORE = _env_float("LIVE_SELECTOR_MIN_SIGNAL_SCORE", live_selector_cfg.get("min_signal_score", 0.60))
LIVE_SELECTOR_MIN_EDGE_SCORE = _env_float("LIVE_SELECTOR_MIN_EDGE_SCORE", live_selector_cfg.get("min_edge_score", 0.55))
LIVE_SELECTOR_MAX_NEW_TRADES_PER_DAY = _env_int("LIVE_SELECTOR_MAX_NEW_TRADES_PER_DAY", live_selector_cfg.get("max_new_trades_per_day", 2))
LIVE_SELECTOR_MAX_PER_SYMBOL_PER_DAY = _env_int("LIVE_SELECTOR_MAX_PER_SYMBOL_PER_DAY", live_selector_cfg.get("max_per_symbol_per_day", 1))
LIVE_SELECTOR_MAX_PER_STRATEGY_PER_DAY = _env_int("LIVE_SELECTOR_MAX_PER_STRATEGY_PER_DAY", live_selector_cfg.get("max_per_strategy_per_day", 1))
LIVE_SELECTOR_MAX_POSITIONS_TOTAL = _env_int("LIVE_SELECTOR_MAX_POSITIONS_TOTAL", live_selector_cfg.get("max_positions_total", 1))
LIVE_SELECTOR_MAX_POSITIONS_PER_SYMBOL = _env_int("LIVE_SELECTOR_MAX_POSITIONS_PER_SYMBOL", live_selector_cfg.get("max_positions_per_symbol", 1))
LIVE_SELECTOR_MAX_POSITIONS_PER_STRATEGY = _env_int("LIVE_SELECTOR_MAX_POSITIONS_PER_STRATEGY", live_selector_cfg.get("max_positions_per_strategy", 1))
LIVE_SELECTOR_BASE_INTERVAL = str(_env_str("LIVE_SELECTOR_BASE_INTERVAL", live_selector_cfg.get("base_interval", "15")))
LIVE_SELECTOR_HTF_INTERVAL = str(_env_str("LIVE_SELECTOR_HTF_INTERVAL", live_selector_cfg.get("htf_interval", "60")))
LIVE_SELECTOR_BASE_LOOKBACK_BARS = _env_int("LIVE_SELECTOR_BASE_LOOKBACK_BARS", live_selector_cfg.get("base_lookback_bars", 500))
LIVE_SELECTOR_HTF_LOOKBACK_BARS = _env_int("LIVE_SELECTOR_HTF_LOOKBACK_BARS", live_selector_cfg.get("htf_lookback_bars", 260))
LIVE_SELECTOR_STALE_DATA_MINUTES = _env_int("LIVE_SELECTOR_STALE_DATA_MINUTES", live_selector_cfg.get("stale_data_minutes", 45))
LIVE_SELECTOR_MAX_SPREAD_BPS = _env_float("LIVE_SELECTOR_MAX_SPREAD_BPS", live_selector_cfg.get("max_spread_bps", 8.0))
LIVE_SELECTOR_SCAN_INTERVAL_SECONDS = _env_int("LIVE_SELECTOR_SCAN_INTERVAL_SECONDS", live_selector_cfg.get("scan_interval_seconds", 30))
LIVE_SELECTOR_ADJUST_QTY_TO_EXCHANGE_MIN = _env_bool(
    "LIVE_SELECTOR_ADJUST_QTY_TO_EXCHANGE_MIN",
    bool(live_selector_cfg.get("adjust_qty_to_exchange_min", False)),
)


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
