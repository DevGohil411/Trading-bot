import os
import sys
import logging
import json
from datetime import datetime
from datetime import timezone
import pytz
import warnings
import time
import math
import numpy as np
import pandas as pd
import backoff
import threading
from pathlib import Path
import functools
import traceback
import csv
import shutil
import io
import sys
import os
from collections import defaultdict
import math
import requests
import logging
import threading
import time
import json
import warnings
from datetime import datetime, timedelta, timezone
import pytz
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import backoff

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
    print("[SUCCESS] Environment variables loaded from .env file")
except ImportError:
    print("[WARNING] python-dotenv not installed, using system environment variables only")
except Exception as e:
    print(f"[WARNING] Error loading .env file: {e}")

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from STOCKDATA.file import (
    active_trades, data_lock, last_trade_candle, partial_trade_tracking_map,
    setup_file_logger, ensure_trades_table_exists, log_trade_to_csv, update_trade_in_csv,
    place_order, get_account_equity_for_file_py, save_active_trades_for_file_py,
    load_active_trades_from_file_py, manage_open_positions, calculate_dynamic_lot,
    fast_trailing_sl, split_lot, calculate_momentum, calculate_liquidity_sl, calculate_tp,
    validate_lot_size, is_market_open, round_lot, get_dynamic_lot, has_open_trade,
    has_opposite_trade, get_recent_trade_performance, send_telegram_alert,
    get_forex_factory_news, news_filter, process_strategy_signal,
    mmxm, msb_retest, order_block, amd_strategy, judas_swing, mmc_strategy,
    MMCXAUUSDStrategy, OTEStrategy,
    init_llm_sentiment_analyzer, init_ml_trade_filter,
    init_telegram_settings, daily_trade_counts, last_executed_candle_for_strategy,
    save_last_executed_candle_state_for_file_py,
    load_last_executed_candle_state_from_file_py,
    check_trade_limits, repair_trade_log_csv, update_trailing_stops
)
# Try importing modules using relative imports (works when running as a module)
# If that fails, try absolute imports (works when running as a script)
try:
    from .modules.indicators import calculate_atr, calculate_rsi
    from .modules.ml_filter import MLTradeFilter
    from .modules.llm_sentiment_analyzer import LLMSentimentAnalyzer
except (ImportError, ValueError):
    # This will work when running as a script
    from modules.indicators import calculate_atr, calculate_rsi
    from modules.ml_filter import MLTradeFilter
    from modules.llm_sentiment_analyzer import LLMSentimentAnalyzer


# Initialize main logger at the very top
logger = None
def setup_main_logger():
    # Simplified logger to print directly to console
    logger = logging.getLogger('main_bot')
    logger.setLevel(logging.INFO)
    # If handlers are already present, clear them
    if logger.hasHandlers():
        logger.handlers.clear()
    # Add a stream handler to output to console, forcing UTF-8
    try:
        # Try to wrap stdout in a UTF-8 wrapper to prevent UnicodeEncodeError on Windows
        utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        stream_handler = logging.StreamHandler(utf8_stdout)
    except (TypeError, ValueError, AttributeError):
        # Fallback for environments where stdout has no buffer (e.g., some IDEs, redirection)
        stream_handler = logging.StreamHandler(sys.stdout)
        
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False # Prevent duplicate logs
    logger.info("Main logger initialized for console output.")
    return logger

logger = setup_main_logger()

def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    if logger:
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    else:
        print("Uncaught exception (logger not initialized):", exc_type, exc_value)
        traceback.print_exception(exc_type, exc_value, exc_traceback)

sys.excepthook = handle_uncaught_exception

def validate_config_lot_size(lot):
    min_lot, max_lot = 0.01, 50.0
    if not isinstance(lot, (int, float)) or lot < min_lot or lot > max_lot:
        if logger: logger.error(f"Invalid lot size in config: {lot}. Must be between {min_lot} and {max_lot}.")
        return False
    return True

def validate_risk_percent(risk):
    if not isinstance(risk, (int, float)) or risk <= 0 or risk > 100:
        if logger: logger.error(f"Invalid risk percent in config: {risk}. Must be >0 and <=100.")
        return False
    return True

def validate_symbol_in_config(symbol, allowed_symbols):
    if symbol not in allowed_symbols:
        if logger: logger.error(f"Invalid symbol in config: {symbol}. Allowed: {allowed_symbols}")
        return False
    return True

warnings.filterwarnings("ignore", category=FutureWarning)

TRADE_TIMES_FILE = "../last_trade_times_main.json" # FIX: Path adjusted for main.py being in STOCKDATA
DAILY_PNL_FILE = '../logs/daily_pnl.json' # FIX: Path adjusted
TRADE_COUNT_FILE = '../logs/daily_trade_count.json' # FIX: Path adjusted

last_trade_times_main = {}
daily_pnl_data = {}
daily_trade_counts_main = {}

# --- Global Variables for Strategy Management ---
bot_startup_time = None
STARTUP_TIME = datetime.now(timezone.utc)
strategy_cooldowns = {}  # {(symbol, strategy_name): last_execution_time}

# Per-strategy cooldown settings (in seconds)
STRATEGY_COOLDOWNS = {
    "judas_swing": 300,      # 5 minutes
    "mmc_combo_strategy": 300, # 5 minutes
    "order_block": 120,      # 2 minutes
    "ote": 120,              # 2 minutes
    "msb_retest": 180,       # 3 minutes
    "mmxm": 180,             # 3 minutes
    "amd_strategy": 180      # 3 minutes
}

# Updated strategy priority mapping
strategy_priority = {
    # Priority 1 (run always)
    # none configured by user request

    # Priority 2 (run up to 3 concurrently)
    "judas_swing": 2,
    "mmxm": 2,
    "mmc_combo_strategy": 2,

    # Priority 3 (run up to 2 concurrently)
    "msb_retest": 3,
    "amd_strategy": 3,
    "order_block": 3,
    "ote": 3,
}

# Special handling for order_block and ote (both are low priority)
LOW_PRIORITY_STRATEGIES = {"order_block", "ote"}

def get_today_str():
    return datetime.now(pytz.UTC).strftime('%Y-%m-%d')

def load_daily_pnl_main():
    global daily_pnl_data
    if os.path.exists(DAILY_PNL_FILE):
        try:
            with open(DAILY_PNL_FILE, 'r') as f:
                daily_pnl_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading daily pnl from {DAILY_PNL_FILE}: {str(e)}")
            daily_pnl_data = {}
    return daily_pnl_data

def save_daily_pnl_main():
    try:
        os.makedirs(os.path.dirname(DAILY_PNL_FILE) or '.', exist_ok=True)
        with open(DAILY_PNL_FILE, 'w') as f:
            json.dump(daily_pnl_data, f)
    except Exception as e:
        logger.error(f"Error saving daily pnl to {DAILY_PNL_FILE}: {str(e)}")

def update_daily_pnl_main(pnl):
    global daily_pnl_data
    today = get_today_str()
    if today not in daily_pnl_data:
        daily_pnl_data[today] = 0.0
    daily_pnl_data[today] = round(daily_pnl_data[today] + pnl, 2)
    save_daily_pnl_main()

def get_daily_pnl_main():
    today = get_today_str()
    return daily_pnl_data.get(today, 0.0)

def load_trade_count_main():
    global daily_trade_counts_main
    if os.path.exists(TRADE_COUNT_FILE):
        try:
            with open(TRADE_COUNT_FILE, 'r') as f:
                daily_trade_counts_main = json.load(f)
        except Exception as e:
            logger.error(f"Error loading daily trade count from {TRADE_COUNT_FILE}: {str(e)}")
            daily_trade_counts_main = {}
    return daily_trade_counts_main

def save_trade_count_main():
    try:
        os.makedirs(os.path.dirname(TRADE_COUNT_FILE) or '.', exist_ok=True)
        with open(TRADE_COUNT_FILE, 'w') as f:
            json.dump(daily_trade_counts_main, f)
    except Exception as e:
        logger.error(f"Error saving daily trade count to {TRADE_COUNT_FILE}: {str(e)}")

def increment_total_trade_count_main(symbol):
    global daily_trade_counts_main
    today = get_today_str()

    if today not in daily_trade_counts_main:
        daily_trade_counts_main[today] = {}
    if symbol not in daily_trade_counts_main[today]:
        daily_trade_counts_main[today][symbol] = 0

    daily_trade_counts_main[today][symbol] += 1
    save_trade_count_main()
    logger.info(f"Global trade count for {symbol} today: {daily_trade_counts_main[today][symbol]}")

def get_total_trade_count_main(symbol):
    today = get_today_str()
    return daily_trade_counts_main.get(today, {}).get(symbol, 0)

def check_max_daily_loss_global(max_loss_limit_usd):
    current_daily_pnl = get_daily_pnl_main()
    if current_daily_pnl < -abs(max_loss_limit_usd):
        logger.warning(f"Daily loss limit of -{max_loss_limit_usd}$ reached. Current daily PnL: {current_daily_pnl}$. Trading paused globally for the day.")
        send_telegram_alert(f"[ALERT] GLOBAL ALERT: Daily Loss Limit Reached! Trading paused for the day. Current PnL: ${current_daily_pnl:.2f}")
        return False
    return True

def check_max_trades_per_day_global(symbol, max_trades_limit):
    current_trade_count = get_total_trade_count_main(symbol)
    if current_trade_count >= max_trades_limit:
        logger.warning(f"Max total trades per day ({max_trades_limit}) reached for {symbol}. Trading paused for this symbol globally for the day.")
        return False
    return True

def check_strategy_cooldown(symbol, strategy_name):
    """
    Check if strategy is in cooldown period for the given symbol.
    Returns True if strategy can execute, False if in cooldown.
    """
    global strategy_cooldowns
    
    key = (symbol, strategy_name)
    if key not in strategy_cooldowns:
        return True
    
    last_execution = strategy_cooldowns[key]
    cooldown_seconds = STRATEGY_COOLDOWNS.get(strategy_name, 180)  # Default to 3 minutes
    time_since_last = (datetime.now(timezone.utc) - last_execution).total_seconds()
    
    if time_since_last < cooldown_seconds:
        remaining = cooldown_seconds - time_since_last
        logger.info(f"[COOLDOWN] {strategy_name} for {symbol}: {remaining:.0f}s remaining in cooldown")
        return False
    
    return True

def update_strategy_cooldown(symbol, strategy_name):
    """Update the cooldown timestamp for a strategy after successful execution."""
    global strategy_cooldowns
    strategy_cooldowns[(symbol, strategy_name)] = datetime.now(timezone.utc)
    cooldown_seconds = STRATEGY_COOLDOWNS.get(strategy_name, 180)
    logger.info(f"[COOLDOWN] Updated cooldown for {strategy_name} on {symbol} ({cooldown_seconds}s)")

def get_strategy_confluence_score(strategy_name, signal_result):
    """
    Calculate a confluence score for order_block and ote strategies.
    Higher score means stronger signal.
    """
    if strategy_name not in LOW_PRIORITY_STRATEGIES:
        return 0
    
    # Base score from signal strength indicators
    score = 0
    
    # Add points for various signal strength indicators
    if signal_result.get("success", False):
        score += 10
        
        # Add points for specific indicators if available
        if "features" in signal_result and signal_result["features"]:
            score += len(signal_result["features"]) * 2
        
        # Add points for price action confirmation if available
        if "price_action_score" in signal_result:
            score += signal_result["price_action_score"]
    
    return score

def select_best_strategy_signals(symbol, strategy_signals):
    """
    Select multiple strategy signals based on priority with caps.
    - Priority 1: run all
    - Priority 2: up to 3
    - Priority 3: up to 2 (pick highest confluence)
    Returns list of selected signals (can be empty).
    """
    if not strategy_signals:
        return []

    # Group signals by priority
    priority_groups = {}
    for sig in strategy_signals:
        priority_groups.setdefault(sig["priority"], []).append(sig)

    selected = []
    for priority in sorted(priority_groups.keys()):
        signals = priority_groups[priority]
        if priority == 1:
            selected.extend(signals)
            logger.info(f"[PRIORITY SELECTION] Selected all {len(signals)} priority-1 signals for {symbol}")
        elif priority == 2:
            # up to 3; sort by confluence desc
            top = sorted(signals, key=lambda x: x.get("confluence_score", 0), reverse=True)[:3]
            selected.extend(top)
            logger.info(f"[PRIORITY SELECTION] Selected {len(top)} of {len(signals)} priority-2 signals for {symbol}")
        elif priority == 3:
            # up to 2; sort by confluence desc
            top = sorted(signals, key=lambda x: x.get("confluence_score", 0), reverse=True)[:2]
            selected.extend(top)
            logger.info(f"[PRIORITY SELECTION] Selected {len(top)} of {len(signals)} priority-3 signals for {symbol}")

    return selected

def load_config():
    try:
        # Construct an absolute path to the config file in the project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        allowed_symbols = config.get("symbols", [])
        for symbol in allowed_symbols:
            if not validate_symbol_in_config(symbol, allowed_symbols):
                pass

        if "mt5" not in config or not all(k in config["mt5"] for k in ["login", "password", "server"]):
            raise ValueError("MT5 credentials missing in config.json")

        risk_settings = config.get("risk_settings", {})
        if not validate_risk_percent(risk_settings.get("risk_per_trade", 0.0)):
            logger.warning("Invalid 'risk_per_trade' in config. Using default.")
        if not isinstance(risk_settings.get("max_daily_loss", 0), (int, float)) or risk_settings.get("max_daily_loss", 0) <= 0:
            logger.warning("Invalid 'max_daily_loss' in config. Using default.")

        if "telegram" not in config:
            config["telegram"] = {"bot_token": "YOUR_TELEGRAM_BOT_TOKEN", "chat_id": "YOUR_TELEGRAM_CHAT_ID"}

        if "gemini" not in config:
            config["gemini"] = {"api_key": os.environ.get('GEMINI_API_KEY', "YOUR_GEMINI_API_KEY")}

        strategy_filters = config.get("strategy_filters", {})
        if "llm_sentiment_threshold" not in strategy_filters:
            strategy_filters["llm_sentiment_threshold"] = 0.7
        if "ml_filter_threshold" not in strategy_filters:
            strategy_filters["ml_filter_threshold"] = 0.55  # Reduced from 0.65 to 55% for more permissive filtering
        config["strategy_filters"] = strategy_filters

        logger.info(f"Loaded config MT5 login: {config['mt5']['login']}")
        logger.info("Config loaded successfully.")
        return config
    except FileNotFoundError:
        log_msg = f"CRITICAL: Configuration file not found at '../config.json'. The bot cannot start without a valid config file."
        if logger:
            logger.critical(log_msg)
        else:
            print(log_msg)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"FATAL: Error loading/creating config.json: {str(e)}. Bot cannot start properly.", exc_info=True)
        sys.exit(1)

# Import signal after sys is fully configured if it might be an issue
import signal

def signal_handler(sig, frame):
    logger.info("Received shutdown signal. Saving state and shutting down MT5...")
    try:
        save_last_trade_times_main()
        save_daily_pnl_main()
        save_trade_count_main()
        save_active_trades_for_file_py()
        save_last_executed_candle_state_for_file_py()
        mt5.shutdown()
        logger.info("MT5 disconnected. Bot shut down completely.")
    except Exception as e:
        logger.error(f"Error during graceful shutdown: {str(e)}", exc_info=True)
    finally:
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def load_last_trade_times_main():
    global last_trade_times_main
    if os.path.exists(TRADE_TIMES_FILE):
        try:
            with open(TRADE_TIMES_FILE, 'r') as f:
                data = json.load(f)
                if 'last_trade_times_main' in data:
                    last_trade_times_main.update({
                        symbol: datetime.fromisoformat(timestamp).replace(tzinfo=pytz.UTC)
                        for symbol, timestamp in data['last_trade_times_main'].items()
                    })
            logger.info(f"Loaded main last trade times: {last_trade_times_main}")
        except Exception as e:
            logger.error(f"Error loading main last trade times from {TRADE_TIMES_FILE}: {str(e)}")

def save_last_trade_times_main():
    try:
        os.makedirs(os.path.dirname(TRADE_TIMES_FILE) or '.', exist_ok=True)
        data_to_save = {
            'last_trade_times_main': {symbol: timestamp.isoformat() for symbol, timestamp in last_trade_times_main.items()},
        }
        with open(TRADE_TIMES_FILE, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        logger.debug(f"Saved main last trade times: {data_to_save}")
    except Exception as e:
        logger.error(f"Error saving main last trade times to {TRADE_TIMES_FILE}: {str(e)}")

def ensure_mt5_connection(login, password, server, max_retries=5, delay=10):
    import os
    import sys
    MT5_TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"  # <-- Adjust if your working terminal is elsewhere
    for attempt in range(max_retries):
        try:
            logger.info(f"Working directory: {os.getcwd()}")
            try:
                logger.info(f"User: {os.getlogin()}")
            except Exception as e:
                logger.info(f"User: <unknown> ({e})")
            logger.info(f"Python executable: {sys.executable}")
            logger.info(f"MT5 login params: login={login}, password={'*' * len(str(password))}, server={server}")
            if not mt5.initialize(path=MT5_TERMINAL_PATH):
                logger.error(f"MT5 initialization failed. Error: {mt5.last_error()}. Attempt {attempt + 1}/{max_retries}")
                time.sleep(delay)
                continue

            if not mt5.login(login=login, password=password, server=server):
                logger.error(f"MT5 login failed. Error: {mt5.last_error()}. Attempt {attempt + 1}/{max_retries}")
                mt5.shutdown()
                time.sleep(delay)
                continue

            account_info = mt5.account_info()
            if account_info is None:
                logger.error(f"Failed to get account info. Attempt {attempt + 1}/{max_retries}")
                mt5.shutdown()
                time.sleep(delay)
                continue

            symbol_info = mt5.symbol_info("XAUUSD")
            if symbol_info is None:
                logger.error(f"Failed to get symbol info for XAUUSD. Attempt {attempt + 1}/{max_retries}")
                mt5.symbol_select("XAUUSD", True)
                symbol_info = mt5.symbol_info("XAUUSD")
                if symbol_info is None:
                    logger.error(f"Symbol XAUUSD still not available after selection. Cannot verify connection.")
                    mt5.shutdown()
                    time.sleep(delay)
                    continue

            logger.info("MT5 connection and login ensured.")
            return True

        except Exception as e:
            logger.error(f"Error ensuring MT5 connection: {str(e)}. Attempt {attempt + 1}/{max_retries}")
            try:
                mt5.shutdown()
            except:
                pass
            time.sleep(delay)

    logger.critical("Failed to ensure MT5 connection after all attempts. Exiting bot.")
    return False

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def fetch_candles(symbol, timeframe, count):
    """
    Fetch candle data with multi-timeframe fallback logic.
    Tries 1m first, then 5m, then 15m if previous timeframes have stale data.
    """
    try:
        # Define timeframe priority order: 1m -> 5m -> 15m
        timeframe_priority = [
            (mt5.TIMEFRAME_M1, "1m"),
            (mt5.TIMEFRAME_M5, "5m"),
            (mt5.TIMEFRAME_M15, "15m")
        ]

        for tf_code, tf_name in timeframe_priority:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None or not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    logger.warning(f"Symbol {symbol} not found or could not be selected in Market Watch.")
                    continue
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    logger.warning(f"Symbol {symbol} still not available after selection.")
                    continue

            tick = mt5.symbol_info_tick(symbol)
            if tick is None or tick.bid == 0 or tick.ask == 0:
                logger.warning(f"No valid quotes for {symbol} on {tf_name}. Market may be closed or disconnected.")
                continue

            if tick:
                logger.debug(f"Broker server time: {datetime.fromtimestamp(tick.time, timezone.utc)} (UTC)")
            else:
                logger.warning("Could not fetch broker server time.")

            rates = mt5.copy_rates_from_pos(symbol, tf_code, 0, count)
            if rates is None or len(rates) == 0:
                logger.warning(f"Failed to fetch {tf_name} candles for {symbol}. Rates is None or empty.")
                continue

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            for col in ['open', 'high', 'low', 'close', 'tick_volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            if df[['open', 'high', 'low', 'close']].isnull().any().any():
                logger.warning(f"DataFrame contains NaN values in price columns for {symbol} on {tf_name}. Dropping NaNs.")
                df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
                if df.empty:
                    logger.warning(f"DataFrame became empty after dropping NaNs for {symbol} on {tf_name}.")
                    continue

            prices = df['close'].to_numpy(dtype=np.float64)
            latest_candle_time = df['time'].iloc[-1]

            # Check if data is fresh (within last 5 minutes)
            broker_tick = mt5.symbol_info_tick(symbol)
            if broker_tick and getattr(broker_tick, 'time', None):
                reference_time = datetime.fromtimestamp(broker_tick.time, timezone.utc)
            else:
                reference_time = datetime.now(timezone.utc)

            age = (reference_time - latest_candle_time).total_seconds()

            if age <= 900:  # 15 minutes threshold
                logger.info(f"Successfully fetched fresh {tf_name} data for {symbol}. Latest candle time: {latest_candle_time}, Age: {age:.0f}s")
                return df, prices

            logger.warning(f"Stale {tf_name} data for {symbol} (age {age:.0f}s). Trying next timeframe...")

        # If we reach here, all timeframes were stale
        logger.error(f"All timeframes have stale data for {symbol}. Cannot fetch fresh candle data.")
        return None, None

    except Exception as e:
        logger.error(f"Error fetching candles for {symbol}: {str(e)}", exc_info=True)
        return None, None

def get_timeframe_seconds(timeframe):
    """Convert MT5 timeframe to seconds"""
    if timeframe == mt5.TIMEFRAME_M1:
        return 60
    elif timeframe == mt5.TIMEFRAME_M5:
        return 5 * 60
    elif timeframe == mt5.TIMEFRAME_M15:
        return 15 * 60
    elif timeframe == mt5.TIMEFRAME_M30:
        return 30 * 60
    elif timeframe == mt5.TIMEFRAME_H1:
        return 60 * 60
    elif timeframe == mt5.TIMEFRAME_H4:
        return 4 * 60 * 60
    elif timeframe == mt5.TIMEFRAME_D1:
        return 24 * 60 * 60
    else:
        return None

# Define MAX_PRICE_DIFF_PCT globally in main.py if it's used here,
# or ensure it's passed from config.
MAX_PRICE_DIFF_PCT = 1.0 # Default value, can be loaded from config if preferred.

def run_strategy_for_symbol(symbol, timeframe, candle_count, strategies, config):
    tf_map = {
        mt5.TIMEFRAME_M1: 1, mt5.TIMEFRAME_M5: 5, mt5.TIMEFRAME_M15: 15, mt5.TIMEFRAME_M30: 30,
        mt5.TIMEFRAME_H1: 60, mt5.TIMEFRAME_H4: 240, mt5.TIMEFRAME_D1: 1440
    }

    # Use the global strategy priority mapping
    # strategy_priority is already defined globally with updated values

    try:
        cycle_start_time = time.time()
        # We'll prefer broker server time for age calculations whenever available
        broker_tick = mt5.symbol_info_tick(symbol)
        if broker_tick and getattr(broker_tick, 'time', None):
            current_utc_time = datetime.fromtimestamp(broker_tick.time, timezone.utc)
        else:
            current_utc_time = datetime.now(timezone.utc)
        logger.info(f"\n{'='*50}\nNEW CYCLE FOR {symbol} STARTED at {current_utc_time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n{'='*50}")

        # Fresh data fetch with retry if candle too old (>5 minutes). Use broker time if possible to avoid negative ages.
        max_age_seconds = 300
        max_retries = 3
        retry_delay = 1.5
        attempt = 0
        while True:
            stage_t0 = time.time()
            df, prices = fetch_candles(symbol, timeframe, candle_count)
            stage_fetch_ms = (time.time() - stage_t0) * 1000.0
            if df is None or df.empty or prices is None or len(prices) == 0:
                logger.error(f"No valid price data for {symbol}. Skipping strategy checks for this cycle.")
                return
            latest_candle_time = df['time'].iloc[-1]
            # Get broker server time for accurate age calculation
            broker_tick = mt5.symbol_info_tick(symbol)
            if broker_tick and getattr(broker_tick, 'time', None):
                reference_time = datetime.fromtimestamp(broker_tick.time, timezone.utc)
            else:
                reference_time = datetime.now(timezone.utc)
            age = (reference_time - latest_candle_time).total_seconds()
            if age <= max_age_seconds:
                break
            attempt += 1
            if attempt >= max_retries:
                logger.warning(f"Stale candles for {symbol} after {attempt} attempts (age {max(0, age):.0f}s). Proceeding but will block trades downstream.")
                break
            logger.info(f"Stale candles for {symbol} (age {age:.0f}s). Refetching (attempt {attempt+1}/{max_retries})...")
            time.sleep(retry_delay)
            
        if df is None or df.empty or prices is None or len(prices) == 0:
            logger.error(f"No valid price data for {symbol}. Skipping strategy checks for this cycle.")
            return

        latest_candle_close = prices[-1]
        latest_candle_time = df['time'].iloc[-1]

        tick = mt5.symbol_info_tick(symbol)
        if tick is None or tick.bid == 0 or tick.ask == 0:
            logger.error(f"Failed to fetch real-time tick data for {symbol}. Skipping strategy checks for this cycle.")
            return

        tick_time_utc = datetime.fromtimestamp(tick.time, timezone.utc)
        live_price = tick.ask if symbol == "XAUUSD" else tick.bid
        # For non-XAUUSD symbols, typically you use bid for sell, ask for buy
        # Here, it's simplified to bid for non-XAUUSD. Consider adjusting based on exact requirement.

        logger.info(f"Price Data for {symbol}: Live Price: {live_price:.5f}, Latest Candle Close: {latest_candle_close:.5f}")
        price_diff_pct = abs(live_price - latest_candle_close) / latest_candle_close * 100
        logger.info(f"Price Diff %: {price_diff_pct:.2f}%. Max allowed: {MAX_PRICE_DIFF_PCT:.2f}%")

        if price_diff_pct > MAX_PRICE_DIFF_PCT:
            logger.info(f"Price difference too large ({price_diff_pct:.2f}% > {MAX_PRICE_DIFF_PCT}%). Skipping trades for {symbol}.")
            return

        candle_freshness_threshold_seconds = config.get("candle_freshness_threshold_seconds", tf_map.get(timeframe, 60) * 60 * 1.5)

        # Use broker server time as reference to avoid negative ages from clock skews
        reference_time = tick_time_utc
        candle_age_seconds = (reference_time - latest_candle_time).total_seconds()
        tick_age_seconds = (reference_time - tick_time_utc).total_seconds()
        # Clamp any tiny negatives to zero
        candle_age_seconds = max(0.0, candle_age_seconds)
        tick_age_seconds = max(0.0, tick_age_seconds)

        logger.info(f"Time Data for {symbol}: Market Open: {is_market_open(symbol, current_utc_time)} | "
                        f"Candle Age: {candle_age_seconds:.0f}s (max {candle_freshness_threshold_seconds}s) | "
                        f"Tick Age: {tick_age_seconds:.0f}s (max {candle_freshness_threshold_seconds}s)")

        if not is_market_open(symbol, current_utc_time):
            logger.info(f"{symbol} market is CLOSED. Skipping trading for this symbol.")
            return

        if candle_age_seconds > candle_freshness_threshold_seconds or tick_age_seconds > candle_freshness_threshold_seconds:
            # Only warn about stale data if market is open
            if is_market_open(symbol, current_utc_time):
                logger.warning(f"Stale data detected for {symbol} - Candle age: {candle_age_seconds:.1f}s, Tick age: {tick_age_seconds:.1f}s (Threshold: {candle_freshness_threshold_seconds}s). Trading may be affected.")
            else:
                logger.info(f"Market for {symbol} is closed. Stale data warnings suppressed.")
            from mt5_utils import data_freshness_check
            data_freshness_check(candle_age_seconds, tick_age_seconds, candle_freshness_threshold_seconds, symbol)

        market_data_for_strategies = {}
        stage_calc_t0 = time.time()
        try:
            df['SMA20'] = df['close'].rolling(window=20).mean()
            df['SMA50'] = df['close'].rolling(window=50).mean()

            if 'close' in df and len(df) >= 14:
                df['RSI'] = calculate_rsi(df['close'])
                current_rsi = df['RSI'].iloc[-1] if not df['RSI'].isna().iloc[-1] else 50
            else:
                current_rsi = 50

            atr = calculate_atr(df)
            if atr is None or math.isnan(atr) or atr <= 0:
                logger.warning(f"ATR is invalid ({atr}) for {symbol}. Using fallback value 0.01.")
                atr = 0.01

            if len(df['close']) > 1:
                volatility = df['close'].diff().abs().mean()
            else:
                volatility = 0.0

            if volatility is None or math.isnan(volatility) or volatility <= 0:
                logger.warning(f"Volatility is invalid ({volatility}) for {symbol}. Using fallback value 0.001.")
                volatility = 0.001

            trend_calc = 'up' if df['close'].iloc[-1] > df['close'].iloc[-min(10, len(df)-1)] else 'down'

            equity = get_account_equity_for_file_py()

            market_data_for_strategies = {
                "df": df, "prices": prices, "atr": atr, "equity": equity,
                "volatility": volatility, "trend": trend_calc, "rsi": current_rsi,
                "current_utc_time": current_utc_time,
                "latest_candle_time": latest_candle_time,
                "tick_time_utc": tick_time_utc,
                "reference_time": tick_time_utc,
                "live_price": live_price,
                # expose runtime-configurable risk knobs to processor
                "default_lot_size": config.get("risk_settings", {}).get("default_lot_size", 0.05),
                "max_spread_points": config.get("advanced_settings", {}).get("max_spread", 20),
            }
            logger.info(f"Market Data Summary for {symbol}: ATR={atr:.4f}, Volatility={volatility:.4f}, Trend={trend_calc}, RSI={current_rsi:.2f}, Live Price={live_price:.5f}, Equity=${equity:.2f}")
            calc_ms = (time.time() - stage_calc_t0) * 1000.0
            logger.debug(f"[TIMING] {symbol}: fetch_ms={stage_fetch_ms:.1f}, calc_ms={calc_ms:.1f}")

        except Exception as e:
            logger.error(f"Error calculating market conditions for {symbol}: {str(e)}", exc_info=True)
            return

        # Pass config to news_filter if it needs llm_sentiment_threshold
        # The news_filter now takes market_data which includes llm_sentiment etc.
        news_filter_result = news_filter(symbol, current_utc_time, market_data_for_strategies)

        if not news_filter_result["allowed"]:
            logger.info(f"News filter blocked trading for {symbol}. Reason: {news_filter_result.get('reason', 'Unknown')}. Skipping this cycle.")
            return

        # Update market_data_for_strategies with news filter results
        market_data_for_strategies.update({
            "llm_news_sentiment": news_filter_result.get("llm_news_sentiment", {"sentiment": "NEUTRAL", "score": 0.0}),
            "news_favored_direction": news_filter_result.get("news_favored_direction", "both"),
            "is_within_news_window": news_filter_result.get("is_within_news_window", False)
        })

        manage_open_positions(symbol, atr)
        
        # Update trailing stops for all active trades
        update_trailing_stops()

        # Placeholder functions for risk checks - ensure these are defined or imported
        # They are not in the provided code, so let's define them as simple pass-throughs
        # or mock them for now. In a real scenario, these would contain actual logic.
        def check_equity_curve_protection(equity_val):
            # Implement your equity curve protection logic here.
            # For now, it always allows.
            return True

        def check_portfolio_correlation(sym):
            # Implement your portfolio correlation logic here.
            # For now, it always allows.
            return True

        def check_global_risk_limits():
            # Implement your global risk limits logic here.
            # For now, it always allows.
            return False # Set to False to indicate "not triggered"

        # These functions are imported from file.py
        # check_trade_limits() is already imported from file.py

        if not check_equity_curve_protection(equity):
            logger.warning("Equity curve protection triggered. Skipping all strategies this cycle.")
            return
        if not check_portfolio_correlation(symbol):
            logger.warning(f"Portfolio correlation filter triggered for {symbol}. Skipping all strategies this cycle.")
            return
        if check_global_risk_limits():
            logger.warning("Global risk limits triggered. Skipping all strategies this cycle.")
            return

        if not check_max_daily_loss_global(config["risk_settings"]["max_daily_loss"]):
            logger.warning(f"Max daily loss triggered. Skipping all strategies for {symbol}.")
            return
        if not check_max_trades_per_day_global(symbol, config["risk_settings"]["max_daily_trades"]):
            logger.warning(f"Max global trades per day for {symbol} triggered. Skipping all strategies for this symbol.")
            return

        llm_sentiment_threshold = config["strategy_filters"].get("llm_sentiment_threshold", 0.7)
        ml_filter_threshold = config["strategy_filters"].get("ml_filter_threshold", 0.65)

        # Enforce global max open positions from settings
        try:
            max_open = int(config.get("risk_settings", {}).get("max_open_trades", 20))
            open_positions = mt5.positions_get()
            if open_positions and len(open_positions) >= max_open:
                logger.info(f"[SKIP ALL] Max open positions reached ({len(open_positions)}/{max_open}).")
                return
        except Exception:
            pass

        # Determine which strategies are enabled from UI settings
        enabled_strategy_names = []
        if bot_live_settings.get("all_strategies", False):
            enabled_strategy_names = [name for name, _ in strategies]
        else:
            sel = bot_live_settings.get("selected_strategies", []) or []
            # normalize to list of strings
            enabled_strategy_names = [s for s in sel if isinstance(s, str)]

        # ===== NEW CENTRALIZED PROCESSOR LOGIC =====
        
        # Collect valid strategy signals
        valid_strategy_signals = []
        strategies_eval_t0 = time.time()
        
        for strategy_name, strategy_obj in strategies:
            # Check if strategy is enabled
            if enabled_strategy_names and strategy_name not in enabled_strategy_names:
                logger.debug(f"[SKIP] {strategy_name}: disabled by UI selection.")
                continue
                
            # Check daily trade limits
            if not check_trade_limits(symbol, strategy_name):
                logger.info(f"[SKIP] {strategy_name}: Daily trade limit reached for {symbol}.")
                continue
            
            # Check strategy cooldown
            if not check_strategy_cooldown(symbol, strategy_name):
                continue  # Cooldown message already logged in function
            
            # Check if already processed this candle
            with data_lock:
                if last_executed_candle_for_strategy.get((symbol, strategy_name)) == latest_candle_time:
                    logger.debug(f"[SKIP] {strategy_name}: Already processed for candle {latest_candle_time}.")
                    continue

            logger.info(f"\n--- Evaluating {strategy_name} for {symbol} ---")

            try:
                if not hasattr(strategy_obj, "execute") or not callable(getattr(strategy_obj, "execute")):
                    logger.error(f"[ERROR] Strategy {strategy_name} has no executable 'execute' method. Skipping.")
                    continue
                strategy_signal = strategy_obj.execute(
                    symbol, prices, df, equity, config["risk_settings"].get("allow_multiple_trades", False)
                )

                if strategy_signal and strategy_signal.get("success", False):
                    # Get strategy priority and confluence score
                    priority = strategy_priority.get(strategy_name, 3)
                    confluence_score = get_strategy_confluence_score(strategy_name, strategy_signal)
                    
                    signal_data = {
                        "strategy_name": strategy_name,
                        "strategy_obj": strategy_obj,
                        "signal_result": strategy_signal,
                        "priority": priority,
                        "confluence_score": confluence_score
                    }
                    
                    valid_strategy_signals.append(signal_data)
                    logger.info(f"[SIGNAL] {strategy_name} generated valid signal (Priority: {priority}, Confluence: {confluence_score})")
                else:
                    logger.debug(f"[NO SIGNAL] {strategy_name} - no valid trade signal found.")

            except Exception as e:
                logger.error(f"[ERROR] Error during {strategy_name} execution: {str(e)}", exc_info=True)
                continue
        
        # Select strategies to run based on priority and caps
        strategy_ms = (time.time() - strategies_eval_t0) * 1000.0
        logger.debug(f"[TIMING] {symbol}: strategy_ms={strategy_ms:.1f}")
        if not valid_strategy_signals:
            logger.info(f"[NO SIGNALS] No valid strategy signals for {symbol} in this cycle.")
            return
        
        logger.info(f"[STRATEGY SIGNALS] Found {len(valid_strategy_signals)} valid signals for {symbol}")
        for signal in valid_strategy_signals:
            logger.info(f"  - {signal['strategy_name']}: Priority {signal['priority']}, Confluence {signal['confluence_score']}")
        
        selected_signals = select_best_strategy_signals(symbol, valid_strategy_signals)
        if not selected_signals:
            logger.info(f"[NO SELECTION] No strategies selected after priority filtering for {symbol}")
            return

        for selected_signal in selected_signals:
            strategy_name = selected_signal["strategy_name"]
            logger.info(f"[SELECTED] Executing {strategy_name} for {symbol}")
            try:
                trade_executed = process_strategy_signal(
                    symbol=symbol,
                    strategy_name=strategy_name,
                    signal_result=selected_signal["signal_result"],
                    priority_level=selected_signal["priority"],
                    market_data=market_data_for_strategies,
                    llm_sentiment_threshold=llm_sentiment_threshold,
                    ml_filter_threshold=ml_filter_threshold
                )

                if trade_executed:
                    update_strategy_cooldown(symbol, strategy_name)
                    increment_total_trade_count_main(symbol)
                    with data_lock:
                        last_executed_candle_for_strategy[(symbol, strategy_name)] = latest_candle_time
                    save_last_executed_candle_state_for_file_py()
                    logger.info(f"[SUCCESS] {strategy_name} trade executed successfully for {symbol}")
                else:
                    logger.info(f"[FILTERED] {strategy_name} signal was filtered out during processing for {symbol}")
            except Exception as e:
                logger.error(f"[ERROR] Error during {strategy_name} signal processing: {str(e)}", exc_info=True)

    except Exception as e:
        logger.error(f"Unhandled error in run_strategy_for_symbol for {symbol}: {str(e)}", exc_info=True)
    finally:
        cycle_duration = time.time() - cycle_start_time
        logger.info(f"\nCycle for {symbol} finished in {cycle_duration:.2f} seconds.")

def thread_wrapper(symbol, timeframe, candle_count, strategies, config):
    last_cycle_finish_time = [time.time()]

    def watchdog():
        while True:
            time.sleep(10)
            current_time = time.time()
            if current_time - last_cycle_finish_time[0] > 180:
                logger.critical(f"WATCHDOG: Thread for {symbol} appears to be stuck! Last cycle completion: {current_time - last_cycle_finish_time[0]:.2f} seconds ago. Attempting to send alert.")
                send_telegram_alert(f"🚨 ALERT: Bot thread for {symbol} STUCK! Last cycle {current_time - last_cycle_finish_time[0]:.0f}s ago. Manual intervention needed.")

    watchdog_thread = threading.Thread(target=watchdog, daemon=True)
    watchdog_thread.start()

    while True:
        try:
            if not ensure_mt5_connection(config["mt5"]["login"], config["mt5"]["password"], config["mt5"]["server"]):
                logger.error(f"MT5 connection lost for {symbol} thread. Waiting before retry...")
                time.sleep(30)
                continue

            run_strategy_for_symbol(symbol, timeframe, candle_count, strategies, config)
            last_cycle_finish_time[0] = time.time()

            sleep_interval = config.get("polling_interval_seconds", 60)
            time_since_cycle_start = time.time() - last_cycle_finish_time[0]
            sleep_duration = max(1, sleep_interval - time_since_cycle_start)
            logger.debug(f"Thread for {symbol} sleeping for {sleep_duration:.2f} seconds.")
            time.sleep(sleep_duration)

        except Exception as e:
            logger.error(f"Thread for {symbol} encountered unhandled error: {str(e)}", exc_info=True)
            send_telegram_alert(f"⚠️ ERROR: Thread for {symbol} crashed: {str(e)}. Attempting to continue...")
            time.sleep(60)

def poll_bot_settings_thread(config):
    global bot_live_settings

    api_url = "http://localhost:8000/api/bot/settings"

    while True:
        try:
            # Prefer runtime state file written by API to avoid CORS/race when UI is local
            runtime_state_path = os.path.join("logs", "bot_runtime_settings.json")
            new_settings = {}
            if os.path.exists(runtime_state_path):
                with open(runtime_state_path, "r") as f:
                    new_settings = json.load(f)
            else:
                response = requests.get(api_url, timeout=15)
                response.raise_for_status()
                new_settings = response.json()

            if isinstance(new_settings, dict):
                bot_live_settings.update({
                    "all_strategies": new_settings.get("all_strategies", bot_live_settings.get("all_strategies", False)),
                    "selected_strategies": new_settings.get("selected_strategies", bot_live_settings.get("selected_strategies", [])),
                    "killzone_map": new_settings.get("killzone_map", bot_live_settings.get("killzone_map", {})),
                    "current_session": new_settings.get("current_session", bot_live_settings.get("current_session", "ALL")),
                })
                logger.debug(f"Fetched new bot settings: {bot_live_settings}")

        except requests.exceptions.RequestException as e:
            logger.warning(f"Error polling bot settings from API ({api_url}): {e}. Retrying...")
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON from bot settings API. Response might not be valid JSON. Retrying...")
        except Exception as e:
            logger.error(f"Unexpected error in poll_bot_settings_thread: {str(e)}", exc_info=True)

        time.sleep(10)

bot_live_settings = {
    "bot_active": False,
    "all_strategies": False,
    "selected_strategies": [],
    "killzone_map": {},
    "current_session": "ALL"
}

def main():
    # Reconfigure stdout and stderr to use UTF-8 encoding to prevent UnicodeEncodeError
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (TypeError, ValueError, AttributeError):
        # This might fail in environments where stdout/stderr are not standard streams
        print("Warning: Could not reconfigure stdout/stderr to UTF-8.")

    global config, bot_live_settings
    try:
        config = load_config()

        if logger:
            logger.setLevel(getattr(logging, config.get("logging", {}).get("log_level", "INFO").upper()))

        logger.info("Starting bot initialization...")

        # Set bot startup time 
        global bot_startup_time
        bot_startup_time = datetime.now(timezone.utc)
        logger.info(f"[STARTUP] Bot startup time set to: {bot_startup_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        if not ensure_mt5_connection(config["mt5"]["login"], config["mt5"]["password"], config["mt5"]["server"]):
            logger.critical("Failed to ensure MT5 connection. Exiting bot.")
            sys.exit(1)

        # --- Initialize file.py components now that main logger is ready ---
        logger.info("Initializing components from file.py...")
        setup_file_logger() # This will now use the same handler as main
        ensure_trades_table_exists()
        repair_trade_log_csv()
        logger.info("file.py components initialized.")

        gemini_api_key = os.environ.get('GEMINI_API_KEY', 'AIzaSyD43qNjRNmecJyNiiqF5Yerri27D9U89Y8')
        if gemini_api_key and gemini_api_key != 'YOUR_GEMINI_API_KEY':
            init_llm_sentiment_analyzer(gemini_api_key)
            logger.info("[SUCCESS] LLM Sentiment Analyzer initialized successfully with API key")
        else:
            logger.warning("Gemini API Key not found in environment variables or is invalid. LLM Sentiment Analysis will be disabled.")

        # Set FMP_API_KEY environment variable if not present
        fmp_api_key = os.environ.get('FMP_API_KEY', 'eOTn0m18D2RUTdvyATakDGoZNzlfKmJR')
        if fmp_api_key:
            os.environ['FMP_API_KEY'] = fmp_api_key
            logger.info("[SUCCESS] FMP API Key set in environment variables")
        else:
            logger.warning("FMP API Key not found in environment variables. Economic calendar features may be limited.")

        ml_model_path = config.get("advanced_settings", {}).get("ml_model_path", "ml_trade_filter.pkl")
        logger.info(f"Attempting to initialize ML Trade Filter with model: {ml_model_path}")

        # Check if model file exists
        if not os.path.exists(ml_model_path):
            logger.warning(f"ML model file not found at {ml_model_path}. ML filtering will be disabled.")
            logger.info("To fix this, either:")
            logger.info("1. Create the ML model file at the specified path, or")
            logger.info("2. Update the ml_model_path in config.json, or")
            logger.info("3. Remove ml_model_path from config.json to use default path")
        else:
            try:
                init_ml_trade_filter(ml_model_path)
                logger.info("ML Trade Filter initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize ML Trade Filter: {str(e)}")
                logger.warning("ML filtering will be disabled due to initialization error.")

        telegram_bot_token = config.get("telegram", {}).get("bot_token")
        telegram_chat_id = config.get("telegram", {}).get("chat_id")
        if telegram_bot_token and telegram_chat_id and \
           telegram_bot_token != "YOUR_TELEGRAM_BOT_TOKEN" and telegram_chat_id != "YOUR_TELEGRAM_CHAT_ID":
            init_telegram_settings(telegram_bot_token, telegram_chat_id)
        else:
            logger.warning("Telegram settings not found or are default. Telegram alerts will be disabled.")


        load_active_trades_from_file_py()
        load_last_executed_candle_state_from_file_py()

        load_last_trade_times_main()
        load_daily_pnl_main()
        load_trade_count_main()

        strategies = [
            ("mmc_combo_strategy", MMCXAUUSDStrategy()), # Corrected in file.py, assuming this is the class to import
            ("mmc", mmc_strategy()),
            ("mmxm", mmxm()),
            ("msb_retest", msb_retest()),
            ("order_block", order_block()),
            ("ote", OTEStrategy()),
            ("amd_strategy", amd_strategy()),
            ("judas_swing", judas_swing())
        ]

        timeframe_str = config.get("timeframe", "TIMEFRAME_M15")
        timeframe = getattr(mt5, timeframe_str, mt5.TIMEFRAME_M15)
        candle_count = config.get("candle_count", 50)
        symbols = list(set(config.get("symbols", ["XAUUSD"])))

        logger.info(f"Bot will monitor symbols: {symbols} on timeframe: {timeframe_str}")

        api_poll_thread = threading.Thread(target=poll_bot_settings_thread, args=(config,), daemon=True)
        api_poll_thread.start()
        logger.info("API polling thread started.")

        symbol_threads = []
        for symbol in symbols:
            t = threading.Thread(
                target=thread_wrapper,
                args=(symbol, timeframe, candle_count, strategies, config),
                daemon=True
            )
            symbol_threads.append(t)
            t.start()
            logger.info(f"Started dedicated thread for symbol: {symbol}")

        # Keep the main thread alive while daemon threads run
        while True:
            # Check if any of the threads are still alive
            if not any(t.is_alive() for t in symbol_threads):
                logger.warning("All symbol threads have stopped. Restarting them...")
                # Optional: logic to restart threads could be added here
                break # Exit the loop to allow shutdown
            time.sleep(60) # Heartbeat check every 60 seconds

        logger.info("All symbol threads finished (this should not happen in normal operation unless main loop breaks).")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Initiating graceful shutdown...")
    except Exception as e:
        logger.critical(f"CRITICAL: Unhandled exception in main() function: {str(e)}", exc_info=True)
    finally:
        logger.info("Performing final shutdown procedures...")
        save_last_trade_times_main()
        save_daily_pnl_main()
        save_trade_count_main()
        save_active_trades_for_file_py()
        save_last_executed_candle_state_for_file_py()
        mt5.shutdown()
        logger.info("MT5 disconnected. Bot shut down completely.")
        sys.exit(0)

if __name__ == "__main__":
    main()