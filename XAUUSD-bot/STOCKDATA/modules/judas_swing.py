"""
Judas Swing Strategy Trading Bot - Enhanced Version

Strategy Overview:
This strategy uses a multi-timeframe approach to identify high-probability trade setups:
1. HTF (Higher Timeframe): Establishes market bias (bullish/bearish) using trend direction
2. MTF (Mid Timeframe): Identifies order blocks (OB) or fair value gaps (FVG) as key zones, 
   validated by liquidity grabs and session alignment
3. LTF (Lower Timeframe): Confirms entry signals (BOS/CHoCH, engulfing, wick rejection) within MTF zones

Enhanced Features:
- Configuration-based symbol-specific parameters
- Robust MT5 handling with retry logic
- Dynamic position sizing based on risk percentage
- Candle caching for performance optimization
- Trade journaling and performance metrics
- Trailing stops and enhanced position management
- Comprehensive validation and error handling
"""

from datetime import datetime, time, timedelta
import MetaTrader5 as mt5
import pandas as pd
import pytz
from enum import Enum, auto
import logging
import uuid
from typing import Optional, Dict, Tuple, List
import numpy as np
import json
import os
import asyncio
import time as time_module

# Setup logging
logger = logging.getLogger('trade_bot.judas')
logger.setLevel(logging.INFO)
logger.propagate = False
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("judas_swing_bot.log"), logging.StreamHandler()]
)

# Load configuration with enhanced error handling
def load_config(config_path: str = "config.json") -> Dict:
    """
    Load configuration from JSON file with comprehensive fallback defaults.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict: Configuration dictionary with all required parameters
    """
    try:
        if not os.path.exists(config_path):
            logger.warning(f"Config file {config_path} not found, using defaults")
            return get_default_config()
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate config structure
        required_sections = ['symbols', 'timeframes', 'kill_zones', 'symbol_params', 'general_params']
        for section in required_sections:
            if section not in config:
                logger.warning(f"Missing config section: {section}, using defaults")
                config.update(get_default_config())
                break
                
        logger.info("Configuration loaded successfully")
        return config
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {str(e)}, using defaults")
        return get_default_config()
    except Exception as e:
        logger.error(f"Error loading config file: {str(e)}, using defaults")
        return get_default_config()

def get_default_config() -> Dict:
    """Return comprehensive default configuration."""
    return {
        "symbols": ["XAUUSD", "EURUSD", "GBPUSD"],
        "timeframes": {
            "HTF": "TIMEFRAME_H1",
            "MTF": "TIMEFRAME_M15", 
            "LTF": "TIMEFRAME_M1"
        },
        "kill_zones": {
            "London": ["02:00", "05:00"],
            "New York": ["07:00", "11:00"]
        },
        "symbol_params": {
            "XAUUSD": {
                "liquidity_threshold": 0.002,
                "atr_multiplier": 2.5,
                "fvg_threshold": 0.002,
                "min_spread": 0.5
            },
            "EURUSD": {
                "liquidity_threshold": 0.0001,
                "atr_multiplier": 2.0,
                "fvg_threshold": 0.0001,
                "min_spread": 0.1
            },
            "GBPUSD": {
                "liquidity_threshold": 0.0001,
                "atr_multiplier": 2.0,
                "fvg_threshold": 0.0001,
                "min_spread": 0.1
            }
        },
        "general_params": {
            "volume_threshold": 1.5,
            "min_rr": 2.0,
            "atr_period": 14,
            "ob_lookback": 10,
            "engulfing_body_ratio": 0.6,
            "htf_expiry_hours": 3,
            "mtf_expiry_minutes": 30,
            "zone_score_threshold_valid": 60,
            "zone_score_threshold_weak": 40,
            "default_risk_percentage": 0.01,
            "cache_duration": 300,
            "max_retries": 3,
            "retry_delay": 5
        }
    }

# Load and setup configuration
config = load_config()
SYMBOLS = config.get("symbols", ["XAUUSD", "EURUSD", "GBPUSD"])

try:
    TIMEFRAMES = {
        'HTF': getattr(mt5, config["timeframes"].get("HTF", "TIMEFRAME_H1")),
        'MTF': getattr(mt5, config["timeframes"].get("MTF", "TIMEFRAME_M15")),
        'LTF': getattr(mt5, config["timeframes"].get("LTF", "TIMEFRAME_M1"))
    }
except AttributeError as e:
    logger.error(f"Invalid timeframe in config: {e}, using defaults")
    TIMEFRAMES = {'HTF': mt5.TIMEFRAME_H1, 'MTF': mt5.TIMEFRAME_M15, 'LTF': mt5.TIMEFRAME_M1}

# Parse kill zones with error handling
KILL_ZONES = {}
for session, (start, end) in config.get("kill_zones", {}).items():
    try:
        KILL_ZONES[session] = (
            datetime.strptime(start, "%H:%M").time(),
            datetime.strptime(end, "%H:%M").time()
        )
    except ValueError as e:
        logger.error(f"Invalid time format for {session}: {e}")

# Add missing constants from config after loading
SYMBOL_PARAMS = config.get("symbol_params", {})
GENERAL_PARAMS = config.get("general_params", {})

# Extract parameters with defaults
VOLUME_THRESHOLD = GENERAL_PARAMS.get("volume_threshold", 1.5)
MIN_RR = GENERAL_PARAMS.get("min_rr", 2.0)
ATR_PERIOD = GENERAL_PARAMS.get("atr_period", 14)
OB_LOOKBACK = GENERAL_PARAMS.get("ob_lookback", 10)
ENGULFING_BODY_RATIO = GENERAL_PARAMS.get("engulfing_body_ratio", 0.6)
HTF_EXPIRY = timedelta(hours=GENERAL_PARAMS.get("htf_expiry_hours", 3))
MTF_EXPIRY = timedelta(minutes=GENERAL_PARAMS.get("mtf_expiry_minutes", 30))
ZONE_SCORE_THRESHOLD_VALID = GENERAL_PARAMS.get("zone_score_threshold_valid", 60)
ZONE_SCORE_THRESHOLD_WEAK = GENERAL_PARAMS.get("zone_score_threshold_weak", 40)

# Add missing threshold constants
LIQUIDITY_THRESHOLD = GENERAL_PARAMS.get("liquidity_threshold", 0.002)
FVG_THRESHOLD = GENERAL_PARAMS.get("fvg_threshold", 0.002)

# Enums
class State(Enum):
    IDLE = auto()
    HTF_CONFIRMED = auto()
    MTF_PENDING = auto()
    LTF_READY = auto()
    READY = auto()
    EXECUTED = auto()
    CANCELLED = auto()

class Direction(Enum):
    BULLISH = auto()
    BEARISH = auto()

class ZoneType(Enum):
    OB_STRONG = auto()
    OB_WEAK = auto()
    FVG_LARGE = auto()
    FVG_SMALL = auto()

# Enhanced candle caching system
class CandleCache:
    """Advanced candle cache with automatic cleanup and performance monitoring."""
    
    def __init__(self, cache_duration: int = None, max_cache_size: int = 1000):
        self.cache = {}
        self.cache_duration = cache_duration or GENERAL_PARAMS.get("cache_duration", 300)
        self.max_cache_size = max_cache_size
        self.hit_count = 0
        self.miss_count = 0
    
    def get_cached_candles(self, symbol: str, timeframe: int, count: int) -> pd.DataFrame:
        """Get cached candles with performance tracking."""
        cache_key = f"{symbol}_{timeframe}"
        current_time = datetime.now(pytz.UTC)
        
        # Check cache hit
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if (current_time - timestamp).total_seconds() < self.cache_duration:
                self.hit_count += 1
                return cached_data.tail(count).copy()
        
        # Cache miss - fetch new data
        self.miss_count += 1
        candles = get_candles(symbol, timeframe, count)
        
        if not candles.empty:
            # Manage cache size
            if len(self.cache) >= self.max_cache_size:
                self._cleanup_old_entries()
            
            self.cache[cache_key] = (candles, current_time)
            
        return candles
    
    def _cleanup_old_entries(self):
        """Remove oldest cache entries when size limit is reached."""
        current_time = datetime.now(pytz.UTC)
        expired_keys = []
        
        for key, (data, timestamp) in self.cache.items():
            if (current_time - timestamp).total_seconds() > self.cache_duration:
                expired_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys:
            del self.cache[key]
        
        # If still too many entries, remove oldest
        if len(self.cache) >= self.max_cache_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 2),
            "cache_size": len(self.cache)
        }
    
    def clear_cache(self):
        """Clear all cached data and reset stats."""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0

# Enhanced MT5 initialization with comprehensive retry logic
def initialize_mt5(max_retries: int = None, retry_delay: int = None) -> bool:
    """
    Initialize MT5 with robust retry logic and detailed error reporting.
    
    Args:
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Delay between retry attempts in seconds
        
    Returns:
        bool: True if successful, False otherwise
    """
    max_retries = max_retries or GENERAL_PARAMS.get("max_retries", 3)
    retry_delay = retry_delay or GENERAL_PARAMS.get("retry_delay", 5)
    
    for attempt in range(max_retries):
        try:
            if mt5.initialize():
                # Verify connection by getting account info
                account_info = mt5.account_info()
                if account_info:
                    logger.info(f"MT5 initialized successfully. Account: {account_info.login}, "
                              f"Balance: {account_info.balance}, Server: {account_info.server}")
                    return True
                else:
                    logger.warning("MT5 initialized but account info not available")
            
            error_code, error_msg = mt5.last_error()
            logger.error(f"MT5 initialization failed (Attempt {attempt + 1}/{max_retries}): "
                        f"Error {error_code} - {error_msg}")
                        
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time_module.sleep(retry_delay)
                
        except Exception as e:
            logger.error(f"Exception during MT5 initialization: {str(e)}")
            
    logger.error("MT5 initialization failed after all retries")
    return False

# Enhanced candle fetching with symbol validation and retry logic
def get_candles(symbol: str, timeframe: int, count: int, 
                backtest_df: Optional[pd.DataFrame] = None, 
                max_retries: int = None) -> pd.DataFrame:
    """
    Fetch candles with comprehensive validation and retry logic.
    
    Args:
        symbol (str): Trading symbol
        timeframe (int): MT5 timeframe constant
        count (int): Number of candles to fetch
        backtest_df (Optional[pd.DataFrame]): Backtest data if available
        max_retries (int): Maximum retry attempts
        
    Returns:
        pd.DataFrame: Candle data with proper time index and validation
    """
    if backtest_df is not None:
        result = backtest_df.tail(count).copy()
        # Ensure proper column types
        if not result.empty:
            for col in ['open', 'high', 'low', 'close']:
                if col in result.columns:
                    result[col] = pd.to_numeric(result[col], errors='coerce')
        return result
    
    # Validate symbol exists and is tradeable
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        logger.error(f"Symbol {symbol} not found in MT5")
        return pd.DataFrame()
    
    if not symbol_info.visible:
        # Try to make symbol visible
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Cannot make symbol {symbol} visible")
            return pd.DataFrame()
        logger.info(f"Made symbol {symbol} visible")
    
    max_retries = max_retries or GENERAL_PARAMS.get("max_retries", 3)
    
    for attempt in range(max_retries):
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                
                # Validate data integrity
                if df.isnull().any().any():
                    logger.warning(f"Null values found in candle data for {symbol}")
                    df = df.dropna()
                
                if len(df) == 0:
                    logger.warning(f"No valid candle data after cleaning for {symbol}")
                    continue
                
                # Convert time and set index
                df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
                df.set_index('time', inplace=True)
                
                # Ensure proper data types
                for col in ['open', 'high', 'low', 'close']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Validate OHLC logic
                invalid_candles = df[(df['high'] < df['low']) | 
                                   (df['high'] < df['open']) | 
                                   (df['high'] < df['close']) |
                                   (df['low'] > df['open']) | 
                                   (df['low'] > df['close'])]
                
                if len(invalid_candles) > 0:
                    logger.warning(f"Found {len(invalid_candles)} invalid candles for {symbol}")
                    df = df.drop(invalid_candles.index)
                
                if len(df) >= count * 0.8:  # Accept if we have at least 80% of requested data
                    logger.debug(f"Successfully fetched {len(df)} candles for {symbol}")
                    return df.tail(count)
                else:
                    logger.warning(f"Insufficient valid data for {symbol}: {len(df)}/{count}")
                    
            error_code, error_msg = mt5.last_error()
            logger.warning(f"Failed to fetch candles for {symbol} "
                          f"(Attempt {attempt + 1}/{max_retries}): "
                          f"Error {error_code} - {error_msg}")
                          
            if attempt < max_retries - 1:
                time_module.sleep(1)  # Brief pause before retry
                
        except Exception as e:
            logger.error(f"Exception fetching candles for {symbol}: {str(e)}")
    
    logger.error(f"Failed to fetch candles for {symbol} after {max_retries} attempts")
    return pd.DataFrame()

# Detect session
def detect_session(timestamp: datetime, symbol: str) -> Optional[str]:
    """
    Detect trading session with comprehensive timezone handling.
    
    Args:
        timestamp (datetime): Current timestamp
        symbol (str): Trading symbol for logging context
        
    Returns:
        Optional[str]: Session name ('London', 'New York', 'Any') or None if error
    """
    try:
        # Handle naive timestamps
        if timestamp.tzinfo is None:
            logger.warning(f"Naive timestamp for {symbol}, assuming UTC: {timestamp}")
            timestamp = timestamp.replace(tzinfo=pytz.UTC)
        
        # Convert to New York time for session detection
        nyt_tz = pytz.timezone('America/New_York')
        try:
            ny_time = timestamp.astimezone(nyt_tz)
        except Exception as e:
            logger.error(f"Timezone conversion failed for {symbol}: {e}")
            return None
            
        current_time = ny_time.time()
        
        # Check kill zones
        for session, (start, end) in KILL_ZONES.items():
            # Handle sessions that cross midnight
            if start <= end:
                if start <= current_time < end:
                    logger.info(f"{symbol}: In {session} session ({current_time})")
                    return session
            else:  # Session crosses midnight
                if current_time >= start or current_time < end:
                    logger.info(f"{symbol}: In {session} session ({current_time})")
                    return session
                    
        logger.debug(f"{symbol}: Outside kill zones ({current_time}), general trading")
        return 'Any'
        
    except Exception as e:
        logger.error(f"Error in detect_session for {symbol}: {str(e)}")
        return None

# Dynamic lot size calculation with comprehensive validation
def calculate_lot_size(equity: float, risk_percentage: float, risk_pips: float, symbol: str) -> float:
    """
    Calculate optimal position size based on account equity and risk parameters.
    
    Args:
        equity (float): Account equity
        risk_percentage (float): Risk percentage (e.g., 0.01 for 1%)
        risk_pips (float): Risk in pips/points
        symbol (str): Trading symbol
        
    Returns:
        float: Calculated lot size within symbol limits
    """
    try:
        # Validate inputs
        if equity <= 0:
            logger.error(f"Invalid equity for {symbol}: {equity}")
            return 0.01
            
        if risk_percentage <= 0 or risk_percentage > 0.1:  # Max 10% risk
            logger.error(f"Invalid risk percentage for {symbol}: {risk_percentage}")
            return 0.01
            
        if risk_pips <= 0:
            logger.error(f"Invalid risk pips for {symbol}: {risk_pips}")
            return 0.01
        
        # Get symbol information
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Cannot get symbol info for {symbol}")
            return 0.01
        
        # Calculate pip value (value of 1 pip for 1 lot)
        if symbol_info.trade_tick_value > 0 and symbol_info.trade_tick_size > 0:
            pip_value = symbol_info.trade_tick_value / symbol_info.trade_tick_size
        else:
            # Fallback calculation based on symbol type
            if 'JPY' in symbol:
                pip_value = 0.01  # For JPY pairs
            elif symbol in ['XAUUSD', 'XAGUSD']:
                pip_value = 1.0   # For precious metals
            else:
                pip_value = 10.0  # For major currency pairs
            
            logger.warning(f"Using fallback pip value for {symbol}: {pip_value}")
        
        # Calculate risk amount in account currency
        risk_amount = equity * risk_percentage
        
        # Calculate lot size: Risk Amount / (Risk in Pips × Pip Value)
        lot_size = risk_amount / (risk_pips * pip_value)
        
        # Apply symbol constraints
        min_volume = getattr(symbol_info, 'volume_min', 0.01)
        max_volume = getattr(symbol_info, 'volume_max', 100.0)
        volume_step = getattr(symbol_info, 'volume_step', 0.01)
        
        # Round to volume step
        lot_size = round(lot_size / volume_step) * volume_step
        
        # Ensure within bounds
        lot_size = max(min_volume, min(lot_size, max_volume))
        
        # Final validation
        if lot_size < min_volume:
            logger.warning(f"Calculated lot size too small for {symbol}: {lot_size}, using minimum: {min_volume}")
            lot_size = min_volume
        
        logger.info(f"Calculated lot size for {symbol}: {lot_size} "
                   f"(Risk: {risk_percentage*100:.1f}%, Pips: {risk_pips}, Equity: {equity})")
        return lot_size
        
    except Exception as e:
        logger.error(f"Error calculating lot size for {symbol}: {str(e)}")
        return 0.01

# Trade journaling with enhanced data capture
def log_trade_to_csv(trade: Dict, filename: str = "trade_journal.csv"):
    """
    Log comprehensive trade details to CSV file for analysis.
    
    Args:
        trade (Dict): Complete trade details dictionary
        filename (str): CSV filename for trade journal
    """
    try:
        # Prepare comprehensive trade data
        trade_data = {
            "timestamp": datetime.now(pytz.UTC).isoformat(),
            "setup_id": trade.get("id", "")[:8],  # Short ID for readability
            "symbol": trade.get("symbol", ""),
            "direction": trade.get("direction", ""),
            "state": trade.get("state", ""),
            "session": trade.get("session", ""),
            "entry_price": trade.get("entry_price"),
            "sl_price": trade.get("sl_price"),
            "tp1_price": trade.get("tp1_price"),
            "tp2_price": trade.get("tp2_price"),
            "lot_size": trade.get("lot_size"),
            "asian_high": trade.get("asian_high"),
            "asian_low": trade.get("asian_low"),
            "zone_score": trade.get("zone_score", 0),
            "zone_status": trade.get("zone_status", ""),
            "profit": trade.get("profit"),
            "outcome": trade.get("outcome"),
            "order_id": trade.get("order_id"),
            "created_time": trade.get("created_time"),
            
            # Calculate additional metrics
            "risk_reward_ratio": None,
            "risk_pips": None,
            "asian_range": None
        }
        
        # Calculate derived metrics
        if trade.get("entry_price") and trade.get("sl_price") and trade.get("tp1_price"):
            entry = float(trade["entry_price"])
            sl = float(trade["sl_price"])
            tp1 = float(trade["tp1_price"])
            
            if trade.get("direction") == "BULLISH":
                risk_pips = entry - sl
                reward_pips = tp1 - entry
            else:
                risk_pips = sl - entry
                reward_pips = entry - tp1
                
            if risk_pips > 0:
                trade_data["risk_reward_ratio"] = round(reward_pips / risk_pips, 2)
                trade_data["risk_pips"] = round(risk_pips, 5)
        
        if trade.get("asian_high") and trade.get("asian_low"):
            trade_data["asian_range"] = round(float(trade["asian_high"]) - float(trade["asian_low"]), 5)
        
        # Write to CSV
        df = pd.DataFrame([trade_data])
        
        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(filename)
        
        df.to_csv(filename, mode='a', index=False, header=not file_exists)
        
        logger.info(f"Trade logged to {filename}: {trade['symbol']} - {trade.get('state')} "
                   f"(RR: {trade_data.get('risk_reward_ratio', 'N/A')})")
        
    except Exception as e:
        logger.error(f"Error logging trade to CSV: {str(e)}")
        # Don't let logging failures affect trading

# Performance metrics calculation with advanced analytics
def calculate_performance_metrics(trades: List[Dict]) -> Dict:
    """
    Calculate comprehensive performance metrics for strategy evaluation.
    
    Args:
        trades (List[Dict]): List of completed trades
        
    Returns:
        Dict: Detailed performance metrics
    """
    try:
        if not trades:
            return {
                "total_trades": 0, "executed_trades": 0, "win_rate": 0,
                "total_profit": 0, "average_rr": 0, "max_drawdown": 0,
                "profit_factor": 0, "sharpe_ratio": 0
            }
        
        # Filter executed trades
        executed_trades = [t for t in trades if t.get("outcome") in ["Win", "Loss", "Breakeven"]]
        
        if not executed_trades:
            return {
                "total_trades": len(trades), "executed_trades": 0, "win_rate": 0,
                "total_profit": 0, "average_rr": 0, "max_drawdown": 0,
                "profit_factor": 0, "sharpe_ratio": 0
            }
        
        # Basic metrics
        total_trades = len(trades)
        executed_count = len(executed_trades)
        wins = sum(1 for t in executed_trades if t.get("outcome") == "Win")
        losses = sum(1 for t in executed_trades if t.get("outcome") == "Loss")
        
        win_rate = (wins / executed_count * 100) if executed_count > 0 else 0
        
        # Profit calculations
        profits = [float(t.get("profit", 0) or 0) for t in executed_trades]
        total_profit = sum(profits)
        
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        # Risk-reward calculations
        rr_ratios = []
        for t in executed_trades:
            try:
                entry = float(t.get("entry_price", 0) or 0)
                sl = float(t.get("sl_price", 0) or 0)
                tp = float(t.get("tp1_price", 0) or 0)
                
                if entry and sl and tp and entry != sl:
                    if t.get("direction") == "BULLISH":
                        rr = (tp - entry) / abs(entry - sl)
                    else:
                        rr = (entry - tp) / abs(sl - entry)
                    rr_ratios.append(rr)
            except:
                continue
        
        avg_rr = sum(rr_ratios) / len(rr_ratios) if rr_ratios else 0
        
        # Advanced metrics
        profit_factor = (sum(winning_trades) / abs(sum(losing_trades))) if losing_trades else float('inf')
        
        # Maximum drawdown calculation
        cumulative_profits = np.cumsum([0] + profits)
        running_max = np.maximum.accumulate(cumulative_profits)
        drawdowns = cumulative_profits - running_max
        max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0
        
        # Sharpe ratio (simplified - using profit standard deviation)
        if len(profits) > 1:
            profit_std = np.std(profits)
            sharpe_ratio = (np.mean(profits) / profit_std) if profit_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Compile metrics
        metrics = {
            "total_trades": total_trades,
            "executed_trades": executed_count,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 2),
            "total_profit": round(total_profit, 2),
            "average_profit_per_trade": round(total_profit / executed_count, 2) if executed_count > 0 else 0,
            "average_rr": round(avg_rr, 2),
            "max_drawdown": round(max_drawdown, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else 0,
            "sharpe_ratio": round(sharpe_ratio, 2),
            "largest_win": max(winning_trades) if winning_trades else 0,
            "largest_loss": min(losing_trades) if losing_trades else 0,
            "average_win": round(sum(winning_trades) / len(winning_trades), 2) if winning_trades else 0,
            "average_loss": round(sum(losing_trades) / len(losing_trades), 2) if losing_trades else 0
        }
        
        logger.info(f"Performance: WR={metrics['win_rate']}%, "
                   f"P/L={metrics['total_profit']}, RR={metrics['average_rr']}, "
                   f"PF={metrics['profit_factor']}")
        
        # Save detailed metrics to file
        timestamp = datetime.now(pytz.UTC).isoformat()
        with open("performance_metrics.txt", "a") as f:
            f.write(f"{timestamp}: {json.dumps(metrics, indent=2)}\n")
            
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {str(e)}")
        return {
            "total_trades": len(trades) if trades else 0,
            "error": str(e)
        }

# ZoneScorer class for zone validation
class ZoneScorer:
    def score_zone(self, zone: Dict, candles: pd.DataFrame, session: str, liquidity_grab: Optional[str]) -> Tuple[int, str]:
        """
        Score a detected zone (OB/FVG) based on specified criteria.
        Returns: (score, status) where status is 'VALID', 'WEAK', or 'INVALID'
        """
        try:
            score = 0
            zone_type = zone.get('type')
            
            # Order Block scoring
            if zone_type in ['OB_STRONG', 'OB_WEAK']:
                if zone_type == 'OB_STRONG':
                    score += 40
                    logger.debug(f"Strong OB detected: +40")
                else:
                    score += 20
                    logger.debug(f"Weak OB detected: +20")
            
            # FVG scoring
            elif zone_type in ['FVG_LARGE', 'FVG_SMALL']:
                if zone_type == 'FVG_LARGE':
                    score += 30
                    logger.debug(f"Large FVG detected: +30")
                else:
                    score += 15
                    logger.debug(f"Small FVG detected: +15")
            
            # Volume confirmation
            if 'tick_volume' in candles and len(candles) >= 20:
                avg_vol = candles['tick_volume'].rolling(window=20).mean().iloc[-1]
                if candles['tick_volume'].iloc[-1] > VOLUME_THRESHOLD * avg_vol:
                    score += 20
                    logger.debug(f"Volume spike detected: +20")
                else:
                    score += 10
                    logger.debug(f"Normal volume: +10")
            else:
                score += 10  # Default normal volume score
                logger.debug(f"No volume data, default: +10")
            
            # Liquidity sweep
            if liquidity_grab:
                score += 15
                logger.debug(f"Liquidity sweep detected: +15")
            
            # Session alignment
            if session in ['London', 'New York']:
                score += 10
                logger.debug(f"Zone in London/NY session: +10")
            else:
                logger.debug(f"Zone in Asian session: +0")
            
            # Determine status
            if score >= ZONE_SCORE_THRESHOLD_VALID:
                status = 'VALID'
                logger.info(f"Valid zone confirmed with score {score}")
            elif ZONE_SCORE_THRESHOLD_WEAK <= score < ZONE_SCORE_THRESHOLD_VALID:
                status = 'WEAK'
                logger.info(f"Weak zone detected with score {score}")
            else:
                status = 'INVALID'
                logger.info(f"Zone ignored with score {score}")
            
            return score, status
        
        except Exception as e:
            logger.error(f"Error in score_zone: {str(e)}")
            return 0, 'INVALID'

# Setup class
class Setup:
    """Enhanced Setup class with advanced features and better tracking."""
    
    def __init__(self, symbol: str, direction: Direction, limit_at_zone: bool = False, 
                 equity: float = 10000, risk_percentage: float = None):
        self.id = str(uuid.uuid4())
        self.symbol = symbol
        self.direction = direction
        self.state = State.IDLE
        self.htf_confirmed_time = None
        self.mtf_confirmed_time = None
        self.asian_high = None
        self.asian_low = None
        self.liquidity_grab_level = None
        self.order_block_zone = None
        self.fvg_zone = None
        self.entry_price = None
        self.sl_price = None
        self.tp1_price = None
        self.tp2_price = None
        self.features = {}
        self.session = None
        self.limit_at_zone = limit_at_zone
        self.ltf_signals = []
        self.order_id = None
        self.zone_score = 0
        self.zone_status = 'INVALID'
        self.equity = equity
        self.risk_percentage = risk_percentage or GENERAL_PARAMS.get("default_risk_percentage", 0.01)
        
        # Performance tracking
        self.created_time = datetime.now(pytz.UTC)
        self.profit = None
        self.outcome = None
        
        # Initialize with minimum lot size (will be updated when risk is known)
        symbol_info = mt5.symbol_info(symbol)
        self.lot_size = getattr(symbol_info, 'volume_min', 0.01) if symbol_info else 0.01

    def update_lot_size_with_risk(self, risk_pips: float):
        """Update lot size based on actual calculated risk in pips."""
        if risk_pips > 0:
            self.lot_size = calculate_lot_size(
                self.equity, 
                self.risk_percentage, 
                risk_pips, 
                self.symbol
            )
            logger.info(f"Updated lot size for {self.symbol}: {self.lot_size} "
                       f"(Risk: {risk_pips:.1f} pips)")

    def update_state(self, new_state: State, reason: str):
        """Update state with enhanced logging."""
        old_state = self.state
        self.state = new_state
        
        # Calculate time in previous state
        time_in_state = ""
        if old_state != State.IDLE and hasattr(self, 'created_time'):
            elapsed = datetime.now(pytz.UTC) - self.created_time
            time_in_state = f" (elapsed: {elapsed.total_seconds():.1f}s)"
        
        logger.info(f"Setup {self.id[:8]} ({self.symbol}, {self.direction.name}): "
                   f"{old_state.name} → {new_state.name} - {reason}{time_in_state}")

    def cancel_if_invalid(self, current_time: datetime, htf_data: pd.DataFrame) -> bool:
        if self.state == State.HTF_CONFIRMED and self.htf_confirmed_time:
            if current_time - self.htf_confirmed_time > HTF_EXPIRY:
                self.update_state(State.CANCELLED, "HTF confirmation expired")
                return True
        if self.state == State.MTF_PENDING and self.mtf_confirmed_time:
            if current_time - self.mtf_confirmed_time > MTF_EXPIRY:
                self.update_state(State.CANCELLED, "MTF confirmation expired")
                return True
        if self.state in [State.MTF_PENDING, State.LTF_READY, State.READY]:
            if not check_htf_bias(self.symbol, htf_data, self.direction):
                self.update_state(State.CANCELLED, "HTF bias flipped")
                return True
        return self.state == State.CANCELLED

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "direction": self.direction.name,
            "state": self.state.name,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp1_price": self.tp1_price,
            "tp2_price": self.tp2_price,
            "asian_high": self.asian_high,
            "asian_low": self.asian_low,
            "features": self.features,
            "session": self.session,
            "order_id": self.order_id,
            "zone_score": self.zone_score,
            "zone_status": self.zone_status
        }

# Initialize MT5
def initialize_mt5() -> bool:
    if not mt5.initialize():
        logger.error("MT5 initialization failed")
        return False
    logger.info("MT5 initialized successfully")
    return True

# Detect Asian session range
def detect_session_range(symbol: str, backtest_df: Optional[pd.DataFrame] = None) -> Tuple[Optional[float], Optional[float]]:
    nyt_tz = pytz.timezone('America/New_York')
    now = datetime.now(pytz.UTC).astimezone(nyt_tz)
    session_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    session_end = now.replace(hour=5, minute=0, second=0, microsecond=0)
    
    utc_start = session_start.astimezone(pytz.UTC)
    utc_end = session_end.astimezone(pytz.UTC)
    
    if backtest_df is not None:
        df = backtest_df[(backtest_df.index >= utc_start) & (backtest_df.index < utc_end)]
    else:
        candles = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, utc_start.timestamp(), utc_end.timestamp())
        if candles is None or len(candles) == 0:
            logger.error(f"Failed to fetch Asian session candles for {symbol}")
            return None, None
        df = pd.DataFrame(candles)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
    
    if df.empty:
        logger.info(f"No Asian session data for {symbol}")
        return None, None
    asian_high = df['high'].max()
    asian_low = df['low'].min()
    logger.info(f"Asian session range for {symbol}: High={asian_high}, Low={asian_low}")
    return asian_high, asian_low

# Check liquidity grab
def check_liquidity_grab(df: pd.DataFrame, asian_high: float, asian_low: float, mtf_data: pd.DataFrame = None) -> Tuple[Optional[str], Optional[float]]:
    try:
        if len(df) < 3:
            logger.info("Not enough data for liquidity grab detection")
            return None, None
            
        # Check last 3 candles for liquidity sweep
        for i in range(1, min(4, len(df))):
            candle = df.iloc[-i]
            prev_candle = df.iloc[-i-1] if len(df) > i else None
            
            # High liquidity grab (sweep above Asian high then reject)
            if (candle['high'] > asian_high * (1 + LIQUIDITY_THRESHOLD) and 
                candle['close'] < asian_high and 
                candle['close'] < candle['open']):
                
                # Confirm with wick rejection
                wick_ratio = (candle['high'] - candle['close']) / (candle['high'] - candle['low'] + 1e-6)
                if wick_ratio > 0.6:  # Strong upper wick
                    logger.info(f"Liquidity Grab High at {candle['high']} with wick rejection")
                    return 'Liquidity Grab High', candle['high']
            
            # Low liquidity grab (sweep below Asian low then reject)
            if (candle['low'] < asian_low * (1 - LIQUIDITY_THRESHOLD) and 
                candle['close'] > asian_low and 
                candle['close'] > candle['open']):
                
                # Confirm with wick rejection
                wick_ratio = (candle['close'] - candle['low']) / (candle['high'] - candle['low'] + 1e-6)
                if wick_ratio > 0.6:  # Strong lower wick
                    logger.info(f"Liquidity Grab Low at {candle['low']} with wick rejection")
                    return 'Liquidity Grab Low', candle['low']
        
        return None, None
    except Exception as e:
        logger.error(f"Error in check_liquidity_grab: {str(e)}")
        return None, None

# Check HTF bias
def check_htf_bias(symbol: str, df: pd.DataFrame, direction: Direction) -> bool:
    try:
        if df.empty or len(df) < 5:
            logger.info(f"No HTF data for {symbol}")
            return False
        highs, lows = df['high'].iloc[-5:], df['low'].iloc[-5:]
        if direction == Direction.BULLISH:
            is_bullish = highs.iloc[-1] > highs.iloc[-2] and lows.iloc[-1] > lows.iloc[-2]
            logger.info(f"HTF Bullish bias for {symbol}: {is_bullish}")
            return is_bullish
        else:
            is_bearish = highs.iloc[-1] < highs.iloc[-2] and lows.iloc[-1] < lows.iloc[-2]
            logger.info(f"HTF Bearish bias for {symbol}: {is_bearish}")
            return is_bearish
    except Exception as e:
        logger.error(f"Error in check_htf_bias for {symbol}: {str(e)}")
        return False

# Enhanced MTF zone detection with proper validation
def check_mtf_zone(symbol: str, df: pd.DataFrame, direction: Direction, session: str, liquidity_grab: Optional[str], zone_scorer: ZoneScorer, htf_data: pd.DataFrame = None) -> Tuple[Optional[Dict], int, str]:
    try:
        if df.empty or len(df) < 10:
            logger.info(f"Insufficient MTF data for {symbol}")
            return None, 0, 'INVALID'
        
        # First ensure MTF structure aligns with HTF bias
        if htf_data is not None and len(htf_data) >= 5:
            htf_trend_bullish = htf_data['close'].iloc[-1] > htf_data['close'].iloc[-5]
            mtf_trend_bullish = df['close'].iloc[-1] > df['close'].iloc[-5]
            
            # MTF should align with HTF trend
            if direction == Direction.BULLISH and not (htf_trend_bullish and mtf_trend_bullish):
                logger.info(f"MTF trend misalignment with HTF for {symbol}")
                return None, 0, 'INVALID'
            elif direction == Direction.BEARISH and not (not htf_trend_bullish and not mtf_trend_bullish):
                logger.info(f"MTF trend misalignment with HTF for {symbol}")
                return None, 0, 'INVALID'
        
        # Look for Order Blocks with improved logic
        atr = calculate_atr(df) or 0.01
        for i in range(-3, -min(OB_LOOKBACK+1, len(df)), -1):
            candle = df.iloc[i]
            next_candles = df.iloc[i+1:]
            
            if len(next_candles) == 0:
                continue
                
            candle_range = candle['high'] - candle['low']
            body_size = abs(candle['close'] - candle['open'])
            is_strong_candle = body_size > candle_range * 0.6  # Strong body
            
            zone = None
            if direction == Direction.BULLISH and candle['close'] > candle['open'] and is_strong_candle:
                ob_low, ob_high = candle['low'], candle['close']  # Use close for bullish OB
                
                # Check if price has returned to this zone
                price_tested_zone = any(c['low'] <= ob_high and c['close'] >= ob_low for _, c in next_candles.iterrows())
                current_price = df['close'].iloc[-1]
                
                if price_tested_zone and current_price >= ob_low:
                    zone_type = 'OB_STRONG' if candle_range > atr * 1.5 else 'OB_WEAK'
                    zone = {'type': zone_type, 'low': ob_low, 'high': ob_high, 'candle_time': candle.name}
                    
            elif direction == Direction.BEARISH and candle['close'] < candle['open'] and is_strong_candle:
                ob_low, ob_high = candle['close'], candle['high']  # Use close for bearish OB
                
                # Check if price has returned to this zone
                price_tested_zone = any(c['high'] >= ob_low and c['close'] <= ob_high for _, c in next_candles.iterrows())
                current_price = df['close'].iloc[-1]
                
                if price_tested_zone and current_price <= ob_high:
                    zone_type = 'OB_STRONG' if candle_range > atr * 1.5 else 'OB_WEAK'
                    zone = {'type': zone_type, 'low': ob_low, 'high': ob_high, 'candle_time': candle.name}
            
            if zone:
                score, status = zone_scorer.score_zone(zone, df, session, liquidity_grab)
                if status != 'INVALID':
                    logger.info(f"MTF Order Block detected for {symbol}: {zone_type} at {ob_low}-{ob_high}")
                    return zone, score, status
        
        # Look for Fair Value Gaps with enhanced detection
        for i in range(len(df)-2, max(0, len(df)-6), -1):
            if i < 2:
                continue
                
            curr, mid, prev = df.iloc[i], df.iloc[i-1], df.iloc[i-2]
            
            # Bullish FVG: curr.low > prev.high (gap up)
            if (direction == Direction.BULLISH and 
                curr['low'] > prev['high'] * (1 + FVG_THRESHOLD) and
                curr['close'] > curr['open']):
                
                gap_size = curr['low'] - prev['high']
                zone_type = 'FVG_LARGE' if gap_size > atr * 1.2 else 'FVG_SMALL'
                zone = {'type': zone_type, 'low': prev['high'], 'high': curr['low'], 'gap_size': gap_size}
                
                score, status = zone_scorer.score_zone(zone, df, session, liquidity_grab)
                if status != 'INVALID':
                    logger.info(f"MTF Bullish FVG detected for {symbol}: {prev['high']}-{curr['low']}")
                    return zone, score, status
                    
            # Bearish FVG: curr.high < prev.low (gap down)
            elif (direction == Direction.BEARISH and 
                  curr['high'] < prev['low'] * (1 - FVG_THRESHOLD) and
                  curr['close'] < curr['open']):
                
                gap_size = prev['low'] - curr['high']
                zone_type = 'FVG_LARGE' if gap_size > atr * 1.2 else 'FVG_SMALL'
                zone = {'type': zone_type, 'low': curr['high'], 'high': prev['low'], 'gap_size': gap_size}
                
                score, status = zone_scorer.score_zone(zone, df, session, liquidity_grab)
                if status != 'INVALID':
                    logger.info(f"MTF Bearish FVG detected for {symbol}: {curr['high']}-{prev['low']}")
                    return zone, score, status
        
        logger.info(f"No valid MTF OB/FVG for {symbol}")
        return None, 0, 'INVALID'
    except Exception as e:
        logger.error(f"Error in check_mtf_zone for {symbol}: {str(e)}")
        return None, 0, 'INVALID'

# Queue LTF signals
def queue_ltf_signal(symbol: str, df: pd.DataFrame, direction: Direction, zone: Dict) -> Optional[Dict]:
    try:
        if df.empty or len(df) < 5:
            logger.info(f"Insufficient LTF data for {symbol}")
            return None
            
        zone_low, zone_high = zone['low'], zone['high']
        curr = df.iloc[-1]
        
        # Enhanced BOS/CHoCH Detection
        if len(df) >= 10:
            recent_highs = df['high'].iloc[-10:]
            recent_lows = df['low'].iloc[-10:]
            
            if direction == Direction.BULLISH:
                # Look for break of structure (higher high)
                prev_swing_high = recent_highs.iloc[-5:-1].max()
                if (curr['close'] > zone_high and 
                    curr['high'] > prev_swing_high * 1.001 and  # 0.1% buffer
                    curr['close'] > curr['open']):
                    
                    # Confirm with volume if available
                    volume_confirmed = True
                    if 'tick_volume' in curr:
                        avg_volume = df['tick_volume'].iloc[-10:].mean()
                        volume_confirmed = curr['tick_volume'] > avg_volume * 1.2
                    
                    if volume_confirmed:
                        signal = {'type': 'BOS/CHoCH', 'price': curr['close'], 'time': curr.name, 'swing_break': prev_swing_high}
                        logger.info(f"Bullish BOS/CHoCH detected for {symbol} at {curr['close']}, broke {prev_swing_high}")
                        return signal
                        
            else:  # BEARISH
                # Look for break of structure (lower low)
                prev_swing_low = recent_lows.iloc[-5:-1].min()
                if (curr['close'] < zone_low and 
                    curr['low'] < prev_swing_low * 0.999 and  # 0.1% buffer
                    curr['close'] < curr['open']):
                    
                    # Confirm with volume if available
                    volume_confirmed = True
                    if 'tick_volume' in curr:
                        avg_volume = df['tick_volume'].iloc[-10:].mean()
                        volume_confirmed = curr['tick_volume'] > avg_volume * 1.2
                    
                    if volume_confirmed:
                        signal = {'type': 'BOS/CHoCH', 'price': curr['close'], 'time': curr.name, 'swing_break': prev_swing_low}
                        logger.info(f"Bearish BOS/CHoCH detected for {symbol} at {curr['close']}, broke {prev_swing_low}")
                        return signal
        
        # Enhanced Engulfing Pattern Detection
        if len(df) >= 3:
            prev = df.iloc[-2]
            curr_body = abs(curr['close'] - curr['open'])
            prev_body = abs(prev['close'] - prev['open'])
            candle_size = curr['high'] - curr['low']
            
            # Strong engulfing criteria
            if curr_body / (candle_size + 1e-6) >= ENGULFING_BODY_RATIO and curr_body > prev_body * 1.2:
                if (direction == Direction.BULLISH and 
                    curr['close'] > curr['open'] and curr['close'] > zone_high and
                    curr['open'] <= prev['close'] and curr['close'] >= prev['open']):
                    
                    signal = {'type': 'Engulfing', 'price': curr['close'], 'time': curr.name}
                    logger.info(f"Strong Bullish engulfing for {symbol} at {curr['close']}")
                    return signal
                    
                elif (direction == Direction.BEARISH and 
                      curr['close'] < curr['open'] and curr['close'] < zone_low and
                      curr['open'] >= prev['close'] and curr['close'] <= prev['open']):
                    
                    signal = {'type': 'Engulfing', 'price': curr['close'], 'time': curr.name}
                    logger.info(f"Strong Bearish engulfing for {symbol} at {curr['close']}")
                    return signal
        
        # Enhanced Wick Rejection Detection
        body = abs(curr['close'] - curr['open'])
        candle_range = curr['high'] - curr['low']
        
        if direction == Direction.BULLISH:
            # Bullish wick rejection: price touches/enters zone then rejects upward
            lower_wick = curr['close'] - curr['low'] if curr['close'] > curr['open'] else curr['open'] - curr['low']
            if (curr['low'] <= zone_high and curr['close'] > zone_high and
                lower_wick > body * 1.5 and  # Strong lower wick
                curr['close'] > curr['open']):
                
                signal = {'type': 'Wick Rejection', 'price': curr['close'], 'time': curr.name}
                logger.info(f"Bullish wick rejection for {symbol} at {curr['close']}")
                return signal
                
        else:  # BEARISH
            # Bearish wick rejection: price touches/enters zone then rejects downward
            upper_wick = curr['high'] - curr['close'] if curr['close'] < curr['open'] else curr['high'] - curr['open']
            if (curr['high'] >= zone_low and curr['close'] < zone_low and
                upper_wick > body * 1.5 and  # Strong upper wick
                curr['close'] < curr['open']):
                
                signal = {'type': 'Wick Rejection', 'price': curr['close'], 'time': curr.name}
                logger.info(f"Bearish wick rejection for {symbol} at {curr['close']}")
                return signal
        
        return None
    except Exception as e:
        logger.error(f"Error in queue_ltf_signal for {symbol}: {str(e)}")
        return None

# Confirm LTF signals
def confirm_ltf_signals(symbol: str, signals: List[Dict], df: pd.DataFrame, zone_status: str) -> Optional[float]:
    try:
        if len(signals) < 1:
            return None
        latest_signal = signals[-1]
        signal_time = latest_signal['time']
        signal_type = latest_signal['type']
        signal_price = latest_signal['price']
        subsequent_candles = df[df.index > signal_time]
        if len(subsequent_candles) < 1:
            return None
        confirm_count = 0
        for _, candle in subsequent_candles.iterrows():
            if 'tick_volume' in candle and candle['tick_volume'] > VOLUME_THRESHOLD * subsequent_candles['tick_volume'].mean():
                confirm_count += 1
            if signal_type in ['BOS/CHoCH', 'Engulfing', 'Wick Rejection']:
                if latest_signal['type'] == 'BOS/CHoCH':
                    highs, lows = subsequent_candles['high'], subsequent_candles['low']
                    if len(highs) >= 2 and ((latest_signal['direction'] == Direction.BULLISH and highs.iloc[-1] > highs.iloc[-2] and lows.iloc[-1] > lows.iloc[-2]) or \
                        (latest_signal['direction'] == Direction.BEARISH and highs.iloc[-1] < highs.iloc[-2] and lows.iloc[-1] < lows.iloc[-2])):
                        confirm_count += 1
                elif latest_signal['type'] == 'Engulfing':
                    if latest_signal['direction'] == Direction.BULLISH and candle['close'] > candle['open']:
                        confirm_count += 1
                    elif latest_signal['direction'] == Direction.BEARISH and candle['close'] < candle['open']:
                        confirm_count += 1
        required_confirmations = 2 if zone_status == 'WEAK' else 1
        if confirm_count >= required_confirmations:
            logger.info(f"LTF signal confirmed for {symbol}: {signal_type} at {signal_price} (Zone Status: {zone_status})")
            return signal_price
        return None
    except Exception as e:
        logger.error(f"Error in confirm_ltf_signals for {symbol}: {str(e)}")
        return None

# Enhanced ML feature extraction with decision integration
class MLSignalFilter:
    """Simple ML-based signal filtering using extracted features."""
    
    def __init__(self):
        self.enabled = GENERAL_PARAMS.get("use_ml_filtering", False)
        self.min_score_threshold = 0.6  # Minimum score to accept signal
        self.feature_weights = {
            'volume_spike': 0.2,
            'liquidity_sweep': 0.25,
            'is_london': 0.1,
            'is_new_york': 0.1,
            'range_width': 0.15,
            'candle_range_ratio': 0.1,
            'wick_rejection_strength': 0.1
        }
    
    def should_accept_signal(self, features: Dict, zone_status: str, signal_type: str) -> Tuple[bool, float]:
        """
        Determine if a signal should be accepted based on ML features.
        
        Args:
            features (Dict): Extracted ML features
            zone_status (str): Zone status (VALID, WEAK, INVALID)
            signal_type (str): Type of signal detected
            
        Returns:
            Tuple[bool, float]: (accept_signal, confidence_score)
        """
        if not self.enabled:
            return True, 1.0
        
        try:
            score = 0.0
            
            # Base score from zone status
            if zone_status == 'VALID':
                score += 0.4
            elif zone_status == 'WEAK':
                score += 0.2
            else:
                return False, 0.0
            
            # Feature-based scoring
            for feature, weight in self.feature_weights.items():
                if feature in features:
                    feature_value = features[feature]
                    
                    if feature == 'volume_spike':
                        score += weight if feature_value else 0
                    elif feature == 'liquidity_sweep':
                        score += weight if feature_value else 0
                    elif feature in ['is_london', 'is_new_york']:
                        score += weight if feature_value else 0
                    elif feature == 'range_width':
                        # Prefer moderate range widths (not too tight, not too wide)
                        if 0.5 <= feature_value <= 2.0:
                            score += weight
                    elif feature == 'candle_range_ratio':
                        # Prefer stronger candles
                        if feature_value >= 1.0:
                            score += weight
                    elif feature == 'wick_rejection_strength':
                        # Calculate wick rejection strength from features
                        wick_up = features.get('wick_up_ratio', 0)
                        wick_dn = features.get('wick_dn_ratio', 0)
                        wick_strength = max(wick_up, wick_dn)
                        if wick_strength >= 1.5:
                            score += weight
            
            # Signal type bonus
            if signal_type in ['BOS/CHoCH', 'Wick Rejection']:
                score += 0.1
            elif signal_type == 'Engulfing':
                score += 0.05
            
            accept = score >= self.min_score_threshold
            
            logger.debug(f"ML Filter: Score={score:.3f}, Threshold={self.min_score_threshold}, "
                        f"Accept={accept}, Zone={zone_status}, Signal={signal_type}")
            
            return accept, score
            
        except Exception as e:
            logger.error(f"Error in ML signal filter: {str(e)}")
            return True, 0.5  # Default accept with low confidence

# Enhanced ATR calculation with symbol-specific periods
def calculate_atr(df: pd.DataFrame, symbol: str = None, period: int = None) -> float:
    """
    Calculate ATR with symbol-specific period configuration.
    
    Args:
        df (pd.DataFrame): OHLC data
        symbol (str, optional): Trading symbol for parameter lookup
        period (int, optional): ATR period override
        
    Returns:
        float: Calculated ATR value
    """
    if df.empty:
        return 0.0
        
    # Get symbol-specific ATR period from config
    if period is None:
        if symbol and symbol in SYMBOL_PARAMS:
            period = SYMBOL_PARAMS[symbol].get("atr_period", ATR_PERIOD)
        else:
            period = ATR_PERIOD
    
    if len(df) < period + 1:
        logger.warning(f"Insufficient data for ATR calculation: {len(df)} < {period + 1}")
        return 0.0
        
    try:
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        # Validate ATR value
        if pd.isna(atr) or atr <= 0:
            logger.warning(f"Invalid ATR calculated for {symbol}: {atr}")
            return 0.0
            
        logger.debug(f"ATR for {symbol} (period {period}): {atr:.5f}")
        return atr
        
    except Exception as e:
        logger.error(f"Error calculating ATR for {symbol}: {str(e)}")
        return 0.0

# Enhanced SL/TP calculation with spread, slippage, and symbol-specific parameters
def calculate_sl_tp(symbol: str, entry: float, direction: Direction, atr: float, 
                   asian_levels: Tuple[float, float], zone: Dict, 
                   mtf_data: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calculate SL/TP with comprehensive spread, slippage, and symbol-specific parameters.
    
    Args:
        symbol (str): Trading symbol
        entry (float): Entry price
        direction (Direction): Trade direction
        atr (float): Average True Range
        asian_levels (Tuple[float, float]): Asian session high and low
        zone (Dict): Order block or FVG zone
        mtf_data (pd.DataFrame): MTF candle data
        
    Returns:
        Tuple[Optional[float], Optional[float], Optional[float]]: SL, TP1, TP2 prices
    """
    try:
        asian_high, asian_low = asian_levels
        zone_low, zone_high = zone['low'], zone['high']
        
        # Get symbol info and parameters
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Cannot get symbol info for {symbol}")
            return None, None, None
            
        decimals = symbol_info.digits
        point = symbol_info.point
        
        # Get symbol-specific parameters with enhanced defaults
        symbol_params = SYMBOL_PARAMS.get(symbol, {})
        atr_multiplier = symbol_params.get("atr_multiplier", 2.0)
        slippage_buffer = symbol_params.get("slippage_buffer", 0.5)  # pips
        max_spread_pips = symbol_params.get("max_spread_pips", 1.0)
        
        # Get current spread and validate trading conditions
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            current_spread = (tick.ask - tick.bid)
            spread_pips = current_spread / point
            
            # Check if spread is acceptable for trading
            if spread_pips > max_spread_pips:
                logger.warning(f"High spread for {symbol}: {spread_pips:.1f} pips > {max_spread_pips}")
                # Consider rejecting the trade if spread is too high
                if spread_pips > max_spread_pips * 2:
                    logger.error(f"Spread too high for {symbol}, rejecting trade")
                    return None, None, None
        else:
            current_spread = point * 2  # Estimate 2 pips if tick unavailable
            spread_pips = 2.0
            logger.warning(f"Cannot get current spread for {symbol}, using estimate: {spread_pips} pips")
        
        # Use MTF swing points for better SL placement
        mtf_recent = mtf_data.tail(10) if len(mtf_data) >= 10 else mtf_data
        
        if direction == Direction.BULLISH:
            # Find recent MTF low for better SL placement
            recent_low = mtf_recent['low'].min()
            sl_candidates = [
                zone_low - (atr * atr_multiplier),
                recent_low - (atr * 0.5)
            ]
            if asian_low:
                sl_candidates.append(asian_low - (atr * atr_multiplier))
            sl = min(sl_candidates)
            
            risk = entry - sl
            if risk <= 0:
                logger.warning(f"Invalid risk for {symbol}: {risk}")
                return None, None, None
            
            # Account for spread and slippage in effective entry
            spread_cost = current_spread / 2  # Half spread cost
            slippage_cost = slippage_buffer * point
            effective_entry = entry + spread_cost + slippage_cost
            effective_risk = effective_entry - sl
            
            # Calculate TP levels with spread/slippage consideration
            tp1 = effective_entry + (effective_risk * MIN_RR)
            tp2 = effective_entry + (effective_risk * MIN_RR * 1.5)
            
        else:  # BEARISH
            # Find recent MTF high for better SL placement
            recent_high = mtf_recent['high'].max()
            sl_candidates = [
                zone_high + (atr * atr_multiplier),
                recent_high + (atr * 0.5)
            ]
            if asian_high:
                sl_candidates.append(asian_high + (atr * atr_multiplier))
            sl = max(sl_candidates)
            
            risk = sl - entry
            if risk <= 0:
                logger.warning(f"Invalid risk for {symbol}: {risk}")
                return None, None, None
            
            # Account for spread and slippage in effective entry
            spread_cost = current_spread / 2  # Half spread cost
            slippage_cost = slippage_buffer * point
            effective_entry = entry - spread_cost - slippage_cost
            effective_risk = sl - effective_entry
            
            # Calculate TP levels with spread/slippage consideration
            tp1 = effective_entry - (effective_risk * MIN_RR)
            tp2 = effective_entry - (effective_risk * MIN_RR * 1.5)
        
        # Final validation - ensure minimum RR after spread/slippage
        actual_rr = abs(tp1 - effective_entry) / abs(effective_entry - sl)
        min_acceptable_rr = MIN_RR * 0.9  # Allow 10% tolerance
        
        if actual_rr < min_acceptable_rr:
            logger.info(f"Effective RR too low for {symbol}: {actual_rr:.2f} < {min_acceptable_rr:.2f} "
                       f"(spread: {spread_pips:.1f} pips, slippage: {slippage_buffer} pips)")
            return None, None, None
        
        # Round to symbol precision
        sl = round(sl, decimals)
        tp1 = round(tp1, decimals)
        tp2 = round(tp2, decimals)
        
        # Calculate risk in pips for lot size adjustment
        risk_pips = abs(effective_entry - sl) / point
        
        logger.info(f"Enhanced SL/TP for {symbol}: SL={sl}, TP1={tp1}, TP2={tp2} "
                   f"(Effective RR: {actual_rr:.2f}, Spread: {spread_pips:.1f} pips, "
                   f"Risk: {risk_pips:.1f} pips)")
        
        return sl, tp1, tp2
        
    except Exception as e:
        logger.error(f"Error in calculate_sl_tp for {symbol}: {str(e)}")
        return None, None, None

# Enhanced position management with complete trade outcome tracking
def manage_position(setup: Setup, debug_mode: bool = False, 
                   use_trailing_stop: bool = None) -> bool:
    """
    Enhanced position management with complete outcome tracking and trailing stops.
    
    Args:
        setup (Setup): Trade setup object
        debug_mode (bool): Debug mode flag
        use_trailing_stop (bool): Enable trailing stop
        
    Returns:
        bool: True if position is still active, False if closed
    """
    if debug_mode or setup.order_id is None:
        return True
    
    try:
        # Check if position still exists
        positions = mt5.positions_get(symbol=setup.symbol)
        position = None
        
        for pos in positions:
            if pos.ticket == setup.order_id:
                position = pos
                break
        
        if not position:
            # Position closed - determine outcome and profit
            return _handle_closed_position(setup)
        
        # Position is still active - manage it
        return _handle_active_position(setup, position, use_trailing_stop)
        
    except Exception as e:
        logger.error(f"Error in manage_position for {setup.symbol}: {str(e)}")
        return True

def _handle_closed_position(setup: Setup) -> bool:
    """Handle closed position with comprehensive outcome tracking."""
    try:
        # Get deal history for this position
        deals = mt5.history_deals_get(position=setup.order_id)
        
        if deals and len(deals) > 0:
            # Calculate total profit from all deals
            total_profit = sum(deal.profit for deal in deals)
            setup.profit = total_profit
            
            # Get entry and exit details
            entry_deal = next((d for d in deals if d.entry == 1), None)  # Entry deal
            exit_deals = [d for d in deals if d.entry == 0]  # Exit deals
            
            # Determine outcome based on profit and exit reason
            if total_profit > 0.01:  # Account for small rounding differences
                setup.outcome = "Win"
            elif total_profit < -0.01:
                setup.outcome = "Loss"
            else:
                setup.outcome = "Breakeven"
            
            # Enhanced logging with deal details
            if entry_deal:
                actual_entry = entry_deal.price
                exit_prices = [d.price for d in exit_deals] if exit_deals else []
                
                logger.info(f"Position closed for {setup.symbol}: "
                           f"Entry={actual_entry}, Exit(s)={exit_prices}, "
                           f"Profit={total_profit:.2f}, Outcome={setup.outcome}")
                
                # Update setup with actual entry if different from planned
                if abs(actual_entry - (setup.entry_price or 0)) > 0.0001:
                    logger.info(f"Actual entry price differs: planned={setup.entry_price}, "
                               f"actual={actual_entry}")
                    setup.entry_price = actual_entry
                
                # Calculate actual risk-reward if we have all data
                if setup.sl_price and setup.tp1_price:
                    if setup.direction == Direction.BULLISH:
                        actual_risk = actual_entry - setup.sl_price
                        actual_reward = setup.tp1_price - actual_entry
                    else:
                        actual_risk = setup.sl_price - actual_entry
                        actual_reward = actual_entry - setup.tp1_price
                    
                    if actual_risk > 0:
                        actual_rr = actual_reward / actual_risk
                        logger.info(f"Actual R/R for {setup.symbol}: {actual_rr:.2f}")
            
        else:
            # No deal history found - position may have been cancelled or closed manually
            setup.outcome = "Cancelled"
            setup.profit = 0
            logger.info(f"No deal history found for {setup.symbol} position {setup.order_id}")
        
        # Update trade journal with final outcome
        trade_data = setup.to_dict()
        trade_data["profit"] = setup.profit
        trade_data["outcome"] = setup.outcome
        log_trade_to_csv(trade_data)
        
        return False  # Position is closed
        
    except Exception as e:
        logger.error(f"Error handling closed position for {setup.symbol}: {str(e)}")
        setup.outcome = "Error"
        setup.profit = 0
        return False

def _handle_active_position(setup: Setup, position, use_trailing_stop: bool = None) -> bool:
    """Handle active position management with enhanced trailing stops."""
    try:
        if use_trailing_stop is None:
            use_trailing_stop = GENERAL_PARAMS.get("use_trailing_stop", False)
        
        current_price_tick = mt5.symbol_info_tick(setup.symbol)
        if not current_price_tick:
            logger.warning(f"Cannot get current price for {setup.symbol}")
            return True
        
        # Use appropriate price for direction
        current_price = current_price_tick.bid if setup.direction == Direction.BEARISH else current_price_tick.ask
        
        # Check for TP1 hit and partial close
        tp1_hit = False
        if setup.tp1_price:
            if setup.direction == Direction.BULLISH:
                tp1_hit = current_price >= setup.tp1_price
            else:
                tp1_hit = current_price <= setup.tp1_price
        
        # Partial close at TP1 (only if we haven't done it already)
        if tp1_hit and abs(position.volume - setup.lot_size) < 0.001:  # First TP1 hit
            success = _handle_partial_close(setup, position, current_price)
            if success:
                logger.info(f"Successfully handled partial close for {setup.symbol}")
            else:
                logger.warning(f"Failed to handle partial close for {setup.symbol}")
        
        # Implement trailing stop after partial close
        if use_trailing_stop and position.volume < setup.lot_size * 0.9:  # After partial close
            success = _update_trailing_stop(setup, position, current_price)
            if not success:
                logger.debug(f"Trailing stop not updated for {setup.symbol}")
        
        return True  # Position still active
        
    except Exception as e:
        logger.error(f"Error handling active position for {setup.symbol}: {str(e)}")
        return True

def _handle_partial_close(setup: Setup, position, current_price: float) -> bool:
    """Handle partial position close at TP1 with enhanced error handling."""
    try:
        # Close 50% of position
        close_volume = round(setup.lot_size / 2, 2)  # Ensure proper rounding
        
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": setup.symbol,
            "volume": close_volume,
            "type": mt5.ORDER_TYPE_SELL if setup.direction == Direction.BULLISH else mt5.ORDER_TYPE_BUY,
            "position": position.ticket,
            "price": current_price,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "comment": f"TP1_50%_{setup.id[:8]}"
        }
        
        result = mt5.order_send(close_request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_code, error_msg = mt5.last_error()
            logger.error(f"Failed to close 50% for {setup.symbol}: {result.comment}, "
                        f"MT5 Error: {error_code} - {error_msg}")
            return False
        
        logger.info(f"Closed 50% ({close_volume} lots) at TP1 for {setup.symbol}")
        
        # Move SL to breakeven
        if setup.entry_price:
            sl_request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": setup.symbol,
                "position": position.ticket,
                "sl": setup.entry_price,
                "tp": setup.tp2_price,
                "comment": f"SL_to_BE_{setup.id[:8]}"
            }
            
            result = mt5.order_send(sl_request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_code, error_msg = mt5.last_error()
                logger.error(f"Failed to move SL to BE for {setup.symbol}: {result.comment}, "
                            f"MT5 Error: {error_code} - {error_msg}")
            else:
                logger.info(f"Moved SL to breakeven ({setup.entry_price}) for {setup.symbol}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in partial close for {setup.symbol}: {str(e)}")
        return False

def _update_trailing_stop(setup: Setup, position, current_price: float) -> bool:
    """Update trailing stop based on symbol-specific ATR multiplier."""
    try:
        # Get fresh candle data for ATR calculation
        ltf_candles = get_candles(setup.symbol, TIMEFRAMES['LTF'], 25)  # Extra candles for ATR
        if ltf_candles.empty:
            logger.warning(f"No LTF data for trailing stop calculation: {setup.symbol}")
            return False
        
        atr = calculate_atr(ltf_candles, setup.symbol)
        if atr <= 0:
            logger.warning(f"Invalid ATR for trailing stop: {setup.symbol}")
            return False
        
        # Get symbol-specific trailing multiplier
        symbol_params = SYMBOL_PARAMS.get(setup.symbol, {})
        trailing_multiplier = symbol_params.get("trailing_atr_multiplier", 1.5)
        
        # Calculate new trailing stop
        if setup.direction == Direction.BULLISH:
            new_sl = current_price - (atr * trailing_multiplier)
            should_update = new_sl > position.sl and new_sl > setup.entry_price
        else:
            new_sl = current_price + (atr * trailing_multiplier)
            should_update = new_sl < position.sl and new_sl < setup.entry_price
        
        if should_update:
            # Round to symbol precision
            symbol_info = mt5.symbol_info(setup.symbol)
            if symbol_info:
                new_sl = round(new_sl, symbol_info.digits)
            
            sl_request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": setup.symbol,
                "position": position.ticket,
                "sl": new_sl,
                "tp": position.tp,
                "comment": f"Trail_SL_{setup.id[:8]}"
            }
            
            result = mt5.order_send(sl_request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Updated trailing stop for {setup.symbol}: {position.sl:.5f} -> {new_sl:.5f}")
                return True
            else:
                error_code, error_msg = mt5.last_error()
                logger.error(f"Failed to update trailing stop for {setup.symbol}: {result.comment}, "
                            f"MT5 Error: {error_code} - {error_msg}")
        
        return False
        
    except Exception as e:
        logger.error(f"Error updating trailing stop for {setup.symbol}: {str(e)}")
        return False

# Backtesting position management simulation
def simulate_position_closure(setup: Setup, df: pd.DataFrame) -> bool:
    """
    Simulate position closure for backtesting mode.
    
    Args:
        setup (Setup): Trade setup object
        df (pd.DataFrame): OHLC data for simulation
        
    Returns:
        bool: True if position is still active, False if closed
    """
    try:
        if not setup.entry_price or not setup.sl_price or not setup.tp1_price:
            return True
        
        # Get current candle
        if df.empty:
            return True
            
        current_candle = df.iloc[-1]
        
        # Check for SL hit
        if setup.direction == Direction.BULLISH:
            if current_candle['low'] <= setup.sl_price:
                # SL hit
                setup.outcome = "Loss"
                setup.profit = -abs(setup.entry_price - setup.sl_price) * setup.lot_size * 100000  # Rough calculation
                logger.info(f"Backtest: SL hit for {setup.symbol} at {current_candle['low']}")
                return False
                
            elif current_candle['high'] >= setup.tp1_price:
                # TP1 hit
                setup.outcome = "Win"
                setup.profit = abs(setup.tp1_price - setup.entry_price) * setup.lot_size * 100000  # Rough calculation
                logger.info(f"Backtest: TP1 hit for {setup.symbol} at {current_candle['high']}")
                return False
                
        else:  # BEARISH
            if current_candle['high'] >= setup.sl_price:
                # SL hit
                setup.outcome = "Loss"
                setup.profit = -abs(setup.sl_price - setup.entry_price) * setup.lot_size * 100000  # Rough calculation
                logger.info(f"Backtest: SL hit for {setup.symbol} at {current_candle['high']}")
                return False
                
            elif current_candle['low'] <= setup.tp1_price:
                # TP1 hit
                setup.outcome = "Win"
                setup.profit = abs(setup.entry_price - setup.tp1_price) * setup.lot_size * 100000  # Rough calculation
                logger.info(f"Backtest: TP1 hit for {setup.symbol} at {current_candle['low']}")
                return False
        
        return True  # Position still active
        
    except Exception as e:
        logger.error(f"Error in simulate_position_closure for {setup.symbol}: {str(e)}")
        return True

# Updated state machine with enhanced risk calculations
def manage_state(setup: Setup, df_dict: Dict[str, pd.DataFrame], timeframes: Dict[str, int], 
                backtest_mode: bool = False, zone_scorer: ZoneScorer = None, 
                use_trailing_stop: bool = False) -> None:
    """Enhanced state machine with improved risk management and outcome tracking."""
    zone_scorer = zone_scorer or ZoneScorer()
    current_time = datetime.now(pytz.UTC) if not backtest_mode else df_dict['LTF'].index[-1]
    
    if setup.cancel_if_invalid(current_time, df_dict.get('HTF')):
        return

    if setup.state == State.IDLE:
        # Step 1: Get HTF bias and Asian session range
        setup.asian_high, setup.asian_low = detect_session_range(setup.symbol, df_dict.get('HTF'))
        setup.session = detect_session(current_time, setup.symbol)
        
        if setup.asian_high is None or setup.asian_low is None:
            setup.update_state(State.CANCELLED, "No Asian session range available")
            return
            
        if check_htf_bias(setup.symbol, df_dict.get('HTF'), setup.direction):
            setup.htf_confirmed_time = current_time
            setup.update_state(State.HTF_CONFIRMED, "HTF bias confirmed")
        else:
            setup.update_state(State.CANCELLED, "HTF bias not confirmed")

    elif setup.state == State.HTF_CONFIRMED:
        # Step 2: Check for liquidity grab and MTF zone
        ltf_df = df_dict.get('LTF')
        liquidity_grab, grab_level = check_liquidity_grab(ltf_df, setup.asian_high, setup.asian_low, df_dict.get('MTF')) if ltf_df is not None else (None, None)
        setup.liquidity_grab_level = grab_level
        
        # Enhanced MTF zone detection with HTF confirmation
        zone, score, status = check_mtf_zone(setup.symbol, df_dict.get('MTF'), setup.direction, setup.session, liquidity_grab, zone_scorer, df_dict.get('HTF'))
        
        if zone and status != 'INVALID':
            setup.order_block_zone = zone
            setup.zone_score = score
            setup.zone_status = status
            setup.mtf_confirmed_time = current_time
            setup.update_state(State.MTF_PENDING, f"MTF zone confirmed: {zone['type']} with score {score}")
        else:
            setup.update_state(State.CANCELLED, f"No valid MTF zone found, score: {score}")

    elif setup.state == State.MTF_PENDING:
        # Step 3: Wait for LTF entry signals
        ltf_df = df_dict.get('LTF')
        if ltf_df is None or len(ltf_df) < 5:
            return
            
        signal = queue_ltf_signal(setup.symbol, ltf_df, setup.direction, setup.order_block_zone)
        if signal:
            signal['direction'] = setup.direction
            setup.ltf_signals.append(signal)
            
            # Keep only last 3 signals
            if len(setup.ltf_signals) > 3:
                setup.ltf_signals = setup.ltf_signals[-3:]
            
            entry_price = confirm_ltf_signals(setup.symbol, setup.ltf_signals, ltf_df, setup.zone_status)
            if entry_price:
                setup.entry_price = entry_price
                
                # Calculate ATR with symbol-specific period
                atr = calculate_atr(df_dict.get('MTF'), setup.symbol)
                setup.features = extract_ml_features(ltf_df, setup.asian_high, setup.asian_low, setup.session, setup.direction, setup.liquidity_grab_level, atr)
                
                # Enhanced SL/TP calculation with spread/slippage
                sl, tp1, tp2 = calculate_sl_tp(setup.symbol, entry_price, setup.direction, atr, 
                                             (setup.asian_high, setup.asian_low), setup.order_block_zone, df_dict.get('MTF'))
                
                if sl is None:
                    setup.update_state(State.CANCELLED, "Invalid SL/TP calculation (spread/slippage too high)")
                    return
                    
                setup.sl_price = sl
                setup.tp1_price = tp1
                setup.tp2_price = tp2
                
                # Update lot size based on calculated risk
                risk_pips = abs(entry_price - sl) / (mt5.symbol_info(setup.symbol).point if mt5.symbol_info(setup.symbol) else 0.00001)
                setup.update_lot_size_with_risk(risk_pips)
                
                setup.update_state(State.LTF_READY, f"LTF entry confirmed at {entry_price}")

    elif setup.state == State.LTF_READY:
        # Step 4: Final validation before order placement
        # Re-confirm HTF bias
        if not check_htf_bias(setup.symbol, df_dict.get('HTF'), setup.direction):
            setup.update_state(State.CANCELLED, "HTF bias no longer valid")
            return
            
        # Re-confirm MTF zone
        zone, score, status = check_mtf_zone(setup.symbol, df_dict.get('MTF'), setup.direction, setup.session, setup.liquidity_grab_level, zone_scorer, df_dict.get('HTF'))
        if not zone or status == 'INVALID':
            setup.update_state(State.CANCELLED, f"MTF zone invalidated, score: {score}")
            return
            
        setup.update_state(State.READY, f"All confirmations passed, ready to trade")

    elif setup.state == State.READY:
        if place_order(setup, backtest_mode):
            setup.update_state(State.EXECUTED, "Order placed successfully")
        else:
            setup.update_state(State.CANCELLED, "Order placement failed")

    elif setup.state == State.EXECUTED:
        # Enhanced position management
        if backtest_mode:
            # Use simulation for backtesting
            position_active = simulate_position_closure(setup, df_dict.get('LTF'))
            if not position_active:
                # Position closed in backtest
                trade_data = setup.to_dict()
                trade_data["profit"] = setup.profit
                trade_data["outcome"] = setup.outcome
                log_trade_to_csv(trade_data)
        else:
            # Use real position management for live trading
            manage_position(setup, debug_mode=False, use_trailing_stop=use_trailing_stop)

# Add missing extract_ml_features function
def extract_ml_features(df: pd.DataFrame, asian_high: float, asian_low: float, 
                       session: str, direction: Direction, liquidity_grab: Optional[str], 
                       atr: float) -> Dict:
    """
    Extract ML features from market data for signal filtering.
    
    Args:
        df: LTF candlestick data
        asian_high: Asian session high
        asian_low: Asian session low
        session: Current trading session
        direction: Trade direction
        liquidity_grab: Liquidity grab type if detected
        atr: Current ATR value
        
    Returns:
        Dict: Extracted features for ML filtering
    """
    try:
        if df.empty:
            return {}
            
        last_candle = df.iloc[-1]
        
        # Basic candle features
        candle_range = last_candle['high'] - last_candle['low']
        body_size = abs(last_candle['close'] - last_candle['open'])
        upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
        
        # Volume features
        volume_spike = False
        if 'tick_volume' in last_candle and len(df) >= 20:
            avg_volume = df['tick_volume'].rolling(window=20).mean().iloc[-1]
            volume_spike = last_candle['tick_volume'] > VOLUME_THRESHOLD * avg_volume
        
        # Calculate features
        features = {
            'atr': atr,
            'range_width': (asian_high - asian_low) / max(atr, 0.0001),
            'candle_range_ratio': candle_range / max(atr, 0.0001),
            'wick_up_ratio': upper_wick / max(body_size, 0.0001),
            'wick_dn_ratio': lower_wick / max(body_size, 0.0001),
            'is_london': 1 if session == 'London' else 0,
            'is_new_york': 1 if session == 'New York' else 0,
            'is_bullish_bias': 1 if direction == Direction.BULLISH else 0,
            'volume_spike': 1 if volume_spike else 0,
            'liquidity_sweep': 1 if liquidity_grab is not None else 0,
            'distance_from_high': abs(asian_high - last_candle['close']) / max(atr, 0.0001),
            'distance_from_low': abs(last_candle['close'] - asian_low) / max(atr, 0.0001)
        }
        
        logger.debug(f"Extracted ML features: {features}")
        return features
        
    except Exception as e:
        logger.error(f"Error extracting ML features: {str(e)}")
        return {}

# Add missing place_order function
def place_order(setup: Setup, debug_mode: bool = False) -> bool:
    """
    Place trading order based on setup configuration.
    
    Args:
        setup: Trade setup object
        debug_mode: If True, simulate order placement
        
    Returns:
        bool: True if order placed successfully
    """
    try:
        if debug_mode:
            logger.info(f"[DEBUG MODE] Would place order for {setup.symbol}: "
                       f"Direction={setup.direction.name}, Entry={setup.entry_price}, "
                       f"SL={setup.sl_price}, TP1={setup.tp1_price}, Lot Size={setup.lot_size}")
            setup.order_id = f"DEBUG_{setup.id[:8]}"
            return True
        
        # Validate setup has required data
        if not all([setup.entry_price, setup.sl_price, setup.tp1_price, setup.lot_size]):
            logger.error(f"Missing required order data for {setup.symbol}")
            return False
        
        # Get symbol info
        symbol_info = mt5.symbol_info(setup.symbol)
        if not symbol_info:
            logger.error(f"Cannot get symbol info for {setup.symbol}")
            return False
        
        # Determine order type
        if setup.limit_at_zone:
            # Place limit order at zone level
            order_type = mt5.ORDER_TYPE_BUY_LIMIT if setup.direction == Direction.BULLISH else mt5.ORDER_TYPE_SELL_LIMIT
            price = setup.order_block_zone.get('high') if setup.direction == Direction.BULLISH else setup.order_block_zone.get('low')
        else:
            # Place market order
            order_type = mt5.ORDER_TYPE_BUY if setup.direction == Direction.BULLISH else mt5.ORDER_TYPE_SELL
            price = setup.entry_price
        
        # Prepare order request
        request = {
            "action": mt5.TRADE_ACTION_PENDING if setup.limit_at_zone else mt5.TRADE_ACTION_DEAL,
            "symbol": setup.symbol,
            "volume": setup.lot_size,
            "type": order_type,
            "price": price,
            "sl": setup.sl_price,
            "tp": setup.tp1_price,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "comment": f"Judas_{setup.direction.name}_{setup.id[:8]}",
            "magic": 123456
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_code, error_msg = mt5.last_error()
            logger.error(f"Order failed for {setup.symbol}: {result.comment}, "
                        f"MT5 Error: {error_code} - {error_msg}")
            return False
        
        # Store order/position ID
        setup.order_id = result.order if setup.limit_at_zone else result.deal
        
        logger.info(f"Order placed successfully for {setup.symbol}: "
                   f"ID={setup.order_id}, Type={order_type}, Price={price}, "
                   f"Volume={setup.lot_size}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error placing order for {setup.symbol}: {str(e)}")
        return False

# === Strategy Wrapper (added to satisfy external import: JudasSwingStrategy) ===
class JudasSwingStrategy:
    """
    Lightweight wrapper exposing a step/run API around the core state machine.
    Fixes ImportError: cannot import name 'JudasSwingStrategy'.
    """
    def __init__(self, symbols: Optional[List[str]] = None, backtest_mode: bool = False,
                 use_trailing_stop: bool = False, equity: float = 10000):
        self.symbols = symbols or SYMBOLS
        self.backtest_mode = backtest_mode
        self.use_trailing_stop = use_trailing_stop
        self.equity = equity
        self.zone_scorer = ZoneScorer()
        self.setups: Dict[str, Setup] = {}
        self.completed_trades: List[Dict] = []
        self._initialized = False
        if not self.backtest_mode:
            self._initialized = initialize_mt5()

    def _fetch_symbol_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        return {
            'HTF': get_candles(symbol, TIMEFRAMES['HTF'], 60),
            'MTF': get_candles(symbol, TIMEFRAMES['MTF'], 40),
            'LTF': get_candles(symbol, TIMEFRAMES['LTF'], 25)
        }

    def _ensure_setups(self, symbol: str):
        active = [s for s in self.setups.values()
                  if s.symbol == symbol and s.state not in (State.CANCELLED, State.EXECUTED)]
        if active:
            return
        # Create fresh bullish & bearish setups
        for direction in (Direction.BULLISH, Direction.BEARISH):
            setup = Setup(symbol, direction, limit_at_zone=False, equity=self.equity)
            key = f"{symbol}_{direction.name}_{uuid.uuid4().hex[:6]}"
            self.setups[key] = setup

    def step(self) -> List[Dict]:
        """Process one iteration across all symbols."""
        for symbol in self.symbols:
            df_bundle = self._fetch_symbol_data(symbol)
            if any(df.empty for df in df_bundle.values()):
                continue
            self._ensure_setups(symbol)
            for key, setup in list(self.setups.items()):
                if setup.symbol != symbol:
                    continue
                manage_state(
                    setup,
                    {'HTF': df_bundle['HTF'], 'MTF': df_bundle['MTF'], 'LTF': df_bundle['LTF']},
                    TIMEFRAMES,
                    backtest_mode=self.backtest_mode,
                    zone_scorer=self.zone_scorer,
                    use_trailing_stop=self.use_trailing_stop
                )
                # Collect & purge finished setups
                if setup.state == State.CANCELLED:
                    del self.setups[key]
                if setup.state == State.EXECUTED and setup.outcome is not None:
                    self.completed_trades.append(setup.to_dict())
                    del self.setups[key]
        return self.completed_trades

    def run(self, iterations: Optional[int] = None, sleep_seconds: int = 60):
        """
        Run continuous loop. iterations=None => infinite until interrupted (live mode).
        Backtest mode executes a single pass.
        """
        count = 0
        while True:
            self.step()
            count += 1
            if self.backtest_mode:
                break
            if iterations is not None and count >= iterations:
                break
            time_module.sleep(sleep_seconds)

    def results(self) -> Dict:
        """Return performance summary plus raw trades."""
        metrics = calculate_performance_metrics(self.completed_trades)
        return {"metrics": metrics, "trades": self.completed_trades}

    def execute(self, symbol: str, prices, df: pd.DataFrame, equity: float, allow_multiple_trades: bool = False) -> dict:
        """Compatibility wrapper expected by main.py.

        Runs a single evaluation for `symbol` and returns a signal dict with
        'success': True/False and optional trade parameters when a setup
        reaches a ready state.
        """
        try:
            # Try to fetch the needed candle bundles (safe - uses internal helpers)
            df_bundle = {
                'HTF': get_candles(symbol, TIMEFRAMES['HTF'], 60),
                'MTF': get_candles(symbol, TIMEFRAMES['MTF'], 40),
                'LTF': get_candles(symbol, TIMEFRAMES['LTF'], 25)
            }

            # If any timeframe missing/empty, no signal
            if any(d is None or d.empty for d in df_bundle.values()):
                return {"success": False}

            # Ensure we have active setups for this symbol
            self._ensure_setups(symbol)

            # Run the state machine for each active setup once and check for READY
            for key, setup in list(self.setups.items()):
                if setup.symbol != symbol:
                    continue
                manage_state(
                    setup,
                    {'HTF': df_bundle['HTF'], 'MTF': df_bundle['MTF'], 'LTF': df_bundle['LTF']},
                    TIMEFRAMES,
                    backtest_mode=self.backtest_mode,
                    zone_scorer=self.zone_scorer,
                    use_trailing_stop=self.use_trailing_stop
                )

                # If setup reached READY, convert into a normalized signal dict
                if getattr(setup, 'state', None) == State.READY:
                    signal = {
                        'success': True,
                        'entry': getattr(setup, 'entry_price', None),
                        'sl': getattr(setup, 'sl_price', None),
                        'tp': getattr(setup, 'tp1_price', None),
                        'lot': getattr(setup, 'lot_size', None),
                        'direction': getattr(setup, 'direction', None).name if getattr(setup, 'direction', None) else None,
                        'features': setup.to_dict() if hasattr(setup, 'to_dict') else {},
                    }
                    return signal

            return {"success": False}

        except Exception as e:
            logger.error(f"Error in JudasSwingStrategy.execute for {symbol}: {e}", exc_info=True)
            return {"success": False}

# Export symbol for external importers
__all__ = ['JudasSwingStrategy']

# === Optional direct execution helper (kept minimal) ===
if __name__ == "__main__":
    strategy = JudasSwingStrategy(symbols=SYMBOLS, backtest_mode=True)
    strategy.run()
    print(strategy.results())