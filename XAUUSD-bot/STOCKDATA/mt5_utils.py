import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone
import logging

logger = logging.getLogger("mt5_utils")

def is_mt5_connected():
    try:
        info = mt5.account_info()
        return info is not None
    except Exception as e:
        logger.error(f"MT5 connection check failed: {str(e)}")
        return False

def safe_positions_get(*args, **kwargs):
    if not is_mt5_connected():
        logger.error("MT5 not connected. Cannot fetch positions.")
        return None
    try:
        positions = mt5.positions_get(*args, **kwargs)
        if positions is None:
            logger.error("MT5 returned None for positions_get()")
        return positions
    except Exception as e:
        logger.error(f"Exception in positions_get: {str(e)}")
        return None

def connect_to_mt5():
    # Removed mt5.initialize() call. Assume MT5 is already initialized by main bot.
    if not is_mt5_connected():
        logger.error(f"MT5 not connected: {mt5.last_error()}")
        return False
    symbols = mt5.symbols_get()
    logger.info(f"Available symbols: {[s.name for s in symbols if 'XAUUSD' in s.name or 'GOLD' in s.name]}")
    logger.info("MT5 connected successfully")
    return True

def fetch_data():
    symbol = "XAUUSD"
    timeframe = mt5.TIMEFRAME_M15
    utc_to = datetime.now(timezone.utc)
    logging.info(f"Fetching data for {symbol}, timeframe={timeframe}, utc_to={utc_to}, bars=100")
    rates = mt5.copy_rates_from(symbol, timeframe, utc_to, 100)
    if rates is None or len(rates) == 0:
        logging.error(f"No data fetched for {symbol}. MT5 error: {mt5.last_error()}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    logging.info(f"Data fetched: rows={len(df)}, columns={list(df.columns)}")
    return df

def data_freshness_check(candle_age, tick_age, max_age_seconds, symbol=None):
    import logging
    if candle_age > max_age_seconds or tick_age > max_age_seconds:
        if symbol:
            logging.warning(f"Stale data for {symbol}: Candle Age={candle_age:.0f}s, Tick Age={tick_age:.0f}s. (Threshold: {max_age_seconds}s). Proceeding with trade.")
        else:
            logging.warning(f"Stale data: Candle Age={candle_age:.0f}s, Tick Age={tick_age:.0f}s. (Threshold: {max_age_seconds}s). Proceeding with trade.")