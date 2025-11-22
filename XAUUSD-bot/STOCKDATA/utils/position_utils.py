import MetaTrader5 as mt5
import math
import logging

logger = logging.getLogger('trade_bot.position_utils')

def ensure_mt5_connection(max_retries=5, delay=10):
    """Ensure MT5 connection is active and working properly"""
    for attempt in range(max_retries):
        try:
            # Test connection by getting account info
            account_info = mt5.account_info()
            if account_info is None:
                logger.error(f"Failed to get account info. Attempt {attempt + 1}/{max_retries}")
                import time
                time.sleep(delay)
                continue
            # Test market data access
            symbol_info = mt5.symbol_info("XAUUSD")
            if symbol_info is None:
                logger.error(f"Failed to get symbol info. Attempt {attempt + 1}/{max_retries}")
                import time
                time.sleep(delay)
                continue
            logger.info("MT5 connection ensured.")
            return True
        except Exception as e:
            logger.error(f"Error ensuring MT5 connection: {str(e)}. Attempt {attempt + 1}/{max_retries}")
            import time
            time.sleep(delay)
    logger.error("Failed to ensure MT5 connection after all attempts")
    return False

def calculate_position_size(symbol, equity, atr, sl_points):
    try:
        if not ensure_mt5_connection():
            logger.error("MT5 not connected. Aborting calculate_position_size.")
            return 0.04
        if atr is None or math.isnan(atr) or sl_points is None or sl_points == 0 or math.isnan(sl_points):
            logger.error("Invalid ATR or SL points for position size calc")
            return 0.01
        max_risk_amount = equity * 0.01
        point_value = 100  # For XAUUSD, adjust as needed for other symbols
        risk_per_lot = sl_points * point_value
        # Calculate base lot size
        lot_size = max_risk_amount / risk_per_lot if risk_per_lot > 0 else 0.01
        # Round to 2 decimal places for most pairs, but ensure minimum 0.01 for XAUUSD
        lot_size = round(lot_size, 2)
        # Ensure minimum lot size of 0.01 and maximum of 1.0
        lot_size = max(0.01, min(lot_size, 1.0))
        logger.info(f"Position Size Calculation:")
        logger.info(f"  * Equity: ${equity:.2f}")
        logger.info(f"  * Max Risk: ${max_risk_amount:.2f}")
        logger.info(f"  * Risk per Lot: ${risk_per_lot:.2f}")
        logger.info(f"  * Calculated Lot Size: {lot_size}")
        return lot_size
    except Exception as e:
        logger.error(f"Error calculating position size: {str(e)}")
        return 0.04 