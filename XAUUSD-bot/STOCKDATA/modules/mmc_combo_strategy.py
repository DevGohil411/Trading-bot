import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import pytz
from .indicators import calculate_atr

# Setup logging
logger = logging.getLogger('trade_bot.mmc_xauusd')
logger.setLevel(logging.INFO)
logger.propagate = False

# Configuration
SYMBOL = "XAUUSD"
TIMEFRAME_H1 = mt5.TIMEFRAME_H1
TIMEFRAME_M15 = mt5.TIMEFRAME_M15
TIMEFRAME_M5 = mt5.TIMEFRAME_M5
BARS = 500
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 1.5
VOLUME_SPIKE_THRESHOLD = 1.5
ACCUMULATION_LOOKBACK = 20
ACCUMULATION_RANGE_PCT = 0.002  # 0.2% range for accumulation
LIQUIDITY_GRAB_THRESHOLD = 0.002  # 0.2% move for sweep
EQUAL_LEVEL_TOLERANCE = 0.001
OB_LOOKBACK = 10
CHOCH_LOOKBACK = 5

# Utility Functions
def _get_data(symbol, timeframe, n=50):
    """Fetch price data from MT5."""
    try:
        utc_to = datetime.utcnow().replace(tzinfo=pytz.UTC)
        rates = mt5.copy_rates_from(symbol, timeframe, utc_to, n)
        if rates is None or len(rates) == 0:
            logger.warning(f"No data for {symbol} timeframe {timeframe}")
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        logger.info(f"Data fetched for {symbol}, TF {timeframe}, length: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Error in _get_data: {str(e)}", exc_info=True)
        return None

def detect_accumulation(df):
    """Detect H1 accumulation zone."""
    try:
        if len(df) < ACCUMULATION_LOOKBACK:
            logger.info("Not enough data for accumulation detection")
            return False, None
        recent = df.iloc[-ACCUMULATION_LOOKBACK:]
        price_range = recent['high'].max() - recent['low'].min()
        avg_price = recent['close'].mean()
        if 'tick_volume' in recent and recent['tick_volume'].mean() > 0:
            avg_volume = recent['tick_volume'].iloc[:-1].mean()
            last_volume = recent['tick_volume'].iloc[-1]
            if last_volume > avg_volume * 0.8:
                return False, None
        if price_range / avg_price < ACCUMULATION_RANGE_PCT:
            logger.info("H1 accumulation zone detected")
            return True, (recent['low'].min(), recent['high'].max())
        return False, None
    except Exception as e:
        logger.error(f"Error in detect_accumulation: {str(e)}", exc_info=True)
        return False, None

def detect_liquidity_pools(df):
    """Detect liquidity pools based on equal highs/lows."""
    try:
        highs = df['high'].iloc[-ACCUMULATION_LOOKBACK:]
        lows = df['low'].iloc[-ACCUMULATION_LOOKBACK:]
        equal_highs = highs[highs.diff().abs() < EQUAL_LEVEL_TOLERANCE * highs.mean()].index
        equal_lows = lows[lows.diff().abs() < EQUAL_LEVEL_TOLERANCE * lows.mean()].index
        high_pool = highs[equal_highs].mean() if len(equal_highs) >= 2 else highs.max()
        low_pool = lows[equal_lows].mean() if len(equal_lows) >= 2 else lows.min()
        logger.info(f"Liquidity pools: High={high_pool}, Low={low_pool}")
        return high_pool, low_pool
    except Exception as e:
        logger.error(f"Error in detect_liquidity_pools: {str(e)}", exc_info=True)
        return None, None

def detect_ob_or_fvg(df):
    """Detect order block or fair value gap."""
    try:
        for i in range(len(df)-2, max(0, len(df)-OB_LOOKBACK-1), -1):
            prev = df.iloc[i-1]
            curr = df.iloc[i]
            if curr['open'] < curr['close'] and prev['open'] > prev['close'] and curr['close'] > prev['open']:
                logger.info(f"Bullish OB found at {curr['low']} - {curr['high']}")
                return 'bullish', (curr['low'], curr['high'])
            if curr['open'] > curr['close'] and prev['open'] < prev['close'] and curr['close'] < prev['open']:
                logger.info(f"Bearish OB found at {curr['low']} - {curr['high']}")
                return 'bearish', (curr['low'], curr['high'])
        if len(df) >= 3:
            curr, prev = df.iloc[-1], df.iloc[-2]
            if curr['low'] > prev['high']:
                logger.info(f"Bullish FVG found at {prev['high']} - {curr['low']}")
                return 'bullish', (prev['high'], curr['low'])
            if curr['high'] < prev['low']:
                logger.info(f"Bearish FVG found at {curr['high']} - {prev['low']}")
                return 'bearish', (curr['high'], prev['low'])
        return None, None
    except Exception as e:
        logger.error(f"Error in detect_ob_or_fvg: {str(e)}", exc_info=True)
        return None, None

def detect_liquidity_sweep(df, high_pool, low_pool):
    """Detect liquidity sweep on M15."""
    try:
        last = df.iloc[-1]
        if last['high'] > high_pool:
            logger.info("Liquidity sweep above high pool")
            return True, 'high', last['high']
        if last['low'] < low_pool:
            logger.info("Liquidity sweep below low pool")
            return True, 'low', last['low']
        return False, None, None
    except Exception as e:
        logger.error(f"Error in detect_liquidity_sweep: {str(e)}", exc_info=True)
        return False, None, None

def is_volume_spike(df):
    """Check for volume spike."""
    try:
        if 'tick_volume' not in df or len(df) < 20:
            return False
        avg_vol = df['tick_volume'].iloc[-20:-1].mean()
        last_vol = df['tick_volume'].iloc[-1]
        return last_vol > VOLUME_SPIKE_THRESHOLD * avg_vol
    except Exception as e:
        logger.error(f"Error in is_volume_spike: {str(e)}", exc_info=True)
        return False

def detect_choch_or_bos(df, sweep_direction):
    """Detect CHOCH or BOS on M15/M5."""
    try:
        if len(df) < CHOCH_LOOKBACK:
            return False, None
        last_close = df['close'].iloc[-1]
        prev_highs = df['high'].iloc[-CHOCH_LOOKBACK:-1]
        prev_lows = df['low'].iloc[-CHOCH_LOOKBACK:-1]
        if sweep_direction == 'high':
            if last_close < prev_highs.min():
                logger.info("Bearish CHOCH/BOS confirmed")
                return True, 'bearish'
        elif sweep_direction == 'low':
            if last_close > prev_lows.max():
                logger.info("Bullish CHOCH/BOS confirmed")
                return True, 'bullish'
        return False, None
    except Exception as e:
        logger.error(f"Error in detect_choch_or_bos: {str(e)}", exc_info=True)
        return False, None

def detect_engulfing_candle(df, direction):
    """Detect engulfing candle on M5."""
    try:
        if len(df) < 2:
            return False
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        if direction == 'bullish':
            if (curr['close'] > curr['open'] and prev['close'] < prev['open'] and 
                curr['open'] <= prev['close'] and curr['close'] >= prev['open']):
                logger.info("Bullish engulfing candle detected on M5")
                return True
        elif direction == 'bearish':
            if (curr['close'] < curr['open'] and prev['close'] > prev['open'] and 
                curr['open'] >= prev['close'] and curr['close'] <= prev['open']):
                logger.info("Bearish engulfing candle detected on M5")
                return True
        return False
    except Exception as e:
        logger.error(f"Error in detect_engulfing_candle: {str(e)}", exc_info=True)
        return False

def price_in_zone(price, zone):
    """Check if price is within mitigation zone."""
    try:
        if zone is None:
            return False
        return zone[0] <= price <= zone[1]
    except Exception as e:
        logger.error(f"Error in price_in_zone: {str(e)}", exc_info=True)
        return False

def compute_sl(symbol, entry_price, direction, atr, recent_df=None, symbol_info=None):
	"""Compute SL per project rules:
	   GBPJPY: max(5*ATR, 0.5)
	   Others: max(2*ATR, 0.5% of entry)
	   optional: prefer recent swing low/high if feasible.
	"""
	try:
		if atr is None or atr <= 0:
			return None
		sym = (symbol or "").upper()
		if sym == "GBPJPY":
			sl_dist = max(5.0 * atr, 0.5)
		else:
			sl_dist = max(2.0 * atr, 0.005 * abs(entry_price))  # 0.5% of entry
		# Optional swing-based placement (prefer recent swing if it's within a reasonable delta)
		if recent_df is not None and len(recent_df) >= 3:
			try:
				window = min(20, max(3, len(recent_df)//4))
				if direction in ('bullish', 'Bullish', 'buy', 'BUY'):
					swing_low = recent_df['low'].tail(window).min()
					if swing_low < entry_price and (entry_price - swing_low) <= sl_dist * 1.5:
						sl = float(swing_low)
					else:
						sl = float(entry_price - sl_dist)
				else:
					swing_high = recent_df['high'].tail(window).max()
					if swing_high > entry_price and (swing_high - entry_price) <= sl_dist * 1.5:
						sl = float(swing_high)
					else:
						sl = float(entry_price + sl_dist)
			except Exception:
				sl = float(entry_price - sl_dist) if direction in ('bullish','Bullish','buy') else float(entry_price + sl_dist)
		else:
			sl = float(entry_price - sl_dist) if direction in ('bullish','Bullish','buy') else float(entry_price + sl_dist)
		# Round using symbol_info if available
		if symbol_info and hasattr(symbol_info, 'digits'):
			sl = round(sl, int(symbol_info.digits))
		return sl
	except Exception as e:
		logger.error(f"compute_sl error: {e}")
		return None

# Core MMC Logic
def mmc_logic(symbol, df_h1, df_m15, df_m5):
    """Core MMC trading logic."""
    try:
        # Step 1: H1 Accumulation
        accumulation, acc_zone = detect_accumulation(df_h1)
        if not accumulation:
            logger.info("No H1 accumulation zone")
            return {"success": False}

        # Step 2: H1 Liquidity Pools
        high_pool, low_pool = detect_liquidity_pools(df_h1)
        if high_pool is None or low_pool is None:
            logger.info("No valid liquidity pools")
            return {"success": False}

        # Step 3: H1 OB/FVG
        htf_ob_type, htf_ob_zone = detect_ob_or_fvg(df_h1)
        if not htf_ob_zone:
            logger.info("No H1 OB/FVG found")
            return {"success": False}

        # Step 4: M15 Liquidity Sweep
        liquidity_grabbed, sweep_direction, sweep_price = detect_liquidity_sweep(df_m15, high_pool, low_pool)
        if not liquidity_grabbed or not is_volume_spike(df_m15):
            logger.info("No M15 liquidity sweep or volume spike")
            return {"success": False}

        # Step 5: M15 CHOCH/BOS
        choch_confirmed, choch_dir = detect_choch_or_bos(df_m15, sweep_direction)
        if not choch_confirmed:
            logger.info("No M15 CHOCH/BOS")
            return {"success": False}

        # Step 6: M15 Mitigation Zone
        m15_ob_type, m15_ob_zone = detect_ob_or_fvg(df_m15)
        if not m15_ob_zone:
            logger.info("No M15 OB/FVG found")
            return {"success": False}

        # Step 7: M5 Entry Confirmation
        last_m5_price = df_m5['close'].iloc[-1]
        if not price_in_zone(last_m5_price, m15_ob_zone):
            logger.info("M5 price not in mitigation zone")
            return {"success": False}

        m5_choch_confirmed, m5_dir = detect_choch_or_bos(df_m5, sweep_direction)
        if not m5_choch_confirmed or m5_dir != choch_dir:
            logger.info("No M5 CHOCH/BOS or direction mismatch")
            return {"success": False}

        if not detect_engulfing_candle(df_m5, choch_dir):
            logger.info("No M5 engulfing candle")
            return {"success": False}

        if not is_volume_spike(df_m5):
            logger.info("No M5 volume spike")
            return {"success": False}

        # Setup trade
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None or not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select {symbol}")
                return {"success": False}
        tick = mt5.symbol_info_tick(symbol)
        if tick is None or tick.bid == 0 or tick.ask == 0:
            logger.warning(f"Failed to get real-time tick data for {symbol}. Cannot determine entry price.")
            return {"success": False}

        atr = calculate_atr(df_m15)
        if atr is None or atr == 0:
            logger.warning("ATR calculation failed")
            return {"success": False}

        price = tick.ask if choch_dir == 'bullish' else tick.bid

        # Replace old SL calculation with compute_sl
        sl_candidate = compute_sl(symbol, price, choch_dir, atr, recent_df=df_m15, symbol_info=symbol_info)
        if sl_candidate is None:
            logger.warning("Failed to compute SL")
            return {"success": False}
        sl = sl_candidate

        risk = abs(price - sl)
        tp1 = price + (risk * 2) if choch_dir == 'bullish' else price - (risk * 2)
        tp2 = price + (risk * 3) if choch_dir == 'bullish' else price - (risk * 3)
        tp3 = price + (risk * 4) if choch_dir == 'bullish' else price - (risk * 4)

        # Round values
        price = round(price, symbol_info.digits)
        sl = round(sl, symbol_info.digits)
        tp1 = round(tp1, symbol_info.digits)
        tp2 = round(tp2, symbol_info.digits)
        tp3 = round(tp3, symbol_info.digits)

        logger.info(f"Trade signal: {choch_dir.upper()} at {price}, SL={sl}, TP1={tp1}, TP2={tp2}, TP3={tp3}")
        # Build ML features (ensure order matches your training pipeline)
        last_m15 = df_m15.iloc[-1]
        body = abs(float(last_m15['close']) - float(last_m15['open']))
        rng = float(last_m15['high']) - float(last_m15['low'])
        wick_up = float(last_m15['high']) - float(max(last_m15['open'], last_m15['close']))
        wick_dn = float(min(last_m15['open'], last_m15['close'])) - float(last_m15['low'])
        atr_val = float(atr)
        atr_safe = atr_val if atr_val != 0 else 1.0
        acc_width_norm = 0.0
        if acc_zone:
            acc_width_norm = float(acc_zone[1] - acc_zone[0]) / atr_safe
        range_ratio = rng / atr_safe
        wick_up_ratio = wick_up / (body + 1e-6)
        wick_dn_ratio = wick_dn / (body + 1e-6)
        hour = int(df_m15.index[-1].hour)
        direction_flag = 1 if choch_dir == 'bullish' else 0
        vol_spike_m15 = int(is_volume_spike(df_m15))
        in_zone_flag = int(price_in_zone(last_m5_price, m15_ob_zone))
        features = [
            atr_val,                 # ATR on M15
            acc_width_norm,          # Accumulation width normalized by ATR
            range_ratio,             # Current M15 candle range / ATR
            wick_up_ratio,           # Upper wick/body
            wick_dn_ratio,           # Lower wick/body
            int(sweep_direction == 'low'),   # Sweep below low pool
            int(sweep_direction == 'high'),  # Sweep above high pool
            int(choch_dir == 'bullish'),     # CHOCH/BOS direction flags
            int(choch_dir == 'bearish'),
            in_zone_flag,            # M5 price in mitigation zone
            vol_spike_m15,           # Volume spike on M15
            hour,                    # Hour of M15 bar
            direction_flag,          # Trade direction (1=buy/bullish)
        ]
        return {
            "success": True,
            "direction": choch_dir,
            "price": price,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "features": features,
        }
    except Exception as e:
        logger.error(f"Error in mmc_logic: {str(e)}", exc_info=True)
        return {"success": False}

# Strategy Class
class MMCXAUUSDStrategy:
    def execute(self, symbol, prices, df, equity, allow_multiple_trades):
        """Execute MMC XAUUSD strategy."""
        try:
            # Fetch data using internal _get_data
            df_h1 = _get_data(symbol, TIMEFRAME_H1, BARS)
            df_m15 = _get_data(symbol, TIMEFRAME_M15, BARS)
            df_m5 = _get_data(symbol, TIMEFRAME_M5, 50)
            if df_h1 is None or df_h1.empty or df_m15 is None or df_m15.empty or df_m5 is None or df_m5.empty:
                logger.warning("No data available")
                return {"success": False}

            # Run MMC logic
            return mmc_logic(symbol, df_h1, df_m15, df_m5)
        except Exception as e:
            logger.error(f"Error in MMCXAUUSDStrategy execute: {str(e)}", exc_info=True)
            return {"success": False}
        # Removed mt5.shutdown() to prevent session drops. Shutdown will be handled at bot exit.
