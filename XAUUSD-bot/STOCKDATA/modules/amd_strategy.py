import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import pytz
from modules.indicators import calculate_atr

# Setup logging
logger = logging.getLogger('trade_bot.amd')
logger.setLevel(logging.DEBUG)
logger.propagate = False

# === SETTINGS === #
SYMBOL = "XAUUSD"
HTF_TIMEFRAME = mt5.TIMEFRAME_H1
MID_TIMEFRAME = mt5.TIMEFRAME_M15
LTF_TIMEFRAME = mt5.TIMEFRAME_M5
CONSOLIDATION_RANGE = 0.006  # 0.6% for H1 accumulation
LIQUIDITY_THRESHOLD = 0.01  # 1% move
OB_LOOKBACK = 8
ATR_PERIOD = 14
ATR_MULTIPLIER_MIN = 1.5
ATR_MULTIPLIER_MAX = 3.0

# === OHLC Fetch ===
def _get_data(symbol, timeframe, n=100):
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
        if rates is None or len(rates) == 0:
            logger.error(f"No data for {symbol} TF {timeframe}")
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize('UTC')
        df.set_index('time', inplace=True)
        logger.info(f"Data fetched successfully for {symbol}, TF {timeframe}, length: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Error in _get_data: {str(e)}", exc_info=True)
        return None

# === Accumulation Detection ===
def detect_accumulation(df, tf_name="Unknown"):
    try:
        if len(df) < 10:
            logger.info(f"{tf_name}: Not enough data for accumulation detection (len={len(df)})")
            return False
        recent = df.iloc[-5:]
        price_range = (recent['high'].max() - recent['low'].min()) / recent['close'].mean()
        if price_range <= CONSOLIDATION_RANGE:
            logger.info(f"{tf_name}: Accumulation detected (range: {price_range*100:.2f}%)")
            return True
        logger.info(f"{tf_name}: No accumulation detected (range: {price_range*100:.2f}%)")
        return False
    except Exception as e:
        logger.error(f"Error in detect_accumulation: {str(e)}", exc_info=True)
        return False

# === Liquidity Grab Detection ===
def detect_liquidity_grab(df, tf_name="Unknown"):
    try:
        if len(df) < 3:
            logger.info(f"{tf_name}: Not enough data for liquidity grab detection (len={len(df)})")
            return False
        last_candle = df.iloc[-2]
        prev_candle = df.iloc[-3]
        move = abs(last_candle['close'] - prev_candle['close']) / prev_candle['close']
        if move >= LIQUIDITY_THRESHOLD and last_candle['high'] > prev_candle['high']:
            logger.info(f"{tf_name}: Liquidity grab detected (move: {move*100:.2f}%)")
            return True
        logger.info(f"{tf_name}: No liquidity grab detected (move: {move*100:.2f}%)")
        return False
    except Exception as e:
        logger.error(f"Error in detect_liquidity_grab: {str(e)}", exc_info=True)
        return False

# === BOS/CHOCH Detection ===
def detect_bos_choch(df, tf_name="Unknown"):
    try:
        if len(df) < 10:
            logger.info(f"{tf_name}: Not enough data for BOS/CHOCH detection (len={len(df)})")
            return False, None
        highs = df['high'][:-1]
        lows = df['low'][:-1]
        last_high = df['high'].iloc[-2]
        last_low = df['low'].iloc[-2]
        prev_hh = highs.max()
        prev_ll = lows.min()
        if last_high > prev_hh:
            logger.info(f"{tf_name}: Bullish BOS detected (new high: {last_high})")
            return True, "Bullish"
        elif last_low < prev_ll:
            logger.info(f"{tf_name}: Bearish CHOCH detected (new low: {last_low})")
            return True, "Bearish"
        logger.info(f"{tf_name}: No BOS/CHOCH detected")
        return False, None
    except Exception as e:
        logger.error(f"Error in detect_bos_choch: {str(e)}", exc_info=True)
        return False, None

# === Order Block Detection ===
def find_order_block(df, direction, tf_name="Unknown"):
    try:
        if len(df) < OB_LOOKBACK:
            logger.info(f"{tf_name}: Not enough data for OB detection (len={len(df)})")
            return None, None
        for i in range(-2, -OB_LOOKBACK-1, -1):
            candle = df.iloc[i]
            if direction == "Bullish" and candle['close'] < candle['open']:
                ob_high = candle['high']
                ob_low = candle['low']
                logger.info(f"{tf_name}: Bullish OB found at {ob_low} - {ob_high}")
                return ob_low, ob_high
            elif direction == "Bearish" and candle['close'] > candle['open']:
                ob_high = candle['high']
                ob_low = candle['low']
                logger.info(f"{tf_name}: Bearish OB found at {ob_low} - {ob_high}")
                return ob_low, ob_high
        logger.info(f"{tf_name}: No OB found")
        return None, None
    except Exception as e:
        logger.error(f"Error in find_order_block: {str(e)}", exc_info=True)
        return None, None

# === Multi-Timeframe Confirmation ===
def multi_timeframe_confirmation(symbol):
    try:
        # H1 Analysis
        df_h1 = _get_data(symbol, HTF_TIMEFRAME, n=20)
        if df_h1 is None or df_h1.empty:
            logger.info("H1: Failed to fetch data")
            return False, None

        # Check Accumulation
        if not detect_accumulation(df_h1, "H1"):
            logger.info("H1: Accumulation check failed")
            return False, None

        # Check Liquidity Grab
        if not detect_liquidity_grab(df_h1, "H1"):
            logger.info("H1: Liquidity grab check failed")
            return False, None

        # Check BOS/CHOCH
        bos_detected, direction = detect_bos_choch(df_h1, "H1")
        if not bos_detected:
            logger.info("H1: BOS/CHOCH check failed")
            return False, None

        # M15 Confirmation
        df_m15 = _get_data(symbol, MID_TIMEFRAME, n=20)
        if df_m15 is None or df_m15.empty:
            logger.info("M15: Failed to fetch data")
            return False, None

        # Confirm BOS/CHOCH on M15
        m15_bos, m15_direction = detect_bos_choch(df_m15, "M15")
        if not m15_bos or m15_direction != direction:
            logger.info(f"M15: BOS/CHOCH confirmation failed (Direction: {m15_direction})")
            return False, None

        # Find Order Block on M15
        ob_low_m15, ob_high_m15 = find_order_block(df_m15, direction, "M15")
        if ob_low_m15 is None or ob_high_m15 is None:
            logger.info("M15: No valid OB found")
            return False, None

        # M5 Entry Setup
        df_m5 = _get_data(symbol, LTF_TIMEFRAME, n=20)
        if df_m5 is None or df_m5.empty:
            logger.info("M5: Failed to fetch data")
            return False, None

        # Find Order Block on M5
        ob_low_m5, ob_high_m5 = find_order_block(df_m5, direction, "M5")
        if ob_low_m5 is None or ob_high_m5 is None:
            logger.info("M5: No valid OB found")
            return False, None

        # Check if price retraced to M5 OB
        current_price = df_m5['close'].iloc[-1]
        if ob_low_m5 <= current_price <= ob_high_m5:
            logger.info(f"M5: Price retraced to OB ({ob_low_m5} - {ob_high_m5})")
            return True, {"direction": direction, "ob_low": ob_low_m5, "ob_high": ob_high_m5, "df_m5": df_m5}
        logger.info(f"M5: Price {current_price} not in OB range ({ob_low_m5} - {ob_high_m5})")
        return False, None
    except Exception as e:
        logger.error(f"Error in multi_timeframe_confirmation: {str(e)}", exc_info=True)
        return False, None

# === Core AMD Strategy ===
class AMDStrategy:
    def execute(self, symbol, prices, df, equity, allow_multiple_trades):
        try:
            # df here is df_main, prices is df_main['close'].to_numpy()
            # We expect df (M15) to be provided by main.py
            if df is None or df.empty:
                logger.error("No M15 data provided to AMD strategy.")
                return {"success": False}

            # Load higher and lower timeframe data within the strategy as needed for its logic
            # (assuming main.py handles initial MT5 connection and base data fetch)

            # Run MTF confirmation
            mtf_valid, mtf_data = multi_timeframe_confirmation(symbol)
            if not mtf_valid or not mtf_data:
                logger.info("Multi-timeframe confirmation failed")
                return {"success": False}

            # Setup trade details for return
            direction = mtf_data["direction"]
            ob_low = mtf_data["ob_low"]
            ob_high = mtf_data["ob_high"]
            
            tick = mt5.symbol_info_tick(symbol)
            if tick is None or tick.bid == 0 or tick.ask == 0:
                logger.warning(f"Failed to get real-time tick data for {symbol}. Cannot determine entry price.")
                return {"success": False}
            
            price = tick.ask if direction == "Bullish" else tick.bid

            # Calculate SL based on OB and ATR using compute_sl
            atr = calculate_atr(df) 
            if atr is None:
                logger.warning("ATR could not be calculated. Skipping trade signal.")
                return {"success": False}

            # Use mtf_data's M5 if available for swing detection
            m5_df = mtf_data.get("df_m5") if isinstance(mtf_data, dict) else None
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None or symbol_info.point == 0:
                logger.error(f"Symbol info for {symbol} unavailable or point is zero. Cannot calculate SL/TP precisely.")
                return {"success": False}

            sl_calc = compute_sl(symbol, price, direction, atr, recent_df=m5_df, symbol_info=symbol_info)
            if sl_calc is None:
                logger.warning("SL calculation failed")
                return {"success": False}
            sl = sl_calc

            # Ensure symbol_info is available for point value
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None or symbol_info.point == 0:
                logger.error(f"Symbol info for {symbol} unavailable or point is zero. Cannot calculate SL/TP precisely.")
                return {"success": False}

            sl_points = abs(price - sl) / symbol_info.point

            # Calculate TP levels based on R multiples
            tp1 = price + (sl_points * symbol_info.point) if direction == "Bullish" else price - (sl_points * symbol_info.point)
            tp2 = price + (sl_points * 2 * symbol_info.point) if direction == "Bullish" else price - (sl_points * 2 * symbol_info.point)
            tp3 = price + (sl_points * 3 * symbol_info.point) if direction == "Bullish" else price - (sl_points * 3 * symbol_info.point)

            # Round values
            price = round(price, symbol_info.digits)
            sl = round(sl, symbol_info.digits)
            tp1 = round(tp1, symbol_info.digits)
            tp2 = round(tp2, symbol_info.digits)
            tp3 = round(tp3, symbol_info.digits)

            logger.info(f"Trade signal generated: {direction.upper()} at {price}, SL={sl}, TP1={tp1}, TP2={tp2}, TP3={tp3}")
            
            # Build ML features (13-length vector; standardized with mmc_combo_strategy)
            last = df.iloc[-1]
            body = float(abs(last['close'] - last['open']))
            rng = float(last['high'] - last['low'])
            wick_up = float(last['high'] - max(last['open'], last['close']))
            wick_dn = float(min(last['open'], last['close']) - last['low'])
            atr_val = float(atr)
            atr_safe = atr_val if atr_val != 0 else 1.0
            zone_width_norm = float(ob_high - ob_low) / atr_safe
            range_ratio = rng / atr_safe
            wick_up_ratio = wick_up / (body + 1e-6)
            wick_dn_ratio = wick_dn / (body + 1e-6)
            # Sweep flags from M15 liquidity grab direction proxy
            sweep_detected = detect_liquidity_grab(df, "M15")
            sweep_low_flag = int(direction == "Bullish" and sweep_detected)
            sweep_high_flag = int(direction == "Bearish" and sweep_detected)
            # CHOCH/BOS flags from M15
            choch_ok, choch_dir = detect_bos_choch(df, "M15")
            choch_bullish = int(choch_ok and choch_dir == "Bullish")
            choch_bearish = int(choch_ok and choch_dir == "Bearish")
            # In-zone flag using latest price vs M5 OB
            in_zone_flag = int(ob_low <= price <= ob_high)
            # M15 volume spike
            vol_spike_m15 = 0
            if 'tick_volume' in df and len(df) >= 20:
                avg_vol = float(df['tick_volume'].rolling(window=20).mean().iloc[-1])
                last_vol = float(df['tick_volume'].iloc[-1])
                vol_spike_m15 = int(avg_vol > 0 and last_vol > 1.5 * avg_vol)
            # Hour from M15 data (handle both index or 'time' column)
            if isinstance(df.index, pd.DatetimeIndex):
                hour = int(df.index[-1].hour)
            else:
                hour = int(pd.to_datetime(df['time'].iloc[-1]).hour)
            direction_flag = 1 if direction == "Bullish" else 0
            features = [
                atr_val,               # 1) ATR
                zone_width_norm,       # 2) OB zone width / ATR
                range_ratio,           # 3) M15 candle range / ATR
                wick_up_ratio,         # 4) Upper wick / body
                wick_dn_ratio,         # 5) Lower wick / body
                sweep_low_flag,        # 6) Sweep low flag
                sweep_high_flag,       # 7) Sweep high flag
                choch_bullish,         # 8) CHOCH/BOS bullish
                choch_bearish,         # 9) CHOCH/BOS bearish
                in_zone_flag,          # 10) In zone
                vol_spike_m15,         # 11) M15 volume spike
                hour,                  # 12) Hour
                direction_flag,        # 13) Direction flag (1=Bullish)
            ]
            return {
                "success": True,
                "direction": direction,
                "price": price,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3,
                "features": features,
            }
        except Exception as e:
            logger.error(f"Error in AMD strategy execute: {str(e)}", exc_info=True)
            return {"success": False}

def compute_sl(symbol, entry_price, direction, atr, recent_df=None, symbol_info=None):
	try:
		if atr is None or atr <= 0:
			return None
		if (symbol or "").upper() == "GBPJPY":
			sl_dist = max(5.0 * atr, 0.5)
		else:
			sl_dist = max(2.0 * atr, 0.005 * abs(entry_price))
		if recent_df is not None and len(recent_df) >= 3:
			window = min(20, max(3, len(recent_df)//4))
			if direction == "Bullish":
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
		else:
			sl = float(entry_price - sl_dist) if direction == "Bullish" else float(entry_price + sl_dist)
		if symbol_info and hasattr(symbol_info, 'digits'):
			sl = round(sl, int(symbol_info.digits))
		return sl
	except Exception:
		return None