import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import pytz
from STOCKDATA.utils.trade_logger import logger
from STOCKDATA.mt5_utils import mt5

# Setup logging
logger = logging.getLogger('trade_bot.msb')
logger.setLevel(logging.DEBUG)
logger.propagate = False

# === SETTINGS (only relevant for strategy logic, not general risk/trade management) === #
TIMEFRAME_MAIN = mt5.TIMEFRAME_M15
TIMEFRAME_HIGHER = mt5.TIMEFRAME_H1
TIMEFRAME_LTF = mt5.TIMEFRAME_M5
CONSOLIDATION_RANGE_THRESHOLD = 0.015
LIQUIDITY_MOVE_THRESHOLD = 0.005
OB_LOOKBACK = 8
ATR_PERIOD = 14

# === OHLC Fetch (Internal to strategy, so mt5 dependency is fine) ===
# Renamed to _get_data for internal use
def _get_data(symbol, timeframe, bars=300):
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            logger.error(f"No data for {symbol} TF {timeframe}")
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        if df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize('UTC')
        df.set_index("time", inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error in _get_data: {str(e)}")
        return None

# === ATR Calculation (Keep, as it's part of strategy logic) ===
def calculate_atr(df, period=ATR_PERIOD):
    try:
        if len(df) < period:
            logger.warning(f"DataFrame too small for ATR calculation (rows: {len(df)}, period: {period})")
            return None
        df["tr"] = np.maximum.reduce([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs()
        ])
        atr = df["tr"].rolling(window=period).mean().iloc[-1]
        if pd.isna(atr) or atr == 0:
            logger.warning("ATR calculation resulted in NaN or zero")
            return None
        return atr
    except Exception as e:
        logger.error(f"Error in calculate_atr: {str(e)}")
        return None

# === Swing Highs/Lows Detection (Keep) ===
def detect_swing_highs_lows(df, window=5):
    try:
        swing_highs, swing_lows = [], []
        for i in range(window, len(df)-window):
            if df["high"].iloc[i] == max(df["high"].iloc[i-window:i+window+1]):
                swing_highs.append(i)
            if df["low"].iloc[i] == min(df["low"].iloc[i-window:i+window+1]):
                swing_lows.append(i)
        return swing_highs, swing_lows
    except Exception as e:
        logger.error(f"Error in detect_swing_highs_lows: {str(e)}")
        return [], []

# === Consolidation Detection (Keep) ===
def detect_consolidation(df):
    try:
        recent_data = df.iloc[-8:]
        range_high = recent_data["high"].max()
        range_low = recent_data["low"].min()
        avg_price = (range_high + range_low) / 2
        range_pct = (range_high - range_low) / avg_price
        if range_pct < CONSOLIDATION_RANGE_THRESHOLD:
            logger.info(f"Consolidation detected: Range={range_pct*100:.2f}%")
            return True, (range_low, range_high)
        logger.debug("No consolidation detected")
        return False, None
    except Exception as e:
        logger.error(f"Error in detect_consolidation: {str(e)}")
        return False, None

# === Liquidity Grab Detection (Keep) ===
def detect_liquidity_grab(df):
    try:
        if len(df) < 3:
            logger.debug("Not enough data for liquidity grab detection")
            return None, None
        curr, prev = df.iloc[-1], df.iloc[-2]
        fvg_low, fvg_high = None, None
        grab_type = None
        if curr["low"] > prev["high"]:
            fvg_low, fvg_high = prev["high"], curr["low"]
            grab_type = "bullish"
        elif curr["high"] < prev["low"]:
            fvg_low, fvg_high = curr["high"], prev["low"]
            grab_type = "bearish"
        candle_move_pct = (curr["high"] - curr["low"]) / ((curr["high"] + curr["low"]) / 2)
        is_impulsive = candle_move_pct >= LIQUIDITY_MOVE_THRESHOLD
        if grab_type and is_impulsive:
            logger.info(f"Liquidity grab detected: {grab_type}, FVG: {fvg_low}-{fvg_high}")
            return grab_type, (fvg_low, fvg_high)
        logger.debug("No liquidity grab: Missing FVG or move")
        return None, None
    except Exception as e:
        logger.error(f"Error in detect_liquidity_grab: {str(e)}")
        return None, None

# === Market Structure Break (BOS/CHOCH) (Keep) ===
def detect_market_structure(df, timeframe_str): # Renamed timeframe to timeframe_str to avoid conflict with mt5.TIMEFRAME_H1
    try:
        sh, sl = detect_swing_highs_lows(df)
        if len(sh) < 2 or len(sl) < 2:
            logger.debug("Insufficient swing points")
            return {"type": None, "break_level": None, "trend": None}
        last_high = df["high"].iloc[sh[-1]]
        last_low = df["low"].iloc[sl[-1]]
        second_last_low = df["low"].iloc[sl[-2]]
        second_last_high = df["high"].iloc[sh[-2]]
        last_close = df["close"].iloc[-1]
        ms_type = None
        break_level = None
        trend = None
        if (df["high"].iloc[-1] > last_high and df["low"].iloc[-1] > second_last_low and last_close > last_high):
            ms_type = "bullish_choch"
            break_level = last_high
            trend = "bullish"
            logger.info(f"Bullish CHOCH on {timeframe_str}")
        elif (df["low"].iloc[-1] < last_low and df["high"].iloc[-1] < second_last_high and last_close < last_low):
            ms_type = "bearish_choch"
            break_level = last_low
            trend = "bearish"
            logger.info(f"Bearish CHOCH on {timeframe_str}")
        return {"type": ms_type, "break_level": break_level, "trend": trend}
    except Exception as e:
        logger.error(f"Error in detect_market_structure: {str(e)}")
        return {"type": None, "break_level": None, "trend": None}

# === BOS Confirmation (Keep) ===
def confirm_bos(df, msb):
    try:
        if msb["type"] is None or msb["break_level"] is None:
            return False
        last_candle = df.iloc[-1]
        confirmed = False
        if "bullish" in msb["type"]:
            confirmed = last_candle["high"] > msb["break_level"] and last_candle["close"] > msb["break_level"]
        elif "bearish" in msb["type"]:
            confirmed = last_candle["low"] < msb["break_level"] and last_candle["close"] < msb["break_level"]
        if confirmed:
            logger.info(f"BOS confirmed on {msb['type']}")
        return confirmed
    except Exception as e:
        logger.error(f"Error in confirm_bos: {str(e)}")
        return False

# === Retest Zone Detection (Keep) ===
def detect_retest_zone(df, msb, df_higher, atr):
    try:
        if msb["type"] is None:
            return None, None
        ob_low, ob_high = None, None
        for i in range(-2, -OB_LOOKBACK-1, -1):
            if len(df) < abs(i):
                continue
            curr, prev = df.iloc[i], df.iloc[i+1]
            if "bullish" in msb["type"] and curr["close"] < curr["open"] and curr["open"] > prev["close"] and curr["close"] < prev["open"]:
                ob_low, ob_high = curr["low"], curr["high"]
                break
            elif "bearish" in msb["type"] and curr["close"] > curr["open"] and curr["open"] < prev["close"] and curr["close"] > prev["open"]:
                ob_low, ob_high = curr["low"], curr["high"]
                break
        _, (fvg_low, fvg_high) = detect_liquidity_grab(df)
        htf_avg = df_higher["close"].iloc[-20:].mean()
        is_discount = df["close"].iloc[-1] < htf_avg if "bullish" in msb["type"] else df["close"].iloc[-1] > htf_avg
        last_close = df["close"].iloc[-1]
        poi_type, poi_zone = None, None
        if ob_low and ob_high and ob_low <= last_close <= ob_high and is_discount:
            poi_type = "OB"
            poi_zone = (ob_low, ob_high)
            logger.info(f"Retest detected in OB: {ob_low}-{ob_high}")
        elif fvg_low and fvg_high and fvg_low <= last_close <= fvg_high and is_discount:
            poi_type = "IFVG"
            poi_zone = (fvg_low, fvg_high)
            logger.info(f"Retest detected in IFVG: {fvg_low}-{fvg_high}")
        return poi_type, poi_zone
    except Exception as e:
        logger.error(f"Error in detect_retest_zone: {str(e)}")
        return None, None

# === M5 BOS Confirmation (Keep) ===
def confirm_ltf_entry(symbol, msb, poi_zone):
    try:
        df_ltf = _get_data(symbol, TIMEFRAME_LTF, bars=20) # Use _get_data
        if df_ltf is None or df_ltf.empty:
            logger.warning("No M5 data available")
            return False
        last_candle = df_ltf.iloc[-1]
        confirmed = False
        if "bullish" in msb["type"]:
            confirmed = last_candle["close"] > msb["break_level"]
        elif "bearish" in msb["type"]:
            confirmed = last_candle["close"] < msb["break_level"]
        poi_low, poi_high = poi_zone
        if not (poi_low <= last_candle["close"] <= poi_high):
            logger.debug("M5 price not in POI zone")
            return False
        if confirmed:
            logger.info(f"M5 BOS confirmed for {msb['type']}")
            return True
        logger.debug("No M5 BOS confirmation")
        return False
    except Exception as e:
        logger.error(f"Error in confirm_ltf_entry: {str(e)}")
        return False

# === H1 Trend Bias (Keep) ===
def check_h1_trend(df_higher, df_main):
    try:
        msb_higher = detect_market_structure(df_higher, "H1")
        if not msb_higher["type"]:
            logger.debug("No H1 CHOCH")
            return None
        if not confirm_bos(df_higher, msb_higher):
            logger.debug("No H1 BOS")
            return None
        grab_type, _ = detect_liquidity_grab(df_main)
        if not grab_type or grab_type != msb_higher["trend"]:
            logger.debug("No M15 liquidity grab or direction mismatch")
            return None
        logger.info(f"H1 trend confirmed: {msb_higher['trend']}")
        return msb_higher["trend"]
    except Exception as e:
        logger.error(f"Error in check_h1_trend: {str(e)}")
        return None

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
			if direction == "bullish":
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
			sl = float(entry_price - sl_dist) if direction == "bullish" else float(entry_price + sl_dist)
		if symbol_info and hasattr(symbol_info, 'digits'):
			sl = round(sl, int(symbol_info.digits))
		return sl
	except Exception:
		return None

class msb_retest:
    def execute(self, symbol, prices, df, equity, allow_multiple_trades):
        try:
            # ... existing logic ...
            pass
        except Exception as e:
            logger.error(f"Error in msb_retest execute: {str(e)}", exc_info=True)
            return {"success": False}

# Minimal MSBRetestStrategy class for import compatibility
class MSBRetestStrategy(msb_retest):
    """
    Minimal MSBRetestStrategy class for import compatibility. Inherits msb_retest.
    """
    pass

    def execute(self, symbol, prices, df, equity, allow_multiple_trades):
        try:
            # df here is df_main, prices is df_main['close'].to_numpy()
            if df is None or df.empty:
                logger.error("No M15 data provided to msb_retest strategy.")
                return {"success": False}

            # Fetch higher and lower timeframe data within the strategy, as these are specific to the logic
            df_higher = _get_data(symbol, TIMEFRAME_HIGHER, bars=100)
            if df_higher is None or df_higher.empty:
                logger.error("No H1 data for msb_retest strategy.")
                return {"success": False}

            # H1 trend bias
            h1_trend = check_h1_trend(df_higher, df) # Using df (M15)
            if not h1_trend:
                logger.info("No valid H1 trend")
                return {"success": False}

            # M15 consolidation
            is_consolidated, _ = detect_consolidation(df)
            if not is_consolidated:
                logger.info("No consolidation detected")
                return {"success": False}

            # M15 market structure
            msb_main = detect_market_structure(df, "M15")
            bos_choch_present = msb_main["type"] in ["bullish_choch", "bearish_choch"]
            grab_type, _ = detect_liquidity_grab(df) # FVG zone is not directly used here for signal, only grab_type
            liquidity_grab_present = grab_type is not None
            if not (bos_choch_present or liquidity_grab_present):
                logger.info("No BOS/CHOCH or liquidity grab present")
                return {"success": False}

            # Confirm BOS on M15
            if not confirm_bos(df, msb_main):
                logger.info("No BOS confirmation on M15")
                return {"success": False}

            # Detect retest zone
            atr = calculate_atr(df)
            if atr is None: # ATR might be None if df is too small or calculation fails
                logger.warning("ATR could not be calculated. Skipping trade signal.")
                return {"success": False}

            poi_type, poi_zone = detect_retest_zone(df, msb_main, df_higher, atr)
            if not poi_type or not poi_zone:
                logger.info("No valid retest zone")
                return {"success": False}

            # M5 entry confirmation
            if not confirm_ltf_entry(symbol, msb_main, poi_zone):
                logger.info("M5 BOS confirmation failed")
                return {"success": False}

            # Setup trade details for return
            direction = msb_main["trend"] # Use the trend from msb_main (bullish/bearish)
            tick = mt5.symbol_info_tick(symbol)
            if tick is None or tick.bid == 0 or tick.ask == 0:
                logger.warning(f"Failed to get real-time tick data for {symbol}. Cannot determine entry price.")
                return {"success": False}

            price = tick.ask if direction == "bullish" else tick.bid
            poi_low, poi_high = poi_zone

            # Calculate SL and TPs based on logic and ATR
            # Replace direct ATR * 1.5 usage with compute_sl
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None or symbol_info.point == 0:
                logger.error(f"Symbol info for {symbol} unavailable or point is zero. Cannot calculate SL/TP precisely.")
                return {"success": False}

            sl_calc = compute_sl(symbol, price, direction, atr, recent_df=df, symbol_info=symbol_info)
            if sl_calc is None:
                logger.warning("SL calculation failed")
                return {"success": False}
            sl = sl_calc

            sl_points = abs(price - sl) / symbol_info.point

            # Calculate TP levels based on R multiples
            tp1 = price + (sl_points * symbol_info.point) if direction == "bullish" else price - (sl_points * symbol_info.point)
            tp2 = price + (sl_points * 2 * symbol_info.point) if direction == "bullish" else price - (sl_points * 2 * symbol_info.point)
            tp3 = price + (sl_points * 3 * symbol_info.point) if direction == "bullish" else price - (sl_points * 3 * symbol_info.point)

            # Round values for clean output
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
            zone_width_norm = float(poi_high - poi_low) / atr_safe
            range_ratio = rng / atr_safe
            wick_up_ratio = wick_up / (body + 1e-6)
            wick_dn_ratio = wick_dn / (body + 1e-6)
            # Sweep flags from M15 liquidity grab direction
            sweep_low_flag = int(grab_type == 'bearish')
            sweep_high_flag = int(grab_type == 'bullish')
            choch_bullish = int(msb_main["type"] == "bullish_choch")
            choch_bearish = int(msb_main["type"] == "bearish_choch")
            in_zone_flag = 1  # retest zone validated
            # M15 volume spike
            vol_spike_m15 = 0
            if 'tick_volume' in df and len(df) >= 20:
                avg_vol = float(df['tick_volume'].rolling(window=20).mean().iloc[-1])
                last_vol = float(df['tick_volume'].iloc[-1])
                vol_spike_m15 = int(avg_vol > 0 and last_vol > 1.5 * avg_vol)
            hour = int(df.index[-1].hour)
            direction_flag = 1 if direction == "bullish" else 0
            features = [
                atr_val,               # 1) ATR
                zone_width_norm,       # 2) Retest zone width / ATR
                range_ratio,           # 3) Candle range / ATR
                wick_up_ratio,         # 4) Upper wick / body
                wick_dn_ratio,         # 5) Lower wick / body
                sweep_low_flag,        # 6) Sweep low flag
                sweep_high_flag,       # 7) Sweep high flag
                choch_bullish,         # 8) CHOCH/BOS bullish
                choch_bearish,         # 9) CHOCH/BOS bearish
                in_zone_flag,          # 10) In zone
                vol_spike_m15,         # 11) M15 volume spike
                hour,                  # 12) Hour
                direction_flag,        # 13) Direction flag (1=bullish)
            ]
            return {
                "success": True,
                "direction": direction,
                "price": price,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3,  # Include TP3 for main.py's partials logic
                "features": features,
            }

        except Exception as e:
            logger.error(f"Error in msb_retest execute: {str(e)}", exc_info=True)
            return {"success": False}