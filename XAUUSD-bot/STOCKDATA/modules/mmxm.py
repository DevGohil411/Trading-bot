import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import pytz
import os
import sys

from utils.trade_logger import logger

# === SETTINGS === #
SYMBOLS = ["XAUUSD"]
TIMEFRAME_MAIN = mt5.TIMEFRAME_M15
TIMEFRAME_HIGHER = mt5.TIMEFRAME_H1
TIMEFRAME_LTF = mt5.TIMEFRAME_M5
CONSOLIDATION_RANGE_THRESHOLD = 0.007
LIQUIDITY_SWEEP_THRESHOLD = 0.005
VOLUME_MULTIPLIER = 1.5
OB_LOOKBACK = 8
ATR_PERIOD = 14

def _get_data(symbol, timeframe, bars=300):
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            logger.error(f"No data for {symbol} TF {timeframe}")
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s").dt.tz_localize('UTC')
        df.set_index("time", inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error in _get_data: {str(e)}")
        return None

def calculate_atr(df, period=ATR_PERIOD):
    try:
        if len(df) < period: return None
        df["tr"] = np.maximum.reduce([(df["high"] - df["low"]), (df["high"] - df["close"].shift()).abs(), (df["low"] - df["close"].shift()).abs()])
        atr = df["tr"].rolling(window=period).mean().iloc[-1]
        return None if pd.isna(atr) or atr == 0 else atr
    except Exception as e:
        logger.error(f"Error in calculate_atr: {str(e)}")
        return None

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
        return False, None
    except Exception as e:
        logger.error(f"Error in detect_consolidation: {str(e)}")
        return False, None

def detect_liquidity_sweep(df, consolidation_zone):
    try:
        if not consolidation_zone: return False, None
        range_low, range_high = consolidation_zone
        last_candle = df.iloc[-1]
        direction = None
        avg_vol = df['tick_volume'].iloc[-10:-1].mean() if 'tick_volume' in df else 0
        if last_candle["high"] > range_high + (range_high * LIQUIDITY_SWEEP_THRESHOLD) and last_candle["close"] < range_high:
            if 'tick_volume' in df and last_candle['tick_volume'] > avg_vol * VOLUME_MULTIPLIER:
                direction = "bearish"
                logger.info(f"Bearish liquidity sweep with volume spike: {last_candle['tick_volume']}/{avg_vol}")
        elif last_candle["low"] < range_low - (range_low * LIQUIDITY_SWEEP_THRESHOLD) and last_candle["close"] > range_low:
            if 'tick_volume' in df and last_candle['tick_volume'] > avg_vol * VOLUME_MULTIPLIER:
                direction = "bullish"
                logger.info(f"Bullish liquidity sweep with volume spike: {last_candle['tick_volume']}/{avg_vol}")
        return direction is not None, direction
    except Exception as e:
        logger.error(f"Error in detect_liquidity_sweep: {str(e)}")
        return False, None

def detect_market_structure(df, timeframe_str):
    try:
        sh, sl = detect_swing_highs_lows(df)
        if len(sh) < 2 or len(sl) < 2: return {"type": None, "break_level": None, "trend": None}
        last_high, last_low, last_close = df["high"].iloc[sh[-1]], df["low"].iloc[sl[-1]], df["close"].iloc[-1]
        ms_type, break_level, trend = None, None, None
        if last_close > last_high:
            ms_type, break_level, trend = "bullish_choch", last_high, "bullish"
            logger.info(f"Bullish CHOCH on {timeframe_str}")
        elif last_close < last_low:
            ms_type, break_level, trend = "bearish_choch", last_low, "bearish"
            logger.info(f"Bearish CHOCH on {timeframe_str}")
        return {"type": ms_type, "break_level": break_level, "trend": trend}
    except Exception as e:
        logger.error(f"Error in detect_market_structure: {str(e)}")
        return {"type": None, "break_level": None, "trend": None}

def confirm_bos(df, msb):
    try:
        if msb["type"] is None: return False
        last_candle = df.iloc[-1]
        confirmed = False
        if "bullish" in msb["type"]:
            confirmed = last_candle["high"] > msb["break_level"] and last_candle["close"] > msb["break_level"]
        elif "bearish" in msb["type"]:
            confirmed = last_candle["low"] < msb["break_level"] and last_candle["close"] < msb["break_level"]
        if confirmed: logger.info(f"BOS confirmed on {msb['type']}")
        return confirmed
    except Exception as e:
        logger.error(f"Error in confirm_bos: {str(e)}")
        return False

def detect_retest_zone(df, msb, df_higher, atr):
    try:
        if msb["type"] is None: return None, None
        ob_low, ob_high = None, None
        for i in range(-2, -OB_LOOKBACK-1, -1):
            if len(df) < abs(i): continue
            curr = df.iloc[i]
            if "bullish" in msb["type"] and curr["close"] < curr["open"]:
                ob_low, ob_high = curr["low"], curr["high"]; break
            elif "bearish" in msb["type"] and curr["close"] > curr["open"]:
                ob_low, ob_high = curr["low"], curr["high"]; break
        htf_avg = df_higher["close"].iloc[-20:].mean()
        is_discount = df["close"].iloc[-1] < htf_avg if "bullish" in msb["type"] else df["close"].iloc[-1] > htf_avg
        last_close = df["close"].iloc[-1]
        if ob_low and ob_high and ob_low <= last_close <= ob_high and is_discount:
            logger.info(f"Retest detected in OB: {ob_low}-{ob_high}")
            return "OB", (ob_low, ob_high)
        return None, None
    except Exception as e:
        logger.error(f"Error in detect_retest_zone: {str(e)}")
        return None, None

def confirm_ltf_entry(symbol, msb):
    try:
        df_ltf = _get_data(symbol, TIMEFRAME_LTF, bars=20)
        if df_ltf is None: return False
        ltf_msb = detect_market_structure(df_ltf, "M5")
        if not ltf_msb["type"]: return False
        expected_ltf_type = "bullish_choch" if "bullish" in msb["type"] else "bearish_choch"
        if ltf_msb["type"] != expected_ltf_type: return False
        logger.info(f"M5 CHOCH/BOS confirmation for {msb['type']}")
        return True
    except Exception as e:
        logger.error(f"Error in confirm_ltf_entry: {str(e)}")
        return False

def check_h1_trend(df_higher):
    try:
        sma = df_higher["close"].iloc[-20:].mean()
        last_close = df_higher["close"].iloc[-1]
        if last_close > sma: return "bullish"
        elif last_close < sma: return "bearish"
        return None
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
	except Exception as e:
		logger.error(f"compute_sl error: {e}")
		return None

class mmxm:
    def execute(self, symbol, prices, df, equity, allow_multiple_trades):
        try:
            if df is None or df.empty: return {"success": False}
            df_higher = _get_data(symbol, TIMEFRAME_HIGHER, bars=50)
            if df_higher is None or df_higher.empty: return {"success": False}
            h1_trend = check_h1_trend(df_higher)
            if not h1_trend: return {"success": False}
            atr = calculate_atr(df)
            if atr is None: return {"success": False}
            msb = detect_market_structure(df, "M15")
            if not msb["trend"] or msb["trend"] != h1_trend: return {"success": False}
            is_consolidated, consolidation_zone = detect_consolidation(df)
            if not is_consolidated: return {"success": False}
            is_sweep, direction = detect_liquidity_sweep(df, consolidation_zone)
            if not is_sweep or direction != msb["trend"]: return {"success": False}
            if not confirm_bos(df, msb): return {"success": False}
            poi_type, poi_zone = detect_retest_zone(df, msb, df_higher, atr)
            if not poi_type: return {"success": False}
            ltf_confirmed = confirm_ltf_entry(symbol, msb)
            if not ltf_confirmed: return {"success": False}
            
            trade_direction = msb["trend"]
            tick = mt5.symbol_info_tick(symbol)
            if tick is None or tick.bid == 0 or tick.ask == 0: return {"success": False}
            price = tick.ask if trade_direction == "bullish" else tick.bid
            ob_low, ob_high = poi_zone
            sl = ob_low - (atr * 1.5) if trade_direction == "bullish" else ob_high + (atr * 1.5)
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None or symbol_info.point == 0: return {"success": False}
            
            # Replace old SL calc with compute_sl
            sl_calc = compute_sl(symbol, price, trade_direction, atr, recent_df=df, symbol_info=symbol_info)
            if sl_calc is None:
                return {"success": False}
            sl = sl_calc

            sl_points = abs(price - sl) / symbol_info.point
            tp1 = price + (sl_points * symbol_info.point) if trade_direction == "bullish" else price - (sl_points * symbol_info.point)
            tp2 = price + (sl_points * 2 * symbol_info.point) if trade_direction == "bullish" else price - (sl_points * 2 * symbol_info.point)
            tp3 = price + (sl_points * 3 * symbol_info.point) if trade_direction == "bullish" else price - (sl_points * 3 * symbol_info.point)
            
            price, sl, tp1, tp2, tp3 = round(price, symbol_info.digits), round(sl, symbol_info.digits), round(tp1, symbol_info.digits), round(tp2, symbol_info.digits), round(tp3, symbol_info.digits)
            logger.info(f"Trade signal generated: {trade_direction.upper()} at {price}, SL={sl}, TP1={tp1}, TP2={tp2}, TP3={tp3}")
            # Build ML features
            last = df.iloc[-1]
            body = abs(float(last["close"]) - float(last["open"]))
            rng = float(last["high"]) - float(last["low"])
            wick_up = float(last["high"]) - float(max(last["open"], last["close"]))
            wick_dn = float(min(last["open"], last["close"])) - float(last["low"])
            atr_val = float(atr)
            atr_safe = atr_val if atr_val != 0 else 1.0
            cons_width_norm = 0.0
            if consolidation_zone:
                range_low, range_high = consolidation_zone
                cons_width_norm = float(range_high - range_low) / atr_safe
            range_ratio = rng / atr_safe
            wick_up_ratio = wick_up / (body + 1e-6)
            wick_dn_ratio = wick_dn / (body + 1e-6)
            hour = int(df.index[-1].hour)
            direction_flag = 1 if trade_direction == "bullish" else 0
            features = [
                atr_val,
                cons_width_norm,
                range_ratio,
                wick_up_ratio,
                wick_dn_ratio,
                int(h1_trend == "bullish"),
                int(h1_trend == "bearish"),
                int(msb["type"] == "bullish_choch"),
                int(msb["type"] == "bearish_choch"),
                int(is_consolidated),
                hour,
                direction_flag,
                int(ltf_confirmed),
            ]
            return {"success": True, "direction": trade_direction, "price": price, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "features": features}
        except Exception as e:
            logger.error(f"Error in mmxm execute: {str(e)}", exc_info=True)
            return {"success": False}

# Minimal MMXMStrategy class to satisfy imports
class MMXMStrategy(mmxm):
    """
    Minimal MMXMStrategy class for import compatibility. Inherits from mmxm.
    """
    pass

    def execute(self, symbol, prices, df, equity, allow_multiple_trades):
        try:
            if df is None or df.empty: return {"success": False}
            df_higher = _get_data(symbol, TIMEFRAME_HIGHER, bars=50)
            if df_higher is None or df_higher.empty: return {"success": False}
            h1_trend = check_h1_trend(df_higher)
            if not h1_trend: return {"success": False}
            atr = calculate_atr(df)
            if atr is None: return {"success": False}
            msb = detect_market_structure(df, "M15")
            if not msb["trend"] or msb["trend"] != h1_trend: return {"success": False}
            is_consolidated, consolidation_zone = detect_consolidation(df)
            if not is_consolidated: return {"success": False}
            is_sweep, direction = detect_liquidity_sweep(df, consolidation_zone)
            if not is_sweep or direction != msb["trend"]: return {"success": False}
            if not confirm_bos(df, msb): return {"success": False}
            poi_type, poi_zone = detect_retest_zone(df, msb, df_higher, atr)
            if not poi_type: return {"success": False}
            ltf_confirmed = confirm_ltf_entry(symbol, msb)
            if not ltf_confirmed: return {"success": False}
            
            trade_direction = msb["trend"]
            tick = mt5.symbol_info_tick(symbol)
            if tick is None or tick.bid == 0 or tick.ask == 0: return {"success": False}
            price = tick.ask if trade_direction == "bullish" else tick.bid
            ob_low, ob_high = poi_zone
            sl = ob_low - (atr * 1.5) if trade_direction == "bullish" else ob_high + (atr * 1.5)
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None or symbol_info.point == 0: return {"success": False}
            
            # Replace old SL calc with compute_sl
            sl_calc = compute_sl(symbol, price, trade_direction, atr, recent_df=df, symbol_info=symbol_info)
            if sl_calc is None:
                return {"success": False}
            sl = sl_calc

            sl_points = abs(price - sl) / symbol_info.point
            tp1 = price + (sl_points * symbol_info.point) if trade_direction == "bullish" else price - (sl_points * symbol_info.point)
            tp2 = price + (sl_points * 2 * symbol_info.point) if trade_direction == "bullish" else price - (sl_points * 2 * symbol_info.point)
            tp3 = price + (sl_points * 3 * symbol_info.point) if trade_direction == "bullish" else price - (sl_points * 3 * symbol_info.point)
            
            price, sl, tp1, tp2, tp3 = round(price, symbol_info.digits), round(sl, symbol_info.digits), round(tp1, symbol_info.digits), round(tp2, symbol_info.digits), round(tp3, symbol_info.digits)
            logger.info(f"Trade signal generated: {trade_direction.upper()} at {price}, SL={sl}, TP1={tp1}, TP2={tp2}, TP3={tp3}")
            # Build ML features
            last = df.iloc[-1]
            body = abs(float(last["close"]) - float(last["open"]))
            rng = float(last["high"]) - float(last["low"])
            wick_up = float(last["high"]) - float(max(last["open"], last["close"]))
            wick_dn = float(min(last["open"], last["close"])) - float(last["low"])
            atr_val = float(atr)
            atr_safe = atr_val if atr_val != 0 else 1.0
            cons_width_norm = 0.0
            if consolidation_zone:
                range_low, range_high = consolidation_zone
                cons_width_norm = float(range_high - range_low) / atr_safe
            range_ratio = rng / atr_safe
            wick_up_ratio = wick_up / (body + 1e-6)
            wick_dn_ratio = wick_dn / (body + 1e-6)
            hour = int(df.index[-1].hour)
            direction_flag = 1 if trade_direction == "bullish" else 0
            features = [
                atr_val,
                cons_width_norm,
                range_ratio,
                wick_up_ratio,
                wick_dn_ratio,
                int(h1_trend == "bullish"),
                int(h1_trend == "bearish"),
                int(msb["type"] == "bullish_choch"),
                int(msb["type"] == "bearish_choch"),
                int(is_consolidated),
                hour,
                direction_flag,
                int(ltf_confirmed),
            ]
            return {"success": True, "direction": trade_direction, "price": price, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "features": features}
        except Exception as e:
            logger.error(f"Error in mmxm execute: {str(e)}", exc_info=True)
            return {"success": False}
