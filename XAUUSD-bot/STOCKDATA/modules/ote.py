import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import sys
import os

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from STOCKDATA.utils.position_utils import calculate_position_size
from modules.indicators import calculate_atr

logger = logging.getLogger('trade_bot.ote')

# Minimum candles required per timeframe
MIN_4H = 50
MIN_1H = 30
MIN_15M = 50
MIN_5M = 20

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
            if direction == 'buy':
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
            sl = float(entry_price - sl_dist) if direction == 'buy' else float(entry_price + sl_dist)
        if symbol_info and hasattr(symbol_info, 'digits'):
            sl = round(sl, int(symbol_info.digits))
        return sl
    except Exception:
        return None

class OTEStrategy:
    def __init__(self):
        pass

    def execute(self, symbol, prices, df, equity, allow_multiple_trades):
        """
        Execute method for OTE strategy that matches the interface expected by main.py
        """
        try:
            logger.info(f"Executing OTE strategy | Symbol: {symbol} | Capital: ${equity:.2f}")
            
            # Get different timeframe data (ensure minimum history)
            df_4h, _ = self.fetch_candles(symbol, mt5.TIMEFRAME_H4, max(MIN_4H, 100))
            df_1h, _ = self.fetch_candles(symbol, mt5.TIMEFRAME_H1, max(MIN_1H, 100))
            df_15m, _ = self.fetch_candles(symbol, mt5.TIMEFRAME_M15, max(MIN_15M, 100))
            df_5m, _ = self.fetch_candles(symbol, mt5.TIMEFRAME_M5, max(MIN_5M, 100))
            
            # Defensive checks: ensure we have enough history
            if df_4h is None or len(df_4h) < MIN_4H:
                logger.info(f"[SKIP] OTE: Not enough 4H candles ({0 if df_4h is None else len(df_4h)}) - need >= {MIN_4H}")
                return self.empty_strategy_result()
            if df_1h is None or len(df_1h) < MIN_1H:
                logger.info(f"[SKIP] OTE: Not enough 1H candles ({0 if df_1h is None else len(df_1h)}) - need >= {MIN_1H}")
                return self.empty_strategy_result()
            if df_15m is None or len(df_15m) < MIN_15M:
                logger.info(f"[SKIP] OTE: Not enough 15M candles ({0 if df_15m is None else len(df_15m)}) - need >= {MIN_15M}")
                return self.empty_strategy_result()
            if df_5m is None or len(df_5m) < MIN_5M:
                logger.info(f"[SKIP] OTE: Not enough 5M candles ({0 if df_5m is None else len(df_5m)}) - need >= {MIN_5M}")
                return self.empty_strategy_result()

            # Call the original entry_signal method
            result = self.entry_signal(symbol, df_4h, df_1h, df_15m, df_5m, equity)
            
            if result is None or not result.get('success', False):
                logger.info("[SKIP] OTE: No valid signal")
                return self.empty_strategy_result()
            
            # Build canonical signal dict for centralized processor (no direct order send here)
            direction = result['direction']
            price = result['price']
            sl = result['sl']
            tp1 = result['tp1']
            tp2 = result['tp2']
            lot_size = result['lot_size']
            features = result.get('features', [])

            return {
                "allowed": True,
                "confidence": 0.7,
                "success": True,
                "is_win": False,
                "risk_amount": equity * 0.01,
                "direction": direction,
                "price": price,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "lot_size": lot_size,
                "features": features,
            }
            
        except Exception as e:
            logger.exception(f"Error in OTE execute: {str(e)}")
            return self.empty_strategy_result()

    def empty_strategy_result(self):
        """Return empty strategy result"""
        return {
            "success": False,
            "is_win": False,
            "risk_amount": 0.0,
            "direction": None,
            "price": None,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "lot_size": None,
            "features": [],
        }

    def fetch_candles(self, symbol, timeframe, count):
        """Fetch candles for a specific timeframe and ensure timezone-aware datetime index"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                logger.warning(f"fetch_candles: No data for {symbol} TF {timeframe}")
                return None, None
            df = pd.DataFrame(rates)
            # create utc-aware datetime index
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            df.set_index('time', inplace=True)
            # ensure tick_volume exists (some feeds may not include)
            if 'tick_volume' not in df.columns:
                df['tick_volume'] = 0
            prices = df['close'].to_numpy(dtype=np.float64)
            return df, prices
        except Exception as e:
            logger.exception(f"Error fetching candles for {symbol}: {str(e)}")
            return None, None

    def get_1h_swing_high_low(self, df_1h):
        # choose last N bars to define swing; require at least 10 bars
        length = len(df_1h)
        window = min(max(length // 10, 10), 50)  # dynamic but safe
        if length < 10:
            logger.debug("get_1h_swing_high_low: insufficient bars for swing detection")
            return None, None
        swing_high = df_1h['high'].iloc[-window:].max()
        swing_low = df_1h['low'].iloc[-window:].min()
        return swing_high, swing_low

    def in_ote_zone(self, price, swing_high, swing_low, side):
        # return boolean + zone bounds (low, high)
        if swing_high is None or swing_low is None:
            return False, (None, None)
        if side == 'buy':
            fib_618 = swing_low + 0.618 * (swing_high - swing_low)
            fib_786 = swing_low + 0.786 * (swing_high - swing_low)
            low, high = sorted((fib_618, fib_786))
            return (low <= price <= high), (low, high)
        else:
            fib_618 = swing_high - 0.618 * (swing_high - swing_low)
            fib_786 = swing_high - 0.786 * (swing_high - swing_low)
            low, high = sorted((fib_786, fib_618))
            return (low <= price <= high), (low, high)

    def detect_liquidity_sweep_15m(self, df_15m, side):
        if len(df_15m) < 6:
            return False
        if side == 'buy':
            return df_15m['low'].iloc[-1] < df_15m['low'].iloc[-6:-1].min()
        else:
            return df_15m['high'].iloc[-1] > df_15m['high'].iloc[-6:-1].max()

    def detect_choch_bos_15m(self, df_15m, side):
        if len(df_15m) < 6:
            return False
        if side == 'buy':
            return df_15m['close'].iloc[-1] > df_15m['high'].iloc[-6:-1].max()
        else:
            return df_15m['close'].iloc[-1] < df_15m['low'].iloc[-6:-1].min()

    def check_5m_entry(self, df_5m, side):
        if len(df_5m) < 3:
            return False, None, None
        c = df_5m.iloc[-1]
        p = df_5m.iloc[-2]
        vol_avg = df_5m['tick_volume'].iloc[-5:-1].mean() if 'tick_volume' in df_5m and len(df_5m) >= 6 else 0
        # ensure numeric values (defensive)
        c_close = float(c['close'])
        c_open = float(c['open'])
        p_high = float(p['high'])
        p_low = float(p['low'])
        if side == 'buy':
            is_engulfing = (c_close > c_open) and (c_close > p_high) and (c_open < p_low)
            vol_ok = True if vol_avg == 0 else (float(c.get('tick_volume', 0)) > vol_avg * 1.2)
            if is_engulfing and vol_ok:
                return True, float(c['low']), float(c['close'])
        else:
            is_engulfing = (c_close < c_open) and (c_close < p_low) and (c_open > p_high)
            vol_ok = True if vol_avg == 0 else (float(c.get('tick_volume', 0)) > vol_avg * 1.2)
            if is_engulfing and vol_ok:
                return True, float(c['high']), float(c['close'])
        return False, None, None

    def entry_signal(self, symbol, df_4h, df_1h, df_15m, df_5m, equity):
        """
        Core OTE entry logic. Returns a standard dict or None.
        """
        try:
            # Defensive: already checked for minimum counts in execute, but double-check
            if df_4h is None or df_1h is None or df_15m is None or df_5m is None:
                logger.debug("entry_signal: missing timeframe data")
                return None
            if len(df_4h) < MIN_4H or len(df_1h) < MIN_1H or len(df_15m) < MIN_15M or len(df_5m) < MIN_5M:
                logger.debug("entry_signal: insufficient candles across timeframes")
                return None

            # Determine trend on 4H (use prior closed bars safely)
            try:
                prev_4h_high = df_4h['high'].rolling(window=20).max().shift(1).iloc[-1]
                prev_4h_low = df_4h['low'].rolling(window=20).min().shift(1).iloc[-1]
            except Exception:
                prev_4h_high = None
                prev_4h_low = None

            last_4h_close = float(df_4h['close'].iloc[-1])
            trend = None
            if prev_4h_high is not None and last_4h_close > prev_4h_high:
                trend = 'buy'
            elif prev_4h_low is not None and last_4h_close < prev_4h_low:
                trend = 'sell'
            else:
                logger.debug("entry_signal: 4H trend not established")
                return None

            # 1H swing high/low
            swing_high, swing_low = self.get_1h_swing_high_low(df_1h)
            if swing_high is None or swing_low is None:
                logger.debug("entry_signal: unable to compute 1H swing high/low")
                return None

            # Use latest M15 close for zone check (we already ensured index)
            price = float(df_15m['close'].iloc[-1])
            in_ote, (zone_low, zone_high) = self.in_ote_zone(price, swing_high, swing_low, trend)
            if not in_ote:
                logger.debug(f"entry_signal: price {price} not in OTE zone ({zone_low}, {zone_high})")
                return None

            # Check liquidity sweep or choch on 15m
            sweep = self.detect_liquidity_sweep_15m(df_15m, trend)
            choch = self.detect_choch_bos_15m(df_15m, trend)
            if not (sweep or choch):
                logger.debug("entry_signal: no sweep/choch on 15M")
                return None

            # Check 5m entry pattern
            entry_ok, sl, entry = self.check_5m_entry(df_5m, trend)
            if not entry_ok:
                logger.debug("entry_signal: no valid 5M entry pattern")
                return None

            # ATR and symbol info validation
            atr = calculate_atr(df_1h)
            if atr is None or not np.isfinite(atr) or atr <= 0:
                logger.warning("entry_signal: ATR invalid or zero")
                return None
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None or symbol_info.point == 0:
                logger.warning("entry_signal: symbol_info missing or point==0")
                return None

            # sl_points and lot calculation (guard division by zero)
            # Replace original computed sl with rule-based SL
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None or symbol_info.point == 0:
                logger.warning("entry_signal: symbol_info missing or point==0")
                return None

            # Compute SL according to new policy
            sl_calc = compute_sl(symbol, entry, trend, atr, recent_df=df_5m, symbol_info=symbol_info)
            if sl_calc is None:
                logger.warning("entry_signal: computed SL invalid")
                return None
            sl = sl_calc

            sl_points = abs(entry - sl) / symbol_info.point if symbol_info.point and abs(entry - sl) > 0 else None
            if sl_points is None or sl_points == 0 or not np.isfinite(sl_points):
                logger.warning("entry_signal: computed sl_points invalid")
                return None

            lot_size = calculate_position_size(symbol, equity, atr, sl_points)
            if lot_size is None or lot_size <= 0:
                logger.warning("entry_signal: calculated lot_size invalid")
                return None

            # calculate TP levels
            if trend == 'buy':
                tp1 = entry + (sl_points * symbol_info.point)
                tp2 = entry + (sl_points * 2 * symbol_info.point)
            else:
                tp1 = entry - (sl_points * symbol_info.point)
                tp2 = entry - (sl_points * 2 * symbol_info.point)

            # Round based on symbol digits
            digits = int(symbol_info.digits) if symbol_info and hasattr(symbol_info, 'digits') else 5
            entry = round(entry, digits)
            sl = round(sl, digits)
            tp1 = round(tp1, digits)
            tp2 = round(tp2, digits)

            # Build ML features (13-length vector)
            last_m15 = df_15m.iloc[-1]
            body = float(abs(last_m15['close'] - last_m15['open']))
            rng = float(last_m15['high'] - last_m15['low'])
            wick_up = float(last_m15['high'] - max(last_m15['open'], last_m15['close']))
            wick_dn = float(min(last_m15['open'], last_m15['close']) - last_m15['low'])
            atr_val = float(atr)
            atr_safe = atr_val if atr_val != 0 else 1.0
            zone_width_norm = float(abs(zone_high - zone_low)) / atr_safe if zone_low is not None and zone_high is not None else 0.0
            range_ratio = rng / atr_safe
            wick_up_ratio = wick_up / (body + 1e-6)
            wick_dn_ratio = wick_dn / (body + 1e-6)
            sweep_low_flag = int(trend == 'buy' and sweep)
            sweep_high_flag = int(trend == 'sell' and sweep)
            choch_bullish = int(trend == 'buy' and choch)
            choch_bearish = int(trend == 'sell' and choch)
            in_zone_flag = int(in_ote)
            vol_spike_m15 = 0
            if 'tick_volume' in df_15m and len(df_15m) >= 20:
                avg_vol = float(df_15m['tick_volume'].rolling(window=20).mean().iloc[-1])
                last_vol = float(df_15m['tick_volume'].iloc[-1])
                vol_spike_m15 = int(avg_vol > 0 and last_vol > 1.5 * avg_vol)
            hour = int(df_15m.index[-1].hour) if isinstance(df_15m.index, pd.DatetimeIndex) else int(pd.to_datetime(df_15m['time'].iloc[-1]).hour)
            direction_flag = 1 if trend == 'buy' else 0
            features = [
                atr_val,               # 1) ATR
                zone_width_norm,       # 2) Zone width / ATR
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
                direction_flag,        # 13) Direction flag (1=buy)
            ]

            return {
                'success': True,
                'direction': trend,
                'price': entry,
                'sl': sl,
                'tp1': tp1,
                'tp2': tp2,
                'lot_size': lot_size,
                'features': features,
            }
        except Exception as e:
            logger.exception(f"entry_signal error: {e}")
            return None
