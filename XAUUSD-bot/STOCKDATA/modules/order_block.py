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

logger = logging.getLogger('trade_bot.order_block')

class OrderBlockStrategy:
    def __init__(self):
        self.max_risk_percent = 1.0  # 1% risk per trade

    def detect_htf_trend(self, df_4h):
        if len(df_4h) < 50:
            return None
        ema = df_4h['close'].ewm(span=50, adjust=False).mean()
        if ema.iloc[-1] > ema.iloc[-5] and df_4h['close'].iloc[-1] > ema.iloc[-1]:
            return 'buy'
        elif ema.iloc[-1] < ema.iloc[-5] and df_4h['close'].iloc[-1] < ema.iloc[-1]:
            return 'sell'
        return None

    def find_swing_high_low(self, df, window=None):
        atr = calculate_atr(df)
        window = max(5, int(atr / df['close'].iloc[-1] * 100))  # Dynamic window based on ATR
        swing_high = df['high'].rolling(window).max().iloc[-1]
        swing_low = df['low'].rolling(window).min().iloc[-1]
        return swing_high, swing_low

    def find_order_block(self, df, side, swing_high, swing_low):
        if side == 'buy':
            obs = df[(df['close'] < df['open']) & (df['low'] < swing_low)]
            if obs.empty:
                return None, None
            return obs.iloc[-1]['low'], obs.iloc[-1]['high']  # Most recent OB
        else:
            obs = df[(df['close'] > df['open']) & (df['high'] > swing_high)]
            if obs.empty:
                return None, None
            return obs.iloc[-1]['low'], obs.iloc[-1]['high']  # Most recent OB

    def find_fvg(self, df, ob_zone, side):
        atr = calculate_atr(df)
        for i in range(-2, -12, -1):
            c1, c2 = df.iloc[i], df.iloc[i-1]
            if side == 'buy' and c1['low'] > c2['high'] + 0.5 * atr and ob_zone[0] <= c1['low'] <= ob_zone[1]:
                return c2['high'], c1['low']
            if side == 'sell' and c1['high'] < c2['low'] - 0.5 * atr and ob_zone[1] >= c1['high'] >= ob_zone[0]:
                return c1['high'], c2['low']
        return None, None

    def detect_liquidity_sweep(self, df_5m, side):
        if len(df_5m) < 7:
            return False
        last = df_5m.iloc[-1]
        wick_ratio = (last['high'] - last['close']) / (last['high'] - last['low'] + 1e-5) if side == 'sell' else \
                     (last['close'] - last['low']) / (last['high'] - last['low'] + 1e-5)
        if side == 'buy':
            return last['low'] < df_5m['low'].iloc[-7:-1].min() and wick_ratio > 0.6
        else:
            return last['high'] > df_5m['high'].iloc[-7:-1].max() and wick_ratio > 0.6

    def detect_choch(self, df_5m, side):
        if len(df_5m) < 3:
            return False
        if side == 'buy':
            return df_5m['close'].iloc[-1] > df_5m['high'].iloc[-3:-1].max()
        else:
            return df_5m['close'].iloc[-1] < df_5m['low'].iloc[-3:-1].min()

    def is_confirmation(self, df, side):
        if len(df) < 2:
            return False
        c1, c2 = df.iloc[-2], df.iloc[-1]
        atr = calculate_atr(df)  # Use M5 ATR for context
        if side == 'buy':
            engulfing = c2['close'] > c2['open'] and c2['close'] > c1['open']
            large_momentum = (c2['high'] - c2['low']) > atr
            pin_bar = c2['close'] > c2['open'] and (c2['high'] - c2['close']) < 0.3 * (c2['high'] - c2['low'])
            return (engulfing and large_momentum) or pin_bar
        else:
            engulfing = c2['close'] < c2['open'] and c2['close'] < c1['open']
            large_momentum = (c2['high'] - c2['low']) > atr
            pin_bar = c2['close'] < c2['open'] and (c2['close'] - c2['low']) < 0.3 * (c2['high'] - c2['low'])
            return (engulfing and large_momentum) or pin_bar

    def entry_signal(self, symbol, df_4h, df_1h, df_15m, df_5m, equity):
        if len(df_4h) < 50 or len(df_1h) < 10 or len(df_15m) < 15 or len(df_5m) < 10:
            return None

        trend = self.detect_htf_trend(df_4h)
        if trend not in ['buy', 'sell']:
            return None

        swing_high, swing_low = self.find_swing_high_low(df_1h)
        ob_low, ob_high = self.find_order_block(df_1h, trend, swing_high, swing_low)
        if ob_low is None or ob_high is None:
            return None
        ob_zone = (ob_low, ob_high)

        fvg_low, fvg_high = self.find_fvg(df_15m, ob_zone, trend)
        fvg_zone = (fvg_low, fvg_high) if fvg_low and fvg_high else None

        # Feature extraction for ML model
        features = []
        # 1. Trend strength
        features.append(1 if trend == 'buy' else 0)
        # 2. OB zone size
        features.append(ob_high - ob_low)
        # 3. Distance from current price to OB mid
        current_price = df_5m['close'].iloc[-1]
        features.append(abs(current_price - ((ob_low + ob_high)/2)))
        # 4-7. Technical indicators (RSI, MACD, etc.)
        features.extend([0.5, 0.3, 0.7, 0.2])  # Placeholder values
        # 8-10. Volatility measures
        features.extend([calculate_atr(df_5m, 14), 
                         df_5m['high'].iloc[-5:].max() - df_5m['low'].iloc[-5:].min(),
                         df_5m['close'].pct_change().std()*100])
        # 11-13. Additional context features
        features.extend([1 if fvg_zone else 0, 
                         2, 
                         equity])

        # Ensure we have exactly 13 features
        if len(features) < 13:
            # Add zeros for missing features
            features.extend([0]*(13-len(features)))
        elif len(features) > 13:
            features = features[:13]

        entry_zones = []
        if ob_zone[0] is not None and ob_zone[1] is not None:
            entry_zones.append(('ob', ob_zone))
        if fvg_zone:
            entry_zones.append(('fvg', fvg_zone))

        for zone_type, zone in entry_zones:
            in_zone = zone[0] <= df_5m['close'].iloc[-1] <= zone[1] if trend == 'buy' else zone[1] >= df_5m['close'].iloc[-1] >= zone[0]
            if in_zone:
                if self.detect_liquidity_sweep(df_5m, trend) or self.detect_choch(df_5m, trend):
                    if not self.is_confirmation(df_5m, trend):
                        continue  # No confirmation
                    entry = df_5m['close'].iloc[-1]
                    atr = calculate_atr(df_1h)
                    symbol_info = mt5.symbol_info(symbol)
                    sl_calc = compute_sl(symbol, float(entry), trend, atr, recent_df=df_5m, symbol_info=symbol_info)
                    if sl_calc is None:
                        continue
                    sl = sl_calc

                    # Compute sl_points (risk in points) and validate before using
                    if symbol_info is None or not hasattr(symbol_info, 'point') or symbol_info.point == 0:
                        logger.warning("entry_signal: symbol_info missing or point==0 - cannot compute sl_points")
                        continue
                    # risk in price units
                    risk = abs(float(entry) - float(sl))
                    # sl_points as number of points/pips
                    sl_points = risk / symbol_info.point if symbol_info.point and risk > 0 else None
                    if sl_points is None or sl_points == 0 or not np.isfinite(sl_points):
                        logger.warning("entry_signal: computed sl_points invalid")
                        continue

                    lot_size = calculate_position_size(symbol, equity, atr, sl_points, risk_percent=self.max_risk_percent)

                    # Use risk (price distance) to compute TP levels (preserve original RR logic)
                    tp1 = float(entry) + (risk * 1.5) if trend == 'buy' else float(entry) - (risk * 1.5)
                    tp2 = float(entry) + (risk * 2.0) if trend == 'buy' else float(entry) - (risk * 2.0)

                    return {
                        'success': True,
                        'direction': trend,
                        'price': entry,
                        'sl': sl,
                        'tp1': tp1,
                        'tp2': tp2,
                        'lot_size': lot_size,
                        'features': features,
                        'zone': zone_type,
                        'strategy': 'OrderBlock + FVG + CHOCH + Confirmation'
                    }

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