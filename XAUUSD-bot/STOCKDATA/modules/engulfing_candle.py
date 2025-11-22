import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI issues
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from STOCKDATA.mt5_utils import safe_positions_get
from STOCKDATA.mt5_utils import safe_positions_get
import logging
from time import sleep
from threading import Lock
import os
import glob
import sys  # Added sys import

# Logger setup
logger = logging.getLogger('trade_bot.engulfing')
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler("trade_log.log")
    stream_handler = logging.StreamHandler(sys.stdout)  # Using sys.stdout
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# Constants
SYMBOL = "XAUUSD"
CORRELATED_SYMBOLS = ["XAUEUR", "XAGUSD"]  # Correlated instruments to avoid simultaneous trades
TIMEFRAME = mt5.TIMEFRAME_M5  # LTF for entry
HIGHER_TIMEFRAME = mt5.TIMEFRAME_M15  # HTF for context
RISK_PERCENT_PER_TRADE = 1.0  # Risk per trade as % of equity
MAX_TOTAL_RISK_PERCENT = 5.0  # Max total risk across all open positions
MAX_DAILY_LOSS_PERCENT = 3.0  # Max daily loss limit
MAX_OPEN_POSITIONS = 3
MAX_PLOT_FILES = 100
COOLDOWN_PERIOD = 60 * 5  # 5 minutes for intraday
KILL_ZONES = [(7, 9), (13, 15)]  # UTC hours
CONSOLIDATION_RANGE = 0.005  # 0.5% for liquidity pool
OB_LOOKBACK = 10  # Candles for OB detection
FVG_THRESHOLD = 0.002  # 0.2% for FVG detection
ATR_PERIOD = 14
ATR_MULTIPLIER_MIN = 1.5  # Minimum SL distance as a multiple of ATR
ATR_MULTIPLIER_MAX = 3.0  # Maximum SL distance as a multiple of ATR
PARTIAL_PROFIT_LEVELS = [1.0, 1.5]  # Book partial profits at 1R, 1.5R
PARTIAL_PROFIT_PERCENTAGES = [0.5, 0.3]  # Book 50% at 1R, 30% at 1.5R
TRAILING_SL_START_R = 1.0  # Start trailing SL after 1R profit
TRAILING_SL_DISTANCE_ATR = 0.5  # Trailing SL distance as ATR multiple
MIN_POSITION_SIZE = 0.01
MAX_POSITION_SIZE = 1.0

# Set PLOTS_DIR to an absolute path
script_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(script_dir)
PLOTS_DIR = os.path.join(parent_dir, "plots")
logger.info(f"Plots directory set to: {PLOTS_DIR}")

# Create plots directory
if not os.path.exists(PLOTS_DIR):
    try:
        os.makedirs(PLOTS_DIR)
        logger.info(f"Created plots directory: {PLOTS_DIR}")
    except Exception as e:
        logger.error(f"Failed to create plots directory {PLOTS_DIR}: {str(e)}")

# Lock for plotting
plot_lock = Lock()

# Global variables for tracking
daily_pnl = 0.0
last_pnl_reset_date = None

class EngulfingStrategy:
    def __init__(self):
        self.last_trade_time = None
        self.last_candle_time = None
        self.open_positions = 0

    def ensure_mt5_connection(self):
        """Ensure MT5 is initialized."""
        # Removed mt5.initialize() call. Assume MT5 is already initialized by main bot.
        if not mt5.account_info():
            logger.error("MT5 not connected.")
            return False
        return True

    def is_in_kill_zone(self, now):
        """Check if current time is in kill zone (UTC)."""
        hour = now.hour
        for start, end in KILL_ZONES:
            if start <= hour < end:
                return True
        return False

    def get_data(self, symbol, timeframe, n=50, max_retries=3):
        """Fetch data from MT5 with retry logic."""
        for attempt in range(max_retries):
            utc_to = datetime.utcnow()
            logger.info(f"Fetching data for {symbol}, timeframe {timeframe}, n={n}, attempt {attempt + 1}")
            rates = mt5.copy_rates_from(symbol, timeframe, utc_to, n)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                logger.info(f"Data fetched successfully, length: {len(df)}")
                return df
            logger.warning(f"No data received for {symbol} timeframe {timeframe}, attempt {attempt + 1}")
            if attempt < max_retries - 1:
                sleep(1)
        logger.error(f"Failed to fetch data for {symbol} after {max_retries} attempts")
        return pd.DataFrame()

    def calculate_atr(self, df, period=ATR_PERIOD):
        """Calculate Average True Range (ATR)."""
        df['tr'] = np.maximum.reduce([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ])
        atr = df['tr'].rolling(window=period).mean().iloc[-1]
        logger.debug(f"ATR calculated: {atr}")
        return atr

    def update_daily_pnl(self):
        """Update daily P/L for loss limit tracking."""
        global daily_pnl, last_pnl_reset_date
        now_utc = datetime.utcnow()
        today = now_utc.date()
        if last_pnl_reset_date != today:
            daily_pnl = 0.0
            last_pnl_reset_date = today
            logger.info("Daily P/L reset for new trading day.")

        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to get account info for P/L tracking.")
            return

        positions = safe_positions_get()
        total_pnl = 0.0
        if positions:
            for pos in positions:
                total_pnl += pos.profit
        daily_pnl = total_pnl
        logger.debug(f"Updated daily P/L: {daily_pnl}")

    def check_daily_loss_limit(self, equity):
        """Check if daily loss limit has been reached."""
        self.update_daily_pnl()
        max_loss = equity * (MAX_DAILY_LOSS_PERCENT / 100)
        if daily_pnl <= -max_loss:
            logger.warning(f"Daily loss limit reached ({daily_pnl:.2f} <= {-max_loss:.2f}). Stopping trading for the day.")
            return False
        return True

    def check_max_open_trades(self):
        """Check if max open positions limit is reached."""
        positions = safe_positions_get()
        open_trades = len(positions) if positions else 0
        if open_trades >= MAX_OPEN_POSITIONS:
            logger.info(f"Max open trades ({MAX_OPEN_POSITIONS}) reached. Skipping new trade.")
            return False
        return True

    def check_correlation_filter(self):
        """Avoid trades if correlated symbols are open."""
        positions = safe_positions_get()
        if not positions:
            return True
        open_symbols = {pos.symbol for pos in positions}
        correlated_open = open_symbols.intersection(CORRELATED_SYMBOLS)
        if correlated_open:
            logger.info(f"Correlated symbols open: {correlated_open}. Skipping trade to avoid correlation risk.")
            return False
        return True

    def get_account_equity(self):
        """Get current account equity."""
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to get account info")
            return None
        logger.debug(f"Account equity: {account_info.equity}")
        return account_info.equity

    def calculate_position_size(self, equity, risk_percent, stoploss_points, atr, point_value=0.1):
        """Calculate position size based on risk and total exposure."""
        risk_amount = equity * (risk_percent / 100)
        position_size = risk_amount / (stoploss_points * point_value)
        position_size = min(max(position_size, MIN_POSITION_SIZE), MAX_POSITION_SIZE)

        positions = safe_positions_get()
        total_risk = risk_amount
        if positions:
            for pos in positions:
                sl_points = abs(pos.price_open - pos.sl) / mt5.symbol_info(pos.symbol).point
                pos_risk = pos.volume * sl_points * point_value
                total_risk += pos_risk

        max_risk = equity * (MAX_TOTAL_RISK_PERCENT / 100)
        if total_risk > max_risk:
            logger.warning(f"Total risk ({total_risk:.2f}) exceeds max allowed risk ({max_risk:.2f}). Reducing position size.")
            position_size *= (max_risk - (total_risk - risk_amount)) / risk_amount

        position_size = min(max(position_size, MIN_POSITION_SIZE), MAX_POSITION_SIZE)
        logger.debug(f"Position size calculated: risk_amount={risk_amount}, stoploss_points={stoploss_points}, position_size={position_size}")
        return round(position_size, 2)

    def modify_position(self, position, new_sl=None, new_tp=None):
        """Modify SL/TP of an open position."""
        if not self.ensure_mt5_connection():
            return False
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": position.ticket,
            "sl": new_sl if new_sl is not None else position.sl,
            "tp": new_tp if new_tp is not None else position.tp,
        }
        result = self.order_send_with_retry(request)
        if result:
            logger.info(f"Position modified: ticket={position.ticket}, new_sl={new_sl}, new_tp={new_tp}")
            return True
        return False

    def manage_open_positions(self, df):
        """Manage open positions: partial profits, trailing SL."""
        positions = safe_positions_get(symbol=SYMBOL)
        if not positions:
            return
        last_price = df['close'].iloc[-1]
        atr = self.calculate_atr(df)

        for pos in positions:
            risk = abs(pos.price_open - pos.sl)
            reward = abs(last_price - pos.price_open)
            rr = reward / risk if risk != 0 else 0

            # Partial profit booking
            for level, percentage in zip(PARTIAL_PROFIT_LEVELS, PARTIAL_PROFIT_PERCENTAGES):
                if rr >= level:
                    close_volume = pos.volume * percentage
                    if close_volume > 0:
                        request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": SYMBOL,
                            "volume": close_volume,
                            "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                            "position": pos.ticket,
                            "price": last_price,
                            "deviation": 10,
                            "magic": pos.magic,
                            "comment": f"Partial close at {level}R"
                        }
                        result = self.order_send_with_retry(request)
                        if result:
                            logger.info(f"Partial profit booked at {level}R: ticket={pos.ticket}, volume={close_volume}")
                            if level == PARTIAL_PROFIT_LEVELS[0]:
                                new_sl = pos.price_open
                                self.modify_position(pos, new_sl=new_sl)

            # Trailing SL
            if rr >= TRAILING_SL_START_R:
                trailing_distance = atr * TRAILING_SL_DISTANCE_ATR * mt5.symbol_info(SYMBOL).point
                if pos.type == mt5.POSITION_TYPE_BUY:
                    new_sl = last_price - trailing_distance
                    if new_sl > pos.sl:
                        self.modify_position(pos, new_sl=new_sl)
                else:  # Sell position
                    new_sl = last_price + trailing_distance
                    if new_sl < pos.sl:
                        self.modify_position(pos, new_sl=new_sl)

    def detect_context(self, df, timeframe_name="M15"):
        """Detect HTF context: OB, Liquidity Pool, or FVG."""
        if len(df) < OB_LOOKBACK + 2:
            logger.info(f"{timeframe_name}: Not enough data for context detection (len={len(df)}).")
            return False, None, None
        # Liquidity Pool (equal highs/lows or consolidation)
        recent = df.iloc[-5:]
        price_range = (recent['high'].max() - recent['low'].min()) / recent['close'].mean()
        if price_range <= CONSOLIDATION_RANGE:
            logger.info(f"{timeframe_name}: Liquidity pool detected (range: {price_range*100:.2f}%).")
            return True, None, None
        # Order Block (last bearish/bullish candle before recent move)
        last_candle = df.iloc[-2]
        if last_candle['close'] > last_candle['open']:  # Bullish move
            for i in range(-2, -OB_LOOKBACK-1, -1):
                candle = df.iloc[i]
                if candle['close'] < candle['open']:
                    ob_low, ob_high = candle['low'], candle['high']
                    logger.info(f"{timeframe_name}: Bullish OB found at {ob_low} - {ob_high}.")
                    return True, ob_low, ob_high
        else:  # Bearish move
            for i in range(-2, -OB_LOOKBACK-1, -1):
                candle = df.iloc[i]
                if candle['close'] > candle['open']:
                    ob_low, ob_high = candle['low'], candle['high']
                    logger.info(f"{timeframe_name}: Bearish OB found at {ob_low} - {ob_high}.")
                    return True, ob_low, ob_high
        # FVG (gap between candles)
        for i in range(-3, -len(df), -1):
            curr = df.iloc[i]
            prev = df.iloc[i+1]
            if curr['low'] > prev['high'] + prev['close'] * FVG_THRESHOLD:
                fvg_low, fvg_high = prev['high'], curr['low']
                logger.info(f"{timeframe_name}: Bullish FVG found at {fvg_low} - {fvg_high}.")
                return True, fvg_low, fvg_high
            elif curr['high'] < prev['low'] - prev['close'] * FVG_THRESHOLD:
                fvg_low, fvg_high = curr['high'], prev['low']
                logger.info(f"{timeframe_name}: Bearish FVG found at {fvg_low} - {fvg_high}.")
                return True, fvg_low, fvg_high
        logger.info(f"{timeframe_name}: No context (OB/Liquidity/FVG) detected.")
        return False, None, None

    def plot_trade(self, df, entry_idx, sl_price=None, tp_price=None, context_low=None, context_high=None):
        """Plot trade setup with context zones."""
        with plot_lock:
            try:
                plot_files = sorted(glob.glob(os.path.join(PLOTS_DIR, "engulfing_trade_setup_*.png")), key=os.path.getmtime)
                if len(plot_files) > MAX_PLOT_FILES:
                    for old_file in plot_files[:-MAX_PLOT_FILES]:
                        try:
                            os.remove(old_file)
                            logger.info(f"Deleted old plot file: {old_file}")
                        except Exception as e:
                            logger.warning(f"Failed to delete old plot file {old_file}: {str(e)}")

                if df.empty:
                    logger.warning("Cannot plot trade: DataFrame is empty")
                    return

                times = df['time']
                fig = plt.figure(figsize=(12, 6))
                ax = fig.add_subplot(111)
                ax.set_title(f"{SYMBOL} - Trade Setup Visualization (5M)")
                ax.grid(True)

                o = df['open'].values
                h = df['high'].values
                l = df['low'].values
                c = df['close'].values
                times = mdates.date2num(times.to_list())
                
                for i in range(len(df)):
                    color = 'green' if c[i] >= o[i] else 'red'
                    ax.plot([times[i], times[i]], [l[i], h[i]], color=color)
                    ax.plot([times[i], times[i]], [o[i], c[i]], linewidth=5, color=color)
                
                ax.scatter(times[entry_idx], c[entry_idx], marker='^', color='blue', s=150, label='Entry Candle')

                if sl_price:
                    ax.axhline(sl_price, color='red', linestyle='--', label='Stop Loss')
                if tp_price:
                    ax.axhline(tp_price, color='green', linestyle='--', label='Take Profit')
                if context_low and context_high:
                    ax.axhspan(context_low, context_high, color='blue', alpha=0.1, label='Context Zone (OB/FVG)')

                kz_start = df['time'].iloc[0].replace(hour=KILL_ZONES[0][0], minute=0, second=0)
                kz_end = df['time'].iloc[0].replace(hour=KILL_ZONES[0][1], minute=0, second=0)
                ax.axvspan(mdates.date2num(kz_start), mdates.date2num(kz_end), color='yellow', alpha=0.2, label='Kill Zone')

                ax.legend()
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.xticks(rotation=45)
                plt.tight_layout()

                fig.canvas.draw()
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                filename = os.path.join(PLOTS_DIR, f'engulfing_trade_setup_{timestamp}.png')
                fig.savefig(filename, bbox_inches='tight')
                logger.info(f"Trade plot saved as {filename}")
                sleep(0.1)
                plt.close(fig)
            except Exception as e:
                logger.error(f"Failed to plot trade: {str(e)}")
                plt.close('all')

    def check_confirmation_candle(self, df):
        """Check for Bullish/Bearish Engulfing pattern with wick rejection."""
        if len(df) < 3:
            return None
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        # Bullish Engulfing
        if (curr['close'] > curr['open'] and 
            curr['open'] <= prev['close'] and 
            curr['close'] >= prev['open'] and 
            prev['close'] < prev['open']):  # Previous candle bearish
            wick_rejection = curr['low'] <= prev['low'] * 1.001  # Wick tests lower level
            if wick_rejection:
                logger.info("Bullish Engulfing with wick rejection detected.")
                return 'buy'
        # Bearish Engulfing
        elif (curr['close'] < curr['open'] and 
              curr['open'] >= prev['close'] and 
              curr['close'] <= prev['open'] and 
              prev['close'] > prev['open']):  # Previous candle bullish
            wick_rejection = curr['high'] >= prev['high'] * 0.999  # Wick tests higher level
            if wick_rejection:
                logger.info("Bearish Engulfing with wick rejection detected.")
                return 'sell'
        logger.info("No engulfing pattern detected.")
        return None

    def check_higher_timeframe_confirmation(self, trade_direction):
        """Check if price is near a 15M context zone (OB, Liquidity, FVG)."""
        df_h1 = self.get_data(SYMBOL, HIGHER_TIMEFRAME, n=20)
        if df_h1.empty:
            logger.info("M15: Failed to fetch HTF data.")
            return False, None, None
        context_detected, context_low, context_high = self.detect_context(df_h1)
        if not context_detected:
            logger.info("M15: No valid context (OB/Liquidity/FVG) found.")
            return False, None, None
        current_price = df_h1['close'].iloc[-1]
        if context_low and context_high and context_low <= current_price <= context_high:
            logger.info(f"M15: Price in context zone ({context_low} - {context_high}).")
            return True, context_low, context_high
        logger.info(f"M15: Price {current_price} not in context zone.")
        return False, None, None

    def order_send_with_retry(self, request, max_retries=5):
        """Send order with retry logic."""
        retries = 0
        while retries < max_retries:
            logger.info(f"Attempting to send order (attempt {retries + 1})...")
            result = mt5.order_send(request)
            logger.info(f"Order send result: {result}")
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Order placed successfully: {result}")
                return result
            elif result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                logger.warning("Requote received, adjusting price...")
                request["price"] = result.price
                retries += 1
                sleep(1)
                continue
            elif result.retcode == mt5.TRADE_RETCODE_REQUEST_REJECTED:
                logger.warning("Request rejected, retrying...")
                retries += 1
                sleep(0.5)
                continue
            elif result.retcode == mt5.TRADE_RETCODE_PRICE_OFF:
                logger.warning("Price off, adjusting price...")
                request["price"] += 0.01 if request["type"] == mt5.ORDER_TYPE_BUY else -0.01
                retries += 1
                sleep(1)
                continue
            else:
                logger.error(f"Order failed with retcode {result.retcode}")
                retries += 1
                sleep(1)
        logger.error("Order failed after maximum retries")
        return None

    def place_order(self, symbol, lot, price, sl, tp, trade_direction, max_retries=5):
        """Place a trade order with retry logic."""
        if not self.ensure_mt5_connection():
            logger.error("Cannot place order: MT5 not initialized")
            return False
        # --- Universal validation ---
        rounded_lot = round_lot(symbol, lot)
        if not is_market_open(symbol):
            logger.error(f"[SKIP] Market is closed or trading disabled for {symbol}. Skipping order.")
            return False
        if not validate_lot_size(symbol, rounded_lot):
            logger.error(f"[SKIP] Invalid lot size {rounded_lot} for {symbol}. Skipping order.")
            return False
        order_type = mt5.ORDER_TYPE_BUY if trade_direction == 'buy' else mt5.ORDER_TYPE_SELL
        symbol_info = mt5.symbol_info(symbol)
        filling_type = symbol_info.filling_mode if symbol_info and symbol_info.filling_mode else mt5.ORDER_FILLING_IOC
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": rounded_lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "magic": 234000,
            "comment": "Engulfing Trade",
            "type_filling": filling_type,
        }
        result = self.order_send_with_retry(request, max_retries)
        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE

    def run_engulfing_candle(self, data=None):
        """Run the engulfing candle strategy for intraday trading with advanced risk management."""
        logger.info("Running updated ENGULFING strategy with advanced risk management")
        
        if not self.ensure_mt5_connection():
            logger.error("MT5 not initialized, stopping strategy")
            return False

        equity = self.get_account_equity()
        if equity is None:
            logger.error("Cannot run strategy without equity.")
            return True

        # Risk management checks
        if not self.check_daily_loss_limit(equity):
            return True
        if not self.check_max_open_trades():
            return True
        if not self.check_correlation_filter():
            return True

        now = datetime.utcnow()

        if not self.is_in_kill_zone(now):
            logger.info("Outside kill zone.")
            return True

        if self.last_trade_time and (now - self.last_trade_time).total_seconds() < COOLDOWN_PERIOD:
            logger.info(f"In cooldown period, {int(COOLDOWN_PERIOD - (now - self.last_trade_time).total_seconds())} seconds left.")
            return True

        df = data if data is not None else self.get_data(SYMBOL, TIMEFRAME)
        if df.empty:
            logger.warning("No data available for engulfing candle strategy.")
            return True

        atr = self.calculate_atr(df)
        if atr is None or atr == 0:
            logger.warning("ATR calculation failed. Skipping trade.")
            return True

        current_candle_time = df['time'].iloc[-1]
        if self.last_candle_time == current_candle_time:
            logger.info("Trade already checked for this candle.")
            return True

        trade_direction = self.check_confirmation_candle(df)
        if trade_direction:
            context_valid, context_low, context_high = self.check_higher_timeframe_confirmation(trade_direction)
            if context_valid:
                tick = mt5.symbol_info_tick(SYMBOL)
                if not tick:
                    logger.error("Failed to get tick data.")
                    return True
                entry_price = tick.ask if trade_direction == 'buy' else tick.bid
                engulfing_candle = df.iloc[-1]

                # Adjust SL based on ATR
                sl_distance = atr * mt5.symbol_info(SYMBOL).point * ATR_MULTIPLIER_MIN
                if sl_distance < atr * mt5.symbol_info(SYMBOL).point * ATR_MULTIPLIER_MIN:
                    sl_distance = atr * mt5.symbol_info(SYMBOL).point * ATR_MULTIPLIER_MIN
                elif sl_distance > atr * mt5.symbol_info(SYMBOL).point * ATR_MULTIPLIER_MAX:
                    sl_distance = atr * mt5.symbol_info(SYMBOL).point * ATR_MULTIPLIER_MAX

                if trade_direction == 'buy':
                    sl = engulfing_candle['low'] - sl_distance
                    risk = entry_price - sl
                    tp = entry_price + max(risk * PARTIAL_PROFIT_LEVELS[-1], 2 * risk)  # Use partial profit level or 1:2 RR
                else:
                    sl = engulfing_candle['high'] + sl_distance
                    risk = sl - entry_price
                    tp = entry_price - max(risk * PARTIAL_PROFIT_LEVELS[-1], 2 * risk)

                stoploss_points = risk / mt5.symbol_info(SYMBOL).point
                lot = self.calculate_position_size(equity, RISK_PERCENT_PER_TRADE, stoploss_points, atr)
                logger.info(f"Trying to enter {trade_direction} trade: Entry={entry_price}, SL={sl}, TP={tp}, Lot={lot}")
                if self.place_order(SYMBOL, lot, entry_price, sl, tp, trade_direction):
                    self.last_trade_time = now
                    self.last_candle_time = current_candle_time
                    self.open_positions += 1
                    self.plot_trade(df, entry_idx=len(df)-1, sl_price=sl, tp_price=tp, context_low=context_low, context_high=context_high)
                    logger.info("Trade executed successfully.")
                else:
                    logger.error("Order placement failed.")
            else:
                logger.info("No HTF context confirmation.")
        else:
            logger.info("No engulfing pattern detected.")

        # Manage open positions (partial profits, trailing SL)
        self.manage_open_positions(df)
        
        logger.info(f"Logger handlers at end of run_engulfing_candle: {[handler.__class__.__name__ for handler in logger.handlers]}")
        return True

# Top-level function
def run_engulfing_candle(data=None):
    """Run the engulfing candle strategy."""
    strategy = EngulfingStrategy()
    return strategy.run_engulfing_candle(data)