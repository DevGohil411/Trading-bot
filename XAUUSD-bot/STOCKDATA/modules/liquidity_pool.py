import MetaTrader5 as mt5
import pandas as pd
import datetime
import pytz
import logging
import requests
import sys
import os
import numpy as np
from time import sleep
import time
from utils.trade_logger import log_trade_execution

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ------------- CONFIGURATION -----------------
SYMBOL = "XAUUSD"
CORRELATED_SYMBOLS = ["XAUEUR", "XAGUSD"]
TIMEFRAME = mt5.TIMEFRAME_M15
HIGHER_TF = mt5.TIMEFRAME_H1
DAILY_TF = mt5.TIMEFRAME_D1
BARS = 500
RISK_PERCENT_PER_TRADE = 1.0
MAX_TOTAL_RISK_PERCENT = 5.0
MAX_DAILY_LOSS_PERCENT = 3.0
MAX_OPEN_TRADES = 3
MIN_POSITION_SIZE = 0.01
MAX_POSITION_SIZE = 1.0
ATR_PERIOD = 14
ATR_MULTIPLIER_MIN = 1.5
ATR_MULTIPLIER_MAX = 3.0
VOLUME_SPIKE_THRESHOLD = 1.5
SESSION_CLOSE_BUFFER_MINUTES = 10
PARTIAL_PROFIT_LEVELS = [1.0, 1.5]
PARTIAL_PROFIT_PERCENTAGES = [0.5, 0.3]
TRAILING_SL_START_R = 1.0
TRAILING_SL_DISTANCE_ATR = 0.5

TELEGRAM_TOKEN = 'your_telegram_bot_token'
TELEGRAM_CHAT_ID = 'your_chat_id'

LONDON_OPEN = datetime.time(7, 0)
LONDON_CLOSE = datetime.time(16, 30)
NY_OPEN = datetime.time(12, 30)
NY_CLOSE = datetime.time(21, 0)

MAX_ORDER_RETRIES = 3
RETRY_DELAY_SECONDS = 5
EQUAL_LEVEL_TOLERANCE = 0.001
LIQUIDITY_GRAB_THRESHOLD = 0.002

daily_pnl = 0.0
last_pnl_reset_date = None

# Set the current date and time (01:09 PM IST, June 06, 2025)
CURRENT_TIME_IST = datetime.datetime(2025, 6, 6, 13, 9, tzinfo=pytz.timezone('Asia/Kolkata'))
CURRENT_TIME_UTC = CURRENT_TIME_IST.astimezone(pytz.UTC)

# ------------- LOGGING SETUP -----------------
logger = logging.getLogger('trade_bot.liquidity')
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler("trade_log.log")
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# ------------- TELEGRAM ALERT FUNCTION -------------
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
        logger.debug(f"Telegram message sent: {message}")
    except Exception as e:
        logger.error(f"Telegram send error: {e}")

# ------------- DATA FETCH ---------------------
def get_data(symbol, timeframe, bars):
    utc_from = CURRENT_TIME_UTC - datetime.timedelta(days=10)
    rates = mt5.copy_rates_from(symbol, timeframe, utc_from, bars)
    if rates is None or len(rates) == 0:
        logger.warning(f"No data received for {symbol} timeframe {timeframe}")
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True, drop=True)
    df.index = df.index.tz_localize('UTC')
    for col in ['open', 'high', 'low', 'close', 'tick_volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[['open', 'high', 'low', 'close']].isnull().any().any():
        logger.error(f"Non-numeric data found in OHLC columns for {symbol}. Aborting.")
        return pd.DataFrame()
    logger.debug(f"Data fetched: shape={df.shape}, columns={list(df.columns)}")
    return df

# ------------- PREVIOUS DAY HIGH/LOW ------------------
def get_previous_day_levels(symbol):
    df_daily = get_data(symbol, DAILY_TF, 2)
    if len(df_daily) < 2:
        logger.warning("Not enough daily data for PDH/PDL.")
        return None, None
    previous_day = df_daily.iloc[-2]
    pdh = float(previous_day['high'])
    pdl = float(previous_day['low'])
    logger.info(f"Previous Day High: {pdh}, Previous Day Low: {pdl}")
    return pdh, pdl

# ------------- MARKET SESSION CHECK --------------
def is_market_open():
    now_utc = CURRENT_TIME_UTC
    now_time = now_utc.time()
    in_london = LONDON_OPEN <= now_time <= LONDON_CLOSE
    in_ny = NY_OPEN <= now_time <= NY_CLOSE
    is_open = in_london or in_ny

    logger.info(f"Checking market session at {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')} (UTC)")

    if in_london:
        close_time = datetime.datetime.combine(now_utc.date(), LONDON_CLOSE, tzinfo=pytz.UTC)
        time_to_close = (close_time - now_utc).total_seconds() / 60
        if time_to_close <= SESSION_CLOSE_BUFFER_MINUTES:
            logger.info(f"Near London session close ({time_to_close:.1f} minutes remaining). Skipping trade.")
            return False
    if in_ny:
        close_time = datetime.datetime.combine(now_utc.date(), NY_CLOSE, tzinfo=pytz.UTC)
        time_to_close = (close_time - now_utc).total_seconds() / 60
        if time_to_close <= SESSION_CLOSE_BUFFER_MINUTES:
            logger.info(f"Near NY session close ({time_to_close:.1f} minutes remaining). Skipping trade.")
            return False

    logger.debug(f"Market session check: in_london={in_london}, in_ny={in_ny}, is_open={is_open}")
    return is_open

def get_session_high_low(df, session_start, session_end):
    df = df.copy()
    df['time_utc'] = df.index.tz_convert('UTC')
    mask = (df['time_utc'].dt.time >= session_start) & (df['time_utc'].dt.time <= session_end)
    session_data = df[mask]
    if session_data.empty:
        return None, None
    session_high = float(session_data['high'].max())
    session_low = float(session_data['low'].min())
    logger.info(f"Session High: {session_high}, Session Low: {session_low}")
    return session_high, session_low

# ------------- NEWS FILTER PLACEHOLDER --------------
def is_high_impact_news():
    return False

# ------------- MT5 RECONNECT ------------------
def ensure_mt5_connection():
    account_info = mt5.account_info()
    if account_info is None:
        logger.warning("MT5 connection lost. Attempting reconnect...")
        sleep(10)
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("MT5 reconnect failed.")
            return False
    logger.debug("MT5 connection ensured.")
    return True

# ------------- RISK MANAGEMENT FUNCTIONS -------------
def update_daily_pnl():
    global daily_pnl, last_pnl_reset_date
    now_utc = CURRENT_TIME_UTC
    today = now_utc.date()
    if last_pnl_reset_date != today:
        daily_pnl = 0.0
        last_pnl_reset_date = today
        logger.info("Daily P/L reset for new trading day.")

    account_info = mt5.account_info()
    if account_info is None:
        logger.error("Failed to get account info for P/L tracking.")
        return

    total_pnl = 0.0
    positions = safe_positions_get()
    if positions:
        for pos in positions:
            total_pnl += pos.profit
    daily_pnl = total_pnl
    logger.debug(f"Updated daily P/L: {daily_pnl}")

def check_daily_loss_limit(equity):
    update_daily_pnl()
    max_loss = equity * (MAX_DAILY_LOSS_PERCENT / 100)
    if daily_pnl <= -max_loss:
        logger.warning(f"Daily loss limit reached ({daily_pnl:.2f} <= {-max_loss:.2f}). Stopping trading for the day.")
        return False
    return True

def check_max_open_trades():
    positions = safe_positions_get()
    open_trades = len(positions) if positions else 0
    if open_trades >= MAX_OPEN_TRADES:
        logger.info(f"Max open trades ({MAX_OPEN_TRADES}) reached. Skipping new trade.")
        return False
    return True

def check_correlation_filter():
    positions = safe_positions_get()
    if not positions:
        return True
    open_symbols = {pos.symbol for pos in positions}
    correlated_open = open_symbols.intersection(CORRELATED_SYMBOLS)
    if correlated_open:
        logger.info(f"Correlated symbols open: {correlated_open}. Skipping trade to avoid correlation risk.")
        return False
    return True

def calculate_atr(df, period=ATR_PERIOD):
    df['tr'] = np.maximum.reduce([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ])
    atr = float(df['tr'].rolling(window=period).mean().iloc[-1])
    logger.debug(f"ATR calculated: {atr}")
    return atr

def get_account_equity():
    account_info = mt5.account_info()
    if account_info is None:
        logger.error("Failed to get account info")
        return None
    logger.debug(f"Account equity: {account_info.equity}")
    return account_info.equity

def calculate_position_size(equity, risk_percent, stoploss_points, atr, point_value=0.01):
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

def modify_position(position, new_sl=None, new_tp=None):
    if not ensure_mt5_connection():
        return False
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": position.ticket,
        "sl": new_sl if new_sl is not None else position.sl,
        "tp": new_tp if new_tp is not None else position.tp,
    }
    result = order_send_with_retry(request)
    if result:
        logger.info(f"Position modified: ticket={position.ticket}, new_sl={new_sl}, new_tp={new_tp}")
        return True
    return False

def manage_open_positions(df):
    positions = safe_positions_get(symbol=SYMBOL)
    if not positions:
        return
    last_price = float(df['close'].iloc[-1])
    atr = calculate_atr(df)

    for pos in positions:
        risk = abs(pos.price_open - pos.sl)
        reward = abs(last_price - pos.price_open)
        rr = reward / risk if risk != 0 else 0

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
                    result = order_send_with_retry(request)
                    if result:
                        logger.info(f"Partial profit booked at {level}R: ticket={pos.ticket}, volume={close_volume}")
                        if level == PARTIAL_PROFIT_LEVELS[0]:
                            new_sl = pos.price_open
                            modify_position(pos, new_sl=new_sl)

        if rr >= TRAILING_SL_START_R:
            trailing_distance = atr * TRAILING_SL_DISTANCE_ATR * mt5.symbol_info(SYMBOL).point
            if pos.type == mt5.POSITION_TYPE_BUY:
                new_sl = last_price - trailing_distance
                if new_sl > pos.sl:
                    modify_position(pos, new_sl=new_sl)
            else:
                new_sl = last_price + trailing_distance
                if new_sl < pos.sl:
                    modify_position(pos, new_sl=new_sl)

# ------------- ORDER PLACEMENT WITH RETRY --------------
def order_send_with_retry(request):
    for attempt in range(1, MAX_ORDER_RETRIES + 1):
        if not ensure_mt5_connection():
            logger.error("MT5 not connected. Order send aborted.")
            return None
        start_time = time.time()
        result = mt5.order_send(request)
        end_time = time.time()
        latency = end_time - start_time
        executed_price = getattr(result, 'price', request['price']) if result else None
        slippage = abs(request['price'] - executed_price) if executed_price else None
        log_trade_execution(
            strategy_name="liquidity_pool",
            symbol=request['symbol'],
            order_type=request['type'],
            requested_price=request['price'],
            executed_price=executed_price,
            sl=request['sl'],
            tp=request['tp'],
            lot_size=request['volume'],
            result=result,
            latency=latency,
            slippage=slippage,
            comment=request.get('comment', '')
        )
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Order placed successfully: {result}")
            return result
        else:
            logger.warning(f"Order send failed (attempt {attempt}): {result.retcode}")
            sleep(RETRY_DELAY_SECONDS)
    logger.error("Max order retries reached. Order placement failed.")
    return None

def place_buy_order(price, sl, tp, volume):
    # --- Universal validation ---
    rounded_volume = round_lot(SYMBOL, volume)
    if not is_market_open(SYMBOL):
        logger.error(f"[SKIP] Market is closed or trading disabled for {SYMBOL}. Skipping buy order.")
        return
    if not validate_lot_size(SYMBOL, rounded_volume):
        logger.error(f"[SKIP] Invalid lot size {rounded_volume} for {SYMBOL}. Skipping buy order.")
        return
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": rounded_volume,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 234000,
        "comment": "Auto buy order"
    }
    result = order_send_with_retry(request)
    if result:
        send_telegram_message(f"Buy order placed for {SYMBOL} at {price} with SL {sl} and TP {tp}")

def place_sell_order(price, sl, tp, volume):
    # --- Universal validation ---
    rounded_volume = round_lot(SYMBOL, volume)
    if not is_market_open(SYMBOL):
        logger.error(f"[SKIP] Market is closed or trading disabled for {SYMBOL}. Skipping sell order.")
        return
    if not validate_lot_size(SYMBOL, rounded_volume):
        logger.error(f"[SKIP] Invalid lot size {rounded_volume} for {SYMBOL}. Skipping sell order.")
        return
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": rounded_volume,
        "type": mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 234000,
        "comment": "Auto sell order"
    }
    result = order_send_with_retry(request)
    if result:
        send_telegram_message(f"Sell order placed for {SYMBOL} at {price} with SL {sl} and TP {tp}")

# ----------- SIGNAL DETECTION FUNCTIONS -------------
def detect_swing_highs_lows(df, lookback=5):
    swing_highs = []
    swing_lows = []
    for i in range(lookback, len(df) - lookback):
        if all(df['high'].iloc[i] >= df['high'].iloc[i-lookback:i]) and all(df['high'].iloc[i] >= df['high'].iloc[i+1:i+lookback+1]):
            swing_highs.append((df.index[i], float(df['high'].iloc[i])))
        if all(df['low'].iloc[i] <= df['low'].iloc[i-lookback:i]) and all(df['low'].iloc[i] <= df['low'].iloc[i+1:i+lookback+1]):
            swing_lows.append((df.index[i], float(df['low'].iloc[i])))
    logger.debug(f"Swing highs detected: {len(swing_highs)}, Swing lows detected: {len(swing_lows)}")
    return swing_highs, swing_lows

def detect_equal_highs_lows(df):
    tolerance = EQUAL_LEVEL_TOLERANCE
    highs = []
    lows = []
    for i in range(1, len(df)):
        if i < len(df) - 1:
            high_diff = abs(df['high'].iloc[i] - df['high'].iloc[i-1]) / df['high'].iloc[i]
            if high_diff < tolerance:
                highs.append((df.index[i], float(df['high'].iloc[i])))
            low_diff = abs(df['low'].iloc[i] - df['low'].iloc[i-1]) / df['low'].iloc[i]
            if low_diff < tolerance:
                lows.append((df.index[i], float(df['low'].iloc[i])))
    logger.debug(f"Equal highs detected: {len(highs)}, Equal lows detected: {len(lows)}")
    return highs, lows

def detect_liquidity_grab(df, levels, direction='high'):
    last_candle = df.iloc[-1]
    last_high, last_low, last_close = float(last_candle['high']), float(last_candle['low']), float(last_candle['close'])
    for _, level in levels:
        level = float(level)
        if direction == 'high' and last_high > level * (1 + LIQUIDITY_GRAB_THRESHOLD) and last_close < level:
            return 'Liquidity Grab High', level
        if direction == 'low' and last_low < level * (1 - LIQUIDITY_GRAB_THRESHOLD) and last_close > level:
            return 'Liquidity Grab Low', level
    return None, None

def detect_choch(df):
    if len(df) < 5:
        return None
    highs = df['high'].iloc[-5:].astype(float)
    lows = df['low'].iloc[-5:].astype(float)
    if highs.iloc[-1] > highs.iloc[-2] and lows.iloc[-1] > lows.iloc[-2]:
        return 'Uptrend Shift (CHOCH)'
    if highs.iloc[-1] < highs.iloc[-2] and lows.iloc[-1] < lows.iloc[-2]:
        return 'Downtrend Shift (CHOCH)'
    return None

def detect_bos(df):
    if len(df) < 3:
        return None
    highs = df['high'].iloc[-3:].astype(float)
    lows = df['low'].iloc[-3:].astype(float)
    if highs.iloc[-1] > highs.iloc[-2] > highs.iloc[-3]:
        return 'Bullish BOS'
    if lows.iloc[-1] < lows.iloc[-2] < lows.iloc[-3]:
        return 'Bearish BOS'
    return None

def detect_order_blocks(df):
    order_blocks = []
    for i in range(1, len(df)):
        if df['close'].iloc[i] < df['open'].iloc[i] and df['close'].iloc[i-1] > df['open'].iloc[i-1]:
            ob_data = {
                'type': 'bearish_ob',
                'high': float(df['high'].iloc[i-1]),
                'low': float(df['low'].iloc[i-1]),
                'timestamp': df.index[i-1]
            }
            order_blocks.append(('bearish_ob', ob_data))
        elif df['close'].iloc[i] > df['open'].iloc[i] and df['close'].iloc[i-1] < df['open'].iloc[i-1]:
            ob_data = {
                'type': 'bullish_ob',
                'high': float(df['high'].iloc[i-1]),
                'low': float(df['low'].iloc[i-1]),
                'timestamp': df.index[i-1]
            }
            order_blocks.append(('bullish_ob', ob_data))
    logger.debug(f"Order blocks detected: {len(order_blocks)}")
    return order_blocks

def detect_fair_value_gap(df):
    fvgs = []
    for i in range(2, len(df)):
        if df['low'].iloc[i] > df['high'].iloc[i-2]:
            fvg_data = {
                'type': 'bullish_fvg',
                'high': float(df['low'].iloc[i]),
                'low': float(df['high'].iloc[i-2]),
                'timestamp': df.index[i]
            }
            fvgs.append(('bullish_fvg', fvg_data))
        elif df['high'].iloc[i] < df['low'].iloc[i-2]:
            fvg_data = {
                'type': 'bearish_fvg',
                'high': float(df['low'].iloc[i-2]),
                'low': float(df['high'].iloc[i]),
                'timestamp': df.index[i]
            }
            fvgs.append(('bearish_fvg', fvg_data))
    logger.debug(f"Fair value gaps detected: {len(fvgs)}")
    return fvgs

def volume_spike_filter(volume_series, threshold=VOLUME_SPIKE_THRESHOLD):
    if len(volume_series) < 20:
        logger.info("Not enough volume data for spike detection.")
        return False
    volume_series = volume_series.astype(float)
    avg_vol = volume_series.rolling(window=20).mean()
    spike = volume_series.iloc[-1] > threshold * avg_vol.iloc[-1]
    logger.info(f"Volume spike: {spike}, Last volume: {volume_series.iloc[-1]}, Avg: {avg_vol.iloc[-1]}")
    return spike

def check_overlap(level1, level2, tolerance=0.001):
    level1, level2 = float(level1), float(level2)
    return abs(level1 - level2) / level1 < tolerance

def find_internal_liquidity(df, direction, entry_price):
    swing_highs, swing_lows = detect_swing_highs_lows(df, lookback=3)
    fvgs = detect_fair_value_gap(df)
    
    entry_price = float(entry_price)
    if direction == 'buy':
        potential_targets = [(idx, lvl) for idx, lvl in swing_lows if lvl > entry_price]
        for fvg_type, fvg_data in fvgs:
            if fvg_type == 'bearish_fvg' and fvg_data['low'] > entry_price:
                potential_targets.append((fvg_data['timestamp'], float(fvg_data['low'])))
        if potential_targets:
            nearest_target = min(potential_targets, key=lambda x: x[1])[1]
            logger.info(f"Dynamic TP for buy: nearest internal liquidity at {nearest_target}")
            return nearest_target
    elif direction == 'sell':
        potential_targets = [(idx, lvl) for idx, lvl in swing_highs if lvl < entry_price]
        for fvg_type, fvg_data in fvgs:
            if fvg_type == 'bullish_fvg' and fvg_data['high'] < entry_price:
                potential_targets.append((fvg_data['timestamp'], float(fvg_data['high'])))
        if potential_targets:
            nearest_target = max(potential_targets, key=lambda x: x[1])[1]
            logger.info(f"Dynamic TP for sell: nearest internal liquidity at {nearest_target}")
            return nearest_target
    
    logger.info("No internal liquidity found for dynamic TP.")
    return None

# ------------- STRATEGY FUNCTION -------------
def run_liquidity_pool(data=None):
    logger.info("Starting run_liquidity_pool")
    if not ensure_mt5_connection():
        logger.error("MT5 connection not available. Strategy aborted.")
        return False

    equity = get_account_equity()
    if equity is None:
        logger.error("Cannot run strategy without equity.")
        return False

    if not check_daily_loss_limit(equity):
        return False
    if not check_max_open_trades():
        return False
    if not check_correlation_filter():
        return False

    if not is_market_open():
        logger.info("Market session closed or near close. No trades allowed now.")
        return False

    if is_high_impact_news():
        logger.info("High impact news event upcoming. No trades allowed now.")
        return False

    if data is None:
        df = get_data(SYMBOL, TIMEFRAME, BARS)
    else:
        df = data.copy()

    if df.empty:
        logger.warning("No data available for Liquidity Pool strategy.")
        return False

    if 'tick_volume' not in df:
        logger.error("tick_volume column missing in price_data.")
        return False

    df_htf = get_data(SYMBOL, HIGHER_TF, 200)
    if df_htf.empty:
        logger.warning("No HTF data available.")
        return False

    atr = calculate_atr(df)
    if pd.isna(atr) or atr == 0:
        logger.warning("ATR calculation failed or zero. Skipping trade.")
        return False
    logger.info(f"ATR check: ATR={atr}, Threshold=0 (volatility filter disabled)")

    pdh, pdl = get_previous_day_levels(SYMBOL)
    if pdh is None or pdl is None:
        logger.warning("Cannot proceed without PDH/PDL.")
        return False

    htf_swing_highs, htf_swing_lows = detect_swing_highs_lows(df_htf)
    ltf_swing_highs, ltf_swing_lows = detect_swing_highs_lows(df)
    equal_highs, equal_lows = detect_equal_highs_lows(df)

    now_utc = CURRENT_TIME_UTC
    today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    london_high, london_low = get_session_high_low(df[df.index >= today_start], LONDON_OPEN, LONDON_CLOSE)
    ny_high, ny_low = get_session_high_low(df[df.index >= today_start], NY_OPEN, NY_CLOSE)

    buy_side_levels = [(idx, level) for idx, level in equal_highs] + [(idx, level) for idx, level in htf_swing_highs] + [(idx, level) for idx, level in ltf_swing_highs]
    sell_side_levels = [(idx, level) for idx, level in equal_lows] + [(idx, level) for idx, level in htf_swing_lows] + [(idx, level) for idx, level in ltf_swing_lows]

    if london_high:
        buy_side_levels.append((None, london_high))
    if ny_high:
        buy_side_levels.append((None, ny_high))
    if london_low:
        sell_side_levels.append((None, london_low))
    if ny_low:
        sell_side_levels.append((None, ny_low))
    buy_side_levels.append((None, pdh))
    sell_side_levels.append((None, pdl))

    high_confluence_buy = []
    high_confluence_sell = []
    for i, (idx1, level1) in enumerate(buy_side_levels):
        for idx2, level2 in buy_side_levels[i+1:]:
            if check_overlap(level1, level2):
                high_confluence_buy.append(level1)
    for i, (idx1, level1) in enumerate(sell_side_levels):
        for idx2, level2 in sell_side_levels[i+1:]:
            if check_overlap(level1, level2):
                high_confluence_sell.append(level1)

    logger.info(f"High confluence buy zones: {high_confluence_buy}")
    logger.info(f"High confluence sell zones: {high_confluence_sell}")

    liquidity_grab_high, grab_level_high = detect_liquidity_grab(df, list(set([(None, lvl) for lvl in high_confluence_buy])), direction='high')
    liquidity_grab_low, grab_level_low = detect_liquidity_grab(df, list(set([(None, lvl) for lvl in high_confluence_sell])), direction='low')

    if not (liquidity_grab_high or liquidity_grab_low):
        logger.info("No liquidity grab detected.")
        return False

    if not volume_spike_filter(df['tick_volume']):
        logger.info("No volume spike during liquidity grab. Skipping trade.")
        return False

    choch = detect_choch(df)
    bos = detect_bos(df)
    order_blocks = detect_order_blocks(df)
    fvgs = detect_fair_value_gap(df)

    last_candle = df.iloc[-1]
    close_price = float(last_candle['close'])

    buy_signal = False
    sell_signal = False
    entry_price = None
    stop_loss = None
    take_profit = None
    volume = None

    sl_distance = atr * mt5.symbol_info(SYMBOL).point * ATR_MULTIPLIER_MIN
    if sl_distance < atr * mt5.symbol_info(SYMBOL).point * ATR_MULTIPLIER_MIN:
        sl_distance = atr * mt5.symbol_info(SYMBOL).point * ATR_MULTIPLIER_MIN
    elif sl_distance > atr * mt5.symbol_info(SYMBOL).point * ATR_MULTIPLIER_MAX:
        sl_distance = atr * mt5.symbol_info(SYMBOL).point * ATR_MULTIPLIER_MAX

    # Sell Trade: Liquidity grab at buy-side level (high), bearish structure, bearish OB/FVG
    if liquidity_grab_high:
        if choch != 'Downtrend Shift (CHOCH)' or bos != 'Bearish BOS':
            logger.info("Sell trade rejected: Market structure not bearish (CHOCH/BOS mismatch).")
        else:
            for ob_type, ob_data in order_blocks:
                if ob_type == 'bearish_ob' and ob_data['low'] <= close_price <= ob_data['high']:
                    entry_price = close_price
                    stop_loss = float(grab_level_high) + sl_distance
                    internal_tp = find_internal_liquidity(df, 'sell', entry_price)
                    if internal_tp:
                        take_profit = internal_tp
                    else:
                        take_profit = entry_price - (stop_loss - entry_price) * max(PARTIAL_PROFIT_LEVELS[-1], 2.0)
                    stoploss_points = (stop_loss - entry_price) / mt5.symbol_info(SYMBOL).point
                    volume = calculate_position_size(equity, RISK_PERCENT_PER_TRADE, stoploss_points, atr)
                    sell_signal = True
                    break
            if not sell_signal:
                for fvg_type, fvg_data in fvgs:
                    if fvg_type == 'bearish_fvg' and fvg_data['low'] <= close_price <= fvg_data['high']:
                        entry_price = close_price
                        stop_loss = float(grab_level_high) + sl_distance
                        internal_tp = find_internal_liquidity(df, 'sell', entry_price)
                        if internal_tp:
                            take_profit = internal_tp
                        else:
                            take_profit = entry_price - (stop_loss - entry_price) * max(PARTIAL_PROFIT_LEVELS[-1], 2.0)
                        stoploss_points = (stop_loss - entry_price) / mt5.symbol_info(SYMBOL).point
                        volume = calculate_position_size(equity, RISK_PERCENT_PER_TRADE, stoploss_points, atr)
                        sell_signal = True
                        break

    # Buy Trade: Liquidity grab at sell-side level (low), bullish structure, bullish OB/FVG
    if liquidity_grab_low:
        if choch != 'Uptrend Shift (CHOCH)' or bos != 'Bullish BOS':
            logger.info("Buy trade rejected: Market structure not bullish (CHOCH/BOS mismatch).")
        else:
            for ob_type, ob_data in order_blocks:
                if ob_type == 'bullish_ob' and ob_data['low'] <= close_price <= ob_data['high']:
                    entry_price = close_price
                    stop_loss = float(grab_level_low) - sl_distance
                    internal_tp = find_internal_liquidity(df, 'buy', entry_price)
                    if internal_tp:
                        take_profit = internal_tp
                    else:
                        take_profit = entry_price + (entry_price - stop_loss) * max(PARTIAL_PROFIT_LEVELS[-1], 2.0)
                    stoploss_points = (entry_price - stop_loss) / mt5.symbol_info(SYMBOL).point
                    volume = calculate_position_size(equity, RISK_PERCENT_PER_TRADE, stoploss_points, atr)
                    buy_signal = True
                    break
            if not buy_signal:
                for fvg_type, fvg_data in fvgs:
                    if fvg_type == 'bullish_fvg' and fvg_data['low'] <= close_price <= fvg_data['high']:
                        entry_price = close_price
                        stop_loss = float(grab_level_low) - sl_distance
                        internal_tp = find_internal_liquidity(df, 'buy', entry_price)
                        if internal_tp:
                            take_profit = internal_tp
                        else:
                            take_profit = entry_price + (entry_price - stop_loss) * max(PARTIAL_PROFIT_LEVELS[-1], 2.0)
                        stoploss_points = (entry_price - stop_loss) / mt5.symbol_info(SYMBOL).point
                        volume = calculate_position_size(equity, RISK_PERCENT_PER_TRADE, stoploss_points, atr)
                        buy_signal = True
                        break

    if buy_signal:
        logger.info(f"Placing BUY order: price={entry_price}, SL={stop_loss}, TP={take_profit}, volume={volume}")
        place_buy_order(entry_price, stop_loss, take_profit, volume)
    elif sell_signal:
        logger.info(f"Placing SELL order: price={entry_price}, SL={stop_loss}, TP={take_profit}, volume={volume}")
        place_sell_order(entry_price, stop_loss, take_profit, volume)
    else:
        logger.info("No trade executed: missing confirmation signals.")
        return False

    manage_open_positions(df)
    logger.info("run_liquidity_pool completed")
    return True

def main():
    if not ensure_mt5_connection():
        logger.error("Failed to initialize MT5. Exiting.")
        return

    try:
        while True:
            success = run_liquidity_pool()
            logger.info(f"Liquidity Pool Strategy executed: {'Success' if success else 'No Trade'}")
            logger.info("Sleeping for 60 seconds...")
            sleep(60)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    finally:
        # Removed mt5.shutdown() call. MT5 shutdown is handled by main bot.
        logger.info("MT5 shutdown successfully.")

if __name__ == "__main__":
    main()