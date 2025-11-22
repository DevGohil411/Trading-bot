import pandas as pd
import numpy as np
import math
import logging

indicators_logger = logging.getLogger(__name__)
indicators_logger.setLevel(logging.INFO)

if not indicators_logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    console_handler.setFormatter(formatter)
    indicators_logger.addHandler(console_handler)
    indicators_logger.propagate = False

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculates the Average True Range (ATR).
    ATR = Moving Average of True Range.
    True Range = max[(high - low), abs(high - previous close), abs(low - previous close)]
    """
    if df.empty or len(df) < period + 1: # Need at least period + 1 candles for ATR
        indicators_logger.warning(f"Not enough data for ATR calculation (need >{period} candles, got {len(df)}). Returning 0.0.")
        return 0.0

    df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning
    
    # Calculate True Range (TR)
    high_low = df_copy['high'] - df_copy['low']
    high_prev_close = np.abs(df_copy['high'] - df_copy['close'].shift())
    low_prev_close = np.abs(df_copy['low'] - df_copy['close'].shift())
    
    # Combine the three components and take the maximum for each row
    true_range = pd.DataFrame({'high_low': high_low, 
                               'high_prev_close': high_prev_close, 
                               'low_prev_close': low_prev_close}).max(axis=1)
    
    # Calculate ATR as the simple moving average of True Range
    atr_series = true_range.ewm(span=period, adjust=False, min_periods=period).mean() # Using EMA for ATR as is common
    
    last_atr = atr_series.iloc[-1]
    
    if math.isnan(last_atr) or last_atr <= 0:
        indicators_logger.warning(f"Calculated ATR is NaN or non-positive ({last_atr}). Returning 0.0.")
        return 0.0
        
    return last_atr

def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    """
    Calculates the Relative Strength Index (RSI).
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """
    if series.empty or len(series) < period + 1:
        indicators_logger.warning(f"Not enough data for RSI calculation (need >{period} values, got {len(series)}). Returning 50.0 (neutral).")
        return 50.0

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(span=period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(span=period, adjust=False, min_periods=period).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan) # Replace 0 with NaN to avoid division by zero
    
    rsi_series = 100 - (100 / (1 + rs))
    
    last_rsi = rsi_series.iloc[-1]

    if math.isnan(last_rsi):
        indicators_logger.warning("Calculated RSI is NaN. Returning 50.0 (neutral).")
        return 50.0
        
    return last_rsi

# Example Usage (for testing this module independently)
if __name__ == "__main__":
    print("\n--- Testing Indicators Module ---")
    
    # Create a dummy DataFrame for testing
    data = {
        'open': [100, 102, 101, 103, 105, 104, 106, 107, 109, 108, 110, 112, 111, 113, 115, 114, 116, 118, 117, 119],
        'high': [103, 104, 103, 106, 107, 106, 108, 109, 111, 110, 112, 114, 113, 115, 117, 116, 118, 120, 119, 121],
        'low': [99, 100, 99, 102, 103, 102, 104, 105, 107, 106, 108, 110, 109, 111, 113, 112, 114, 116, 115, 117],
        'close': [102, 101, 103, 105, 104, 106, 107, 109, 108, 110, 112, 111, 113, 115, 114, 116, 118, 117, 119, 120]
    }
    test_df = pd.DataFrame(data)

    # Test ATR
    atr_val = calculate_atr(test_df)
    print(f"Calculated ATR: {atr_val:.4f}")
    if atr_val > 0:
        print("ATR calculation seems correct (non-zero).")
    else:
        print("ATR calculation might have issues.")

    # Test RSI
    rsi_val = calculate_rsi(test_df['close'])
    print(f"Calculated RSI: {rsi_val:.2f}")
    if 0 <= rsi_val <= 100:
        print("RSI calculation seems correct (within 0-100 range).")
    else:
        print("RSI calculation might have issues.")

    # Test with insufficient data
    print("\nTesting with insufficient data:")
    small_df = test_df.head(5)
    atr_small = calculate_atr(small_df)
    rsi_small = calculate_rsi(small_df['close'])
    print(f"ATR with 5 candles: {atr_small:.4f} (Expected near 0)")
    print(f"RSI with 5 candles: {rsi_small:.2f} (Expected near 50)")