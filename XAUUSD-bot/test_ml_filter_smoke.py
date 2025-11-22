import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from STOCKDATA.modules.ml_filter import MLTradeFilter


def main():
    model_path = os.path.join(PROJECT_ROOT, 'models', 'ml_trade_filter.pkl')
    filt = MLTradeFilter(model_path=model_path)
    # Dummy 13-length features vector (values roughly within plausible ranges)
    features = [
        1.2,   # ATR
        0.5,   # zone_width_norm
        0.8,   # range_ratio
        0.3,   # wick_up_ratio
        0.2,   # wick_dn_ratio
        1,     # sweep_low_flag
        0,     # sweep_high_flag
        1,     # choch_bullish
        0,     # choch_bearish
        1,     # in_zone_flag
        0,     # vol_spike_m15
        14,    # hour
        1,     # direction_flag
    ]

    prob = filt.get_win_probability(features)
    allowed = filt.allow_trade(features, threshold=0.65)
    print(f"Win probability: {prob:.4f}")
    print(f"Allowed (threshold=0.65): {allowed}")


if __name__ == '__main__':
    main()
