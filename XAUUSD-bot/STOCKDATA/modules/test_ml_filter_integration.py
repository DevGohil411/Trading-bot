import os
import json
import sys

# Ensure project root and STOCKDATA are on sys.path (root first to pick correct 'utils')
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))  # d:/XAUUSD-bot/STOCKDATA
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))               # d:/XAUUSD-bot

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from utils.trade_logger import logger as log
from modules.ml_filter import MLTradeFilter


def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        log.error(f"Failed to read config at {config_path}: {e}")
        return {}


def main():
    project_root = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
    config_path = os.path.join(project_root, 'config.json')
    cfg = load_config(config_path)

    ml_model_path = cfg.get('advanced_settings', {}).get('ml_model_path', 'models/ml_trade_filter.pkl')
    # Expand to absolute path if relative
    if not os.path.isabs(ml_model_path):
        ml_model_path = os.path.join(project_root, ml_model_path)

    log.info(f"[TEST] Loading MLTradeFilter from: {ml_model_path}")
    filt = MLTradeFilter(model_path=ml_model_path)

    # Construct a deterministic 13-feature vector matching our strategies
    features = [0.0] * 13

    # Force a filtered decision by using a very high threshold
    high_threshold = 0.99
    prob = filt.get_win_probability(features)
    allowed = filt.allow_trade(features, threshold=high_threshold)
    log.info(f"[TEST] High threshold={high_threshold} -> prob={prob:.4f}, allowed={allowed}")

    # Force an allowed decision by using a very low threshold
    low_threshold = 0.01
    prob2 = filt.get_win_probability(features)
    allowed2 = filt.allow_trade(features, threshold=low_threshold)
    log.info(f"[TEST] Low threshold={low_threshold} -> prob={prob2:.4f}, allowed={allowed2}")

    print("model_loaded", bool(filt.model))
    print("sample_prob", prob)
    print("sample_prob2", prob2)


if __name__ == '__main__':
    main()
