"""
Train a simple 13-feature logistic model without scikit-learn and save it for MLTradeFilter.

Usage (from project root):
  python -m STOCKDATA.modules.train_simple_model --output models/ml_trade_filter.pkl
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import joblib

from .simple_ml import SimpleLogisticModel

DEF_OUTPUT = "models/ml_trade_filter.pkl"
N_SAMPLES = 12000
N_FEATURES = 13
RANDOM_STATE = 42


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def make_synthetic_data(n_samples: int, n_features: int, rng: np.random.Generator):
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    # True weights to mimic some signal structure
    true_w = rng.normal(0.0, 0.8, size=(n_features,))
    true_b = -0.15
    noise = rng.normal(0.0, 0.6, size=(n_samples,))
    logits = X @ true_w + true_b + noise
    p = sigmoid(logits)
    y = (p >= 0.5).astype(int)
    return X, y


def train_model(output_path: Path):
    rng = np.random.default_rng(RANDOM_STATE)
    X, y = make_synthetic_data(N_SAMPLES, N_FEATURES, rng)

    # Simple split
    idx = np.arange(N_SAMPLES)
    rng.shuffle(idx)
    split = int(0.8 * N_SAMPLES)
    tr_idx, te_idx = idx[:split], idx[split:]

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]

    model = SimpleLogisticModel(n_features=N_FEATURES, lr=0.08, l2=1e-3, epochs=1200)
    model.fit(X_tr, y_tr)

    # Evaluate
    p_te = model.predict_proba(X_te)[:, 1]
    y_pred = (p_te >= 0.5).astype(int)
    acc = (y_pred == y_te).mean()
    print(f"Trained simple model. Test ACC: {acc:.3f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(output_path))
    print(f"Saved model to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train and save a simple ML trade filter model (13 features)")
    parser.add_argument("--output", type=str, default=DEF_OUTPUT, help=f"Output path (default: {DEF_OUTPUT})")
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    train_model(output_path)


if __name__ == "__main__":
    main()
