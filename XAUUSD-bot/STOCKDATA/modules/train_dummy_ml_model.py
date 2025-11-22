"""
Train a heavier RandomForestClassifier model (13 features) and save to models/ml_trade_filter.pkl.
Compatible with MLTradeFilter (expects predict_proba()).

Usage (from project root):
  python -m STOCKDATA.modules.train_dummy_ml_model --output models/ml_trade_filter.pkl
"""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DEF_OUTPUT = "models/ml_trade_filter.pkl"
N_SAMPLES = 20000
N_FEATURES = 13
RANDOM_STATE = 42


def make_synthetic_data(n_samples: int, n_features: int, rng: np.random.Generator):
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    # Nonlinear target via random tree-like rule
    weights = rng.normal(0.0, 1.0, size=(n_features,))
    logits = X @ weights + 0.5 * (X[:, 0] * X[:, 3]) - 0.4 * (X[:, 5] > 0).astype(float)
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
    y = (p >= 0.5).astype(int)
    return X, y


def train_model(output_path: Path):
    rng = np.random.default_rng(RANDOM_STATE)
    X, y = make_synthetic_data(N_SAMPLES, N_FEATURES, rng)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    clf = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
    )
    clf.fit(X_tr, y_tr)

    acc = clf.score(X_te, y_te)
    print(f"Trained RandomForest. Test ACC: {acc:.3f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, str(output_path))
    print(f"Saved model to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train and save a heavy RandomForest model (13 features)")
    parser.add_argument("--output", type=str, default=DEF_OUTPUT, help=f"Output path (default: {DEF_OUTPUT})")
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    train_model(output_path)


if __name__ == "__main__":
    main()
