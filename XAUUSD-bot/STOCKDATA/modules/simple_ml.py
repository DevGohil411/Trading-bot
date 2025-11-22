"""
Pure-Python logistic regression with L2 regularization.
Provides predict_proba() API compatible with scikit-learn for use by MLTradeFilter.
"""
from __future__ import annotations
import numpy as np


class SimpleLogisticModel:
    def __init__(self, n_features: int, lr: float = 0.05, l2: float = 1e-3, epochs: int = 800):
        self.n_features = n_features
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        # Initialize weights and bias
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleLogisticModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        assert d == self.n_features, f"Expected {self.n_features} features, got {d}"

        for _ in range(self.epochs):
            z = X @ self.w + self.b
            p = self._sigmoid(z)
            # Gradients
            grad_w = (X.T @ (p - y)) / n + self.l2 * self.w
            grad_b = np.sum(p - y) / n
            # Update
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        z = X @ self.w + self.b
        p1 = self._sigmoid(z)
        p0 = 1.0 - p1
        # Return shape (n_samples, 2): [P(class0), P(class1)]
        return np.vstack([p0, p1]).T

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)
