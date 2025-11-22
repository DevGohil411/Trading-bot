# Requires 'joblib' package. Install with: pip install joblib
import joblib
import os
import logging
import numpy as np # Added numpy for array handling if features are lists

"""
Prefer routing ML filter logs through the shared 'trade_bot_logger' so messages
appear in logs/trade_bot.log. Fall back to a simple console logger if unavailable.
"""
try:
    # Shared app logger writes to file and console
    from utils.trade_logger import logger as trade_bot_logger
    ml_filter_logger = trade_bot_logger
except Exception:
    # Fallback: module-specific logger to console
    ml_filter_logger = logging.getLogger(__name__)
    ml_filter_logger.setLevel(logging.INFO)
    if not ml_filter_logger.handlers:
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        console_handler.setFormatter(formatter)
        ml_filter_logger.addHandler(console_handler)
    # Allow propagation so a root logger (if any) can also capture
    ml_filter_logger.propagate = True

class MLTradeFilter:
    """
    Institutional ML filter for trade setups.
    Loads a pre-trained sklearn model (RandomForest/XGBoost/LightGBM).
    Use allow_trade(features, threshold) to filter low-probability setups.
    """
    def __init__(self, model_path="ml_trade_filter.pkl"):
        # Adjusted model_path to look in common places, or expect absolute path
        # Assuming ml_trade_filter.pkl is in the root directory or 'models' subfolder
        possible_paths = [
            model_path, # Direct path provided
            os.path.join(os.getcwd(), model_path), # Current working directory
            os.path.join(os.getcwd(), 'models', model_path) # Common 'models' folder
        ]
        
        self.model = None
        self.model_path_used = None

        for p in possible_paths:
            if os.path.exists(p):
                self.model_path_used = p
                break
        
        if self.model_path_used:
            self._load_model()
        else:
            ml_filter_logger.info(f"ML model file not found (checked paths: {possible_paths}). ML-based trade filtering is disabled. This is expected if a model has not been configured.")

    def _load_model(self):
        """Attempt to load the ML model"""
        try:
            self.model = joblib.load(self.model_path_used)
            ml_filter_logger.info(f"Successfully loaded ML model from {self.model_path_used}")
            # Log model details where possible
            try:
                n_features = getattr(self.model, "n_features_in_", None)
                if n_features is not None:
                    ml_filter_logger.info(f"ML model expects n_features_in_={n_features}")
                else:
                    ml_filter_logger.info("ML model does not expose n_features_in_.")
            except Exception:
                pass
        except Exception as e:
            ml_filter_logger.error(f"Error loading ML model from {self.model_path_used}: {str(e)}. All trades will be allowed by ML filter.", exc_info=True)
            self.model = None # Ensure model is None if loading fails

    def _prepare_features(self, features: list):
        """Validate and reshape features to 2D array (1, n_features)."""
        # Convert to numpy array
        arr = np.array(features)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError(f"Features must be 1D or 2D, got ndim={arr.ndim}")

        # Validate expected feature length if model provides it
        expected = getattr(self.model, "n_features_in_", None) if self.model is not None else None
        if expected is not None and arr.shape[1] != expected:
            raise ValueError(f"Feature length mismatch: got {arr.shape[1]}, expected {expected}")

        return arr

    def _predict_proba(self, features_array: np.ndarray) -> float:
        """Return probability for the positive class using model, with fallbacks."""
        # Primary: predict_proba if available
        if self.model is None:
            return 0.5
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features_array)
            # Expect shape (n_samples, 2) or (n_samples, 1) depending on model
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return float(proba[0][1])
            return float(proba[0][0])
        # Fallbacks: decision_function or predict
        try:
            if hasattr(self.model, "decision_function"):
                score = self.model.decision_function(features_array)
                # Map score to (0,1) via logistic transform for a rough probability
                prob = 1.0 / (1.0 + np.exp(-float(score[0])))
                return prob
        except Exception:
            pass
        try:
            pred = self.model.predict(features_array)
            # Convert hard prediction to probability-like value
            return float(pred[0]) if np.isscalar(pred[0]) else 0.5
        except Exception:
            pass
        return 0.5

    def allow_trade(self, features: list, threshold: float = 0.6) -> bool:
        """
        Returns True if model predicts win-probability > threshold, else False.
        If model is not loaded, always returns True (allow all trades).
        Features should be a list/array of numerical values that the model expects.
        """
        if self.model is None:
            return True # If model didn't load, allow all trades

        try:
            # Prepare features and compute probability safely
            features_array = self._prepare_features(features)
            prob = self._predict_proba(features_array)
            allowed = prob > threshold
            
            if not allowed:
                ml_filter_logger.info(f"Trade filtered by ML model. Win probability: {prob:.2f} < threshold: {threshold}")
            else:
                ml_filter_logger.debug(f"Trade allowed by ML model. Win probability: {prob:.2f} >= threshold: {threshold}")

            return allowed
        except Exception as e:
            ml_filter_logger.error(f"Error in ML prediction (allow_trade): {str(e)}. Allowing trade.", exc_info=True)
            return True # Fail-safe: if prediction fails, allow trade

    def get_win_probability(self, features: list) -> float:
        """
        Get the raw win probability from the model.
        Features should be a list/array of numerical values.
        """
        if self.model is None:
            return 0.5  # Neutral probability when model is not available
        try:
            features_array = self._prepare_features(features)
            return self._predict_proba(features_array)
        except Exception as e:
            ml_filter_logger.error(f"Error getting win probability: {str(e)}", exc_info=True)
            return 0.5 # Fail-safe: return neutral probability on error

# Example Usage (for testing this module independently)
if __name__ == "__main__":
    # To test this, you need a dummy 'ml_trade_filter.pkl' file.
    # You can create one quickly for testing:
    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.datasets import make_classification
    # import joblib
    # X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_redundant=0, random_state=42)
    # model = RandomForestClassifier(random_state=42)
    # model.fit(X, y)
    # joblib.dump(model, "ml_trade_filter.pkl")
    # print("Dummy model 'ml_trade_filter.pkl' created for testing.")

    print("\n--- Testing MLTradeFilter ---")
    
    # Test case when model file does not exist
    print("Testing with non-existent model:")
    temp_filter = MLTradeFilter(model_path="non_existent_model.pkl")
    print(f"Should allow trade (no model): {temp_filter.allow_trade([1, 2, 3, 4, 5])}\n")

    # Assuming 'ml_trade_filter.pkl' is in the current directory for this test
    # If your model is elsewhere, provide the correct path.
    model_file_path = "ml_trade_filter.pkl"
    if not os.path.exists(model_file_path):
        print(f"'{model_file_path}' not found. Please create a dummy model or place your actual model for testing.")
        print("Skipping further ML filter tests without a model.")
    else:
        print(f"Testing with existing model: {model_file_path}")
        ml_filter = MLTradeFilter(model_path=model_file_path)

        if ml_filter.model:
            # Example features (replace with actual features your model expects)
            # The length of this list should match n_features of your trained model (e.g., 5 if make_classification used 5 features)
            good_features = [0.1, 0.2, 0.7, 0.3, 0.8] # Example features that might lead to high probability
            bad_features = [-0.5, 0.1, -0.2, 0.9, 0.1] # Example features that might lead to low probability

            print(f"Features (Good): {good_features}")
            allowed_good = ml_filter.allow_trade(good_features)
            prob_good = ml_filter.get_win_probability(good_features)
            print(f"Allowed: {allowed_good}, Win Prob: {prob_good:.2f}\n")

            print(f"Features (Bad): {bad_features}")
            allowed_bad = ml_filter.allow_trade(bad_features)
            prob_bad = ml_filter.get_win_probability(bad_features)
            print(f"Allowed: {allowed_bad}, Win Prob: {prob_bad:.2f}\n")
        else:
            print("ML model could not be loaded for testing.")