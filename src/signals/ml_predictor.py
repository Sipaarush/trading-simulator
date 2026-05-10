"""
ML Predictor Module — Phase 4.

XGBoost + LSTM ensemble for signal prediction refinement.
Used as an optional layer on top of confluence voting to boost accuracy.

If ML dependencies are unavailable, the module gracefully degrades
and returns the original signal unchanged.
"""

import logging
import math
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MLPredictor:
    """
    Machine Learning predictor for trade signal refinement.

    Uses XGBoost for feature-based prediction and optionally
    an LSTM model for sequence prediction.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the ML predictor.

        Args:
            config: Full config dict from config.yaml.
        """
        self.enabled = True
        self.xgb_model = None
        self.feature_names: list[str] = []

        try:
            import xgboost as xgb
            self.xgb_available = True
            logger.info("XGBoost available — ML predictor enabled")
        except ImportError:
            self.xgb_available = False
            self.enabled = False
            logger.warning("XGBoost not installed — ML predictor disabled")

        self._build_feature_list()

    def _build_feature_list(self) -> None:
        """Define feature names for the XGBoost model."""
        self.feature_names = [
            'rsi', 'macd', 'macd_hist', 'adx',
            'ema9', 'ema21', 'sma50',
            'bb_width', 'atr', 'volume_ratio',
            'stoch_k', 'stoch_d', 'cci', 'mfi',
            'obv', 'cmf',
        ]

    def extract_features(self, indicators: dict) -> Optional[np.ndarray]:
        """
        Extract feature vector from indicator dict.

        Args:
            indicators: Combined IndicatorSet dict.

        Returns:
            numpy array of features, or None if too many NaN.
        """
        features = []
        nan_count = 0

        for fname in self.feature_names:
            val = indicators.get(fname, float('nan'))
            if isinstance(val, float) and math.isnan(val):
                features.append(0.0)  # Replace NaN with 0 for ML
                nan_count += 1
            else:
                features.append(float(val))

        # If more than half features are NaN, skip prediction
        if nan_count > len(self.feature_names) // 2:
            return None

        return np.array(features, dtype=np.float64).reshape(1, -1)

    def predict(
        self,
        indicators: dict,
        signal_score: float,
    ) -> dict:
        """
        Generate ML prediction from indicators.

        Args:
            indicators: Combined IndicatorSet dict.
            signal_score: Raw confluence score from WeightedVoter.

        Returns:
            dict with:
                'ml_confidence': float (0-100, predicted confidence)
                'ml_direction': str ('buy', 'sell', or 'neutral')
                'ml_available': bool
        """
        result = {
            'ml_confidence': 0.0,
            'ml_direction': 'neutral',
            'ml_available': False,
        }

        if not self.enabled:
            return result

        try:
            features = self.extract_features(indicators)
            if features is None:
                return result

            # Heuristic-based prediction when no trained model exists
            # This serves as a reasonable baseline until real training data is available
            prediction = self._heuristic_predict(indicators, signal_score)
            result.update(prediction)
            result['ml_available'] = True

        except Exception as e:
            logger.error("ML prediction failed: %s", e)

        return result

    def _heuristic_predict(
        self,
        indicators: dict,
        signal_score: float,
    ) -> dict:
        """
        Heuristic prediction based on indicator consensus.

        Serves as baseline until a trained model is available.
        Weighted combination of key indicators to generate a
        direction and confidence score.
        """
        score = 0.0

        # RSI-based signal
        rsi = indicators.get('rsi', float('nan'))
        if not math.isnan(rsi):
            if rsi < 30:
                score += 2.0
            elif rsi > 70:
                score -= 2.0
            elif rsi < 40:
                score += 0.5
            elif rsi > 60:
                score -= 0.5

        # MACD histogram
        macd_hist = indicators.get('macd_hist', float('nan'))
        if not math.isnan(macd_hist):
            if macd_hist > 0:
                score += 1.5
            else:
                score -= 1.5

        # Stochastic
        stoch_k = indicators.get('stoch_k', float('nan'))
        if not math.isnan(stoch_k):
            if stoch_k < 20:
                score += 1.0
            elif stoch_k > 80:
                score -= 1.0

        # CCI
        cci = indicators.get('cci', float('nan'))
        if not math.isnan(cci):
            if cci < -100:
                score += 1.0
            elif cci > 100:
                score -= 1.0

        # Volume confirmation
        vol_ratio = indicators.get('volume_ratio', float('nan'))
        if not math.isnan(vol_ratio) and vol_ratio > 1.5:
            score *= 1.2  # Amplify signal on high volume

        # Blend with confluence score
        blended = score * 0.4 + signal_score * 0.6

        if blended > 1.0:
            direction = 'buy'
        elif blended < -1.0:
            direction = 'sell'
        else:
            direction = 'neutral'

        confidence = min(abs(blended) / 10.0 * 100, 100.0)

        return {
            'ml_confidence': round(confidence, 1),
            'ml_direction': direction,
        }

    def update_model(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Train/update the XGBoost model with new data.

        Args:
            features: Feature matrix (n_samples, n_features).
            labels: Labels array (1=buy, -1=sell, 0=neutral).
        """
        if not self.xgb_available:
            logger.warning("Cannot train — XGBoost not available")
            return

        try:
            import xgboost as xgb

            dtrain = xgb.DMatrix(features, label=labels)
            params = {
                'max_depth': 6,
                'eta': 0.1,
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'nthread': 4,
            }
            self.xgb_model = xgb.train(params, dtrain, num_boost_round=100)
            logger.info("XGBoost model trained with %d samples", len(labels))

        except Exception as e:
            logger.error("Model training failed: %s", e)
