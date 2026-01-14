"""
SMC Ultra V2 - ML Confidence Scorer
===================================
Machine Learning basiertes Confidence Scoring.

Das Herzstück für 70%+ Winrate:
- Lernt aus historischen Trades
- Erkennt Muster die Menschen übersehen
- Adaptiert sich automatisch
"""

import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("Warning: ML libraries not installed. Using fallback scoring.")

from config.settings import config
from .features import FeatureExtractor, FeatureSet


@dataclass
class PredictionResult:
    """Result of ML prediction"""
    confidence: int  # 0-100
    win_probability: float
    should_trade: bool
    direction_confidence: float  # How confident in direction
    feature_importance: Dict[str, float] = None


class MLConfidenceScorer:
    """
    ML-based confidence scoring using XGBoost.

    Features:
    - Learns optimal feature weights from data
    - Recency weighting for recent performance
    - Automatic feature importance analysis
    """

    def __init__(self, model_dir: str = None):
        self.model_dir = Path(model_dir or config.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.scaler = StandardScaler() if HAS_ML else None
        self.feature_extractor = FeatureExtractor()
        self.feature_names = FeatureExtractor.FEATURE_NAMES
        self.is_trained = False

        self.ml_config = config.ml

    def train(
        self,
        trades_df: pd.DataFrame,
        features_df: pd.DataFrame = None,
        save: bool = True
    ) -> Dict:
        """
        Train the ML model on historical trades.

        Args:
            trades_df: DataFrame with trades and outcomes
                Required columns: 'outcome' (1=win, 0=loss), 'timestamp'
                Optional: 'pnl_pct' for weighted training
            features_df: Pre-calculated features (if None, will calculate)
            save: Whether to save trained model

        Returns:
            Training metrics
        """
        if not HAS_ML:
            return {'error': 'ML libraries not installed'}

        if len(trades_df) < self.ml_config.min_training_samples:
            return {'error': f'Need at least {self.ml_config.min_training_samples} samples'}

        print(f"Training ML model on {len(trades_df)} trades...")

        # Prepare features
        if features_df is None:
            # Assume trades_df has feature columns
            X = trades_df[self.feature_names].fillna(0)
        else:
            X = features_df[self.feature_names].fillna(0)

        y = trades_df['outcome'].values

        # Calculate sample weights (recency weighting)
        if self.ml_config.enable_recency_weighting and 'timestamp' in trades_df.columns:
            weights = self._calc_recency_weights(trades_df['timestamp'])
        else:
            weights = np.ones(len(y))

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train/test split
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X_scaled, y, weights,
            test_size=self.ml_config.train_test_split,
            random_state=42,
            stratify=y
        )

        # Train XGBoost
        self.model = XGBClassifier(
            n_estimators=self.ml_config.n_estimators,
            max_depth=self.ml_config.max_depth,
            learning_rate=self.ml_config.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            random_state=42
        )

        self.model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'samples': len(y),
            'feature_importance': self._get_feature_importance()
        }

        print(f"Model trained - Accuracy: {metrics['accuracy']:.2%}")
        print(f"Precision: {metrics['precision']:.2%}, Recall: {metrics['recall']:.2%}")

        self.is_trained = True

        if save:
            self.save()

        return metrics

    def predict(self, features: FeatureSet) -> PredictionResult:
        """
        Predict confidence for a potential trade.

        Args:
            features: FeatureSet from FeatureExtractor

        Returns:
            PredictionResult with confidence and metadata
        """
        if not self.is_trained:
            return self._fallback_predict(features)

        # Prepare features
        X = features.to_array(self.feature_names).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Predict
        proba = self.model.predict_proba(X_scaled)[0]
        win_prob = proba[1]
        confidence = int(win_prob * 100)

        # Direction confidence
        direction_conf = abs(win_prob - 0.5) * 2  # 0-1 scale

        # Get feature importance for this prediction
        importance = None
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(self.feature_names, self.model.feature_importances_))

        return PredictionResult(
            confidence=confidence,
            win_probability=win_prob,
            should_trade=confidence >= config.entry.min_confidence,
            direction_confidence=direction_conf,
            feature_importance=importance
        )

    def _fallback_predict(self, features: FeatureSet) -> PredictionResult:
        """
        Fallback scoring when ML model not trained.
        Uses weighted rules-based scoring.
        """
        score = 50  # Base score
        f = features.features

        # SMC factors (max +30)
        if f.get('has_bullish_ob') or f.get('has_bearish_ob'):
            score += 10 * f.get('ob_strength', 0.5)
        if f.get('has_bullish_fvg') or f.get('has_bearish_fvg'):
            score += 8
        if f.get('has_bullish_sweep') or f.get('has_bearish_sweep'):
            score += 12 * f.get('sweep_strength', 0.5)

        # MTF alignment (max +15)
        if f.get('mtf_aligned'):
            score += 10
        score += f.get('htf_strength', 0) * 5

        # Technical (max +10)
        if f.get('rsi_oversold') or f.get('rsi_overbought'):
            score += 5
        if f.get('macd_cross_up') or f.get('macd_cross_down'):
            score += 5

        # Regime adjustment
        if f.get('regime_choppy'):
            score -= 15
        elif f.get('regime_trending'):
            score += 5

        # Session adjustment
        score *= f.get('session_weight', 1.0)

        confidence = int(np.clip(score, 0, 100))

        return PredictionResult(
            confidence=confidence,
            win_probability=confidence / 100,
            should_trade=confidence >= config.entry.min_confidence,
            direction_confidence=0.5,
            feature_importance=None
        )

    def _calc_recency_weights(self, timestamps: pd.Series) -> np.ndarray:
        """
        Calculate sample weights based on recency.

        Recent trades get higher weights.
        """
        now = pd.Timestamp.now()
        days_ago = (now - pd.to_datetime(timestamps)).dt.days

        weights = np.zeros(len(days_ago))
        recency_config = self.ml_config.recency_weights

        for max_days, weight in sorted(recency_config.items()):
            mask = days_ago <= max_days
            weights[mask] = weight

        # Anything older gets minimum weight
        weights[weights == 0] = 0.1

        return weights

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if not self.is_trained or self.model is None:
            return {}

        importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))

        # Sort by importance
        return dict(sorted(importance.items(), key=lambda x: -x[1]))

    def save(self, path: str = None):
        """Save model and scaler"""
        if not self.is_trained:
            return

        model_path = Path(path) if path else self.model_dir / "ml_scorer.pkl"

        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'trained_at': datetime.now().isoformat()
            }, f)

        print(f"Model saved to {model_path}")

    def load(self, path: str = None) -> bool:
        """Load model and scaler"""
        model_path = Path(path) if path else self.model_dir / "ml_scorer.pkl"

        if not model_path.exists():
            print(f"No model found at {model_path}")
            return False

        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)

            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.is_trained = True

            print(f"Model loaded (trained at {data.get('trained_at', 'unknown')})")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def retrain_incremental(
        self,
        new_trades: pd.DataFrame,
        existing_trades: pd.DataFrame = None
    ) -> Dict:
        """
        Retrain model with new data.

        Combines new trades with existing, applies recency weighting.
        """
        if existing_trades is not None:
            all_trades = pd.concat([existing_trades, new_trades])
        else:
            all_trades = new_trades

        return self.train(all_trades, save=True)


class EnsembleScorer:
    """
    Ensemble of multiple scorers for more robust predictions.

    Combines:
    - ML scorer
    - Rules-based scorer
    - Historical pattern matcher
    """

    def __init__(self):
        self.ml_scorer = MLConfidenceScorer()
        self.weights = {
            'ml': 0.6,
            'rules': 0.3,
            'history': 0.1
        }

    def predict(
        self,
        features: FeatureSet,
        historical_patterns: List[Dict] = None
    ) -> PredictionResult:
        """
        Ensemble prediction combining multiple methods.
        """
        scores = []

        # ML prediction
        if self.ml_scorer.is_trained:
            ml_result = self.ml_scorer.predict(features)
            scores.append(('ml', ml_result.confidence, self.weights['ml']))
        else:
            scores.append(('ml', 50, 0))  # No contribution if not trained

        # Rules-based
        rules_result = self.ml_scorer._fallback_predict(features)
        scores.append(('rules', rules_result.confidence, self.weights['rules']))

        # Historical pattern matching
        if historical_patterns:
            pattern_score = self._match_patterns(features, historical_patterns)
            scores.append(('history', pattern_score, self.weights['history']))
        else:
            scores.append(('history', 50, 0))

        # Weighted average
        total_weight = sum(w for _, _, w in scores)
        if total_weight == 0:
            total_weight = 1

        final_confidence = sum(s * w for _, s, w in scores) / total_weight
        final_confidence = int(np.clip(final_confidence, 0, 100))

        return PredictionResult(
            confidence=final_confidence,
            win_probability=final_confidence / 100,
            should_trade=final_confidence >= config.entry.min_confidence,
            direction_confidence=0.5,
            feature_importance=None
        )

    def _match_patterns(
        self,
        features: FeatureSet,
        historical_patterns: List[Dict]
    ) -> int:
        """Match current features against historical winning patterns"""
        if not historical_patterns:
            return 50

        # Simple similarity matching
        best_match = 0

        for pattern in historical_patterns:
            if pattern.get('outcome') != 1:  # Only match winners
                continue

            similarity = self._calc_similarity(features.features, pattern.get('features', {}))
            if similarity > best_match:
                best_match = similarity

        return int(best_match * 100)

    def _calc_similarity(self, f1: Dict, f2: Dict) -> float:
        """Calculate similarity between two feature sets"""
        if not f1 or not f2:
            return 0.5

        common_keys = set(f1.keys()) & set(f2.keys())
        if not common_keys:
            return 0.5

        similarities = []
        for key in common_keys:
            v1, v2 = f1[key], f2[key]
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                # Normalized difference
                max_val = max(abs(v1), abs(v2), 1)
                sim = 1 - abs(v1 - v2) / max_val
                similarities.append(sim)

        return np.mean(similarities) if similarities else 0.5
