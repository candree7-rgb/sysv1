"""
SMC Ultra V2 - ML Coin Filter
=============================
Filtert Coins basierend auf Tradability Score.

Predicted welche Coins heute/diese Stunde "tradeable" sind:
- Ausreichend Volatilität
- Guter Spread
- Historisch gute Performance
- Passendes Regime
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler
    HAS_ML = True
except ImportError:
    HAS_ML = False

from config.settings import config

# Simple CoinConfig for ML scoring (coin_db removed for simplicity)
@dataclass
class CoinConfig:
    symbol: str
    tier: int = 2
    max_leverage: int = 50
    min_volume_24h: float = 1000000
    typical_spread_pct: float = 0.05
    volatility_mult: float = 1.0
    enabled: bool = True


@dataclass
class CoinScore:
    """Score for a coin's tradability"""
    symbol: str
    score: float  # 0-100
    is_tradeable: bool
    reasons: List[str]

    # Detailed metrics
    volatility_score: float = 0.0
    spread_score: float = 0.0
    volume_score: float = 0.0
    historical_wr: float = 0.0
    regime_score: float = 0.0


class CoinFilter:
    """
    Filters coins based on tradability.

    Used to pre-filter coins before signal detection.
    Only the top N coins are scanned for signals.
    """

    def __init__(self):
        self.historical_stats: Dict[str, Dict] = {}
        self.min_score = 60  # Minimum score to be tradeable

    def score_coins(
        self,
        coins: List[str],
        current_data: Dict[str, pd.DataFrame] = None,
        regime_data: Dict[str, str] = None
    ) -> List[CoinScore]:
        """
        Score all coins and return sorted list.

        Args:
            coins: List of coin symbols
            current_data: Current price data for each coin
            regime_data: Current regime for each coin

        Returns:
            Sorted list of CoinScore (best first)
        """
        scores = []

        for symbol in coins:
            score = self._score_coin(symbol, current_data, regime_data)
            scores.append(score)

        # Sort by score (highest first)
        scores.sort(key=lambda x: -x.score)

        return scores

    def get_tradeable(
        self,
        coins: List[str],
        top_n: int = 20,
        current_data: Dict[str, pd.DataFrame] = None,
        regime_data: Dict[str, str] = None
    ) -> List[str]:
        """
        Get top N tradeable coins.

        Returns:
            List of symbol names
        """
        scores = self.score_coins(coins, current_data, regime_data)
        tradeable = [s for s in scores if s.is_tradeable]
        return [s.symbol for s in tradeable[:top_n]]

    def _score_coin(
        self,
        symbol: str,
        current_data: Dict[str, pd.DataFrame],
        regime_data: Dict[str, str]
    ) -> CoinScore:
        """Score a single coin"""
        score = 50.0  # Base score
        reasons = []

        # Get coin config (simplified - all coins treated equally)
        coin_config = CoinConfig(symbol=symbol, tier=2)

        # 1. Tier bonus (max 15 points)
        tier_bonus = {1: 15, 2: 10, 3: 5, 4: 0}
        tier_score = tier_bonus.get(coin_config.tier, 0)
        score += tier_score
        if tier_score > 10:
            reasons.append("top_tier")

        # 2. Volatility score (max 20 points)
        vol_score = 10  # Default
        if current_data and symbol in current_data:
            df = current_data[symbol]
            if 'atr_pct' in df.columns:
                atr_pct = df['atr_pct'].iloc[-1]
                if 0.2 <= atr_pct <= 1.5:  # Good volatility range
                    vol_score = 20
                    reasons.append("good_volatility")
                elif atr_pct < 0.1:
                    vol_score = 0
                    reasons.append("low_volatility")
                elif atr_pct > 2.0:
                    vol_score = 5
                    reasons.append("high_volatility")
        score += vol_score

        # 3. Volume score (max 15 points)
        vol_ratio_score = 7.5  # Default
        if current_data and symbol in current_data:
            df = current_data[symbol]
            if 'volume_ratio' in df.columns:
                vol_ratio = df['volume_ratio'].iloc[-1]
                if vol_ratio >= 1.5:
                    vol_ratio_score = 15
                    reasons.append("high_volume")
                elif vol_ratio >= 1.0:
                    vol_ratio_score = 10
                elif vol_ratio < 0.5:
                    vol_ratio_score = 0
                    reasons.append("low_volume")
        score += vol_ratio_score

        # 4. Spread score (max 10 points)
        spread_score = 5  # Default
        typical_spread = coin_config.typical_spread_pct
        if typical_spread <= 0.05:
            spread_score = 10
            reasons.append("tight_spread")
        elif typical_spread > 0.1:
            spread_score = 0
            reasons.append("wide_spread")
        score += spread_score

        # 5. Historical win rate (max 20 points)
        wr_score = 10  # Default
        historical_wr = 0.5
        if symbol in self.historical_stats:
            historical_wr = self.historical_stats[symbol].get('win_rate', 0.5)
            if historical_wr >= 0.7:
                wr_score = 20
                reasons.append("high_historical_wr")
            elif historical_wr >= 0.6:
                wr_score = 15
            elif historical_wr < 0.45:
                wr_score = 0
                reasons.append("low_historical_wr")
        score += wr_score

        # 6. Regime score (max 10 points)
        regime_score = 5  # Default
        if regime_data and symbol in regime_data:
            regime = regime_data[symbol]
            if 'trend' in regime:
                regime_score = 10
                reasons.append("trending_regime")
            elif regime == 'choppy':
                regime_score = 0
                reasons.append("choppy_regime")
            elif regime == 'ranging':
                regime_score = 5
        score += regime_score

        # Cap at 100
        score = min(score, 100)

        return CoinScore(
            symbol=symbol,
            score=score,
            is_tradeable=score >= self.min_score,
            reasons=reasons,
            volatility_score=vol_score,
            spread_score=spread_score,
            volume_score=vol_ratio_score,
            historical_wr=historical_wr,
            regime_score=regime_score
        )

    def update_historical_stats(
        self,
        trades_df: pd.DataFrame
    ):
        """
        Update historical statistics from trade data.

        Args:
            trades_df: DataFrame with 'symbol', 'outcome' columns
        """
        if 'symbol' not in trades_df.columns:
            return

        for symbol in trades_df['symbol'].unique():
            symbol_trades = trades_df[trades_df['symbol'] == symbol]

            if len(symbol_trades) < 5:  # Need minimum samples
                continue

            self.historical_stats[symbol] = {
                'win_rate': symbol_trades['outcome'].mean(),
                'trades': len(symbol_trades),
                'avg_pnl': symbol_trades.get('pnl_pct', pd.Series([0])).mean(),
                'last_updated': datetime.now()
            }

    def save_stats(self, path: str = None):
        """Save historical stats"""
        path = Path(path) if path else Path(config.model_dir) / "coin_stats.pkl"
        with open(path, 'wb') as f:
            pickle.dump(self.historical_stats, f)

    def load_stats(self, path: str = None) -> bool:
        """Load historical stats"""
        path = Path(path) if path else Path(config.model_dir) / "coin_stats.pkl"
        if not path.exists():
            return False

        try:
            with open(path, 'rb') as f:
                self.historical_stats = pickle.load(f)
            return True
        except:
            return False


class MLCoinFilter(CoinFilter):
    """
    ML-enhanced coin filter.

    Uses ML to predict which coins are likely to produce winning trades.
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.scaler = StandardScaler() if HAS_ML else None
        self.is_trained = False

    def train(self, trades_df: pd.DataFrame) -> Dict:
        """
        Train ML model to predict coin tradability.

        Features per coin:
        - Recent win rate
        - Recent volume
        - Recent volatility
        - Time features
        - Regime
        """
        if not HAS_ML:
            return {'error': 'ML libraries not installed'}

        # Aggregate by coin and hour
        trades_df['hour'] = pd.to_datetime(trades_df['timestamp']).dt.hour

        features = []
        labels = []

        for symbol in trades_df['symbol'].unique():
            symbol_data = trades_df[trades_df['symbol'] == symbol]

            for hour in range(24):
                hour_data = symbol_data[symbol_data['hour'] == hour]

                if len(hour_data) < 3:
                    continue

                feat = {
                    'win_rate': hour_data['outcome'].mean(),
                    'trade_count': len(hour_data),
                    'avg_pnl': hour_data.get('pnl_pct', pd.Series([0])).mean(),
                    'hour_sin': np.sin(2 * np.pi * hour / 24),
                    'hour_cos': np.cos(2 * np.pi * hour / 24)
                }

                features.append(feat)
                labels.append(1 if hour_data['outcome'].mean() > 0.6 else 0)

        if len(features) < 50:
            return {'error': 'Not enough data'}

        X = pd.DataFrame(features)
        y = np.array(labels)

        X_scaled = self.scaler.fit_transform(X)

        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        self.is_trained = True

        return {'trained': True, 'samples': len(features)}
