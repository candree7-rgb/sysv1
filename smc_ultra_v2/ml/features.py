"""
SMC Ultra V2 - Feature Engineering
==================================
Extrahiert Features für Machine Learning.

Features sind kategorisiert:
- SMC Features (Order Blocks, FVG, Liquidity, Structure)
- Technical Features (RSI, MACD, BB, etc.)
- Market Context (Regime, BTC correlation, etc.)
- Time Features (Session, Day of week, etc.)
- Coin-specific Features
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
import numpy as np

from config.settings import config
from smc import OrderBlock, FairValueGap, LiquiditySweep
from analysis import RegimeState, HTFBias, TrendDirection


@dataclass
class FeatureSet:
    """Complete feature set for ML prediction"""
    features: Dict[str, float]
    metadata: Dict[str, Any]

    def to_array(self, feature_names: List[str]) -> np.ndarray:
        """Convert to numpy array in specified order"""
        return np.array([self.features.get(name, 0) for name in feature_names])

    def to_dict(self) -> Dict[str, float]:
        return self.features


class FeatureExtractor:
    """
    Extracts features for ML model.

    Standardizes all features to similar scales for better model performance.
    """

    # All feature names (for consistent ordering)
    FEATURE_NAMES = [
        # SMC Features
        'has_bullish_ob', 'has_bearish_ob', 'ob_strength', 'ob_distance_pct',
        'ob_volume_ratio', 'ob_age_bars',
        'has_bullish_fvg', 'has_bearish_fvg', 'fvg_size_pct', 'fvg_impulse_strength',
        'has_bullish_sweep', 'has_bearish_sweep', 'sweep_strength', 'sweep_recency',

        # MTF Features
        'htf_bias', 'htf_strength', 'htf_adx', 'htf_ema_alignment',
        'mtf_aligned', 'ltf_confirmation', 'ltf_pattern_strength',

        # Structure Features
        'structure_bullish', 'structure_bearish', 'trend_strength',
        'recent_bos_bullish', 'recent_bos_bearish',

        # Technical Features
        'rsi', 'rsi_oversold', 'rsi_overbought', 'rsi_divergence',
        'macd_histogram', 'macd_cross_up', 'macd_cross_down',
        'bb_position', 'bb_squeeze', 'bb_expansion',
        'atr_percentile', 'volume_ratio',

        # Market Context
        'regime_trending', 'regime_ranging', 'regime_choppy', 'regime_volatile',
        'btc_correlation', 'btc_trend_aligned',

        # Time Features
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'is_london_session', 'is_ny_session', 'is_overlap_session',
        'session_weight',

        # Coin-specific
        'coin_historical_wr', 'coin_avg_move_pct', 'coin_tier',
        'spread_pct', 'funding_rate'
    ]

    def __init__(self):
        self.feature_stats = {}  # For normalization

    def extract(
        self,
        df: pd.DataFrame,
        htf_df: pd.DataFrame = None,
        order_blocks: List[OrderBlock] = None,
        fvgs: List[FairValueGap] = None,
        sweeps: List[LiquiditySweep] = None,
        regime: RegimeState = None,
        htf_bias: HTFBias = None,
        timestamp: datetime = None,
        coin_stats: Dict = None
    ) -> FeatureSet:
        """
        Extract all features for a single prediction.

        Args:
            df: LTF DataFrame
            htf_df: HTF DataFrame (optional)
            order_blocks: Detected order blocks
            fvgs: Detected fair value gaps
            sweeps: Detected liquidity sweeps
            regime: Current market regime
            htf_bias: HTF bias analysis
            timestamp: Current timestamp
            coin_stats: Historical stats for this coin

        Returns:
            FeatureSet with all features
        """
        features = {}

        # Current price and candle
        current = df.iloc[-1]
        price = current['close']

        # SMC Features
        smc_features = self._extract_smc_features(
            price, order_blocks, fvgs, sweeps
        )
        features.update(smc_features)

        # MTF Features
        mtf_features = self._extract_mtf_features(htf_bias, regime)
        features.update(mtf_features)

        # Structure Features
        structure_features = self._extract_structure_features(df)
        features.update(structure_features)

        # Technical Features
        tech_features = self._extract_technical_features(df)
        features.update(tech_features)

        # Market Context
        context_features = self._extract_context_features(regime, htf_df)
        features.update(context_features)

        # Time Features
        time_features = self._extract_time_features(timestamp or datetime.utcnow())
        features.update(time_features)

        # Coin-specific Features
        coin_features = self._extract_coin_features(coin_stats, df)
        features.update(coin_features)

        return FeatureSet(
            features=features,
            metadata={
                'timestamp': timestamp,
                'price': price,
                'symbol': coin_stats.get('symbol') if coin_stats else None
            }
        )

    def _extract_smc_features(
        self,
        price: float,
        order_blocks: List[OrderBlock],
        fvgs: List[FairValueGap],
        sweeps: List[LiquiditySweep]
    ) -> Dict[str, float]:
        """Extract SMC-related features"""
        features = {
            'has_bullish_ob': 0, 'has_bearish_ob': 0, 'ob_strength': 0,
            'ob_distance_pct': 0, 'ob_volume_ratio': 0, 'ob_age_bars': 0,
            'has_bullish_fvg': 0, 'has_bearish_fvg': 0, 'fvg_size_pct': 0,
            'fvg_impulse_strength': 0,
            'has_bullish_sweep': 0, 'has_bearish_sweep': 0,
            'sweep_strength': 0, 'sweep_recency': 0
        }

        # Order Blocks
        if order_blocks:
            active_obs = [ob for ob in order_blocks if not ob.is_mitigated]

            for ob in active_obs:
                distance_pct = abs(price - ob.mid) / price * 100

                if distance_pct <= 2.0:  # Within 2%
                    if ob.is_bullish:
                        features['has_bullish_ob'] = 1
                    else:
                        features['has_bearish_ob'] = 1

                    features['ob_strength'] = max(features['ob_strength'], ob.strength)
                    features['ob_distance_pct'] = distance_pct
                    features['ob_volume_ratio'] = ob.volume_ratio
                    features['ob_age_bars'] = min(ob.age_bars / 100, 1)  # Normalize

        # FVGs
        if fvgs:
            active_fvgs = [f for f in fvgs if not f.is_filled]

            for fvg in active_fvgs:
                distance_pct = abs(price - fvg.mid) / price * 100

                if distance_pct <= 2.0:
                    if fvg.is_bullish:
                        features['has_bullish_fvg'] = 1
                    else:
                        features['has_bearish_fvg'] = 1

                    features['fvg_size_pct'] = max(features['fvg_size_pct'], fvg.size_pct)
                    features['fvg_impulse_strength'] = max(
                        features['fvg_impulse_strength'],
                        min(fvg.impulse_strength / 3, 1)
                    )

        # Sweeps
        if sweeps:
            recent_sweeps = sweeps[-5:]  # Last 5 sweeps

            for sweep in recent_sweeps:
                if sweep.is_bullish:
                    features['has_bullish_sweep'] = 1
                else:
                    features['has_bearish_sweep'] = 1

                features['sweep_strength'] = max(
                    features['sweep_strength'],
                    sweep.reversal_strength
                )
                features['sweep_recency'] = 1  # Has recent sweep

        return features

    def _extract_mtf_features(
        self,
        htf_bias: HTFBias,
        regime: RegimeState
    ) -> Dict[str, float]:
        """Extract multi-timeframe features"""
        features = {
            'htf_bias': 0, 'htf_strength': 0, 'htf_adx': 0,
            'htf_ema_alignment': 0, 'mtf_aligned': 0,
            'ltf_confirmation': 0, 'ltf_pattern_strength': 0
        }

        if htf_bias:
            # HTF Bias direction: 1 = bullish, -1 = bearish, 0 = neutral
            if htf_bias.direction == TrendDirection.BULLISH:
                features['htf_bias'] = 1
            elif htf_bias.direction == TrendDirection.BEARISH:
                features['htf_bias'] = -1

            features['htf_strength'] = htf_bias.strength / 100
            features['htf_adx'] = min(htf_bias.adx / 50, 1)  # Normalize to 0-1

            # EMA alignment score
            alignment_scores = {
                'perfect_bull': 1.0, 'bull': 0.7,
                'neutral': 0.5, 'bear': 0.3, 'perfect_bear': 0.0
            }
            features['htf_ema_alignment'] = alignment_scores.get(htf_bias.ema_alignment, 0.5)

        if regime:
            # Check if regime aligns with bias
            if htf_bias:
                if (htf_bias.is_bullish and 'up' in regime.regime.value) or \
                   (htf_bias.is_bearish and 'down' in regime.regime.value):
                    features['mtf_aligned'] = 1

        return features

    def _extract_structure_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract market structure features"""
        features = {
            'structure_bullish': 0, 'structure_bearish': 0,
            'trend_strength': 0, 'recent_bos_bullish': 0, 'recent_bos_bearish': 0
        }

        if len(df) < 20:
            return features

        # Simple structure analysis
        recent = df.tail(20)

        # Higher highs / lows
        highs = recent['high'].values
        lows = recent['low'].values

        hh = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        hl = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
        lh = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
        ll = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])

        total = len(highs) - 1

        if hh + hl > lh + ll:
            features['structure_bullish'] = 1
            features['trend_strength'] = (hh + hl) / (total * 2)
        elif lh + ll > hh + hl:
            features['structure_bearish'] = 1
            features['trend_strength'] = (lh + ll) / (total * 2)

        # Recent break of structure
        current_high = recent['high'].iloc[-1]
        current_low = recent['low'].iloc[-1]
        prev_high = recent['high'].iloc[:-5].max()
        prev_low = recent['low'].iloc[:-5].min()

        if current_high > prev_high:
            features['recent_bos_bullish'] = 1
        if current_low < prev_low:
            features['recent_bos_bearish'] = 1

        return features

    def _extract_technical_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract technical indicator features"""
        features = {
            'rsi': 0.5, 'rsi_oversold': 0, 'rsi_overbought': 0, 'rsi_divergence': 0,
            'macd_histogram': 0, 'macd_cross_up': 0, 'macd_cross_down': 0,
            'bb_position': 0.5, 'bb_squeeze': 0, 'bb_expansion': 0,
            'atr_percentile': 0.5, 'volume_ratio': 1
        }

        if len(df) < 20:
            return features

        current = df.iloc[-1]

        # RSI
        if 'rsi' in df.columns:
            rsi = current['rsi']
            features['rsi'] = rsi / 100
            features['rsi_oversold'] = 1 if rsi < 30 else 0
            features['rsi_overbought'] = 1 if rsi > 70 else 0

            # RSI divergence (simplified)
            price_trend = df['close'].iloc[-5:].mean() - df['close'].iloc[-10:-5].mean()
            rsi_trend = df['rsi'].iloc[-5:].mean() - df['rsi'].iloc[-10:-5].mean()
            if (price_trend > 0 and rsi_trend < 0) or (price_trend < 0 and rsi_trend > 0):
                features['rsi_divergence'] = 1

        # MACD
        if 'macd_hist' in df.columns:
            features['macd_histogram'] = np.clip(current['macd_hist'] / 100, -1, 1)

            if len(df) >= 2:
                prev_hist = df['macd_hist'].iloc[-2]
                curr_hist = current['macd_hist']
                if prev_hist < 0 and curr_hist > 0:
                    features['macd_cross_up'] = 1
                if prev_hist > 0 and curr_hist < 0:
                    features['macd_cross_down'] = 1

        # Bollinger Bands
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            bb_range = current['bb_upper'] - current['bb_lower']
            if bb_range > 0:
                features['bb_position'] = (current['close'] - current['bb_lower']) / bb_range

            if 'bb_width' in df.columns:
                avg_width = df['bb_width'].rolling(20).mean().iloc[-1]
                current_width = current['bb_width']
                features['bb_squeeze'] = 1 if current_width < avg_width * 0.7 else 0
                features['bb_expansion'] = 1 if current_width > avg_width * 1.5 else 0

        # ATR Percentile
        if 'atr' in df.columns:
            atr_values = df['atr'].dropna()
            if len(atr_values) > 0:
                percentile = (atr_values < current['atr']).sum() / len(atr_values)
                features['atr_percentile'] = percentile

        # Volume Ratio
        if 'volume_ratio' in df.columns:
            features['volume_ratio'] = min(current['volume_ratio'], 3) / 3

        return features

    def _extract_context_features(
        self,
        regime: RegimeState,
        htf_df: pd.DataFrame = None
    ) -> Dict[str, float]:
        """Extract market context features"""
        features = {
            'regime_trending': 0, 'regime_ranging': 0,
            'regime_choppy': 0, 'regime_volatile': 0,
            'btc_correlation': 0.5, 'btc_trend_aligned': 0
        }

        if regime:
            if 'trend' in regime.regime.value:
                features['regime_trending'] = 1
            elif regime.regime.value == 'ranging':
                features['regime_ranging'] = 1
            elif regime.regime.value == 'choppy':
                features['regime_choppy'] = 1
            elif regime.regime.value == 'high_volatility':
                features['regime_volatile'] = 1

        return features

    def _extract_time_features(self, timestamp: datetime) -> Dict[str, float]:
        """Extract time-based features"""
        hour = timestamp.hour
        day = timestamp.weekday()

        # Cyclical encoding for hour and day
        features = {
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'day_sin': np.sin(2 * np.pi * day / 7),
            'day_cos': np.cos(2 * np.pi * day / 7),
            'is_london_session': 1 if 8 <= hour < 16 else 0,
            'is_ny_session': 1 if 13 <= hour < 21 else 0,
            'is_overlap_session': 1 if 13 <= hour < 16 else 0,
            'session_weight': 1.0
        }

        # Session weight
        if 13 <= hour < 16:
            features['session_weight'] = 1.2
        elif 8 <= hour < 16 or 16 <= hour < 21:
            features['session_weight'] = 1.0
        elif 0 <= hour < 8:
            features['session_weight'] = 0.7
        else:
            features['session_weight'] = 0.5

        # Day weight
        day_weights = {0: 0.9, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.8, 5: 0.5, 6: 0.5}
        features['session_weight'] *= day_weights.get(day, 1.0)

        return features

    def _extract_coin_features(
        self,
        coin_stats: Dict,
        df: pd.DataFrame
    ) -> Dict[str, float]:
        """Extract coin-specific features"""
        features = {
            'coin_historical_wr': 0.5, 'coin_avg_move_pct': 0,
            'coin_tier': 0.5, 'spread_pct': 0, 'funding_rate': 0
        }

        if coin_stats:
            features['coin_historical_wr'] = coin_stats.get('win_rate', 0.5)
            features['coin_avg_move_pct'] = min(coin_stats.get('avg_move', 0) / 5, 1)
            features['coin_tier'] = coin_stats.get('tier', 2) / 4  # Normalize 1-4 to 0.25-1
            features['spread_pct'] = min(coin_stats.get('spread', 0) / 0.2, 1)
            features['funding_rate'] = np.clip(coin_stats.get('funding', 0) * 100, -1, 1)

        return features
