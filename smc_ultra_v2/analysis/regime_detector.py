"""
SMC Ultra V2 - Market Regime Detector
=====================================
Erkennt automatisch das Marktregime und passt die Strategie an.

KRITISCH für "jede Marktlage" - ohne das funktioniert nichts!
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

from config.settings import config


class MarketRegime(Enum):
    """Market regime types"""
    STRONG_TREND_UP = "strong_trend_up"
    STRONG_TREND_DOWN = "strong_trend_down"
    WEAK_TREND_UP = "weak_trend_up"
    WEAK_TREND_DOWN = "weak_trend_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    CHOPPY = "choppy"
    LOW_VOLUME = "low_volume"


@dataclass
class RegimeState:
    """Current regime state with parameters"""
    regime: MarketRegime
    confidence: float  # 0-100
    should_trade: bool
    direction: str  # 'long_only', 'short_only', 'both', 'none'
    min_confidence: int
    tp_multiplier: float
    leverage_multiplier: float
    reason: str = ""

    # Detailed metrics
    adx: float = 0.0
    choppiness: float = 0.0
    volatility_ratio: float = 1.0
    volume_ratio: float = 1.0
    trend_strength: float = 0.0


class RegimeDetector:
    """
    Detects market regime using multiple indicators.

    Indicators used:
    - ADX: Trend strength
    - Choppiness Index: Trend vs Range
    - Bollinger Band Width: Volatility
    - Volume: Market activity
    - Price vs EMAs: Trend direction
    """

    def __init__(self):
        self.config = config.regime

    def detect(self, df: pd.DataFrame) -> RegimeState:
        """
        Main detection method.

        Args:
            df: DataFrame with OHLCV data and indicators

        Returns:
            RegimeState with all parameters
        """
        if len(df) < 50:
            return self._default_state("insufficient_data")

        # Calculate regime indicators
        adx = self._calc_adx(df)
        chop = self._calc_choppiness(df)
        vol_ratio = self._calc_volatility_ratio(df)
        volume_ratio = self._calc_volume_ratio(df)
        trend_dir, trend_strength = self._calc_trend(df)

        # Classify regime
        regime = self._classify_regime(
            adx=adx,
            choppiness=chop,
            volatility_ratio=vol_ratio,
            volume_ratio=volume_ratio,
            trend_direction=trend_dir,
            trend_strength=trend_strength
        )

        # Get regime settings
        settings = self.config.regime_settings.get(
            regime.value,
            {'trade': False, 'reason': 'unknown_regime'}
        )

        return RegimeState(
            regime=regime,
            confidence=self._calc_regime_confidence(adx, chop, vol_ratio),
            should_trade=settings.get('trade', False),
            direction=settings.get('direction', 'none'),
            min_confidence=settings.get('min_confidence', 95),
            tp_multiplier=settings.get('tp_multiplier', 1.0),
            leverage_multiplier=settings.get('leverage_mult', 1.0),
            reason=settings.get('reason', ''),
            adx=adx,
            choppiness=chop,
            volatility_ratio=vol_ratio,
            volume_ratio=volume_ratio,
            trend_strength=trend_strength
        )

    def _calc_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average Directional Index.

        ADX > 25: Trending
        ADX > 40: Strong trend
        ADX < 20: Ranging/Weak
        """
        # Check if ADX already exists in dataframe
        if 'adx' in df.columns:
            val = df['adx'].iloc[-1]
            return val if not np.isnan(val) else 15.0

        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        # Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Keep index alignment!
        plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr

        # ADX - handle division by zero
        di_sum = plus_di + minus_di
        di_sum = di_sum.replace(0, np.nan)  # Avoid division by zero
        dx = 100 * abs(plus_di - minus_di) / di_sum
        adx = dx.rolling(period).mean()

        result = adx.iloc[-1]
        return result if not np.isnan(result) else 15.0  # Default to weak trend instead of 0

    def _calc_choppiness(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Choppiness Index.

        >61.8: Choppy/Ranging (BAD for trading)
        <38.2: Trending (GOOD for trading)
        38.2-61.8: Mixed
        """
        high = df['high'].rolling(period).max()
        low = df['low'].rolling(period).min()

        # ATR sum
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        ], axis=1).max(axis=1)
        atr_sum = tr.rolling(period).sum()

        # Choppiness
        chop = 100 * np.log10(atr_sum / (high - low + 0.0001)) / np.log10(period)

        return chop.iloc[-1] if not np.isnan(chop.iloc[-1]) else 50

    def _calc_volatility_ratio(self, df: pd.DataFrame) -> float:
        """
        Calculate current volatility vs historical average.

        >1.5: High volatility
        <0.5: Low volatility
        """
        if 'atr' not in df.columns:
            tr = pd.concat([
                df['high'] - df['low'],
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
        else:
            atr = df['atr']

        current_atr = atr.iloc[-1]
        avg_atr = atr.rolling(50).mean().iloc[-1]

        if avg_atr == 0:
            return 1.0

        return current_atr / avg_atr

    def _calc_volume_ratio(self, df: pd.DataFrame) -> float:
        """Calculate current volume vs average"""
        if 'volume_ratio' in df.columns:
            return df['volume_ratio'].iloc[-1]

        current_vol = df['volume'].iloc[-1]
        avg_vol = df['volume'].rolling(20).mean().iloc[-1]

        if avg_vol == 0:
            return 1.0

        return current_vol / avg_vol

    def _calc_trend(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Calculate trend direction and strength.

        Returns:
            (direction, strength) where direction is 'up', 'down', 'neutral'
        """
        close = df['close'].iloc[-1]

        # Use EMAs if available
        if 'ema20' in df.columns and 'ema50' in df.columns:
            ema20 = df['ema20'].iloc[-1]
            ema50 = df['ema50'].iloc[-1]
            ema200 = df['ema200'].iloc[-1] if 'ema200' in df.columns else ema50
        else:
            ema20 = df['close'].ewm(span=20).mean().iloc[-1]
            ema50 = df['close'].ewm(span=50).mean().iloc[-1]
            ema200 = df['close'].ewm(span=200).mean().iloc[-1]

        # Calculate strength
        bullish_score = 0
        bearish_score = 0

        # Price vs EMAs
        if close > ema20:
            bullish_score += 1
        else:
            bearish_score += 1

        if close > ema50:
            bullish_score += 1
        else:
            bearish_score += 1

        if close > ema200:
            bullish_score += 1
        else:
            bearish_score += 1

        # EMA alignment
        if ema20 > ema50 > ema200:
            bullish_score += 2
        elif ema20 < ema50 < ema200:
            bearish_score += 2

        # Higher highs / Lower lows
        recent_highs = df['high'].tail(20)
        recent_lows = df['low'].tail(20)

        if recent_highs.iloc[-1] > recent_highs.iloc[-10]:
            bullish_score += 1
        if recent_lows.iloc[-1] > recent_lows.iloc[-10]:
            bullish_score += 1
        if recent_highs.iloc[-1] < recent_highs.iloc[-10]:
            bearish_score += 1
        if recent_lows.iloc[-1] < recent_lows.iloc[-10]:
            bearish_score += 1

        # Determine direction and strength
        total = bullish_score + bearish_score

        # DEBUG: Log scores (first 5 only)
        if not hasattr(self, '_trend_debug_logged'):
            self._trend_debug_logged = 0
        if self._trend_debug_logged < 5:
            print(f"  [TREND DEBUG] bullish={bullish_score}, bearish={bearish_score}, total={total}")
            self._trend_debug_logged += 1

        if total == 0:
            return 'neutral', 0

        # Simple majority determines direction
        if bullish_score > bearish_score:
            return 'up', bullish_score / total
        elif bearish_score > bullish_score:
            return 'down', bearish_score / total
        else:
            return 'neutral', 0.5

    def _classify_regime(
        self,
        adx: float,
        choppiness: float,
        volatility_ratio: float,
        volume_ratio: float,
        trend_direction: str,
        trend_strength: float
    ) -> MarketRegime:
        """
        Classify market regime based on indicators.
        """
        # DEBUG: Track why we're not getting trends
        if not hasattr(self, '_debug_logged'):
            self._debug_logged = 0
        if self._debug_logged < 5:
            print(f"  [REGIME DEBUG] ADX={adx:.1f}, trend_dir={trend_direction}, chop={choppiness:.1f}, vol={volume_ratio:.2f}")
            self._debug_logged += 1

        # Low volume = No trade
        if volume_ratio < self.config.low_volume_threshold:
            return MarketRegime.LOW_VOLUME

        # Choppy market = No trade
        if choppiness > self.config.choppiness_threshold:
            return MarketRegime.CHOPPY

        # High volatility
        if volatility_ratio > self.config.high_volatility_mult:
            return MarketRegime.HIGH_VOLATILITY

        # Strong trend
        if adx > self.config.adx_strong_trend:
            if trend_direction == 'up':
                return MarketRegime.STRONG_TREND_UP
            elif trend_direction == 'down':
                return MarketRegime.STRONG_TREND_DOWN

        # Weak trend
        if adx > self.config.adx_weak_trend:
            if trend_direction == 'up':
                return MarketRegime.WEAK_TREND_UP
            elif trend_direction == 'down':
                return MarketRegime.WEAK_TREND_DOWN

        # Default to ranging
        return MarketRegime.RANGING

    def _calc_regime_confidence(
        self,
        adx: float,
        choppiness: float,
        volatility_ratio: float
    ) -> float:
        """
        Calculate confidence in regime classification.
        """
        confidence = 50.0

        # ADX clarity
        if adx > 40:
            confidence += 20
        elif adx > 30:
            confidence += 15
        elif adx > 25:
            confidence += 10
        elif adx < 15:
            confidence += 5  # Clear ranging

        # Choppiness clarity
        if choppiness < 38.2:
            confidence += 15  # Clear trend
        elif choppiness > 61.8:
            confidence += 10  # Clear chop

        # Volatility normal
        if 0.8 <= volatility_ratio <= 1.3:
            confidence += 10

        return min(confidence, 100)

    def _default_state(self, reason: str) -> RegimeState:
        """Return default no-trade state"""
        return RegimeState(
            regime=MarketRegime.CHOPPY,
            confidence=0,
            should_trade=False,
            direction='none',
            min_confidence=100,
            tp_multiplier=1.0,
            leverage_multiplier=0.0,
            reason=reason
        )


class MultiTimeframeRegime:
    """
    Analyzes regime across multiple timeframes.

    Confluence across TFs = Higher confidence
    """

    def __init__(self):
        self.detector = RegimeDetector()

    def analyze(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, RegimeState]:
        """
        Analyze regime for each timeframe.
        """
        result = {}

        for tf, df in data.items():
            result[tf] = self.detector.detect(df)

        return result

    def get_consensus(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> RegimeState:
        """
        Get consensus regime across timeframes.

        HTF has more weight than LTF.
        """
        regimes = self.analyze(data)

        if not regimes:
            return self.detector._default_state("no_data")

        # Weight by timeframe (higher TF = more weight)
        weights = {'240': 3, '60': 2.5, '30': 2, '15': 1.5, '5': 1, '1': 0.5}

        # Check if any TF says no trade
        for tf, state in regimes.items():
            if not state.should_trade:
                # If HTF says no, definitely no
                if int(tf) >= 15:
                    return state

        # Find dominant direction
        bull_weight = 0
        bear_weight = 0
        total_weight = 0

        for tf, state in regimes.items():
            w = weights.get(tf, 1)
            total_weight += w

            if 'up' in state.regime.value:
                bull_weight += w * state.confidence
            elif 'down' in state.regime.value:
                bear_weight += w * state.confidence

        # Determine consensus
        htf_key = max(regimes.keys(), key=lambda x: int(x) if x.isdigit() else 0)
        htf_state = regimes[htf_key]

        # Adjust confidence based on alignment
        alignment = abs(bull_weight - bear_weight) / max(bull_weight + bear_weight, 1)

        consensus = RegimeState(
            regime=htf_state.regime,
            confidence=htf_state.confidence * (0.5 + alignment * 0.5),
            should_trade=htf_state.should_trade and alignment > 0.3,
            direction=htf_state.direction,
            min_confidence=htf_state.min_confidence,
            tp_multiplier=htf_state.tp_multiplier,
            leverage_multiplier=htf_state.leverage_multiplier * alignment,
            reason=f"alignment_{alignment:.2f}",
            adx=htf_state.adx,
            choppiness=htf_state.choppiness,
            volatility_ratio=htf_state.volatility_ratio,
            volume_ratio=htf_state.volume_ratio,
            trend_strength=htf_state.trend_strength
        )

        return consensus
