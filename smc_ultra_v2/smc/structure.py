"""
SMC Ultra V2 - Market Structure Analysis
========================================
Analysiert die Marktstruktur (Higher Highs, Lower Lows, BOS, CHoCH).

Struktur-Konzepte:
- HH/HL = Uptrend (Higher Highs, Higher Lows)
- LH/LL = Downtrend (Lower Highs, Lower Lows)
- BOS = Break of Structure (Trend continuation)
- CHoCH = Change of Character (Trend reversal)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np


class StructureType(Enum):
    BULLISH = "bullish"      # HH/HL
    BEARISH = "bearish"      # LH/LL
    RANGING = "ranging"      # Mixed
    TRANSITIONING = "transitioning"  # Changing


class BreakType(Enum):
    BOS = "bos"              # Break of Structure
    CHOCH = "choch"          # Change of Character
    NONE = "none"


@dataclass
class SwingPoint:
    """Represents a swing high or low"""
    price: float
    index: int
    timestamp: pd.Timestamp
    is_high: bool
    is_broken: bool = False
    break_index: Optional[int] = None


@dataclass
class StructureBreak:
    """Represents a structure break event"""
    timestamp: pd.Timestamp
    price: float
    break_type: BreakType
    direction: str  # 'bullish' or 'bearish'
    swing_broken: SwingPoint
    strength: float = 0.0


@dataclass
class StructureState:
    """Current market structure state"""
    structure_type: StructureType
    swing_highs: List[SwingPoint]
    swing_lows: List[SwingPoint]
    last_break: Optional[StructureBreak]
    trend_strength: float  # 0-1

    @property
    def is_bullish(self) -> bool:
        return self.structure_type == StructureType.BULLISH

    @property
    def is_bearish(self) -> bool:
        return self.structure_type == StructureType.BEARISH


class MarketStructure:
    """
    Analyzes market structure using swing points.

    Methods:
    - Identify swing highs/lows
    - Detect BOS (Break of Structure) - Trend continuation
    - Detect CHoCH (Change of Character) - Trend reversal
    """

    def __init__(self, swing_length: int = 5):
        self.swing_length = swing_length

    def analyze(self, df: pd.DataFrame) -> StructureState:
        """
        Analyze complete market structure.

        Returns:
            StructureState with all structure information
        """
        if len(df) < self.swing_length * 4:
            return self._empty_state()

        # Find swing points
        swing_highs = self._find_swing_highs(df)
        swing_lows = self._find_swing_lows(df)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return self._empty_state()

        # Analyze structure
        structure_type = self._determine_structure(swing_highs, swing_lows)

        # Find last break
        last_break = self._find_last_break(df, swing_highs, swing_lows)

        # Calculate trend strength
        trend_strength = self._calc_trend_strength(swing_highs, swing_lows)

        # Update broken status
        self._update_broken_status(df, swing_highs, swing_lows)

        return StructureState(
            structure_type=structure_type,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            last_break=last_break,
            trend_strength=trend_strength
        )

    def _find_swing_highs(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Find all swing highs"""
        swing_highs = []
        highs = df['high'].values

        for i in range(self.swing_length, len(df) - self.swing_length):
            window = highs[i - self.swing_length:i + self.swing_length + 1]

            if highs[i] == max(window):
                swing_highs.append(SwingPoint(
                    price=highs[i],
                    index=i,
                    timestamp=df.iloc[i]['timestamp'] if 'timestamp' in df.columns else df.index[i],
                    is_high=True
                ))

        return swing_highs

    def _find_swing_lows(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Find all swing lows"""
        swing_lows = []
        lows = df['low'].values

        for i in range(self.swing_length, len(df) - self.swing_length):
            window = lows[i - self.swing_length:i + self.swing_length + 1]

            if lows[i] == min(window):
                swing_lows.append(SwingPoint(
                    price=lows[i],
                    index=i,
                    timestamp=df.iloc[i]['timestamp'] if 'timestamp' in df.columns else df.index[i],
                    is_high=False
                ))

        return swing_lows

    def _determine_structure(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint]
    ) -> StructureType:
        """Determine overall structure type"""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return StructureType.RANGING

        # Check last 3 swing points of each type
        recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
        recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows

        # Count higher highs/lows
        higher_highs = sum(
            1 for i in range(1, len(recent_highs))
            if recent_highs[i].price > recent_highs[i-1].price
        )
        higher_lows = sum(
            1 for i in range(1, len(recent_lows))
            if recent_lows[i].price > recent_lows[i-1].price
        )

        # Count lower highs/lows
        lower_highs = sum(
            1 for i in range(1, len(recent_highs))
            if recent_highs[i].price < recent_highs[i-1].price
        )
        lower_lows = sum(
            1 for i in range(1, len(recent_lows))
            if recent_lows[i].price < recent_lows[i-1].price
        )

        # Determine structure
        bullish_signals = higher_highs + higher_lows
        bearish_signals = lower_highs + lower_lows

        if bullish_signals >= 3 and bullish_signals > bearish_signals:
            return StructureType.BULLISH
        elif bearish_signals >= 3 and bearish_signals > bullish_signals:
            return StructureType.BEARISH
        elif abs(bullish_signals - bearish_signals) <= 1:
            return StructureType.RANGING
        else:
            return StructureType.TRANSITIONING

    def _find_last_break(
        self,
        df: pd.DataFrame,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint]
    ) -> Optional[StructureBreak]:
        """Find the most recent structure break"""
        breaks = []

        # Check for breaks of swing lows (bearish break)
        for swing_low in swing_lows[-5:]:  # Last 5 swing lows
            break_info = self._check_break(df, swing_low, is_break_above=False)
            if break_info:
                breaks.append(break_info)

        # Check for breaks of swing highs (bullish break)
        for swing_high in swing_highs[-5:]:  # Last 5 swing highs
            break_info = self._check_break(df, swing_high, is_break_above=True)
            if break_info:
                breaks.append(break_info)

        if not breaks:
            return None

        # Return most recent
        breaks.sort(key=lambda x: x.timestamp)
        return breaks[-1]

    def _check_break(
        self,
        df: pd.DataFrame,
        swing_point: SwingPoint,
        is_break_above: bool
    ) -> Optional[StructureBreak]:
        """Check if a swing point was broken"""
        if swing_point.index + 1 >= len(df):
            return None

        future_data = df.iloc[swing_point.index + 1:]

        for i, (_, candle) in enumerate(future_data.iterrows()):
            if is_break_above:
                # Break above = bullish
                if candle['close'] > swing_point.price:
                    # Determine if BOS or CHoCH
                    # CHoCH = break against previous structure
                    break_type = BreakType.BOS  # Simplified

                    return StructureBreak(
                        timestamp=candle['timestamp'] if 'timestamp' in candle else future_data.index[i],
                        price=candle['close'],
                        break_type=break_type,
                        direction='bullish',
                        swing_broken=swing_point,
                        strength=self._calc_break_strength(candle, swing_point)
                    )
            else:
                # Break below = bearish
                if candle['close'] < swing_point.price:
                    break_type = BreakType.BOS

                    return StructureBreak(
                        timestamp=candle['timestamp'] if 'timestamp' in candle else future_data.index[i],
                        price=candle['close'],
                        break_type=break_type,
                        direction='bearish',
                        swing_broken=swing_point,
                        strength=self._calc_break_strength(candle, swing_point)
                    )

        return None

    def _calc_break_strength(
        self,
        break_candle: pd.Series,
        swing_point: SwingPoint
    ) -> float:
        """Calculate strength of structure break"""
        # Bigger body = stronger break
        body = abs(break_candle['close'] - break_candle['open'])
        range_ = break_candle['high'] - break_candle['low']

        body_ratio = body / range_ if range_ > 0 else 0

        # How far past the level
        if swing_point.is_high:
            depth = (break_candle['close'] - swing_point.price) / swing_point.price
        else:
            depth = (swing_point.price - break_candle['close']) / swing_point.price

        strength = (body_ratio * 0.5) + (min(depth * 10, 0.5))

        return min(strength, 1.0)

    def _calc_trend_strength(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint]
    ) -> float:
        """Calculate trend strength 0-1"""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 0.0

        recent_highs = swing_highs[-4:]
        recent_lows = swing_lows[-4:]

        # Count consecutive higher/lower swings
        hh_count = sum(
            1 for i in range(1, len(recent_highs))
            if recent_highs[i].price > recent_highs[i-1].price
        )
        hl_count = sum(
            1 for i in range(1, len(recent_lows))
            if recent_lows[i].price > recent_lows[i-1].price
        )
        lh_count = sum(
            1 for i in range(1, len(recent_highs))
            if recent_highs[i].price < recent_highs[i-1].price
        )
        ll_count = sum(
            1 for i in range(1, len(recent_lows))
            if recent_lows[i].price < recent_lows[i-1].price
        )

        bull_strength = (hh_count + hl_count) / (len(recent_highs) + len(recent_lows) - 2)
        bear_strength = (lh_count + ll_count) / (len(recent_highs) + len(recent_lows) - 2)

        return max(bull_strength, bear_strength)

    def _update_broken_status(
        self,
        df: pd.DataFrame,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint]
    ):
        """Update is_broken status for all swing points"""
        current_price = df['close'].iloc[-1]

        for sh in swing_highs:
            if current_price > sh.price:
                sh.is_broken = True
                # Find break index
                future = df.iloc[sh.index + 1:]
                for i, (_, c) in enumerate(future.iterrows()):
                    if c['close'] > sh.price:
                        sh.break_index = sh.index + 1 + i
                        break

        for sl in swing_lows:
            if current_price < sl.price:
                sl.is_broken = True
                future = df.iloc[sl.index + 1:]
                for i, (_, c) in enumerate(future.iterrows()):
                    if c['close'] < sl.price:
                        sl.break_index = sl.index + 1 + i
                        break

    def _empty_state(self) -> StructureState:
        """Return empty structure state"""
        return StructureState(
            structure_type=StructureType.RANGING,
            swing_highs=[],
            swing_lows=[],
            last_break=None,
            trend_strength=0.0
        )

    def get_key_levels(
        self,
        df: pd.DataFrame,
        lookback: int = 50
    ) -> dict:
        """
        Get key structural levels.

        Returns:
            Dict with resistance and support levels
        """
        recent = df.tail(lookback)
        state = self.analyze(recent)

        resistance = []
        support = []

        # Unbroken swing highs = resistance
        for sh in state.swing_highs:
            if not sh.is_broken:
                resistance.append(sh.price)

        # Unbroken swing lows = support
        for sl in state.swing_lows:
            if not sl.is_broken:
                support.append(sl.price)

        return {
            'resistance': sorted(resistance),
            'support': sorted(support, reverse=True),
            'structure': state.structure_type.value,
            'trend_strength': state.trend_strength
        }
