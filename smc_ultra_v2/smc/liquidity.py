"""
SMC Ultra V2 - Liquidity Detection
==================================
Detektiert Liquidity Sweeps und Liquidity Pools.

Liquidity Concepts:
- Equal Highs/Lows = Liquidity pools (stop losses)
- Sweep = Price takes out liquidity then reverses
- Inducement = Small liquidity grab before bigger move
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np


@dataclass
class LiquidityLevel:
    """Represents a liquidity level (equal highs/lows)"""
    price: float
    timestamp: pd.Timestamp
    is_high: bool  # True for equal highs, False for equal lows
    touches: int = 2  # Number of times price touched this level
    is_swept: bool = False
    sweep_timestamp: Optional[pd.Timestamp] = None


@dataclass
class LiquiditySweep:
    """Represents a liquidity sweep event"""
    timestamp: pd.Timestamp
    sweep_price: float
    liquidity_level: float
    is_bullish: bool  # True if swept low (bullish setup), False if swept high
    sweep_depth: float  # How far past the level
    reversal_strength: float  # Strength of reversal after sweep
    close_price: float


class LiquidityDetector:
    """
    Detects liquidity levels and sweeps.

    A sweep is a powerful reversal signal when:
    1. Price breaks a key level (takes liquidity)
    2. Price quickly reverses back
    3. Close is on opposite side of the level
    """

    def __init__(
        self,
        swing_lookback: int = 5,
        equal_threshold_pct: float = 0.1,
        sweep_lookback: int = 20
    ):
        self.swing_lookback = swing_lookback
        self.equal_threshold = equal_threshold_pct / 100
        self.sweep_lookback = sweep_lookback

    def find_liquidity_levels(
        self,
        df: pd.DataFrame,
        lookback: int = 50
    ) -> List[LiquidityLevel]:
        """
        Find equal highs and equal lows (liquidity pools).
        """
        if len(df) < lookback:
            return []

        levels = []
        recent = df.tail(lookback)

        # Find swing highs
        swing_highs = self._find_swing_points(recent, is_high=True)

        # Find swing lows
        swing_lows = self._find_swing_points(recent, is_high=False)

        # Find equal highs
        for i, (idx1, high1) in enumerate(swing_highs):
            for idx2, high2 in swing_highs[i+1:]:
                if abs(high1 - high2) / high1 < self.equal_threshold:
                    levels.append(LiquidityLevel(
                        price=(high1 + high2) / 2,
                        timestamp=recent.iloc[idx2]['timestamp'] if 'timestamp' in recent.columns else recent.index[idx2],
                        is_high=True,
                        touches=2
                    ))
                    break

        # Find equal lows
        for i, (idx1, low1) in enumerate(swing_lows):
            for idx2, low2 in swing_lows[i+1:]:
                if abs(low1 - low2) / low1 < self.equal_threshold:
                    levels.append(LiquidityLevel(
                        price=(low1 + low2) / 2,
                        timestamp=recent.iloc[idx2]['timestamp'] if 'timestamp' in recent.columns else recent.index[idx2],
                        is_high=False,
                        touches=2
                    ))
                    break

        return levels

    def _find_swing_points(
        self,
        df: pd.DataFrame,
        is_high: bool
    ) -> List[Tuple[int, float]]:
        """Find swing highs or lows"""
        points = []
        col = 'high' if is_high else 'low'
        values = df[col].values

        for i in range(self.swing_lookback, len(df) - self.swing_lookback):
            window = values[i - self.swing_lookback:i + self.swing_lookback + 1]

            if is_high:
                if values[i] == max(window):
                    points.append((i, values[i]))
            else:
                if values[i] == min(window):
                    points.append((i, values[i]))

        return points

    def find_sweeps(
        self,
        df: pd.DataFrame,
        lookback_bars: int = None
    ) -> List[LiquiditySweep]:
        """
        Find liquidity sweeps in dataframe.

        A sweep occurs when:
        1. Price breaks recent swing high/low
        2. Price reverses and closes back
        """
        lookback = lookback_bars or self.sweep_lookback
        if len(df) < lookback + 5:
            return []

        sweeps = []

        for i in range(lookback, len(df)):
            candle = df.iloc[i]
            recent = df.iloc[i - lookback:i]

            # Get recent swing high/low
            recent_high = recent['high'].max()
            recent_low = recent['low'].min()

            # Bullish Sweep: Break low, close above
            if (candle['low'] < recent_low and
                candle['close'] > recent_low and
                candle['close'] > candle['open']):

                sweep_depth = recent_low - candle['low']
                body = candle['close'] - candle['open']
                reversal_strength = body / (candle['high'] - candle['low']) if (candle['high'] - candle['low']) > 0 else 0

                sweeps.append(LiquiditySweep(
                    timestamp=candle['timestamp'] if 'timestamp' in candle else df.index[i],
                    sweep_price=candle['low'],
                    liquidity_level=recent_low,
                    is_bullish=True,
                    sweep_depth=sweep_depth,
                    reversal_strength=reversal_strength,
                    close_price=candle['close']
                ))

            # Bearish Sweep: Break high, close below
            if (candle['high'] > recent_high and
                candle['close'] < recent_high and
                candle['close'] < candle['open']):

                sweep_depth = candle['high'] - recent_high
                body = candle['open'] - candle['close']
                reversal_strength = body / (candle['high'] - candle['low']) if (candle['high'] - candle['low']) > 0 else 0

                sweeps.append(LiquiditySweep(
                    timestamp=candle['timestamp'] if 'timestamp' in candle else df.index[i],
                    sweep_price=candle['high'],
                    liquidity_level=recent_high,
                    is_bullish=False,
                    sweep_depth=sweep_depth,
                    reversal_strength=reversal_strength,
                    close_price=candle['close']
                ))

        return sweeps

    def has_recent_sweep(
        self,
        sweeps: List[LiquiditySweep],
        timestamp: pd.Timestamp,
        lookback_bars: int = 5,
        bar_minutes: int = 5
    ) -> dict:
        """
        Check if there was a recent sweep.

        Returns:
            Dict with 'bullish_sweep' and 'bearish_sweep' booleans
        """
        lookback_time = pd.Timedelta(minutes=lookback_bars * bar_minutes)
        min_time = timestamp - lookback_time

        recent = [s for s in sweeps if min_time <= s.timestamp <= timestamp]

        return {
            'bullish_sweep': any(s.is_bullish for s in recent),
            'bearish_sweep': any(not s.is_bullish for s in recent),
            'recent_sweeps': recent
        }

    def get_sweep_quality(self, sweep: LiquiditySweep) -> float:
        """
        Rate sweep quality 0-1.

        Higher quality:
        - Deeper sweep
        - Stronger reversal
        - Clean close opposite side
        """
        quality = 0.5

        # Reversal strength (max 0.3)
        quality += min(sweep.reversal_strength * 0.3, 0.3)

        # Sweep depth relative to level (max 0.2)
        depth_pct = sweep.sweep_depth / sweep.liquidity_level
        quality += min(depth_pct * 10, 0.2)

        return min(quality, 1.0)


class InducementDetector:
    """
    Detects inducement patterns - small liquidity grabs before bigger moves.

    Inducement is a trap:
    1. Small break of structure (takes out early entries)
    2. Reversal back
    3. Then the REAL move happens
    """

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def find_inducements(
        self,
        df: pd.DataFrame,
        sweeps: List[LiquiditySweep]
    ) -> List[dict]:
        """
        Find inducement patterns.

        Returns list of inducement events with direction and timing.
        """
        inducements = []

        for sweep in sweeps:
            # Find the candle index for this sweep
            if 'timestamp' not in df.columns:
                continue

            sweep_idx = df[df['timestamp'] == sweep.timestamp].index
            if len(sweep_idx) == 0:
                continue

            sweep_idx = sweep_idx[0]

            # Look at next few candles
            if sweep_idx + 5 >= len(df):
                continue

            next_candles = df.iloc[sweep_idx + 1:sweep_idx + 6]

            # Check for continuation in sweep direction (confirms it was inducement)
            if sweep.is_bullish:
                # After bullish sweep, look for higher prices
                continuation = next_candles['high'].max() > sweep.close_price * 1.005
            else:
                continuation = next_candles['low'].min() < sweep.close_price * 0.995

            if continuation:
                inducements.append({
                    'timestamp': sweep.timestamp,
                    'type': 'bullish' if sweep.is_bullish else 'bearish',
                    'sweep_price': sweep.sweep_price,
                    'quality': self._calc_inducement_quality(sweep, next_candles)
                })

        return inducements

    def _calc_inducement_quality(
        self,
        sweep: LiquiditySweep,
        follow_up: pd.DataFrame
    ) -> float:
        """Calculate inducement quality"""
        quality = sweep.reversal_strength

        # Check follow-through
        if sweep.is_bullish:
            follow_pct = (follow_up['high'].max() - sweep.close_price) / sweep.close_price
        else:
            follow_pct = (sweep.close_price - follow_up['low'].min()) / sweep.close_price

        quality += min(follow_pct * 10, 0.3)

        return min(quality, 1.0)
