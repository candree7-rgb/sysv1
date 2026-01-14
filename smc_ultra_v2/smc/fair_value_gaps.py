"""
SMC Ultra V2 - Fair Value Gap Detection
=======================================
Detektiert Fair Value Gaps (FVG) - Imbalance-Zonen im Markt.

Ein FVG entsteht wenn:
- Bullish FVG: Low von Kerze 0 > High von Kerze 2 (Gap up)
- Bearish FVG: High von Kerze 0 < Low von Kerze 2 (Gap down)

Diese Gaps werden oft "gefüllt" - Preis kehrt zurück.
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

import pandas as pd
import numpy as np


@dataclass
class FairValueGap:
    """Represents a Fair Value Gap zone"""
    top: float
    bottom: float
    timestamp: pd.Timestamp
    is_bullish: bool
    is_filled: bool = False
    fill_timestamp: Optional[pd.Timestamp] = None
    fill_percentage: float = 0.0  # How much of the gap was filled

    # Quality metrics
    size_pct: float = 0.0  # Gap size as % of price
    impulse_strength: float = 0.0  # Strength of impulse creating gap
    volume_ratio: float = 1.0

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def size(self) -> float:
        return self.top - self.bottom


class FVGDetector:
    """
    Detects Fair Value Gaps.

    Parameters:
        min_size_pct: Minimum gap size as % of price
        max_age_bars: Maximum age before FVG becomes invalid
    """

    def __init__(
        self,
        min_size_pct: float = 0.1,
        max_age_bars: int = 100
    ):
        self.min_size_pct = min_size_pct
        self.max_age_bars = max_age_bars

    def detect(self, df: pd.DataFrame) -> List[FairValueGap]:
        """
        Detect all FVGs in dataframe.

        Args:
            df: OHLCV DataFrame

        Returns:
            List of FairValueGap objects
        """
        if len(df) < 5:
            return []

        fvgs = []

        for i in range(2, len(df)):
            c0 = df.iloc[i]      # Current candle
            c1 = df.iloc[i-1]    # Middle candle
            c2 = df.iloc[i-2]    # Two bars ago

            # Bullish FVG: Current low > 2 bars ago high
            if c0['low'] > c2['high']:
                size = c0['low'] - c2['high']
                size_pct = (size / c0['close']) * 100

                if size_pct >= self.min_size_pct:
                    # Calculate impulse strength
                    impulse = abs(c1['close'] - c1['open'])
                    avg_body = df['close'].diff().abs().rolling(20).mean().iloc[i]
                    impulse_strength = impulse / avg_body if avg_body > 0 else 1

                    # Volume ratio
                    volume_ratio = 1.0
                    if 'volume_ratio' in df.columns:
                        volume_ratio = c1['volume_ratio']

                    fvgs.append(FairValueGap(
                        top=c0['low'],
                        bottom=c2['high'],
                        timestamp=c1['timestamp'] if 'timestamp' in c1 else df.index[i-1],
                        is_bullish=True,
                        size_pct=size_pct,
                        impulse_strength=min(impulse_strength, 3),
                        volume_ratio=volume_ratio
                    ))

            # Bearish FVG: Current high < 2 bars ago low
            if c0['high'] < c2['low']:
                size = c2['low'] - c0['high']
                size_pct = (size / c0['close']) * 100

                if size_pct >= self.min_size_pct:
                    impulse = abs(c1['close'] - c1['open'])
                    avg_body = df['close'].diff().abs().rolling(20).mean().iloc[i]
                    impulse_strength = impulse / avg_body if avg_body > 0 else 1

                    volume_ratio = 1.0
                    if 'volume_ratio' in df.columns:
                        volume_ratio = c1['volume_ratio']

                    fvgs.append(FairValueGap(
                        top=c2['low'],
                        bottom=c0['high'],
                        timestamp=c1['timestamp'] if 'timestamp' in c1 else df.index[i-1],
                        is_bullish=False,
                        size_pct=size_pct,
                        impulse_strength=min(impulse_strength, 3),
                        volume_ratio=volume_ratio
                    ))

        # Update fill status
        fvgs = self._update_fill_status(fvgs, df)

        return fvgs

    def _update_fill_status(
        self,
        fvgs: List[FairValueGap],
        df: pd.DataFrame
    ) -> List[FairValueGap]:
        """
        Update fill status for all FVGs.

        An FVG is filled when price returns to the gap.
        """
        for fvg in fvgs:
            # Find data after FVG formation
            if 'timestamp' in df.columns:
                future_data = df[df['timestamp'] > fvg.timestamp]
            else:
                fvg_idx = df.index.get_loc(fvg.timestamp) if fvg.timestamp in df.index else 0
                future_data = df.iloc[fvg_idx + 1:]

            max_fill = 0.0

            for _, candle in future_data.iterrows():
                if fvg.is_bullish:
                    # Bullish FVG filled when low enters gap
                    if candle['low'] <= fvg.top:
                        fill_depth = fvg.top - max(candle['low'], fvg.bottom)
                        fill_pct = fill_depth / fvg.size if fvg.size > 0 else 0
                        max_fill = max(max_fill, fill_pct)

                        if fill_pct >= 0.5:  # 50% fill = filled
                            fvg.is_filled = True
                            fvg.fill_timestamp = candle.get('timestamp')
                            fvg.fill_percentage = fill_pct
                            break
                else:
                    # Bearish FVG filled when high enters gap
                    if candle['high'] >= fvg.bottom:
                        fill_depth = min(candle['high'], fvg.top) - fvg.bottom
                        fill_pct = fill_depth / fvg.size if fvg.size > 0 else 0
                        max_fill = max(max_fill, fill_pct)

                        if fill_pct >= 0.5:
                            fvg.is_filled = True
                            fvg.fill_timestamp = candle.get('timestamp')
                            fvg.fill_percentage = fill_pct
                            break

            if not fvg.is_filled:
                fvg.fill_percentage = max_fill

        return fvgs

    def get_active(
        self,
        fvgs: List[FairValueGap],
        current_price: float,
        max_distance_pct: float = 2.0
    ) -> dict:
        """
        Get active (unfilled) FVGs near current price.

        Returns:
            Dict with 'bullish' and 'bearish' lists
        """
        active_bull = []
        active_bear = []

        for fvg in fvgs:
            if fvg.is_filled:
                continue

            # Calculate distance
            distance_pct = abs(current_price - fvg.mid) / current_price * 100

            if distance_pct <= max_distance_pct:
                if fvg.is_bullish:
                    active_bull.append(fvg)
                else:
                    active_bear.append(fvg)

        # Sort by size (larger = more significant)
        active_bull.sort(key=lambda x: -x.size_pct)
        active_bear.sort(key=lambda x: -x.size_pct)

        return {
            'bullish': active_bull,
            'bearish': active_bear,
            'near_bullish': len(active_bull) > 0,
            'near_bearish': len(active_bear) > 0,
            'in_bullish_fvg': any(fvg.bottom <= current_price <= fvg.top for fvg in active_bull),
            'in_bearish_fvg': any(fvg.bottom <= current_price <= fvg.top for fvg in active_bear)
        }


class InversionFVG:
    """
    Detects FVG Inversions - when a filled FVG acts as opposite zone.

    Example: Bullish FVG gets filled, then acts as resistance (bearish).
    """

    def find_inversions(
        self,
        fvgs: List[FairValueGap],
        df: pd.DataFrame
    ) -> List[FairValueGap]:
        """
        Find FVGs that have inverted.

        An inverted FVG:
        1. Was filled (price entered the gap)
        2. Price then rejected from the opposite side
        3. Now acts as the opposite type of zone
        """
        inversions = []

        for fvg in fvgs:
            if not fvg.is_filled or fvg.fill_timestamp is None:
                continue

            # Get data after fill
            if 'timestamp' in df.columns:
                post_fill = df[df['timestamp'] > fvg.fill_timestamp]
            else:
                continue

            if len(post_fill) < 5:
                continue

            # Look for rejection
            if fvg.is_bullish:
                # Filled bullish FVG -> now resistance
                # Check if price rejected down from the zone
                for _, candle in post_fill.head(10).iterrows():
                    if (candle['high'] >= fvg.bottom and
                        candle['close'] < fvg.bottom and
                        candle['close'] < candle['open']):
                        # Create inverted (bearish) zone
                        inversions.append(FairValueGap(
                            top=fvg.top,
                            bottom=fvg.bottom,
                            timestamp=fvg.fill_timestamp,
                            is_bullish=False,  # Inverted!
                            size_pct=fvg.size_pct,
                            impulse_strength=fvg.impulse_strength * 0.8
                        ))
                        break
            else:
                # Filled bearish FVG -> now support
                for _, candle in post_fill.head(10).iterrows():
                    if (candle['low'] <= fvg.top and
                        candle['close'] > fvg.top and
                        candle['close'] > candle['open']):
                        inversions.append(FairValueGap(
                            top=fvg.top,
                            bottom=fvg.bottom,
                            timestamp=fvg.fill_timestamp,
                            is_bullish=True,  # Inverted!
                            size_pct=fvg.size_pct,
                            impulse_strength=fvg.impulse_strength * 0.8
                        ))
                        break

        return inversions
