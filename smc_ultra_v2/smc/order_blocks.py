"""
SMC Ultra V2 - Order Block Detection
====================================
Detektiert Order Blocks (OB) - Zonen wo institutionelle Orders platziert wurden.

Ein Order Block ist:
- Die letzte bearish Kerze vor einem bullish Impuls (Bullish OB)
- Die letzte bullish Kerze vor einem bearish Impuls (Bearish OB)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np


@dataclass
class OrderBlock:
    """Represents an Order Block zone"""
    top: float
    bottom: float
    timestamp: pd.Timestamp
    is_bullish: bool
    is_mitigated: bool = False
    mitigation_timestamp: Optional[pd.Timestamp] = None

    # Quality metrics
    strength: float = 1.0  # 0-1, based on impulse strength
    volume_ratio: float = 1.0  # Volume vs average
    impulse_size: float = 0.0  # Size of impulse move
    touches: int = 0  # Number of times price touched zone
    age_bars: int = 0  # How old is the OB

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def size(self) -> float:
        return self.top - self.bottom

    @property
    def size_pct(self) -> float:
        return (self.size / self.mid) * 100


class OrderBlockDetector:
    """
    Detects Order Blocks using impulse-based detection.

    Parameters:
        impulse_atr_mult: Minimum impulse size as ATR multiple
        lookback: How far back to look for the OB candle
        min_strength: Minimum strength threshold
    """

    def __init__(
        self,
        impulse_atr_mult: float = 1.5,
        lookback: int = 10,
        min_strength: float = 0.5
    ):
        self.impulse_mult = impulse_atr_mult
        self.lookback = lookback
        self.min_strength = min_strength

    def detect(
        self,
        df: pd.DataFrame,
        atr: pd.Series = None
    ) -> List[OrderBlock]:
        """
        Detect all Order Blocks in dataframe.

        Args:
            df: OHLCV DataFrame
            atr: ATR series (calculated if not provided)

        Returns:
            List of OrderBlock objects
        """
        if len(df) < self.lookback + 10:
            return []

        # Calculate ATR if not provided
        if atr is None:
            atr = self._calc_atr(df)

        order_blocks = []

        for i in range(self.lookback + 5, len(df)):
            candle = df.iloc[i]
            current_atr = atr.iloc[i]

            if pd.isna(current_atr) or current_atr == 0:
                continue

            # Check for bullish impulse
            body = candle['close'] - candle['open']
            if body > current_atr * self.impulse_mult:
                ob = self._find_ob_before_impulse(
                    df, atr, i, is_bullish=True
                )
                if ob and ob.strength >= self.min_strength:
                    order_blocks.append(ob)

            # Check for bearish impulse
            body = candle['open'] - candle['close']
            if body > current_atr * self.impulse_mult:
                ob = self._find_ob_before_impulse(
                    df, atr, i, is_bullish=False
                )
                if ob and ob.strength >= self.min_strength:
                    order_blocks.append(ob)

        # Update mitigation status
        order_blocks = self._update_mitigation(order_blocks, df)

        # Remove duplicates (same zone)
        order_blocks = self._remove_duplicates(order_blocks)

        return order_blocks

    def _find_ob_before_impulse(
        self,
        df: pd.DataFrame,
        atr: pd.Series,
        impulse_idx: int,
        is_bullish: bool
    ) -> Optional[OrderBlock]:
        """
        Find the Order Block candle before an impulse move.

        For bullish impulse: Look for last bearish candle
        For bearish impulse: Look for last bullish candle
        """
        impulse_candle = df.iloc[impulse_idx]
        impulse_size = abs(impulse_candle['close'] - impulse_candle['open'])

        for i in range(impulse_idx - 1, max(0, impulse_idx - self.lookback), -1):
            candle = df.iloc[i]
            candle_body = candle['close'] - candle['open']

            # For bullish OB: Find bearish candle before bullish impulse
            if is_bullish and candle_body < 0:
                strength = self._calc_ob_strength(
                    df, atr, i, impulse_idx, is_bullish
                )

                volume_ratio = 1.0
                if 'volume_ratio' in df.columns:
                    volume_ratio = df['volume_ratio'].iloc[i]
                elif 'volume' in df.columns and 'volume_sma' in df.columns:
                    volume_ratio = candle['volume'] / df['volume_sma'].iloc[i]

                return OrderBlock(
                    top=candle['high'],
                    bottom=candle['low'],
                    timestamp=candle['timestamp'] if 'timestamp' in candle else df.index[i],
                    is_bullish=True,
                    strength=strength,
                    volume_ratio=volume_ratio,
                    impulse_size=impulse_size / atr.iloc[impulse_idx],
                    age_bars=impulse_idx - i
                )

            # For bearish OB: Find bullish candle before bearish impulse
            elif not is_bullish and candle_body > 0:
                strength = self._calc_ob_strength(
                    df, atr, i, impulse_idx, is_bullish
                )

                volume_ratio = 1.0
                if 'volume_ratio' in df.columns:
                    volume_ratio = df['volume_ratio'].iloc[i]
                elif 'volume' in df.columns and 'volume_sma' in df.columns:
                    volume_ratio = candle['volume'] / df['volume_sma'].iloc[i]

                return OrderBlock(
                    top=candle['high'],
                    bottom=candle['low'],
                    timestamp=candle['timestamp'] if 'timestamp' in candle else df.index[i],
                    is_bullish=False,
                    strength=strength,
                    volume_ratio=volume_ratio,
                    impulse_size=impulse_size / atr.iloc[impulse_idx],
                    age_bars=impulse_idx - i
                )

        return None

    def _calc_ob_strength(
        self,
        df: pd.DataFrame,
        atr: pd.Series,
        ob_idx: int,
        impulse_idx: int,
        is_bullish: bool
    ) -> float:
        """
        Calculate Order Block strength based on:
        - Impulse size
        - Volume
        - Clean departure (no wicks into zone)
        - Distance from OB to impulse
        """
        strength = 0.5  # Base strength

        ob_candle = df.iloc[ob_idx]
        impulse_candle = df.iloc[impulse_idx]
        current_atr = atr.iloc[impulse_idx]

        # 1. Impulse size factor (bigger impulse = stronger OB)
        impulse_size = abs(impulse_candle['close'] - impulse_candle['open'])
        impulse_factor = min(impulse_size / (current_atr * 2), 1.0) * 0.2
        strength += impulse_factor

        # 2. Volume factor
        if 'volume_ratio' in df.columns:
            vol_ratio = df['volume_ratio'].iloc[ob_idx]
            if vol_ratio > 1.5:
                strength += 0.15
            elif vol_ratio > 1.0:
                strength += 0.1

        # 3. Clean departure (no wicks back into zone)
        zone_mid = (ob_candle['high'] + ob_candle['low']) / 2
        clean_departure = True

        for j in range(ob_idx + 1, impulse_idx):
            check_candle = df.iloc[j]
            if is_bullish:
                if check_candle['low'] < zone_mid:
                    clean_departure = False
                    break
            else:
                if check_candle['high'] > zone_mid:
                    clean_departure = False
                    break

        if clean_departure:
            strength += 0.15

        # 4. Distance factor (closer = stronger)
        distance = impulse_idx - ob_idx
        if distance <= 3:
            strength += 0.1
        elif distance <= 5:
            strength += 0.05

        return min(strength, 1.0)

    def _update_mitigation(
        self,
        order_blocks: List[OrderBlock],
        df: pd.DataFrame
    ) -> List[OrderBlock]:
        """
        Update mitigation status for all order blocks.

        An OB is mitigated when price returns to the zone.
        """
        for ob in order_blocks:
            # Find data after OB formation
            if 'timestamp' in df.columns:
                future_data = df[df['timestamp'] > ob.timestamp]
            else:
                ob_idx = df.index.get_loc(ob.timestamp) if ob.timestamp in df.index else 0
                future_data = df.iloc[ob_idx + 1:]

            touches = 0

            for _, candle in future_data.iterrows():
                # Check if price entered the zone
                if ob.is_bullish:
                    # Bullish OB mitigated when low enters zone
                    if candle['low'] <= ob.top:
                        touches += 1
                        if candle['low'] <= ob.mid:
                            ob.is_mitigated = True
                            ob.mitigation_timestamp = candle.get('timestamp')
                            break
                else:
                    # Bearish OB mitigated when high enters zone
                    if candle['high'] >= ob.bottom:
                        touches += 1
                        if candle['high'] >= ob.mid:
                            ob.is_mitigated = True
                            ob.mitigation_timestamp = candle.get('timestamp')
                            break

            ob.touches = touches

        return order_blocks

    def _remove_duplicates(
        self,
        order_blocks: List[OrderBlock]
    ) -> List[OrderBlock]:
        """Remove duplicate OBs (overlapping zones)"""
        if len(order_blocks) <= 1:
            return order_blocks

        # Sort by strength (keep strongest)
        order_blocks.sort(key=lambda x: -x.strength)

        unique = []
        for ob in order_blocks:
            is_duplicate = False
            for existing in unique:
                # Check if zones overlap significantly
                overlap = min(ob.top, existing.top) - max(ob.bottom, existing.bottom)
                if overlap > 0:
                    min_size = min(ob.size, existing.size)
                    if overlap / min_size > 0.5:  # >50% overlap
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique.append(ob)

        return unique

    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def get_active(
        self,
        order_blocks: List[OrderBlock],
        current_price: float,
        max_distance_pct: float = 2.0
    ) -> dict:
        """
        Get active (non-mitigated) OBs near current price.

        Returns:
            Dict with 'bullish' and 'bearish' lists
        """
        active_bull = []
        active_bear = []

        for ob in order_blocks:
            if ob.is_mitigated:
                continue

            # Calculate distance
            distance_pct = abs(current_price - ob.mid) / current_price * 100

            if distance_pct <= max_distance_pct:
                if ob.is_bullish:
                    active_bull.append(ob)
                else:
                    active_bear.append(ob)

        # Sort by strength
        active_bull.sort(key=lambda x: -x.strength)
        active_bear.sort(key=lambda x: -x.strength)

        return {
            'bullish': active_bull,
            'bearish': active_bear,
            'near_bullish': len(active_bull) > 0,
            'near_bearish': len(active_bear) > 0,
            'in_bullish_zone': any(ob.bottom <= current_price <= ob.top for ob in active_bull),
            'in_bearish_zone': any(ob.bottom <= current_price <= ob.top for ob in active_bear)
        }
