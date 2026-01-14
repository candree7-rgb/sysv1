"""
SMC Ultra V2 - Multi-Timeframe Analyzer
=======================================
Analysiert mehrere Timeframes für maximale Confluence.

HTF (1H): Trend Direction + Key Levels
MTF (15m): Entry Zone + Structure
LTF (1m): Precision Entry + Trigger
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

from config.settings import config


class TrendDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class HTFBias:
    """Higher Timeframe Bias Analysis"""
    direction: TrendDirection
    strength: float  # 0-100
    key_levels: Dict[str, float]  # resistance, support levels
    ema_alignment: str  # 'perfect_bull', 'bull', 'neutral', 'bear', 'perfect_bear'
    adx: float
    structure: str  # 'hh_hl', 'll_lh', 'mixed'

    @property
    def is_bullish(self) -> bool:
        return self.direction == TrendDirection.BULLISH

    @property
    def is_bearish(self) -> bool:
        return self.direction == TrendDirection.BEARISH


@dataclass
class MTFSetup:
    """Medium Timeframe Setup Analysis"""
    valid: bool
    direction: TrendDirection
    zone: Optional[Dict] = None  # {top, bottom, type}
    structure_break: bool = False
    pullback_level: Optional[float] = None
    reason: str = ""


@dataclass
class LTFTrigger:
    """Lower Timeframe Entry Trigger"""
    triggered: bool
    entry_price: Optional[float] = None
    pattern: str = ""  # 'engulfing', 'pin_bar', 'inside_break', etc.
    confirmation_strength: float = 0.0
    reason: str = ""


class MTFAnalyzer:
    """
    Multi-Timeframe Analysis System.

    Flow:
    1. HTF → Determine bias (only trade in this direction)
    2. MTF → Find entry zones (OB, FVG, key levels)
    3. LTF → Wait for trigger pattern
    """

    def __init__(self):
        self.htf = config.timeframes.htf
        self.mtf = config.timeframes.mtf
        self.ltf = config.timeframes.ltf

    def analyze_htf(self, df: pd.DataFrame) -> HTFBias:
        """
        Analyze Higher Timeframe for directional bias.

        This is the MOST important decision.
        """
        if len(df) < 200:
            return self._neutral_bias("insufficient_data")

        close = df['close'].iloc[-1]

        # EMAs
        ema20 = df['ema20'].iloc[-1] if 'ema20' in df.columns else df['close'].ewm(span=20).mean().iloc[-1]
        ema50 = df['ema50'].iloc[-1] if 'ema50' in df.columns else df['close'].ewm(span=50).mean().iloc[-1]
        ema200 = df['ema200'].iloc[-1] if 'ema200' in df.columns else df['close'].ewm(span=200).mean().iloc[-1]

        # ADX
        adx = self._calc_adx(df) if 'adx' not in df.columns else df['adx'].iloc[-1]

        # Structure analysis
        structure = self._analyze_structure(df)

        # EMA alignment
        ema_alignment = self._get_ema_alignment(close, ema20, ema50, ema200)

        # Calculate bias strength
        bullish_signals = 0
        bearish_signals = 0

        # Price vs EMAs
        if close > ema20:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if close > ema50:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if close > ema200:
            bullish_signals += 2  # More weight
        else:
            bearish_signals += 2

        # EMA stack
        if ema20 > ema50 > ema200:
            bullish_signals += 3
        elif ema20 < ema50 < ema200:
            bearish_signals += 3

        # Structure
        if structure == 'hh_hl':
            bullish_signals += 2
        elif structure == 'll_lh':
            bearish_signals += 2

        # ADX confirms trend
        if adx > 25:
            if bullish_signals > bearish_signals:
                bullish_signals += 1
            elif bearish_signals > bullish_signals:
                bearish_signals += 1

        # Calculate strength
        total = bullish_signals + bearish_signals
        if total == 0:
            return self._neutral_bias("no_signals")

        # Determine direction
        if bullish_signals > bearish_signals + 2:
            direction = TrendDirection.BULLISH
            strength = min((bullish_signals / total) * 100, 100)
        elif bearish_signals > bullish_signals + 2:
            direction = TrendDirection.BEARISH
            strength = min((bearish_signals / total) * 100, 100)
        else:
            direction = TrendDirection.NEUTRAL
            strength = 50

        # Find key levels
        key_levels = self._find_key_levels(df)

        return HTFBias(
            direction=direction,
            strength=strength,
            key_levels=key_levels,
            ema_alignment=ema_alignment,
            adx=adx,
            structure=structure
        )

    def analyze_mtf(
        self,
        df: pd.DataFrame,
        htf_bias: HTFBias,
        ob_zones: List = None,
        fvg_zones: List = None
    ) -> MTFSetup:
        """
        Analyze Medium Timeframe for entry setup.

        Only looks for setups aligned with HTF bias.
        """
        if htf_bias.direction == TrendDirection.NEUTRAL:
            return MTFSetup(valid=False, direction=TrendDirection.NEUTRAL, reason="no_htf_bias")

        if len(df) < 50:
            return MTFSetup(valid=False, direction=htf_bias.direction, reason="insufficient_data")

        close = df['close'].iloc[-1]

        # Check for structure break
        structure_break = self._check_structure_break(df, htf_bias.direction)

        # Find valid zones (OBs and FVGs in direction of bias)
        valid_zones = self._find_valid_zones(
            df, htf_bias.direction, ob_zones, fvg_zones
        )

        if not valid_zones:
            return MTFSetup(
                valid=False,
                direction=htf_bias.direction,
                reason="no_zones"
            )

        # Find best zone (closest to price in direction)
        best_zone = self._select_best_zone(close, valid_zones, htf_bias.direction)

        if not best_zone:
            return MTFSetup(
                valid=False,
                direction=htf_bias.direction,
                reason="no_suitable_zone"
            )

        # Calculate pullback level
        pullback = self._calc_pullback_level(df, htf_bias.direction)

        return MTFSetup(
            valid=True,
            direction=htf_bias.direction,
            zone=best_zone,
            structure_break=structure_break,
            pullback_level=pullback
        )

    def analyze_ltf(
        self,
        df: pd.DataFrame,
        mtf_setup: MTFSetup
    ) -> LTFTrigger:
        """
        Analyze Lower Timeframe for entry trigger.

        Only triggers when price is in zone AND confirmation pattern appears.
        """
        if not mtf_setup.valid:
            return LTFTrigger(triggered=False, reason="invalid_mtf_setup")

        if mtf_setup.zone is None:
            return LTFTrigger(triggered=False, reason="no_zone")

        close = df['close'].iloc[-1]
        zone = mtf_setup.zone

        # Check if price is in zone
        in_zone = zone['bottom'] <= close <= zone['top']

        if not in_zone:
            # Check if close to zone
            zone_mid = (zone['top'] + zone['bottom']) / 2
            zone_size = zone['top'] - zone['bottom']
            distance_pct = abs(close - zone_mid) / close * 100

            if distance_pct > 0.5:  # More than 0.5% away
                return LTFTrigger(triggered=False, reason="price_not_in_zone")

        # Look for confirmation pattern
        pattern = self._detect_confirmation_pattern(df, mtf_setup.direction)

        if not pattern['confirmed']:
            return LTFTrigger(triggered=False, reason="no_confirmation_pattern")

        # Calculate optimal entry
        entry_price = self._calc_optimal_entry(df, zone, mtf_setup.direction)

        return LTFTrigger(
            triggered=True,
            entry_price=entry_price,
            pattern=pattern['type'],
            confirmation_strength=pattern['strength']
        )

    def _analyze_structure(self, df: pd.DataFrame) -> str:
        """Analyze swing structure"""
        highs = df['high'].values
        lows = df['low'].values

        # Find swing points (simplified)
        swing_len = 5
        swing_highs = []
        swing_lows = []

        for i in range(swing_len, len(df) - swing_len):
            if highs[i] == max(highs[i-swing_len:i+swing_len+1]):
                swing_highs.append((i, highs[i]))
            if lows[i] == min(lows[i-swing_len:i+swing_len+1]):
                swing_lows.append((i, lows[i]))

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'mixed'

        # Check last 2 swings
        last_highs = [h[1] for h in swing_highs[-2:]]
        last_lows = [l[1] for l in swing_lows[-2:]]

        higher_high = last_highs[-1] > last_highs[-2]
        higher_low = last_lows[-1] > last_lows[-2]
        lower_high = last_highs[-1] < last_highs[-2]
        lower_low = last_lows[-1] < last_lows[-2]

        if higher_high and higher_low:
            return 'hh_hl'
        elif lower_high and lower_low:
            return 'll_lh'
        else:
            return 'mixed'

    def _get_ema_alignment(
        self,
        close: float,
        ema20: float,
        ema50: float,
        ema200: float
    ) -> str:
        """Get EMA alignment status"""
        if close > ema20 > ema50 > ema200:
            return 'perfect_bull'
        elif close > ema20 and ema20 > ema50:
            return 'bull'
        elif close < ema20 < ema50 < ema200:
            return 'perfect_bear'
        elif close < ema20 and ema20 < ema50:
            return 'bear'
        else:
            return 'neutral'

    def _find_key_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """Find key support/resistance levels"""
        highs = df['high'].values
        lows = df['low'].values
        close = df['close'].iloc[-1]

        # Recent swing highs as resistance
        resistance_levels = []
        support_levels = []

        swing_len = 10
        for i in range(swing_len, len(df) - swing_len):
            if highs[i] == max(highs[i-swing_len:i+swing_len+1]):
                if highs[i] > close:
                    resistance_levels.append(highs[i])
            if lows[i] == min(lows[i-swing_len:i+swing_len+1]):
                if lows[i] < close:
                    support_levels.append(lows[i])

        return {
            'resistance_1': min(resistance_levels) if resistance_levels else close * 1.02,
            'resistance_2': sorted(resistance_levels)[1] if len(resistance_levels) > 1 else close * 1.05,
            'support_1': max(support_levels) if support_levels else close * 0.98,
            'support_2': sorted(support_levels, reverse=True)[1] if len(support_levels) > 1 else close * 0.95
        }

    def _check_structure_break(
        self,
        df: pd.DataFrame,
        direction: TrendDirection
    ) -> bool:
        """Check if there was a recent structure break"""
        close = df['close'].iloc[-1]

        # Find recent swing high/low
        swing_len = 5
        recent_data = df.tail(30)

        if direction == TrendDirection.BULLISH:
            # Look for break above recent swing high
            swing_high = recent_data['high'].max()
            return close > swing_high
        else:
            # Look for break below recent swing low
            swing_low = recent_data['low'].min()
            return close < swing_low

    def _find_valid_zones(
        self,
        df: pd.DataFrame,
        direction: TrendDirection,
        ob_zones: List = None,
        fvg_zones: List = None
    ) -> List[Dict]:
        """Find valid zones in direction of bias"""
        zones = []
        close = df['close'].iloc[-1]

        # Filter Order Blocks
        if ob_zones:
            for ob in ob_zones:
                if direction == TrendDirection.BULLISH and ob.is_bullish and not ob.is_mitigated:
                    if ob.bottom < close:  # Zone is below price (for long entry)
                        zones.append({
                            'top': ob.top,
                            'bottom': ob.bottom,
                            'type': 'ob',
                            'strength': ob.strength
                        })
                elif direction == TrendDirection.BEARISH and not ob.is_bullish and not ob.is_mitigated:
                    if ob.top > close:  # Zone is above price (for short entry)
                        zones.append({
                            'top': ob.top,
                            'bottom': ob.bottom,
                            'type': 'ob',
                            'strength': ob.strength
                        })

        # Filter FVGs
        if fvg_zones:
            for fvg in fvg_zones:
                if direction == TrendDirection.BULLISH and fvg.is_bullish and not fvg.is_filled:
                    if fvg.bottom < close:
                        zones.append({
                            'top': fvg.top,
                            'bottom': fvg.bottom,
                            'type': 'fvg',
                            'strength': 0.8
                        })
                elif direction == TrendDirection.BEARISH and not fvg.is_bullish and not fvg.is_filled:
                    if fvg.top > close:
                        zones.append({
                            'top': fvg.top,
                            'bottom': fvg.bottom,
                            'type': 'fvg',
                            'strength': 0.8
                        })

        return zones

    def _select_best_zone(
        self,
        close: float,
        zones: List[Dict],
        direction: TrendDirection
    ) -> Optional[Dict]:
        """Select the best zone for entry"""
        if not zones:
            return None

        # Sort by distance to price
        if direction == TrendDirection.BULLISH:
            # For longs, want zones below price, closest first
            valid = [z for z in zones if z['top'] < close]
            valid.sort(key=lambda z: close - z['top'])
        else:
            # For shorts, want zones above price, closest first
            valid = [z for z in zones if z['bottom'] > close]
            valid.sort(key=lambda z: z['bottom'] - close)

        # Return best zone (closest with highest strength)
        if valid:
            # Prefer zones with higher strength if similar distance
            best = valid[0]
            for z in valid[1:3]:  # Check top 3
                if z['strength'] > best['strength']:
                    best = z
            return best

        return None

    def _calc_pullback_level(
        self,
        df: pd.DataFrame,
        direction: TrendDirection
    ) -> float:
        """Calculate Fibonacci pullback level"""
        recent = df.tail(50)

        if direction == TrendDirection.BULLISH:
            swing_low = recent['low'].min()
            swing_high = recent['high'].max()
            # 61.8% retracement
            return swing_high - (swing_high - swing_low) * 0.618
        else:
            swing_low = recent['low'].min()
            swing_high = recent['high'].max()
            # 61.8% retracement
            return swing_low + (swing_high - swing_low) * 0.618

    def _detect_confirmation_pattern(
        self,
        df: pd.DataFrame,
        direction: TrendDirection
    ) -> Dict:
        """Detect confirmation pattern for entry"""
        last_3 = df.tail(3)

        c0 = last_3.iloc[-1]  # Current candle
        c1 = last_3.iloc[-2]  # Previous candle

        result = {'confirmed': False, 'type': '', 'strength': 0}

        if direction == TrendDirection.BULLISH:
            # Bullish engulfing
            if (c0['close'] > c0['open'] and
                c1['close'] < c1['open'] and
                c0['close'] > c1['open'] and
                c0['open'] < c1['close']):
                result = {'confirmed': True, 'type': 'bullish_engulfing', 'strength': 0.9}

            # Bullish pin bar
            elif (c0['close'] > c0['open'] and
                  (c0['close'] - c0['open']) < (c0['high'] - c0['low']) * 0.3 and
                  (c0['open'] - c0['low']) > (c0['high'] - c0['low']) * 0.6):
                result = {'confirmed': True, 'type': 'bullish_pin', 'strength': 0.85}

            # Inside bar breakout
            elif (c0['high'] < c1['high'] and c0['low'] > c1['low'] and
                  c0['close'] > c1['high'] * 0.9):
                result = {'confirmed': True, 'type': 'inside_break_up', 'strength': 0.8}

            # Strong bullish candle
            elif (c0['close'] > c0['open'] and
                  (c0['close'] - c0['open']) > df['atr'].iloc[-1] * 0.5):
                result = {'confirmed': True, 'type': 'strong_bullish', 'strength': 0.7}

        else:  # Bearish
            # Bearish engulfing
            if (c0['close'] < c0['open'] and
                c1['close'] > c1['open'] and
                c0['close'] < c1['open'] and
                c0['open'] > c1['close']):
                result = {'confirmed': True, 'type': 'bearish_engulfing', 'strength': 0.9}

            # Bearish pin bar
            elif (c0['close'] < c0['open'] and
                  (c0['open'] - c0['close']) < (c0['high'] - c0['low']) * 0.3 and
                  (c0['high'] - c0['open']) > (c0['high'] - c0['low']) * 0.6):
                result = {'confirmed': True, 'type': 'bearish_pin', 'strength': 0.85}

            # Inside bar breakdown
            elif (c0['high'] < c1['high'] and c0['low'] > c1['low'] and
                  c0['close'] < c1['low'] * 1.1):
                result = {'confirmed': True, 'type': 'inside_break_down', 'strength': 0.8}

            # Strong bearish candle
            elif (c0['close'] < c0['open'] and
                  (c0['open'] - c0['close']) > df['atr'].iloc[-1] * 0.5):
                result = {'confirmed': True, 'type': 'strong_bearish', 'strength': 0.7}

        return result

    def _calc_optimal_entry(
        self,
        df: pd.DataFrame,
        zone: Dict,
        direction: TrendDirection
    ) -> float:
        """Calculate optimal entry price"""
        close = df['close'].iloc[-1]

        if direction == TrendDirection.BULLISH:
            # Enter at zone top (better price)
            return min(close, zone['top'])
        else:
            # Enter at zone bottom (better price)
            return max(close, zone['bottom'])

    def _calc_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        up_move = high - high.shift()
        down_move = low.shift() - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)
        adx = dx.rolling(period).mean()

        return adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else 0

    def _neutral_bias(self, reason: str) -> HTFBias:
        """Return neutral bias"""
        return HTFBias(
            direction=TrendDirection.NEUTRAL,
            strength=0,
            key_levels={},
            ema_alignment='neutral',
            adx=0,
            structure='mixed'
        )
