"""
SMC Ultra V2 - Signal Generator
===============================
Kombiniert alle Komponenten zu Trading-Signalen.

Flow:
1. Regime Detection → Trade or No Trade?
2. Coin Filter → Top 20 tradeable coins
3. HTF Bias → Direction
4. MTF Setup → Entry Zone
5. LTF Trigger → Precise Entry
6. ML Confidence → Final decision
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

from config.settings import config
from config.coins import coin_db

from data import MTFDataLoader
from analysis import (
    RegimeDetector, MTFAnalyzer, SessionFilter,
    RegimeState, HTFBias, MTFSetup, LTFTrigger, TrendDirection
)
from smc import (
    OrderBlockDetector, FVGDetector, LiquidityDetector,
    MarketStructure, OrderBlock, FairValueGap, LiquiditySweep
)
from ml import (
    FeatureExtractor, MLConfidenceScorer, CoinFilter,
    FeatureSet, PredictionResult
)


@dataclass
class Signal:
    """Complete trading signal"""
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    take_profit: float
    stop_loss: float

    confidence: int  # 0-100
    should_trade: bool

    # Detailed info
    regime: str
    htf_bias: str
    pattern: str
    factors: List[str] = field(default_factory=list)

    # Calculated values
    tp_pct: float = 0.0
    sl_pct: float = 0.0
    effective_rr: float = 1.0
    leverage: int = 1
    risk_per_trade: float = 1.0

    # Metadata
    timestamp: datetime = None
    zone: Dict = None


class SignalGenerator:
    """
    Main signal generator that combines all components.

    This is the brain of the system.
    """

    def __init__(self):
        # Data
        self.data_loader = MTFDataLoader()

        # Analysis
        self.regime_detector = RegimeDetector()
        self.mtf_analyzer = MTFAnalyzer()
        self.session_filter = SessionFilter()

        # SMC
        self.ob_detector = OrderBlockDetector()
        self.fvg_detector = FVGDetector()
        self.liq_detector = LiquidityDetector()
        self.structure = MarketStructure()

        # ML
        self.feature_extractor = FeatureExtractor()
        self.ml_scorer = MLConfidenceScorer()
        self.coin_filter = CoinFilter()

        # Config
        self.config = config

    def scan_all(
        self,
        coins: List[str] = None,
        timestamp: datetime = None
    ) -> List[Signal]:
        """
        Scan all coins for signals.

        Returns:
            List of Signal objects, sorted by confidence
        """
        timestamp = timestamp or datetime.utcnow()

        # 1. Session check
        session_ok, session_reason = self.session_filter.should_trade(timestamp)
        if not session_ok:
            print(f"Session filter: {session_reason}")
            return []

        # 2. Get coins to scan
        if coins is None:
            from config.coins import get_top_n_coins
            coins = get_top_n_coins(100)

        # 3. Pre-filter coins
        tradeable_coins = self.coin_filter.get_tradeable(coins, top_n=30)
        print(f"Scanning {len(tradeable_coins)} tradeable coins...")

        # 4. Analyze each coin
        signals = []
        for symbol in tradeable_coins:
            signal = self.analyze_coin(symbol, timestamp)
            if signal and signal.should_trade:
                signals.append(signal)

        # 5. Sort by confidence
        signals.sort(key=lambda x: -x.confidence)

        return signals

    def analyze_coin(
        self,
        symbol: str,
        timestamp: datetime = None
    ) -> Optional[Signal]:
        """
        Complete analysis of a single coin.
        """
        timestamp = timestamp or datetime.utcnow()

        try:
            # 1. Load MTF data
            data = self.data_loader.load_symbol(symbol)
            if not data:
                return None

            htf_df = data.get(self.config.timeframes.htf)
            mtf_df = data.get(self.config.timeframes.mtf)
            ltf_df = data.get(self.config.timeframes.ltf)

            if htf_df is None or len(htf_df) < 50:
                return None

            # 2. Regime detection
            regime = self.regime_detector.detect(htf_df)

            if not regime.should_trade:
                return Signal(
                    symbol=symbol,
                    direction='none',
                    entry_price=0,
                    take_profit=0,
                    stop_loss=0,
                    confidence=0,
                    should_trade=False,
                    regime=regime.regime.value,
                    htf_bias='none',
                    pattern='none',
                    factors=[f'regime_{regime.reason}']
                )

            # 3. HTF Bias
            htf_bias = self.mtf_analyzer.analyze_htf(htf_df)

            if htf_bias.direction == TrendDirection.NEUTRAL:
                return self._no_signal(symbol, regime.regime.value, 'no_htf_bias')

            # Check regime direction alignment
            if regime.direction != 'both':
                if regime.direction == 'long_only' and htf_bias.direction != TrendDirection.BULLISH:
                    return self._no_signal(symbol, regime.regime.value, 'regime_direction_mismatch')
                if regime.direction == 'short_only' and htf_bias.direction != TrendDirection.BEARISH:
                    return self._no_signal(symbol, regime.regime.value, 'regime_direction_mismatch')

            # 4. Detect SMC structures on MTF
            if mtf_df is None or len(mtf_df) < 30:
                mtf_df = htf_df  # Fallback

            atr = mtf_df['atr'] if 'atr' in mtf_df.columns else None
            order_blocks = self.ob_detector.detect(mtf_df, atr)
            fvgs = self.fvg_detector.detect(mtf_df)
            sweeps = self.liq_detector.find_sweeps(mtf_df)

            # 5. MTF Setup
            mtf_setup = self.mtf_analyzer.analyze_mtf(
                mtf_df, htf_bias, order_blocks, fvgs
            )

            if not mtf_setup.valid:
                return self._no_signal(symbol, regime.regime.value, mtf_setup.reason)

            # 6. LTF Trigger
            if ltf_df is None or len(ltf_df) < 20:
                ltf_df = mtf_df  # Fallback

            ltf_trigger = self.mtf_analyzer.analyze_ltf(ltf_df, mtf_setup)

            if not ltf_trigger.triggered:
                return self._no_signal(symbol, regime.regime.value, ltf_trigger.reason)

            # 7. Extract features and get ML confidence
            features = self.feature_extractor.extract(
                df=ltf_df,
                htf_df=htf_df,
                order_blocks=order_blocks,
                fvgs=fvgs,
                sweeps=sweeps,
                regime=regime,
                htf_bias=htf_bias,
                timestamp=timestamp,
                coin_stats=self.coin_filter.historical_stats.get(symbol, {})
            )

            ml_result = self.ml_scorer.predict(features)

            # 8. Apply regime adjustments
            adjusted_confidence = int(ml_result.confidence * regime.leverage_multiplier)
            adjusted_min = max(self.config.entry.min_confidence, regime.min_confidence)

            if adjusted_confidence < adjusted_min:
                return self._no_signal(symbol, regime.regime.value, 'low_confidence')

            # 9. Calculate targets
            direction = 'long' if htf_bias.is_bullish else 'short'
            entry = ltf_trigger.entry_price
            zone = mtf_setup.zone

            targets = self._calculate_targets(
                entry, direction, zone, ltf_df,
                regime.tp_multiplier
            )

            # 10. Calculate leverage
            leverage_info = self._calculate_leverage(
                adjusted_confidence,
                targets['sl_pct'],
                regime.volatility_ratio,
                self.coin_filter.historical_stats.get(symbol, {}).get('win_rate', 0.55)
            )

            # 11. Build signal
            factors = self._collect_factors(
                htf_bias, mtf_setup, ltf_trigger, order_blocks, fvgs, sweeps
            )

            return Signal(
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                take_profit=targets['tp'],
                stop_loss=targets['sl'],
                confidence=adjusted_confidence,
                should_trade=True,
                regime=regime.regime.value,
                htf_bias=htf_bias.direction.value,
                pattern=ltf_trigger.pattern,
                factors=factors,
                tp_pct=targets['tp_pct'],
                sl_pct=targets['sl_pct'],
                effective_rr=targets['effective_rr'],
                leverage=leverage_info['leverage'],
                risk_per_trade=leverage_info['risk_pct'],
                timestamp=timestamp,
                zone=zone
            )

        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None

    def _calculate_targets(
        self,
        entry: float,
        direction: str,
        zone: Dict,
        df: pd.DataFrame,
        tp_multiplier: float = 1.0
    ) -> Dict:
        """Calculate TP/SL with 1:1 RR"""
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else entry * 0.01

        # SL behind zone
        if zone:
            zone_size = zone['top'] - zone['bottom']
            buffer = zone_size * 0.1
        else:
            zone_size = atr
            buffer = atr * 0.2

        if direction == 'long':
            sl = (zone['bottom'] if zone else entry) - buffer
            sl_distance = entry - sl
        else:
            sl = (zone['top'] if zone else entry) + buffer
            sl_distance = sl - entry

        sl_pct = (sl_distance / entry) * 100

        # TP for ~1:1 RR (after fees)
        fee_pct = self.config.backtest.fee_pct * 2  # Entry + exit
        tp_pct = sl_pct * tp_multiplier + fee_pct  # Slightly above 1:1 to account for fees

        # Apply limits
        tp_pct = max(tp_pct, self.config.exit.min_tp_pct)
        tp_pct = min(tp_pct, self.config.exit.max_tp_pct)

        tp_distance = entry * (tp_pct / 100)

        if direction == 'long':
            tp = entry + tp_distance
        else:
            tp = entry - tp_distance

        # Effective RR
        net_tp = tp_pct - fee_pct
        net_sl = sl_pct + fee_pct
        effective_rr = net_tp / net_sl if net_sl > 0 else 1.0

        return {
            'tp': tp,
            'sl': sl,
            'tp_pct': tp_pct,
            'sl_pct': sl_pct,
            'effective_rr': round(effective_rr, 2)
        }

    def _calculate_leverage(
        self,
        confidence: int,
        sl_pct: float,
        volatility_ratio: float,
        historical_wr: float
    ) -> Dict:
        """Calculate leverage using Kelly-inspired formula"""
        # Base leverage from risk config
        max_risk = self.config.risk.max_risk_per_trade_pct
        max_lev = self.config.risk.max_leverage

        # Kelly fraction
        p = historical_wr
        q = 1 - p
        b = 1.0  # 1:1 RR
        kelly = (p * b - q) / b if b > 0 else 0
        kelly = max(kelly, 0) * 0.25  # Use 25% Kelly

        # Confidence adjustment
        conf_mult = confidence / 100

        # Volatility adjustment
        if volatility_ratio > 1.5:
            vol_mult = 0.5
        elif volatility_ratio > 1.2:
            vol_mult = 0.7
        else:
            vol_mult = 1.0

        # Calculate leverage
        raw_leverage = kelly * 100 * conf_mult * vol_mult

        # Respect max risk
        max_lev_for_risk = max_risk / sl_pct * 100 if sl_pct > 0 else max_lev

        final_leverage = min(
            raw_leverage,
            max_lev_for_risk,
            max_lev
        )
        final_leverage = max(int(final_leverage), self.config.risk.min_leverage)

        # Actual risk
        risk_pct = (sl_pct / 100) * final_leverage * 100

        return {
            'leverage': final_leverage,
            'risk_pct': round(risk_pct, 2),
            'kelly': round(kelly, 3)
        }

    def _collect_factors(
        self,
        htf_bias: HTFBias,
        mtf_setup: MTFSetup,
        ltf_trigger: LTFTrigger,
        order_blocks: List,
        fvgs: List,
        sweeps: List
    ) -> List[str]:
        """Collect all confluence factors"""
        factors = []

        # HTF
        factors.append(f'htf_{htf_bias.direction.value}')
        factors.append(f'ema_{htf_bias.ema_alignment}')

        # MTF
        if mtf_setup.zone:
            factors.append(f'zone_{mtf_setup.zone.get("type", "ob")}')
        if mtf_setup.structure_break:
            factors.append('structure_break')

        # LTF
        if ltf_trigger.pattern:
            factors.append(f'pattern_{ltf_trigger.pattern}')

        # SMC
        active_obs = [ob for ob in order_blocks if not ob.is_mitigated]
        active_fvgs = [f for f in fvgs if not f.is_filled]
        recent_sweeps = sweeps[-3:] if sweeps else []

        if any(ob.is_bullish for ob in active_obs):
            factors.append('bullish_ob')
        if any(not ob.is_bullish for ob in active_obs):
            factors.append('bearish_ob')
        if any(f.is_bullish for f in active_fvgs):
            factors.append('bullish_fvg')
        if any(not f.is_bullish for f in active_fvgs):
            factors.append('bearish_fvg')
        if any(s.is_bullish for s in recent_sweeps):
            factors.append('bullish_sweep')
        if any(not s.is_bullish for s in recent_sweeps):
            factors.append('bearish_sweep')

        return factors

    def _no_signal(self, symbol: str, regime: str, reason: str) -> Signal:
        """Return no-trade signal"""
        return Signal(
            symbol=symbol,
            direction='none',
            entry_price=0,
            take_profit=0,
            stop_loss=0,
            confidence=0,
            should_trade=False,
            regime=regime,
            htf_bias='none',
            pattern='none',
            factors=[reason]
        )
