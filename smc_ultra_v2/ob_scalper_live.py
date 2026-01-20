"""
OB Scalper Live - 1:1 Match with Backtest
=========================================
Live signal generator that uses EXACTLY the same logic as ob_scalper.py backtest.

Key differences from signal_generator.py:
- 5min OB detection, 1min entry precision
- 1H + 4H MTF alignment (both must confirm direction)
- Daily filter ONLY for shorts (not longs!)
- OB detection_timestamp check (no look-ahead)
- Partial TP support
"""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

# === CONFIGURATION (same as ob_scalper.py) ===
OB_MIN_STRENGTH = float(os.getenv('OB_MIN_STRENGTH', '0.9'))
OB_MIN_STRENGTH_SHORT = float(os.getenv('OB_MIN_STRENGTH_SHORT', '0.9'))
OB_MAX_AGE = int(os.getenv('OB_MAX_AGE', '50'))  # in 5min candles
RR_TARGET = float(os.getenv('RR_TARGET', '1.5'))
SL_BUFFER_PCT = float(os.getenv('SL_BUFFER_PCT', '0.05'))

# MTF Filters
USE_4H_MTF = os.getenv('USE_4H_MTF', 'true').lower() == 'true'
USE_DAILY_FOR_SHORTS = os.getenv('USE_DAILY_FOR_SHORTS', 'true').lower() == 'true'
USE_DAILY_FOR_LONGS = os.getenv('USE_DAILY_FOR_LONGS', 'false').lower() == 'true'  # OFF!

# Partial TP
USE_PARTIAL_TP = os.getenv('USE_PARTIAL_TP', 'true').lower() == 'true'
PARTIAL_TP_LEVEL = float(os.getenv('PARTIAL_TP_LEVEL', '0.5'))
PARTIAL_SIZE = float(os.getenv('PARTIAL_SIZE', '0.5'))

# Risk & Leverage
MAX_LEVERAGE = int(os.getenv('MAX_LEVERAGE', '20'))
RISK_PER_TRADE_PCT = float(os.getenv('RISK_PER_TRADE_PCT', '2.0'))  # Target risk % per trade


@dataclass
class LiveSignal:
    """Live trading signal - 1:1 match with backtest"""
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    sl_price: float
    tp_price: float

    leverage: int
    ob_top: float
    ob_bottom: float

    # Partial TP settings
    use_partial_tp: bool = True
    partial_tp_price: float = None  # 50% toward TP
    partial_size: float = 0.5  # Close 50% at partial

    # Metadata
    timestamp: datetime = None
    ob_strength: float = 0.0
    ob_age_candles: int = 0


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA indicators - same as backtest"""
    df = df.copy()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    return df


class OBScalperLive:
    """
    Live signal generator with 1:1 backtest logic.

    MTF Alignment:
    - Longs: 1H + 4H must be bullish (NO daily filter)
    - Shorts: 1H + 4H + Daily must be bearish
    """

    def __init__(self):
        from data import BybitDataDownloader
        from smc import OrderBlockDetector

        self.dl = BybitDataDownloader()
        self.ob_detector = OrderBlockDetector(min_strength=0.5)

    def get_signal(self, symbol: str) -> Optional[LiveSignal]:
        """
        Check if there's a valid entry signal for a symbol.

        Returns LiveSignal if valid setup found, None otherwise.
        """
        try:
            # Load data (same timeframes as backtest)
            df_5m = self.dl.load_or_download(symbol, "5", 7)
            df_1m = self.dl.load_or_download(symbol, "1", 2)
            df_1h = self.dl.load_or_download(symbol, "60", 14)
            df_4h = self.dl.load_or_download(symbol, "240", 30) if USE_4H_MTF else None
            df_daily = self.dl.load_or_download(symbol, "D", 60) if USE_DAILY_FOR_SHORTS else None

            if df_5m is None or len(df_5m) < 100:
                return None
            if df_1m is None or len(df_1m) < 50:
                return None
            if df_1h is None or len(df_1h) < 50:
                return None

            # Add indicators
            df_5m = calculate_indicators(df_5m)
            df_1h = calculate_indicators(df_1h)
            df_1m = calculate_indicators(df_1m)
            if df_4h is not None:
                df_4h = calculate_indicators(df_4h)
            if df_daily is not None:
                df_daily = calculate_indicators(df_daily)

            # Detect OBs on 5min
            atr = df_5m['close'].rolling(14).apply(
                lambda x: pd.Series(x).diff().abs().mean() if len(x) > 1 else 0
            )
            obs = self.ob_detector.detect(df_5m, atr)

            if not obs:
                return None

            # Get current candle
            current_1m = df_1m.iloc[-1]
            ts = current_1m['timestamp']
            current_price = current_1m['close']

            # === 1H MTF CHECK ===
            ts_1h = ts.floor('1h')
            h1_candles = df_1h[df_1h['timestamp'] <= ts_1h - pd.Timedelta(hours=1)]
            if len(h1_candles) == 0:
                return None
            h1_candle = h1_candles.iloc[-1]

            h1_bullish = h1_candle['close'] > h1_candle['ema20'] > h1_candle['ema50']
            h1_bearish = h1_candle['close'] < h1_candle['ema20'] < h1_candle['ema50']

            if not h1_bullish and not h1_bearish:
                return None  # No clear 1H trend

            # === 4H MTF CHECK ===
            if USE_4H_MTF and df_4h is not None:
                ts_4h = ts.floor('4h')
                h4_candles = df_4h[df_4h['timestamp'] <= ts_4h - pd.Timedelta(hours=4)]
                if len(h4_candles) > 0:
                    h4_candle = h4_candles.iloc[-1]
                    h4_bullish = h4_candle['close'] > h4_candle['ema20'] > h4_candle['ema50']
                    h4_bearish = h4_candle['close'] < h4_candle['ema20'] < h4_candle['ema50']

                    # 4H must confirm 1H
                    if h1_bullish and not h4_bullish:
                        return None
                    if h1_bearish and not h4_bearish:
                        return None

            # Direction
            direction = 'long' if h1_bullish else 'short'

            # === DAILY FILTER FOR SHORTS ONLY ===
            if USE_DAILY_FOR_SHORTS and direction == 'short' and df_daily is not None:
                ts_daily = ts.floor('D')
                daily_candles = df_daily[df_daily['timestamp'] <= ts_daily - pd.Timedelta(days=1)]
                if len(daily_candles) > 0:
                    daily_candle = daily_candles.iloc[-1]
                    daily_bearish = daily_candle['close'] < daily_candle['ema20'] < daily_candle['ema50']
                    if not daily_bearish:
                        return None  # Daily not bearish - skip short

            # === DAILY FILTER FOR LONGS (disabled by default) ===
            if USE_DAILY_FOR_LONGS and direction == 'long' and df_daily is not None:
                ts_daily = ts.floor('D')
                daily_candles = df_daily[df_daily['timestamp'] <= ts_daily - pd.Timedelta(days=1)]
                if len(daily_candles) > 0:
                    daily_candle = daily_candles.iloc[-1]
                    daily_bullish = daily_candle['close'] > daily_candle['ema20'] > daily_candle['ema50']
                    if not daily_bullish:
                        return None  # Daily not bullish - skip long

            # === FIND VALID OB ===
            matching_ob = None

            for ob in obs:
                # Detection timestamp check (no look-ahead!)
                ob_known_at = ob.detection_timestamp if ob.detection_timestamp else ob.timestamp
                if ob_known_at >= ts:
                    continue

                # Not mitigated
                if ob.is_mitigated:
                    if ob.mitigation_timestamp and ob.mitigation_timestamp <= ts:
                        continue

                # Strength filter
                min_strength = OB_MIN_STRENGTH_SHORT if direction == 'short' else OB_MIN_STRENGTH
                if ob.strength < min_strength:
                    continue

                # Age filter
                ob_age = (ts - ob.timestamp).total_seconds() / 300
                if ob_age > OB_MAX_AGE:
                    continue

                # Direction match
                if direction == 'long' and not ob.is_bullish:
                    continue
                if direction == 'short' and ob.is_bullish:
                    continue

                # Price touching OB zone
                if direction == 'long':
                    if current_1m['low'] <= ob.top <= current_1m['high']:
                        matching_ob = ob
                        break
                else:
                    if current_1m['low'] <= ob.bottom <= current_1m['high']:
                        matching_ob = ob
                        break

            if not matching_ob:
                return None

            # === CREATE SIGNAL ===
            ob = matching_ob

            if direction == 'long':
                entry = ob.top
                sl = ob.bottom * (1 - SL_BUFFER_PCT / 100)
                sl_distance = entry - sl
                tp = entry + (sl_distance * RR_TARGET)
            else:
                entry = ob.bottom
                sl = ob.top * (1 + SL_BUFFER_PCT / 100)
                sl_distance = sl - entry
                tp = entry - (sl_distance * RR_TARGET)

            # Calculate leverage (same as backtest)
            sl_pct = abs(entry - sl) / entry * 100
            leverage = min(MAX_LEVERAGE, max(5, int(RISK_PER_TRADE_PCT / sl_pct)))

            # Partial TP price
            partial_tp_price = None
            if USE_PARTIAL_TP:
                if direction == 'long':
                    partial_tp_price = entry + (sl_distance * RR_TARGET * PARTIAL_TP_LEVEL)
                else:
                    partial_tp_price = entry - (sl_distance * RR_TARGET * PARTIAL_TP_LEVEL)

            ob_age_candles = int((ts - ob.timestamp).total_seconds() / 300)

            return LiveSignal(
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                sl_price=sl,
                tp_price=tp,
                leverage=leverage,
                ob_top=ob.top,
                ob_bottom=ob.bottom,
                use_partial_tp=USE_PARTIAL_TP,
                partial_tp_price=partial_tp_price,
                partial_size=PARTIAL_SIZE,
                timestamp=ts,
                ob_strength=ob.strength,
                ob_age_candles=ob_age_candles
            )

        except Exception as e:
            print(f"Error checking {symbol}: {e}")
            return None

    def scan_coins(self, coins: List[str], timeout_per_coin: int = 10) -> List[LiveSignal]:
        """Scan multiple coins for signals with per-coin timeout"""
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

        signals = []
        skipped = 0

        for symbol in coins:
            try:
                # Use thread with timeout to prevent hanging on slow API calls
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.get_signal, symbol)
                    signal = future.result(timeout=timeout_per_coin)

                if signal:
                    signals.append(signal)
                    print(f"  [SIGNAL] {symbol} {signal.direction.upper()} @ {signal.entry_price:.4f}")
            except FuturesTimeoutError:
                skipped += 1
                if skipped <= 3:  # Only log first few
                    print(f"  [SKIP] {symbol} timeout")
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    print(f"  [SKIP] {symbol}: {str(e)[:30]}")

        if skipped > 3:
            print(f"  ... and {skipped - 3} more skipped")

        return signals


def print_signal(signal: LiveSignal):
    """Print signal details"""
    print(f"\n{'='*50}")
    print(f"SIGNAL: {signal.symbol} {signal.direction.upper()}")
    print(f"{'='*50}")
    print(f"Entry:    {signal.entry_price:.6f}")
    print(f"SL:       {signal.sl_price:.6f} ({abs(signal.entry_price - signal.sl_price) / signal.entry_price * 100:.2f}%)")
    print(f"TP:       {signal.tp_price:.6f} ({abs(signal.tp_price - signal.entry_price) / signal.entry_price * 100:.2f}%)")
    print(f"Leverage: {signal.leverage}x")
    print(f"OB Age:   {signal.ob_age_candles} candles")
    if signal.use_partial_tp:
        print(f"Partial:  Close {signal.partial_size*100:.0f}% @ {signal.partial_tp_price:.6f}")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    from config.coins import get_top_n_coins

    print("OB Scalper Live - Signal Scanner")
    print("="*50)
    print(f"MTF: 1H + {'4H' if USE_4H_MTF else 'none'}")
    print(f"Daily: Shorts={'ON' if USE_DAILY_FOR_SHORTS else 'OFF'}, Longs={'ON' if USE_DAILY_FOR_LONGS else 'OFF'}")
    print(f"Partial TP: {'ON' if USE_PARTIAL_TP else 'OFF'}")
    print("="*50)

    scanner = OBScalperLive()
    coins = get_top_n_coins(30)

    print(f"\nScanning {len(coins)} coins...")
    signals = scanner.scan_coins(coins)

    print(f"\nFound {len(signals)} signals:")
    for signal in signals:
        print_signal(signal)
