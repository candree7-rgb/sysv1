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

        # Data cache - don't re-download every scan
        self._data_cache = {}  # {cache_key: (df, timestamp)}
        self._cache_duration = {
            '1': 60,          # 1m data: cache 1 min
            '5': 5 * 60,      # 5m data: cache 5 min
            '60': 15 * 60,    # 1H data: cache 15 min
            '240': 60 * 60,   # 4H data: cache 1 hour
            'D': 4 * 60 * 60  # Daily data: cache 4 hours
        }
        self._preloaded = False

    def preload_data(self, coins: List[str]):
        """Pre-load HTF data for all coins to speed up scanning"""
        import time
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

        total = len(coins)
        print(f"    [PRELOAD] Loading HTF data for {total} coins...", flush=True)
        start = time.time()
        loaded = 0
        errors = 0
        skipped = []

        def load_single_coin(symbol):
            """Load HTF data for a single coin"""
            self._get_cached(symbol, "60", 3)   # 1H: 3 days
            if USE_4H_MTF:
                self._get_cached(symbol, "240", 7)   # 4H: 7 days
            if USE_DAILY_FOR_SHORTS:
                self._get_cached(symbol, "D", 14)    # Daily: 14 days
            return True

        for i, symbol in enumerate(coins):
            # Progress every 10 coins
            if i > 0 and (i % 10 == 0 or i == total - 1):
                pct = int((i / total) * 100)
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                print(f"    [PRELOAD] {pct}% ({i}/{total}) - {loaded} OK, {errors} skip - ETA {eta:.0f}s", flush=True)

            try:
                # Use thread with 30 second timeout per coin
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(load_single_coin, symbol)
                    future.result(timeout=30)  # 30 second max per coin
                loaded += 1

            except FuturesTimeout:
                errors += 1
                skipped.append(symbol)
                if errors <= 3:
                    print(f"    [PRELOAD] TIMEOUT {symbol} (>30s)", flush=True)

            except Exception as e:
                errors += 1
                skipped.append(symbol)
                if errors <= 3:
                    print(f"    [PRELOAD] Skip {symbol}: {str(e)[:30]}", flush=True)

            # Small delay every 5 coins to avoid rate limits
            if i > 0 and i % 5 == 0:
                time.sleep(0.1)  # 100ms pause

        elapsed = time.time() - start
        print(f"    [PRELOAD] Done! {loaded}/{total} coins in {elapsed:.1f}s", flush=True)
        if skipped:
            print(f"    [PRELOAD] Skipped: {', '.join(skipped[:5])}{'...' if len(skipped) > 5 else ''}", flush=True)
        self._preloaded = True

    def _get_cached(self, symbol: str, interval: str, days: int):
        """Get data with caching to reduce API calls"""
        import time
        now = time.time()

        cache_key = f"{symbol}_{interval}"
        if cache_key in self._data_cache:
            df, cached_at = self._data_cache[cache_key]
            cache_duration = self._cache_duration.get(interval, 300)
            if now - cached_at < cache_duration:
                return df  # Return cached

        # Download fresh
        df = self.dl.load_or_download(symbol, interval, days)
        if df is not None and len(df) > 0:
            self._data_cache[cache_key] = (df, now)
        return df

    def get_signal(self, symbol: str, debug: bool = False) -> Optional[LiveSignal]:
        """
        Check if there's a valid OB setup for a symbol.

        SIMPLIFIED: No 1m check needed! We just detect OB and place limit order.
        Bybit handles whether price reaches it or not.
        """
        try:
            # Load data - NO 1M NEEDED for live!
            # We just detect OB on 5m and check MTF alignment
            df_5m = self._get_cached(symbol, "5", 1)      # 5m for OB detection
            df_1h = self._get_cached(symbol, "60", 7)     # 1H for MTF
            df_4h = self._get_cached(symbol, "240", 14) if USE_4H_MTF else None
            df_daily = self._get_cached(symbol, "D", 30) if USE_DAILY_FOR_SHORTS else None

            if df_5m is None or len(df_5m) < 50:
                if debug: print(f"      {symbol}: No 5m data")
                return None
            if df_1h is None or len(df_1h) < 20:
                if debug: print(f"      {symbol}: No 1h data")
                return None

            # Add indicators
            df_5m = calculate_indicators(df_5m)
            df_1h = calculate_indicators(df_1h)
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

            # Get current 5m candle (no 1m needed!)
            current_5m = df_5m.iloc[-1]
            ts = current_5m['timestamp']
            current_price = current_5m['close']

            # === 1H MTF CHECK ===
            h1_candle = df_1h.iloc[-2]  # Use completed 1H candle

            h1_bullish = h1_candle['close'] > h1_candle['ema20'] > h1_candle['ema50']
            h1_bearish = h1_candle['close'] < h1_candle['ema20'] < h1_candle['ema50']

            if not h1_bullish and not h1_bearish:
                return None  # No clear 1H trend

            # === 4H MTF CHECK ===
            if USE_4H_MTF and df_4h is not None and len(df_4h) > 1:
                h4_candle = df_4h.iloc[-2]  # Use completed 4H candle
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
            if USE_DAILY_FOR_SHORTS and direction == 'short' and df_daily is not None and len(df_daily) > 1:
                daily_candle = df_daily.iloc[-2]  # Use completed daily candle
                daily_bearish = daily_candle['close'] < daily_candle['ema20'] < daily_candle['ema50']
                if not daily_bearish:
                    return None  # Daily not bearish - skip short

            # === DAILY FILTER FOR LONGS (disabled by default) ===
            if USE_DAILY_FOR_LONGS and direction == 'long' and df_daily is not None and len(df_daily) > 1:
                daily_candle = df_daily.iloc[-2]  # Use completed daily candle
                daily_bullish = daily_candle['close'] > daily_candle['ema20'] > daily_candle['ema50']
                if not daily_bullish:
                    return None  # Daily not bullish - skip long

            # === FIND VALID OB (no price touch check - just find best OB!) ===
            best_ob = None
            best_ob_age = float('inf')

            for ob in obs:
                # Not mitigated
                if ob.is_mitigated:
                    continue

                # Strength filter
                min_strength = OB_MIN_STRENGTH_SHORT if direction == 'short' else OB_MIN_STRENGTH
                if ob.strength < min_strength:
                    continue

                # Age filter (in 5min candles)
                ob_age = (ts - ob.timestamp).total_seconds() / 300
                if ob_age > OB_MAX_AGE or ob_age < 0:
                    continue

                # Direction match
                if direction == 'long' and not ob.is_bullish:
                    continue
                if direction == 'short' and ob.is_bullish:
                    continue

                # Pick the FRESHEST valid OB (lowest age)
                if ob_age < best_ob_age:
                    best_ob = ob
                    best_ob_age = ob_age

            if not best_ob:
                return None

            matching_ob = best_ob

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
        """Scan multiple coins for signals with rate limiting"""
        import time

        signals = []
        skipped = 0
        total = len(coins)
        scan_start = time.time()

        print(f"    Scanning {total} coins...", flush=True)

        for i, symbol in enumerate(coins):
            # Progress every 20 coins
            if i > 0 and i % 20 == 0:
                elapsed = time.time() - scan_start
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                print(f"    [{i}/{total}] {len(signals)} signals, {skipped} skip - ETA {eta:.0f}s", flush=True)

            coin_start = time.time()
            try:
                signal = self.get_signal(symbol)
                coin_time = time.time() - coin_start

                if signal:
                    signals.append(signal)
                    print(f"  ★ {symbol} {signal.direction.upper()} @ {signal.entry_price:.4f}", flush=True)

            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    print(f"  [skip] {symbol}: {str(e)[:30]}", flush=True)

            # Small delay every 5 coins to avoid rate limits
            if i > 0 and i % 5 == 0:
                time.sleep(0.05)  # 50ms

        total_time = time.time() - scan_start
        print(f"    Done: {len(signals)} signals in {total_time:.1f}s", flush=True)

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
