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

# === DEBUG MODE (set DEBUG_SIGNALS=true in Railway to see filtering reasons) ===
DEBUG_SIGNALS = os.getenv('DEBUG_SIGNALS', 'false').lower() == 'true'

# === PARITY LOG (set PARITY_LOG=true to log detailed info for backtest comparison) ===
PARITY_LOG = os.getenv('PARITY_LOG', 'false').lower() == 'true'
PARITY_LOG_FILE = os.getenv('PARITY_LOG_FILE', 'parity_live.log')

# === CONFIGURATION (MATCHED to ob_scalper.py backtest!) ===
OB_MIN_STRENGTH = float(os.getenv('OB_MIN_STRENGTH', '0.8'))  # Same as backtest!
OB_MIN_STRENGTH_SHORT = float(os.getenv('OB_MIN_STRENGTH_SHORT', '0.9'))  # Same as backtest!
OB_MAX_AGE = int(os.getenv('OB_MAX_AGE', '100'))  # Same as backtest (was 50)
RR_TARGET = float(os.getenv('RR_TARGET', '2.0'))  # Same as backtest (was 1.5)
SL_BUFFER_PCT = float(os.getenv('SL_BUFFER_PCT', '0.05'))

# Volume Filter - SAME AS BACKTEST (confirms institutional interest)
MIN_VOLUME_RATIO = float(os.getenv('MIN_VOLUME_RATIO', '1.2'))  # 1.2x average volume
USE_VOLUME_FILTER = os.getenv('USE_VOLUME_FILTER', 'true').lower() == 'true'  # ON by default

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

    # For scoring/ranking
    current_price: float = None
    distance_to_entry_pct: float = None  # How far is current price from entry

    @property
    def score(self) -> float:
        """
        Score for ranking signals. Higher = better opportunity.
        Factors: OB strength, distance to entry (closer = higher score)
        """
        strength_score = self.ob_strength * 50  # 0-50 points

        # Distance score: closer to entry = higher score
        # 0% distance = 50 points, 2% distance = 0 points
        if self.distance_to_entry_pct is not None:
            distance_score = max(0, 50 - (self.distance_to_entry_pct * 25))
        else:
            distance_score = 25  # Default middle score

        # Fresher OB = bonus
        age_penalty = min(self.ob_age_candles * 0.5, 10)  # Max -10 points for old OBs

        return strength_score + distance_score - age_penalty


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA indicators - same as backtest"""
    df = df.copy()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    return df


def log_parity(symbol: str, data: dict):
    """Log detailed signal check info for parity comparison with backtest"""
    if not PARITY_LOG:
        return

    import json
    import numpy as np

    # Convert numpy types to Python types for JSON
    def convert(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        return str(obj)

    data['symbol'] = symbol
    data['log_time'] = datetime.utcnow().isoformat()
    clean_data = {k: convert(v) for k, v in data.items()}

    # Print to console (visible in Railway logs)
    print(f"[PARITY] {json.dumps(clean_data)}", flush=True)


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
            if USE_DAILY_FOR_SHORTS or USE_DAILY_FOR_LONGS:
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

    def get_signal(self, symbol: str, debug: bool = None) -> Optional[LiveSignal]:
        # Use ENV variable if debug not explicitly set
        if debug is None:
            debug = DEBUG_SIGNALS
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
            df_daily = self._get_cached(symbol, "D", 30) if (USE_DAILY_FOR_SHORTS or USE_DAILY_FOR_LONGS) else None

            if df_5m is None or len(df_5m) < 50:
                if debug: print(f"      {symbol}: SKIP - No 5m data")
                return None
            if df_1h is None or len(df_1h) < 20:
                if debug: print(f"      {symbol}: SKIP - No 1h data")
                return None

            # Add indicators
            df_5m = calculate_indicators(df_5m)
            df_1h = calculate_indicators(df_1h)
            if df_4h is not None:
                df_4h = calculate_indicators(df_4h)
            if df_daily is not None:
                df_daily = calculate_indicators(df_daily)

            # Add volume SMA for OB volume filter (SAME AS BACKTEST!)
            if 'volume' in df_5m.columns:
                df_5m['volume_sma'] = df_5m['volume'].rolling(20).mean()
                df_5m['volume_ratio'] = df_5m['volume'] / df_5m['volume_sma']

            # Detect OBs on 5min
            atr = df_5m['close'].rolling(14).apply(
                lambda x: pd.Series(x).diff().abs().mean() if len(x) > 1 else 0
            )
            obs = self.ob_detector.detect(df_5m, atr)

            if not obs:
                if debug: print(f"      {symbol}: SKIP - No OBs detected")
                return None

            if debug: print(f"      {symbol}: Found {len(obs)} OBs")

            # Get current 5m candle (no 1m needed!)
            current_5m = df_5m.iloc[-1]
            ts = current_5m['timestamp']
            current_price = current_5m['close']

            # === 1H MTF CHECK ===
            # Use the last COMPLETED 1H candle (same logic as backtest!)
            # Check if last candle is still forming (timestamp is current hour)
            now = datetime.utcnow()
            current_hour = now.replace(minute=0, second=0, microsecond=0)

            if df_1h.iloc[-1]['timestamp'] >= current_hour:
                # Last candle is current (incomplete) - use iloc[-2]
                h1_candle = df_1h.iloc[-2]
            else:
                # Last candle is complete - use iloc[-1]
                h1_candle = df_1h.iloc[-1]

            h1_bullish = h1_candle['close'] > h1_candle['ema20'] > h1_candle['ema50']
            h1_bearish = h1_candle['close'] < h1_candle['ema20'] < h1_candle['ema50']

            if debug:
                print(f"      {symbol}: 1H candle ts={h1_candle['timestamp']} close={h1_candle['close']:.4f} ema20={h1_candle['ema20']:.4f} ema50={h1_candle['ema50']:.4f}")
                print(f"      {symbol}: 1H bullish={h1_bullish} bearish={h1_bearish}")

            if not h1_bullish and not h1_bearish:
                if debug: print(f"      {symbol}: SKIP - No clear 1H trend (neither bullish nor bearish)")
                log_parity(symbol, {
                    'signal': False,
                    'skip_reason': 'no_1h_trend',
                    'check_time': now.isoformat(),
                    '5m_candle_ts': str(ts),
                    '1h_candle_ts': str(h1_candle['timestamp']),
                    '1h_close': float(h1_candle['close']),
                    '1h_ema20': float(h1_candle['ema20']),
                    '1h_ema50': float(h1_candle['ema50']),
                    '1h_bullish': h1_bullish,
                    '1h_bearish': h1_bearish,
                })
                return None  # No clear 1H trend

            # === 4H MTF CHECK ===
            if USE_4H_MTF and df_4h is not None and len(df_4h) > 1:
                # Use the last COMPLETED 4H candle (same logic as backtest!)
                current_4h = now.replace(hour=(now.hour // 4) * 4, minute=0, second=0, microsecond=0)
                if df_4h.iloc[-1]['timestamp'] >= current_4h:
                    h4_candle = df_4h.iloc[-2]
                else:
                    h4_candle = df_4h.iloc[-1]
                h4_bullish = h4_candle['close'] > h4_candle['ema20'] > h4_candle['ema50']
                h4_bearish = h4_candle['close'] < h4_candle['ema20'] < h4_candle['ema50']

                if debug:
                    print(f"      {symbol}: 4H candle ts={h4_candle['timestamp']} close={h4_candle['close']:.4f} ema20={h4_candle['ema20']:.4f} ema50={h4_candle['ema50']:.4f}")
                    print(f"      {symbol}: 4H bullish={h4_bullish} bearish={h4_bearish}")

                # 4H must confirm 1H
                if h1_bullish and not h4_bullish:
                    if debug: print(f"      {symbol}: SKIP - 1H bullish but 4H NOT bullish")
                    log_parity(symbol, {
                        'signal': False,
                        'skip_reason': '4h_mismatch',
                        'check_time': now.isoformat(),
                        '1h_bullish': True,
                        '4h_bullish': h4_bullish,
                        '4h_bearish': h4_bearish,
                    })
                    return None
                if h1_bearish and not h4_bearish:
                    if debug: print(f"      {symbol}: SKIP - 1H bearish but 4H NOT bearish")
                    log_parity(symbol, {
                        'signal': False,
                        'skip_reason': '4h_mismatch',
                        'check_time': now.isoformat(),
                        '1h_bearish': True,
                        '4h_bullish': h4_bullish,
                        '4h_bearish': h4_bearish,
                    })
                    return None

            # Direction
            direction = 'long' if h1_bullish else 'short'
            if debug: print(f"      {symbol}: Direction = {direction}")

            # === DAILY FILTER FOR SHORTS ONLY ===
            if USE_DAILY_FOR_SHORTS and direction == 'short' and df_daily is not None and len(df_daily) > 1:
                # Use the last COMPLETED daily candle (same logic as backtest!)
                current_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
                if df_daily.iloc[-1]['timestamp'] >= current_day:
                    daily_candle = df_daily.iloc[-2]
                else:
                    daily_candle = df_daily.iloc[-1]
                daily_bearish = daily_candle['close'] < daily_candle['ema20'] < daily_candle['ema50']
                if debug:
                    print(f"      {symbol}: Daily close={daily_candle['close']:.4f} ema20={daily_candle['ema20']:.4f} ema50={daily_candle['ema50']:.4f}")
                    print(f"      {symbol}: Daily bearish={daily_bearish}")
                if not daily_bearish:
                    if debug: print(f"      {symbol}: SKIP - Short but Daily NOT bearish")
                    log_parity(symbol, {
                        'signal': False,
                        'skip_reason': 'daily_not_bearish',
                        'check_time': now.isoformat(),
                        'direction': 'short',
                        'daily_bearish': daily_bearish,
                    })
                    return None  # Daily not bearish - skip short

            # === DAILY FILTER FOR LONGS (disabled by default) ===
            if USE_DAILY_FOR_LONGS and direction == 'long' and df_daily is not None and len(df_daily) > 1:
                # Use the last COMPLETED daily candle
                current_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
                if df_daily.iloc[-1]['timestamp'] >= current_day:
                    daily_candle = df_daily.iloc[-2]
                else:
                    daily_candle = df_daily.iloc[-1]
                daily_bullish = daily_candle['close'] > daily_candle['ema20'] > daily_candle['ema50']
                if not daily_bullish:
                    if debug: print(f"      {symbol}: SKIP - Long but Daily NOT bullish")
                    return None  # Daily not bullish - skip long

            # === FIND VALID OB (no price touch check - just find best OB!) ===
            best_ob = None
            best_ob_age = float('inf')

            # Debug counters
            ob_mitigated = 0
            ob_weak = 0
            ob_old = 0
            ob_wrong_dir = 0
            ob_low_vol = 0

            for ob in obs:
                # Not mitigated
                if ob.is_mitigated:
                    ob_mitigated += 1
                    continue

                # Strength filter
                min_strength = OB_MIN_STRENGTH_SHORT if direction == 'short' else OB_MIN_STRENGTH
                if ob.strength < min_strength:
                    ob_weak += 1
                    continue

                # Volume filter (SAME AS BACKTEST - confirms institutional interest)
                if USE_VOLUME_FILTER:
                    if hasattr(ob, 'volume_ratio') and ob.volume_ratio < MIN_VOLUME_RATIO:
                        ob_low_vol += 1
                        continue  # Low volume OB - skip

                # Age filter (in 5min candles)
                ob_age = (ts - ob.timestamp).total_seconds() / 300
                if ob_age > OB_MAX_AGE or ob_age < 0:
                    ob_old += 1
                    continue

                # Direction match
                if direction == 'long' and not ob.is_bullish:
                    ob_wrong_dir += 1
                    continue
                if direction == 'short' and ob.is_bullish:
                    ob_wrong_dir += 1
                    continue

                # Pick the FRESHEST valid OB (lowest age)
                if ob_age < best_ob_age:
                    best_ob = ob
                    best_ob_age = ob_age

            if debug:
                print(f"      {symbol}: OB filter: {len(obs)} total, {ob_mitigated} mitigated, {ob_weak} weak, {ob_low_vol} low_vol, {ob_old} old, {ob_wrong_dir} wrong_dir")

            if not best_ob:
                if debug: print(f"      {symbol}: SKIP - No valid OB after filtering")
                # Parity log for no signal
                log_parity(symbol, {
                    'signal': False,
                    'skip_reason': 'no_valid_ob',
                    'check_time': now.isoformat(),
                    '5m_candle_ts': str(ts),
                    '1h_candle_ts': str(h1_candle['timestamp']),
                    '1h_bullish': h1_bullish,
                    '1h_bearish': h1_bearish,
                    'direction': direction,
                    'obs_total': len(obs),
                    'obs_mitigated': ob_mitigated,
                    'obs_weak': ob_weak,
                    'obs_low_vol': ob_low_vol,
                    'obs_old': ob_old,
                    'obs_wrong_dir': ob_wrong_dir,
                })
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

            # Calculate distance to entry for scoring
            distance_to_entry_pct = abs(current_price - entry) / entry * 100

            # Parity log for signal found
            log_parity(symbol, {
                'signal': True,
                'check_time': now.isoformat(),
                '5m_candle_ts': str(ts),
                '1h_candle_ts': str(h1_candle['timestamp']),
                '1h_bullish': h1_bullish,
                '1h_bearish': h1_bearish,
                'direction': direction,
                'obs_total': len(obs),
                'obs_valid': 1,
                'chosen_ob_ts': str(ob.timestamp),
                'chosen_ob_strength': ob.strength,
                'chosen_ob_age': ob_age_candles,
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'current_price': current_price,
                'distance_to_entry_pct': distance_to_entry_pct,
            })

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
                ob_age_candles=ob_age_candles,
                current_price=current_price,
                distance_to_entry_pct=distance_to_entry_pct,
            )

        except Exception as e:
            print(f"Error checking {symbol}: {e}")
            return None

    def scan_coins(self, coins: List[str], timeout_per_coin: int = 30, debug: bool = None) -> List[LiveSignal]:
        """Scan multiple coins for signals with per-coin timeout"""
        import time
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

        # Use ENV variable if debug not explicitly set
        if debug is None:
            debug = DEBUG_SIGNALS

        signals = []
        skipped = 0
        slow_coins = []
        total = len(coins)
        scan_start = time.time()

        # Filter stats (for debug summary)
        filter_stats = {
            'no_data': 0,
            'no_obs': 0,
            'no_1h_trend': 0,
            '4h_mismatch': 0,
            'daily_mismatch': 0,
            'no_valid_ob': 0,
            'signal': 0,
            'error': 0
        }

        if debug:
            print(f"\n{'='*60}", flush=True)
            print(f"DEBUG MODE - Settings:", flush=True)
            print(f"  OB_MIN_STRENGTH: {OB_MIN_STRENGTH} (short: {OB_MIN_STRENGTH_SHORT})", flush=True)
            print(f"  OB_MAX_AGE: {OB_MAX_AGE} candles ({OB_MAX_AGE * 5}min)", flush=True)
            print(f"  USE_VOLUME_FILTER: {USE_VOLUME_FILTER} (min: {MIN_VOLUME_RATIO}x)", flush=True)
            print(f"  USE_4H_MTF: {USE_4H_MTF}", flush=True)
            print(f"  USE_DAILY_FOR_SHORTS: {USE_DAILY_FOR_SHORTS}", flush=True)
            print(f"  USE_DAILY_FOR_LONGS: {USE_DAILY_FOR_LONGS}", flush=True)
            print(f"{'='*60}\n", flush=True)

        print(f"    Scanning {total} coins... (debug={debug})", flush=True)

        for i, symbol in enumerate(coins):
            # Progress every 10 coins (unless debug mode)
            if not debug and i > 0 and i % 10 == 0:
                elapsed = time.time() - scan_start
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                pct = int(i / total * 100)
                print(f"    {pct}% [{i}/{total}] {len(signals)} sig, {skipped} skip - {eta:.0f}s left", flush=True)

            coin_start = time.time()
            try:
                # Use thread with timeout to prevent hanging
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.get_signal, symbol, debug)
                    signal = future.result(timeout=timeout_per_coin)

                coin_time = time.time() - coin_start

                if signal:
                    signals.append(signal)
                    print(f"  ★ {symbol} {signal.direction.upper()} @ {signal.entry_price:.4f}", flush=True)
                elif coin_time > 10:
                    slow_coins.append(symbol)

            except FuturesTimeout:
                skipped += 1
                slow_coins.append(symbol)
                print(f"  [TIMEOUT] {symbol} >{timeout_per_coin}s - skipping", flush=True)

            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  [skip] {symbol}: {str(e)[:30]}", flush=True)

            # Small delay every 5 coins to avoid rate limits
            if i > 0 and i % 5 == 0:
                time.sleep(0.05)  # 50ms

        total_time = time.time() - scan_start
        print(f"    Done: {len(signals)} signals in {total_time:.1f}s ({skipped} skipped)", flush=True)

        if debug and len(signals) == 0:
            print(f"\n    ⚠️  NO SIGNALS FOUND - Check Railway logs above for per-coin reasons", flush=True)
            print(f"    Common causes:", flush=True)
            print(f"    - 1H trend unclear (close not > ema20 > ema50 for bullish)", flush=True)
            print(f"    - 4H doesn't confirm 1H direction", flush=True)
            print(f"    - Daily not bearish for shorts", flush=True)
            print(f"    - No OBs with strength >= {OB_MIN_STRENGTH}", flush=True)
            print(f"    - All OBs older than {OB_MAX_AGE} candles ({OB_MAX_AGE * 5}min)", flush=True)

        return signals

    def debug_scan(self, coins: List[str], max_coins: int = 10) -> Dict:
        """
        Debug scan that shows exactly why coins are filtered out.
        Returns detailed stats.
        """
        print(f"\n{'='*60}")
        print(f"DEBUG SCAN - Analyzing first {max_coins} coins")
        print(f"{'='*60}")
        print(f"Settings:")
        print(f"  OB_MIN_STRENGTH: {OB_MIN_STRENGTH}")
        print(f"  OB_MAX_AGE: {OB_MAX_AGE} candles")
        print(f"  USE_4H_MTF: {USE_4H_MTF}")
        print(f"  USE_DAILY_FOR_SHORTS: {USE_DAILY_FOR_SHORTS}")
        print(f"  USE_DAILY_FOR_LONGS: {USE_DAILY_FOR_LONGS}")
        print(f"{'='*60}\n")

        signals = []
        for symbol in coins[:max_coins]:
            print(f"\n--- {symbol} ---")
            signal = self.get_signal(symbol, debug=True)
            if signal:
                signals.append(signal)
                print(f"  ✓ SIGNAL: {signal.direction.upper()} @ {signal.entry_price:.4f}")
            else:
                print(f"  ✗ No signal")

        print(f"\n{'='*60}")
        print(f"RESULT: {len(signals)}/{max_coins} coins have signals")
        print(f"{'='*60}\n")

        return {"signals": signals, "total": max_coins, "found": len(signals)}


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
