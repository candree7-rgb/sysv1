"""
OB Scalper Strategy - 1min Precision with OB-based SL
======================================================
Uses 5min OBs for zone detection, 1min for precise entries.
SL at OB edge (logical invalidation), TP at configurable R:R.

NO LOOK-AHEAD BIAS:
- OBs only used after detection_timestamp (impulse confirmation)
- Entry only after signal candle closes
- MTF data uses only past candles
"""

import os
import sys
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configuration via ENV
NUM_WORKERS = int(os.getenv('NUM_WORKERS', str(min(4, cpu_count()))))
TOTAL_TIMEOUT = int(os.getenv('TOTAL_TIMEOUT', '300'))
OB_MIN_STRENGTH = float(os.getenv('OB_MIN_STRENGTH', '0.8'))
OB_MIN_STRENGTH_SHORT = float(os.getenv('OB_MIN_STRENGTH_SHORT', '0.9'))  # Stricter for shorts
OB_MAX_AGE = int(os.getenv('OB_MAX_AGE', '100'))  # in 5min candles
RR_TARGET = float(os.getenv('RR_TARGET', '2.0'))  # TP = RR_TARGET * SL
SL_BUFFER_PCT = float(os.getenv('SL_BUFFER_PCT', '0.05'))  # Buffer beyond OB edge

# ORDER EXPIRY - same as live! If price doesn't touch OB within X minutes, skip
# This simulates limit order cancellation in live trading
MAX_ORDER_AGE_MIN = int(os.getenv('MAX_ORDER_AGE_MIN', '30'))  # 30 min like live

# Volume Filter - OB must have above-average volume (confirms institutional interest)
MIN_VOLUME_RATIO = float(os.getenv('MIN_VOLUME_RATIO', '1.2'))  # 1.2x average volume
USE_VOLUME_FILTER = os.getenv('USE_VOLUME_FILTER', 'true').lower() == 'true'  # ON by default

# RSI Filter - DISABLED by default (too restrictive, filters good trades)
RSI_LONG_MAX = int(os.getenv('RSI_LONG_MAX', '45'))
RSI_SHORT_MIN = int(os.getenv('RSI_SHORT_MIN', '55'))
USE_RSI_FILTER = os.getenv('USE_RSI_FILTER', 'false').lower() == 'true'  # OFF by default

# 4H MTF Filter - requires 4H trend alignment in addition to 1H
USE_4H_MTF = os.getenv('USE_4H_MTF', 'true').lower() == 'true'  # ON by default

# Daily MTF Filter for Shorts - requires Daily bearish for shorts (stricter)
USE_DAILY_FOR_SHORTS = os.getenv('USE_DAILY_FOR_SHORTS', 'true').lower() == 'true'  # ON by default

# Daily MTF Filter for Longs - requires Daily bullish for longs (stricter)
USE_DAILY_FOR_LONGS = os.getenv('USE_DAILY_FOR_LONGS', 'false').lower() == 'true'  # OFF by default

# Trade Direction - allows testing long/short independently
# Options: "both", "long", "short"
TRADE_DIRECTION = os.getenv('TRADE_DIRECTION', 'both').lower()

# Break-Even: Move SL to entry when price reaches X% toward TP
# 80% is conservative - only triggers when almost at TP
BE_THRESHOLD = float(os.getenv('BE_THRESHOLD', '0.8'))  # 80% toward TP (was 50%)
USE_BE = os.getenv('USE_BE', 'false').lower() == 'true'  # OFF by default (global)
USE_BE_SHORTS = os.getenv('USE_BE_SHORTS', 'false').lower() == 'true'  # BE only for shorts

# === DD REDUCTION OPTIONS ===
# Option 1: Trailing Stop (moves SL progressively as price moves in favor)
USE_TRAILING = os.getenv('USE_TRAILING', 'false').lower() == 'true'
TRAIL_START = float(os.getenv('TRAIL_START', '0.3'))  # Start trailing at 30% toward TP
TRAIL_STEP = float(os.getenv('TRAIL_STEP', '0.25'))  # Move SL by 25% of profit

# Option 2: Time Exit (close after X bars if no TP)
USE_TIME_EXIT = os.getenv('USE_TIME_EXIT', 'false').lower() == 'true'
MAX_BARS = int(os.getenv('MAX_BARS', '60'))  # Max 60 1min bars = 1 hour

# Option 3: Max Leverage Cap
MAX_LEVERAGE = int(os.getenv('MAX_LEVERAGE', '50'))  # Max 50x leverage
MAX_MARGIN_PCT = float(os.getenv('MAX_MARGIN_PCT', '0.50'))  # Max 50% margin per trade
RISK_PER_TRADE_PCT = float(os.getenv('RISK_PER_TRADE_PCT', '2.0'))  # Target risk % per trade

# Option 4: Partial Take Profit (lock in profits early, let remainder run)
USE_PARTIAL_TP = os.getenv('USE_PARTIAL_TP', 'false').lower() == 'true'
PARTIAL_TP_LEVEL = float(os.getenv('PARTIAL_TP_LEVEL', '0.5'))  # Close partial at 50% toward TP
PARTIAL_SIZE = float(os.getenv('PARTIAL_SIZE', '0.5'))  # Close 50% of position

# Date Range: Skip recent days for historical testing
# DAYS=30 SKIP_DAYS=0  → Last 30 days (default)
# DAYS=30 SKIP_DAYS=30 → Days 30-60 ago
# DAYS=30 SKIP_DAYS=60 → Days 60-90 ago
SKIP_DAYS = int(os.getenv('SKIP_DAYS', '0'))

# Fees (Bybit Futures)
MAKER_FEE = 0.0002  # 0.02%
TAKER_FEE = 0.00055  # 0.055%

# Position limits
MAX_CONCURRENT = int(os.getenv('MAX_CONCURRENT', '1'))

# Parity Log - for comparing backtest vs live signals
PARITY_LOG = os.getenv('PARITY_LOG', 'false').lower() == 'true'
PARITY_LOG_FILE = os.getenv('PARITY_LOG_FILE', 'parity_backtest.log')


def log_parity(symbol: str, data: dict):
    """Log detailed signal check info for parity comparison with live"""
    if not PARITY_LOG:
        return
    import json
    data['symbol'] = symbol

    # Print to console
    print(f"[PARITY] {json.dumps(data)}", flush=True)


@dataclass
class ScalpTrade:
    """A single scalp trade"""
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    sl_price: float
    tp_price: float
    entry_time: datetime
    ob_top: float
    ob_bottom: float
    leverage: int = 10

    # Dynamic SL for BE/Trailing
    current_sl: float = None  # Tracks SL (may move for BE/trailing)
    be_triggered: bool = False  # Has BE been activated?
    max_profit_price: float = None  # Peak price reached
    trail_level: int = 0  # Current trailing level (0=none, 1=BE, 2+=profit locked)
    bars_in_trade: int = 0  # Bars since entry (for time exit)

    # Partial TP tracking
    partial_closed: bool = False  # Has partial TP been taken?
    partial_pnl: float = 0.0  # PnL from partial close (locked in)

    # Results (filled after exit)
    exit_price: float = None
    exit_time: datetime = None
    exit_reason: str = None  # 'tp', 'sl', 'be'
    pnl_pct: float = None

    pnl_gross: float = None

    def __post_init__(self):
        if self.current_sl is None:
            self.current_sl = self.sl_price
        if self.max_profit_price is None:
            self.max_profit_price = self.entry_price


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_indicators(df: pd.DataFrame, include_rsi: bool = False) -> pd.DataFrame:
    """Add EMA and optionally RSI indicators"""
    df = df.copy()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    if include_rsi:
        df['rsi'] = calculate_rsi(df['close'], 14)
    return df


def process_coin(args) -> List[ScalpTrade]:
    """Process a single coin - worker function"""
    symbol, days = args

    print(f"      [Worker] {symbol}...", flush=True)

    # Disable SSL warnings in worker
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    import urllib3
    urllib3.disable_warnings()

    from data import BybitDataDownloader
    from smc import OrderBlockDetector

    trades = []

    try:
        dl = BybitDataDownloader()
        ob_det = OrderBlockDetector(min_strength=0.5)  # Detect all, filter later

        # Calculate date range with SKIP_DAYS
        # SKIP_DAYS=0: last N days | SKIP_DAYS=30: days 30-60 ago, etc.
        total_days_needed = days + SKIP_DAYS

        # Load data (extra buffer for indicators and OB detection)
        df_5m = dl.load_or_download(symbol, "5", total_days_needed + 10)
        if df_5m is None or len(df_5m) < 200:
            return []

        df_1m = dl.load_or_download(symbol, "1", total_days_needed + 5)
        if df_1m is None or len(df_1m) < 500:
            return []

        df_1h = dl.load_or_download(symbol, "60", total_days_needed + 30)
        if df_1h is None or len(df_1h) < 50:
            return []

        # Load 4H data if MTF filter enabled
        df_4h = None
        if USE_4H_MTF:
            df_4h = dl.load_or_download(symbol, "240", total_days_needed + 60)
            if df_4h is None or len(df_4h) < 20:
                df_4h = None  # Fallback: skip 4H filter for this coin

        # Load Daily data for short filter
        df_daily = None
        if USE_DAILY_FOR_SHORTS:
            df_daily = dl.load_or_download(symbol, "D", total_days_needed + 90)
            if df_daily is None or len(df_daily) < 20:
                df_daily = None

        # Apply date filter: only keep data in target date range
        # Use timezone-naive timestamp to match data format
        now = pd.Timestamp.now()
        # End date: SKIP_DAYS ago | Start date: SKIP_DAYS + days ago
        end_date = now - pd.Timedelta(days=SKIP_DAYS)
        start_date = now - pd.Timedelta(days=SKIP_DAYS + days)

        # Filter ALL timeframes to target range (critical for OB detection!)
        # Need buffer before start_date for indicators and OB detection
        buffer_start = start_date - pd.Timedelta(days=5)  # 5 day buffer for indicators

        df_1m = df_1m[(df_1m['timestamp'] >= start_date) & (df_1m['timestamp'] <= end_date)]
        df_5m = df_5m[(df_5m['timestamp'] >= buffer_start) & (df_5m['timestamp'] <= end_date)]

        if len(df_1m) < 500:
            return []  # Not enough data in range
        if len(df_5m) < 100:
            return []  # Not enough 5m data for OB detection

        # Add indicators (RSI on 1min for pullback confirmation)
        df_5m = calculate_indicators(df_5m)
        df_1h = calculate_indicators(df_1h)
        df_1m = calculate_indicators(df_1m, include_rsi=True)  # RSI for entry filter
        if df_4h is not None:
            df_4h = calculate_indicators(df_4h)
        if df_daily is not None:
            df_daily = calculate_indicators(df_daily)

        # Add volume SMA for OB volume filter
        if 'volume' in df_5m.columns:
            df_5m['volume_sma'] = df_5m['volume'].rolling(20).mean()

        # Detect OBs on 5min (with detection_timestamp!)
        obs = ob_det.detect(df_5m, df_5m['close'].rolling(14).apply(
            lambda x: pd.Series(x).diff().abs().mean() if len(x) > 1 else 0
        ))

        if not obs:
            return []

        # Run backtest
        trades = run_backtest(symbol, df_5m, df_1m, df_1h, obs, df_4h, df_daily)

    except Exception as e:
        print(f"      [Error] {symbol}: {str(e)[:50]}", flush=True)
        return []

    return trades


def run_backtest(
    symbol: str,
    df_5m: pd.DataFrame,
    df_1m: pd.DataFrame,
    df_1h: pd.DataFrame,
    obs: list,
    df_4h: pd.DataFrame = None,
    df_daily: pd.DataFrame = None
) -> List[ScalpTrade]:
    """
    Run backtest for a single coin.

    Logic:
    1. Iterate through 1min candles
    2. Check for valid OB setup (5min OB, 1H MTF aligned)
    3. Entry when 1min touches OB edge
    4. SL at OB opposite edge + buffer
    5. TP at entry + RR_TARGET * SL_distance
    """
    trades = []
    active_trade = None
    used_obs = set()  # Track OBs that have been traded (same as live!)

    # Create timestamp index for fast lookup
    df_5m_indexed = df_5m.set_index('timestamp')
    df_1h_indexed = df_1h.set_index('timestamp')

    # Start from enough data for indicators
    start_idx = 100

    for idx in range(start_idx, len(df_1m)):
        candle = df_1m.iloc[idx]
        ts = candle['timestamp']

        # === CHECK ACTIVE TRADE EXIT ===
        if active_trade:
            t = active_trade  # Shorthand
            t.bars_in_trade += 1

            # === ENTRY CANDLE: Only check SL, no profit tracking ===
            # On the first candle, we don't know if price went up first or down first.
            # Conservative approach: assume no profit on entry candle, only check SL.
            if t.bars_in_trade == 1:
                # Entry candle - only check if SL was hit
                profit_pct = 0
                tp_distance = abs(t.tp_price - t.entry_price)

                # Check SL hit on entry candle
                sl_hit = False
                if t.direction == 'long':
                    sl_hit = candle['low'] <= t.current_sl
                else:
                    sl_hit = candle['high'] >= t.current_sl

                if sl_hit:
                    t.exit_price = t.current_sl
                    t.exit_reason = 'sl'
            else:
                # === CANDLE 2+: Conservative profit tracking ===
                # SIMPLE RULE: If SL could be hit, don't update profit tracking
                # This prevents partial TP triggering when price just drops to SL
                sl_could_be_hit = False
                if t.direction == 'long':
                    sl_could_be_hit = candle['low'] <= t.current_sl
                else:
                    sl_could_be_hit = candle['high'] >= t.current_sl

                if not sl_could_be_hit:
                    # TP was likely first - update profit tracking
                    if t.direction == 'long':
                        t.max_profit_price = max(t.max_profit_price, candle['high'])
                        tp_distance = t.tp_price - t.entry_price
                        current_profit = t.max_profit_price - t.entry_price
                        profit_pct = current_profit / tp_distance if tp_distance > 0 else 0
                    else:
                        t.max_profit_price = min(t.max_profit_price, candle['low'])
                        tp_distance = t.entry_price - t.tp_price
                        current_profit = t.entry_price - t.max_profit_price
                        profit_pct = current_profit / tp_distance if tp_distance > 0 else 0
                else:
                    # SL was first - don't update max_profit
                    if t.direction == 'long':
                        tp_distance = t.tp_price - t.entry_price
                        current_profit = t.max_profit_price - t.entry_price
                        profit_pct = current_profit / tp_distance if tp_distance > 0 else 0
                    else:
                        tp_distance = t.entry_price - t.tp_price
                        current_profit = t.entry_price - t.max_profit_price
                        profit_pct = current_profit / tp_distance if tp_distance > 0 else 0

            # === TRAILING STOP LOGIC ===
            # Skip if already exited (e.g., SL on entry candle)
            if not t.exit_reason and USE_TRAILING and profit_pct >= TRAIL_START:
                # Calculate new SL based on profit locked
                # Start at BE (entry), then move toward TP as profit increases
                profit_to_lock = profit_pct * TRAIL_STEP  # Lock % of current profit
                if t.direction == 'long':
                    new_sl = t.entry_price + (tp_distance * profit_to_lock)
                    if new_sl > t.current_sl:
                        t.current_sl = new_sl
                        # Only count as trail if past entry (actual profit locked)
                        if new_sl > t.entry_price:
                            t.trail_level = max(t.trail_level, 2)  # Profit locked
                        else:
                            t.trail_level = max(t.trail_level, 1)  # BE level
                else:
                    new_sl = t.entry_price - (tp_distance * profit_to_lock)
                    if new_sl < t.current_sl:
                        t.current_sl = new_sl
                        # Only count as trail if past entry (actual profit locked)
                        if new_sl < t.entry_price:
                            t.trail_level = max(t.trail_level, 2)  # Profit locked
                        else:
                            t.trail_level = max(t.trail_level, 1)  # BE level

            # === BE LOGIC (if trailing not used) ===
            elif not t.exit_reason and not USE_TRAILING:
                if t.direction == 'long':
                    if USE_BE and not t.be_triggered:
                        if profit_pct >= BE_THRESHOLD:
                            t.be_triggered = True
                            t.current_sl = t.entry_price
                else:
                    if (USE_BE or USE_BE_SHORTS) and not t.be_triggered:
                        if profit_pct >= BE_THRESHOLD:
                            t.be_triggered = True
                            t.current_sl = t.entry_price

            # === PARTIAL TAKE PROFIT ===
            # Close partial position at intermediate target, let rest run
            # Skip if already exited
            if not t.exit_reason and USE_PARTIAL_TP and not t.partial_closed and profit_pct >= PARTIAL_TP_LEVEL:
                # Calculate PnL for the partial close
                # Use stored tp_distance from trade, not recalculated
                sl_dist = abs(t.entry_price - t.sl_price)
                tp_dist = sl_dist * RR_TARGET  # Recalculate to ensure correct value

                if t.direction == 'long':
                    partial_exit_price = t.entry_price + (tp_dist * PARTIAL_TP_LEVEL)
                    partial_gross = (partial_exit_price - t.entry_price) / t.entry_price
                else:
                    partial_exit_price = t.entry_price - (tp_dist * PARTIAL_TP_LEVEL)
                    # Simplified: partial_gross = (tp_dist * PARTIAL_TP_LEVEL) / entry
                    partial_gross = (tp_dist * PARTIAL_TP_LEVEL) / t.entry_price

                # Lock in partial profit
                # Entry fee was paid on full position, only exit fee for partial
                partial_fees = (MAKER_FEE * PARTIAL_SIZE) + TAKER_FEE
                t.partial_pnl = (partial_gross - partial_fees) * 100 * t.leverage * PARTIAL_SIZE
                t.partial_closed = True

                # DEBUG: Print first few partial calculations
                if len(trades) < 3:
                    print(f"    [DEBUG Partial] {symbol}: tp_dist={tp_dist:.6f}, LEVEL={PARTIAL_TP_LEVEL}, entry={t.entry_price:.4f}, gross={partial_gross:.6f}, partial_pnl={t.partial_pnl:.2f}%")

                # Move SL to lock in tiny profit for remaining position
                # Long: SL above entry (exit higher = profit)
                # Short: SL below entry (exit lower = profit)
                if t.direction == 'long':
                    t.current_sl = t.entry_price * 1.001  # 0.1% above entry = tiny profit
                else:
                    t.current_sl = t.entry_price * 0.999  # 0.1% below entry = tiny profit
                t.be_triggered = True

            # === TIME EXIT ===
            if USE_TIME_EXIT and t.bars_in_trade >= MAX_BARS and not t.exit_reason:
                t.exit_price = candle['close']
                t.exit_reason = 'time'

            # === CHECK SL/TP EXITS ===
            if not t.exit_reason:
                # INTRA-CANDLE LOGIC FIX (ROBUST VERSION):
                # Problem: We don't know if High or Low happened first within a candle.
                # Solution: Be CONSERVATIVE - if both TP and SL could have been hit,
                # assume the WORSE outcome (SL hit).

                # Also use candle CLOSE as sanity check:
                # - For LONG: If close is below entry after "TP hit", something is wrong
                # - For SHORT: If close is above entry after "TP hit", something is wrong

                tp_touched = False
                sl_touched = False

                if t.direction == 'long':
                    tp_touched = candle['high'] >= t.tp_price
                    sl_touched = candle['low'] <= t.current_sl

                    # CONSERVATIVE: If BOTH touched, assume SL was first (worst case)
                    if tp_touched and sl_touched:
                        t.exit_price = t.current_sl
                        t.exit_reason = 'trail' if t.trail_level > 0 else ('be' if t.be_triggered else 'sl')
                    elif sl_touched:
                        t.exit_price = t.current_sl
                        t.exit_reason = 'trail' if t.trail_level > 0 else ('be' if t.be_triggered else 'sl')
                    elif tp_touched:
                        # Extra validation: If TP touched but close is below entry,
                        # the candle was extremely volatile (likely flash crash/wick)
                        # In reality, you'd likely get stopped out or exit badly
                        if candle['close'] < t.entry_price:
                            # Conservative: Exit at the close price (not TP)
                            # This reflects realistic slippage/panic in volatile conditions
                            t.exit_price = candle['close']
                            t.exit_reason = 'volatile'  # New exit reason for tracking
                        elif candle['low'] <= t.entry_price * 0.99:  # Low dipped >1% below entry
                            # Suspicious wick - price dipped significantly then recovered
                            # In reality, you might have been stopped out
                            t.exit_price = t.entry_price  # Exit at break-even (conservative)
                            t.exit_reason = 'be'
                        else:
                            t.exit_price = t.tp_price
                            t.exit_reason = 'tp'

                else:  # Short
                    tp_touched = candle['low'] <= t.tp_price
                    sl_touched = candle['high'] >= t.current_sl

                    # CONSERVATIVE: If BOTH touched, assume SL was first (worst case)
                    if tp_touched and sl_touched:
                        t.exit_price = t.current_sl
                        t.exit_reason = 'trail' if t.trail_level > 0 else ('be' if t.be_triggered else 'sl')
                    elif sl_touched:
                        t.exit_price = t.current_sl
                        t.exit_reason = 'trail' if t.trail_level > 0 else ('be' if t.be_triggered else 'sl')
                    elif tp_touched:
                        # Extra validation: If TP touched but close is above entry,
                        # the candle was extremely volatile (likely pump/wick)
                        # In reality, you'd likely get stopped out or exit badly
                        if candle['close'] > t.entry_price:
                            # Conservative: Exit at the close price (not TP)
                            # This reflects realistic slippage/panic in volatile conditions
                            t.exit_price = candle['close']
                            t.exit_reason = 'volatile'  # New exit reason for tracking
                        elif candle['high'] >= t.entry_price * 1.01:  # High spiked >1% above entry
                            # Suspicious wick - price spiked significantly then dropped
                            # In reality, you might have been stopped out
                            t.exit_price = t.entry_price  # Exit at break-even (conservative)
                            t.exit_reason = 'be'
                        else:
                            t.exit_price = t.tp_price
                            t.exit_reason = 'tp'

            if t.exit_reason:
                t.exit_time = ts

                # Calculate PnL with fees
                if t.direction == 'long':
                    gross = (t.exit_price - t.entry_price) / t.entry_price
                else:
                    gross = (t.entry_price - t.exit_price) / t.entry_price

                # Fees: maker entry (limit), taker exit (market on SL/TP)
                fees = MAKER_FEE + TAKER_FEE
                t.pnl_gross = gross * 100

                # If partial TP was taken, calculate combined PnL
                if t.partial_closed:
                    # Remaining position: entry fee was shared, only pay exit fee
                    remaining_fees = (MAKER_FEE * (1 - PARTIAL_SIZE)) + TAKER_FEE
                    remaining_pnl = (gross - remaining_fees) * 100 * t.leverage * (1 - PARTIAL_SIZE)
                    # Total = locked partial profit + remaining position result
                    t.pnl_pct = t.partial_pnl + remaining_pnl

                    # DEBUG: Print first few combined calculations
                    if len(trades) < 3:
                        print(f"    [DEBUG Exit] {symbol}: exit={t.exit_reason}, gross={gross:.5f}, partial_pnl={t.partial_pnl:.2f}%, remaining={remaining_pnl:.2f}%, total={t.pnl_pct:.2f}%")
                else:
                    t.pnl_pct = (gross - fees) * 100 * t.leverage

                trades.append(t)
                active_trade = None

        # === LOOK FOR NEW ENTRY (only if no active trade) ===
        if active_trade:
            continue

        # Get corresponding 5min candle (for OB checking)
        # Use the 5min candle that contains this 1min timestamp
        ts_5m = ts.floor('5min')

        # Get corresponding 1H candle for MTF
        ts_1h = ts.floor('1h')

        # Find 1H candle (must be COMPLETED, so use previous)
        h1_candles = df_1h[df_1h['timestamp'] <= ts_1h - pd.Timedelta(hours=1)]
        if len(h1_candles) == 0:
            continue
        h1_candle = h1_candles.iloc[-1]

        # Determine 1H trend
        h1_bullish = h1_candle['close'] > h1_candle['ema20'] > h1_candle['ema50']
        h1_bearish = h1_candle['close'] < h1_candle['ema20'] < h1_candle['ema50']

        if not h1_bullish and not h1_bearish:
            continue  # No clear 1H trend - skip

        # === 4H MTF FILTER ===
        # Requires 4H trend to align with 1H for higher probability trades
        if USE_4H_MTF and df_4h is not None:
            ts_4h = ts.floor('4h')
            h4_candles = df_4h[df_4h['timestamp'] <= ts_4h - pd.Timedelta(hours=4)]
            if len(h4_candles) > 0:
                h4_candle = h4_candles.iloc[-1]
                h4_bullish = h4_candle['close'] > h4_candle['ema20'] > h4_candle['ema50']
                h4_bearish = h4_candle['close'] < h4_candle['ema20'] < h4_candle['ema50']

                # Skip if 4H doesn't confirm 1H direction
                if h1_bullish and not h4_bullish:
                    continue  # 1H bullish but 4H not confirming
                if h1_bearish and not h4_bearish:
                    continue  # 1H bearish but 4H not confirming

        # Direction based on 1H trend (now confirmed by 4H if enabled)
        direction = 'long' if h1_bullish else 'short'

        # === DAILY MTF FILTER FOR SHORTS ===
        # Shorts require Daily timeframe to also be bearish (stricter filter)
        if USE_DAILY_FOR_SHORTS and direction == 'short' and df_daily is not None:
            ts_daily = ts.floor('D')
            daily_candles = df_daily[df_daily['timestamp'] <= ts_daily - pd.Timedelta(days=1)]
            if len(daily_candles) > 0:
                daily_candle = daily_candles.iloc[-1]
                daily_bearish = daily_candle['close'] < daily_candle['ema20'] < daily_candle['ema50']
                if not daily_bearish:
                    continue  # Daily not bearish - skip short

        # === DAILY MTF FILTER FOR LONGS ===
        # Longs require Daily timeframe to also be bullish (stricter filter)
        if USE_DAILY_FOR_LONGS and direction == 'long' and df_daily is not None:
            ts_daily = ts.floor('D')
            daily_candles = df_daily[df_daily['timestamp'] <= ts_daily - pd.Timedelta(days=1)]
            if len(daily_candles) > 0:
                daily_candle = daily_candles.iloc[-1]
                daily_bullish = daily_candle['close'] > daily_candle['ema20'] > daily_candle['ema50']
                if not daily_bullish:
                    continue  # Daily not bullish - skip long

        # === DIRECTION FILTER ===
        # Allows testing long/short independently
        if TRADE_DIRECTION == 'long' and direction == 'short':
            continue  # Skip shorts in long-only mode
        if TRADE_DIRECTION == 'short' and direction == 'long':
            continue  # Skip longs in short-only mode

        # === RSI FILTER ===
        # Confirms we're actually in a pullback (not chasing)
        if USE_RSI_FILTER:
            rsi = candle.get('rsi', 50)  # Default 50 if not available
            if pd.isna(rsi):
                rsi = 50
            if direction == 'long' and rsi > RSI_LONG_MAX:
                continue  # RSI too high - not a pullback, skip
            if direction == 'short' and rsi < RSI_SHORT_MIN:
                continue  # RSI too low - not a pullback, skip

        # === FIND VALID OB ===
        # Pick the FRESHEST matching OB (same logic as live!)
        current_price = candle['close']
        matching_ob = None
        best_ob_age = float('inf')

        for ob in obs:
            # CRITICAL: Only use OBs we KNOW about (detection_timestamp check)
            ob_known_at = ob.detection_timestamp if ob.detection_timestamp else ob.timestamp
            if ob_known_at >= ts:
                continue  # Don't know about this OB yet!

            # Check OB is not mitigated (or mitigation is in future)
            if ob.is_mitigated:
                if ob.mitigation_timestamp and ob.mitigation_timestamp <= ts:
                    continue  # Already mitigated

            # Skip OBs that have already been traded (same as live!)
            ob_key = f"{symbol}_{ob.top}_{ob.bottom}"
            if ob_key in used_obs:
                continue  # Already traded this OB

            # Filter by strength (stricter for shorts)
            min_strength = OB_MIN_STRENGTH_SHORT if direction == 'short' else OB_MIN_STRENGTH
            if ob.strength < min_strength:
                continue

            # Filter by volume (confirms institutional interest)
            if USE_VOLUME_FILTER:
                if ob.volume_ratio < MIN_VOLUME_RATIO:
                    continue  # Low volume OB - skip

            # Filter by age (in 5min candles)
            ob_age = (ts - ob.timestamp).total_seconds() / 300  # 5min = 300sec
            if ob_age > OB_MAX_AGE:
                continue

            # === ORDER EXPIRY (same as live!) ===
            # In live, we place a limit order when signal is detected.
            # If price doesn't reach OB within MAX_ORDER_AGE_MIN, order is cancelled.
            # Simulate this by checking time since detection.
            ob_known_at = ob.detection_timestamp if ob.detection_timestamp else ob.timestamp
            time_since_detection_min = (ts - ob_known_at).total_seconds() / 60
            if time_since_detection_min > MAX_ORDER_AGE_MIN:
                continue  # Order would have expired in live!

            # Check direction matches
            if direction == 'long' and not ob.is_bullish:
                continue
            if direction == 'short' and ob.is_bullish:
                continue

            # Check if price is touching OB zone on THIS 1min candle
            # Pick the FRESHEST OB if multiple match (same as live!)
            if direction == 'long':
                # For long: price should touch OB top (entry level)
                if candle['low'] <= ob.top <= candle['high']:
                    if ob_age < best_ob_age:
                        matching_ob = ob
                        best_ob_age = ob_age
            else:
                # For short: price should touch OB bottom (entry level)
                if candle['low'] <= ob.bottom <= candle['high']:
                    if ob_age < best_ob_age:
                        matching_ob = ob
                        best_ob_age = ob_age

        if not matching_ob:
            continue

        # === CREATE TRADE ===
        ob = matching_ob

        if direction == 'long':
            entry = ob.top
            sl = ob.bottom * (1 - SL_BUFFER_PCT / 100)  # Below OB with buffer
            sl_distance = entry - sl
            tp = entry + (sl_distance * RR_TARGET)
        else:
            entry = ob.bottom
            sl = ob.top * (1 + SL_BUFFER_PCT / 100)  # Above OB with buffer
            sl_distance = sl - entry
            tp = entry - (sl_distance * RR_TARGET)

        # === DYNAMIC LEVERAGE CALCULATION ===
        # Calculate minimum leverage needed to achieve 2% risk within margin limits
        sl_pct = abs(entry - sl) / entry * 100
        sl_pct_decimal = sl_pct / 100

        # min_leverage = risk_pct / (sl_pct * max_margin_pct)
        min_leverage_needed = (RISK_PER_TRADE_PCT / 100) / (sl_pct_decimal * MAX_MARGIN_PCT)

        if min_leverage_needed > MAX_LEVERAGE:
            # Can't achieve 2% risk within constraints - skip
            continue

        # Use minimum leverage that achieves target (round up)
        leverage = min(MAX_LEVERAGE, max(5, int(min_leverage_needed) + 1))

        active_trade = ScalpTrade(
            symbol=symbol,
            direction=direction,
            entry_price=entry,
            sl_price=sl,
            tp_price=tp,
            entry_time=ts,
            ob_top=ob.top,
            ob_bottom=ob.bottom,
            leverage=leverage
        )

        # Mark OB as used (same as live - prevents re-entry on same OB)
        ob_key = f"{symbol}_{ob.top}_{ob.bottom}"
        used_obs.add(ob_key)

        # Parity log for signal found
        log_parity(symbol, {
            'signal': True,
            'candle_ts': str(ts),
            '1h_candle_ts': str(h1_candle['timestamp']),
            'direction': direction,
            'chosen_ob_ts': str(ob.timestamp),
            'chosen_ob_strength': ob.strength,
            'entry': entry,
            'sl': sl,
            'tp': tp,
            'current_price': float(candle['close']),
        })

    return trades


def run_ob_scalper(num_coins: int = 50, days: int = 30):
    """Run the OB Scalper backtest"""
    from config.coins import get_top_n_coins

    print("=" * 80)
    print("OB SCALPER BACKTEST - 1min Precision")
    print("=" * 80)
    # Show date range info
    if SKIP_DAYS > 0:
        print(f"Coins: {num_coins} | Days: {days} (skipping last {SKIP_DAYS} days → testing days {SKIP_DAYS}-{SKIP_DAYS+days})")
    else:
        print(f"Coins: {num_coins} | Days: {days}")
    print(f"OB Strength: Long >= {OB_MIN_STRENGTH}, Short >= {OB_MIN_STRENGTH_SHORT} | Max Age: {OB_MAX_AGE}")
    print(f"Volume Filter: {'ON (>=' + str(MIN_VOLUME_RATIO) + 'x avg)' if USE_VOLUME_FILTER else 'OFF'}")
    print(f"R:R Target: {RR_TARGET}:1 | SL Buffer: {SL_BUFFER_PCT}%")
    daily_info = []
    if USE_DAILY_FOR_SHORTS:
        daily_info.append("Daily(shorts)")
    if USE_DAILY_FOR_LONGS:
        daily_info.append("Daily(longs)")
    print(f"MTF: 1H + {'4H' if USE_4H_MTF else 'none'} + {' + '.join(daily_info) if daily_info else ''}")
    print(f"Direction: {TRADE_DIRECTION.upper()}")

    # DD Reduction Options
    dd_opts = []
    if USE_TRAILING:
        dd_opts.append(f"Trailing(start={TRAIL_START}, step={TRAIL_STEP})")
    if USE_TIME_EXIT:
        dd_opts.append(f"TimeExit({MAX_BARS}bars)")
    if USE_PARTIAL_TP:
        dd_opts.append(f"PartialTP({int(PARTIAL_SIZE*100)}%@{int(PARTIAL_TP_LEVEL*100)}%)")
    if MAX_LEVERAGE < 20:
        dd_opts.append(f"MaxLev={MAX_LEVERAGE}")
    print(f"DD Reduction: {', '.join(dd_opts) if dd_opts else 'None'}")
    print(f"Order Expiry: {MAX_ORDER_AGE_MIN} min (same as live)")

    print(f"Workers: {NUM_WORKERS} | Timeout: {TOTAL_TIMEOUT}s")
    print("=" * 80)

    # Get coins
    coins = get_top_n_coins(num_coins)

    # Skip problematic coins (same as live)
    SKIP_COINS = {
        'USDCUSDT', 'FDUSDUSDT',  # Stablecoins
        'APEUSDT', 'MATICUSDT', 'OCEANUSDT', 'EOSUSDT',  # Often timeout
        'RNDRUSDT', 'FETUSDT', 'AGIXUSDT', 'MKRUSDT',  # API issues
        'FOGOUSDT', 'FHEUSDT', 'SKRUSDT',  # New/problematic
    }
    coins = [c for c in coins if c not in SKIP_COINS]

    print(f"Testing {len(coins)} coins (skipped {len(SKIP_COINS)} problematic)...")

    # Process coins with per-coin timeout (same as live)
    import multiprocessing as mp
    COIN_TIMEOUT = int(os.getenv('COIN_TIMEOUT', '60'))  # seconds per coin (ENV configurable)

    def process_coin_worker(coin, days, result_queue):
        """Worker that runs in subprocess - can be killed on timeout"""
        try:
            trades = process_coin((coin, days))
            result_queue.put(('ok', trades))
        except Exception as e:
            result_queue.put(('error', str(e)[:50]))

    all_trades = []
    completed = 0
    skipped = 0
    runtime_skip = set()  # Coins that timeout get added here

    for i, coin in enumerate(coins):
        # Skip coins that timed out previously
        if coin in runtime_skip:
            print(f"    [{i+1}/{len(coins)}] {coin}... SKIP (auto-skip)", flush=True)
            skipped += 1
            continue

        print(f"    [{i+1}/{len(coins)}] {coin}...", end="", flush=True)

        try:
            result_queue = mp.Queue()
            proc = mp.Process(target=process_coin_worker, args=(coin, days, result_queue))
            proc.start()
            proc.join(timeout=COIN_TIMEOUT)

            if proc.is_alive():
                # Process hung - kill it!
                proc.terminate()
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
                print(" TIMEOUT!", flush=True)
                runtime_skip.add(coin)
                skipped += 1
            elif not result_queue.empty():
                status, data = result_queue.get_nowait()
                if status == 'ok' and data:
                    all_trades.extend(data)
                    completed += 1
                    print(f" {len(data)} trades", flush=True)
                elif status == 'ok':
                    completed += 1
                    print(" 0 trades", flush=True)
                else:
                    skipped += 1
                    print(f" err: {data}", flush=True)
            else:
                skipped += 1
                print(" no result", flush=True)

        except Exception as e:
            skipped += 1
            print(f" err: {str(e)[:30]}", flush=True)

    print(f"\nCompleted: {completed}/{len(coins)} coins")
    print(f"Skipped: {skipped} coins")
    if runtime_skip:
        print(f"Auto-skipped (timeout): {', '.join(runtime_skip)}")
    print(f"Total trades: {len(all_trades)}")

    if not all_trades:
        print("\nNo trades found!")
        return

    # === CALCULATE STATS ===
    tp_exits = [t for t in all_trades if t.exit_reason == 'tp']
    sl_exits = [t for t in all_trades if t.exit_reason == 'sl']
    be_exits = [t for t in all_trades if t.exit_reason == 'be']
    trail_exits = [t for t in all_trades if t.exit_reason == 'trail']
    time_exits = [t for t in all_trades if t.exit_reason == 'time']
    partial_trades = [t for t in all_trades if t.partial_closed]

    # Wins = TP + positive exits from BE/trail/time
    other_exits = be_exits + trail_exits + time_exits
    wins = tp_exits + [t for t in other_exits if t.pnl_pct >= 0]
    losses = sl_exits + [t for t in other_exits if t.pnl_pct < 0]

    total_trades = len(all_trades)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total_trades * 100 if total_trades > 0 else 0

    # PnL
    total_pnl = sum(t.pnl_pct for t in all_trades)
    avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0

    # Profit Factor
    gross_profit = sum(t.pnl_pct for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl_pct for t in losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Max Drawdown (simplified)
    equity = 10000
    peak = equity
    max_dd = 0
    for t in sorted(all_trades, key=lambda x: x.entry_time):
        equity *= (1 + t.pnl_pct / 100)
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100
        max_dd = max(max_dd, dd)

    final_equity = 10000
    for t in sorted(all_trades, key=lambda x: x.entry_time):
        final_equity *= (1 + t.pnl_pct / 100)

    # === PRINT RESULTS ===
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Trades:        {total_trades}")
    print(f"Wins:          {win_count} ({win_rate:.1f}%)")
    print(f"Losses:        {loss_count}")
    print(f"Win Rate:      {win_rate:.1f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Total PnL:     {total_pnl:+.1f}%")
    print(f"Max Drawdown:  {max_dd:.1f}%")
    print(f"$10,000 ->     ${final_equity:,.0f}")
    print("-" * 40)
    print(f"Avg Win:       {avg_win:+.2f}%")
    print(f"Avg Loss:      {avg_loss:+.2f}%")
    print(f"Avg R:R:       {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "Avg R:R:       N/A")
    print("-" * 40)
    print("EXIT BREAKDOWN:")
    print(f"  TP Hits:     {len(tp_exits)} ({len(tp_exits)/total_trades*100:.1f}%)")
    print(f"  SL Hits:     {len(sl_exits)} ({len(sl_exits)/total_trades*100:.1f}%)")
    if be_exits:
        print(f"  BE Exits:    {len(be_exits)} ({len(be_exits)/total_trades*100:.1f}%)")
    if trail_exits:
        print(f"  Trail Exits: {len(trail_exits)} ({len(trail_exits)/total_trades*100:.1f}%)")
    if time_exits:
        print(f"  Time Exits:  {len(time_exits)} ({len(time_exits)/total_trades*100:.1f}%)")
    if partial_trades:
        print(f"  Partial TP:  {len(partial_trades)} ({len(partial_trades)/total_trades*100:.1f}%)")
    print("=" * 80)

    # Breakdown by direction
    longs = [t for t in all_trades if t.direction == 'long']
    shorts = [t for t in all_trades if t.direction == 'short']
    long_wins = len([t for t in longs if t.exit_reason in ['tp', 'be'] and t.pnl_pct >= 0])
    short_wins = len([t for t in shorts if t.exit_reason in ['tp', 'be'] and t.pnl_pct >= 0])
    long_wr = long_wins / len(longs) * 100 if longs else 0
    short_wr = short_wins / len(shorts) * 100 if shorts else 0

    print(f"\nLongs:  {len(longs)} trades, {long_wr:.1f}% WR")
    print(f"Shorts: {len(shorts)} trades, {short_wr:.1f}% WR")

    return all_trades


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--coins', type=int, default=50)
    parser.add_argument('--days', type=int, default=30)
    args = parser.parse_args()

    run_ob_scalper(args.coins, args.days)
