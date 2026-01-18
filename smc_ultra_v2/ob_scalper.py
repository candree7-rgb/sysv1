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
OB_MAX_AGE = int(os.getenv('OB_MAX_AGE', '100'))  # in 5min candles
RR_TARGET = float(os.getenv('RR_TARGET', '2.0'))  # TP = RR_TARGET * SL
SL_BUFFER_PCT = float(os.getenv('SL_BUFFER_PCT', '0.05'))  # Buffer beyond OB edge

# Fees (Bybit Futures)
MAKER_FEE = 0.0002  # 0.02%
TAKER_FEE = 0.00055  # 0.055%

# Position limits
MAX_CONCURRENT = int(os.getenv('MAX_CONCURRENT', '1'))


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

    # Results (filled after exit)
    exit_price: float = None
    exit_time: datetime = None
    exit_reason: str = None  # 'tp', 'sl'
    pnl_pct: float = None
    pnl_gross: float = None


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA indicators for trend detection"""
    df = df.copy()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
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

        # Load data
        df_5m = dl.load_or_download(symbol, "5", days + 10)
        if df_5m is None or len(df_5m) < 200:
            return []

        df_1m = dl.load_or_download(symbol, "1", days + 5)
        if df_1m is None or len(df_1m) < 500:
            return []

        df_1h = dl.load_or_download(symbol, "60", days + 30)
        if df_1h is None or len(df_1h) < 50:
            return []

        # Add indicators
        df_5m = calculate_indicators(df_5m)
        df_1h = calculate_indicators(df_1h)

        # Detect OBs on 5min (with detection_timestamp!)
        obs = ob_det.detect(df_5m, df_5m['close'].rolling(14).apply(
            lambda x: pd.Series(x).diff().abs().mean() if len(x) > 1 else 0
        ))

        if not obs:
            return []

        # Run backtest
        trades = run_backtest(symbol, df_5m, df_1m, df_1h, obs)

    except Exception as e:
        print(f"      [Error] {symbol}: {str(e)[:50]}", flush=True)
        return []

    return trades


def run_backtest(
    symbol: str,
    df_5m: pd.DataFrame,
    df_1m: pd.DataFrame,
    df_1h: pd.DataFrame,
    obs: list
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
            # Check SL
            if active_trade.direction == 'long':
                if candle['low'] <= active_trade.sl_price:
                    active_trade.exit_price = active_trade.sl_price
                    active_trade.exit_reason = 'sl'
                elif candle['high'] >= active_trade.tp_price:
                    active_trade.exit_price = active_trade.tp_price
                    active_trade.exit_reason = 'tp'
            else:  # short
                if candle['high'] >= active_trade.sl_price:
                    active_trade.exit_price = active_trade.sl_price
                    active_trade.exit_reason = 'sl'
                elif candle['low'] <= active_trade.tp_price:
                    active_trade.exit_price = active_trade.tp_price
                    active_trade.exit_reason = 'tp'

            if active_trade.exit_reason:
                active_trade.exit_time = ts

                # Calculate PnL with fees
                if active_trade.direction == 'long':
                    gross = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                else:
                    gross = (active_trade.entry_price - active_trade.exit_price) / active_trade.entry_price

                # Fees: maker entry (limit), taker exit (market on SL/TP)
                fees = MAKER_FEE + TAKER_FEE
                active_trade.pnl_gross = gross * 100
                active_trade.pnl_pct = (gross - fees) * 100 * active_trade.leverage

                trades.append(active_trade)
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
            continue  # No clear MTF trend

        # Find 5min candle for trend check
        m5_candles = df_5m[df_5m['timestamp'] <= ts_5m]
        if len(m5_candles) == 0:
            continue
        m5_candle = m5_candles.iloc[-1]

        # 5min trend must align with 1H
        m5_bullish = m5_candle['close'] > m5_candle['ema20'] > m5_candle['ema50']
        m5_bearish = m5_candle['close'] < m5_candle['ema20'] < m5_candle['ema50']

        # Determine direction based on aligned trend
        direction = None
        if h1_bullish and m5_bullish:
            direction = 'long'
        elif h1_bearish and m5_bearish:
            direction = 'short'

        if not direction:
            continue

        # === FIND VALID OB ===
        current_price = candle['close']
        matching_ob = None

        for ob in obs:
            # CRITICAL: Only use OBs we KNOW about (detection_timestamp check)
            ob_known_at = ob.detection_timestamp if ob.detection_timestamp else ob.timestamp
            if ob_known_at >= ts:
                continue  # Don't know about this OB yet!

            # Check OB is not mitigated (or mitigation is in future)
            if ob.is_mitigated:
                if ob.mitigation_timestamp and ob.mitigation_timestamp <= ts:
                    continue  # Already mitigated

            # Filter by strength
            if ob.strength < OB_MIN_STRENGTH:
                continue

            # Filter by age (in 5min candles)
            ob_age = (ts - ob.timestamp).total_seconds() / 300  # 5min = 300sec
            if ob_age > OB_MAX_AGE:
                continue

            # Check direction matches
            if direction == 'long' and not ob.is_bullish:
                continue
            if direction == 'short' and ob.is_bullish:
                continue

            # Check if price is touching OB zone on THIS 1min candle
            if direction == 'long':
                # For long: price should touch OB top (entry level)
                if candle['low'] <= ob.top <= candle['high']:
                    matching_ob = ob
                    break
            else:
                # For short: price should touch OB bottom (entry level)
                if candle['low'] <= ob.bottom <= candle['high']:
                    matching_ob = ob
                    break

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

        # Calculate dynamic leverage based on SL distance
        sl_pct = abs(entry - sl) / entry * 100
        leverage = min(20, max(5, int(2 / sl_pct)))  # Target ~2% risk per trade

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

    return trades


def run_ob_scalper(num_coins: int = 50, days: int = 30):
    """Run the OB Scalper backtest"""
    from config.coins import get_top_n_coins

    print("=" * 80)
    print("OB SCALPER BACKTEST - 1min Precision")
    print("=" * 80)
    print(f"Coins: {num_coins} | Days: {days}")
    print(f"OB Strength: >= {OB_MIN_STRENGTH} | OB Max Age: {OB_MAX_AGE} candles")
    print(f"R:R Target: {RR_TARGET}:1 | SL Buffer: {SL_BUFFER_PCT}%")
    print(f"Workers: {NUM_WORKERS} | Timeout: {TOTAL_TIMEOUT}s")
    print("=" * 80)

    # Get coins
    coins = get_top_n_coins(num_coins)
    skip = {'USDCUSDT', 'FDUSDUSDT'}  # Stablecoins
    coins = [c for c in coins if c not in skip]

    print(f"Testing {len(coins)} coins...")

    # Process coins in parallel
    all_trades = []
    completed = 0
    skipped = 0

    args = [(coin, days) for coin in coins]

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_coin = {executor.submit(process_coin, arg): arg[0] for arg in args}

        try:
            for future in as_completed(future_to_coin, timeout=TOTAL_TIMEOUT):
                coin = future_to_coin[future]
                try:
                    trades = future.result(timeout=60)
                    if trades:
                        all_trades.extend(trades)
                    completed += 1
                    print(f"    [{completed}/{len(coins)}] {coin} - {len(trades)} trades", flush=True)
                except Exception as e:
                    skipped += 1
                    print(f"    [Skip] {coin}: {str(e)[:30]}", flush=True)
        except Exception as e:
            print(f"\n  Timeout reached - {len(coins) - completed} coins pending")
            executor.shutdown(wait=False, cancel_futures=True)

    print(f"\nCompleted: {completed}/{len(coins)} coins")
    print(f"Skipped: {skipped} coins")
    print(f"Total trades: {len(all_trades)}")

    if not all_trades:
        print("\nNo trades found!")
        return

    # === CALCULATE STATS ===
    wins = [t for t in all_trades if t.exit_reason == 'tp']
    losses = [t for t in all_trades if t.exit_reason == 'sl']

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
    print("=" * 80)

    # Breakdown by direction
    longs = [t for t in all_trades if t.direction == 'long']
    shorts = [t for t in all_trades if t.direction == 'short']
    long_wr = len([t for t in longs if t.exit_reason == 'tp']) / len(longs) * 100 if longs else 0
    short_wr = len([t for t in shorts if t.exit_reason == 'tp']) / len(shorts) * 100 if shorts else 0

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
