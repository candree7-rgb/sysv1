"""
Mean Reversion Short Strategy
=============================
Shorts overbought conditions in ANY market phase.
Works in bull, bear, and sideways markets.

Logic:
- Entry: RSI > 70 + Price extended above EMA + Red candle
- TP: Back to EMA20
- SL: Above recent high
"""

import os
import sys
from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configuration via ENV
NUM_WORKERS = int(os.getenv('NUM_WORKERS', str(min(4, cpu_count()))))
TOTAL_TIMEOUT = int(os.getenv('TOTAL_TIMEOUT', '300'))

# Mean Reversion Settings
RSI_OVERBOUGHT = int(os.getenv('RSI_OVERBOUGHT', '70'))  # Entry when RSI > this
ATR_EXTENSION = float(os.getenv('ATR_EXTENSION', '1.5'))  # Price must be X ATR above EMA
LOOKBACK_HIGH = int(os.getenv('LOOKBACK_HIGH', '10'))  # Bars to find recent high for SL

# Fees (Bybit Futures)
MAKER_FEE = 0.0002
TAKER_FEE = 0.00055


@dataclass
class MRTrade:
    """Mean Reversion Trade"""
    symbol: str
    direction: str  # Always 'short' for this strategy
    entry_price: float
    sl_price: float
    tp_price: float
    entry_time: datetime
    rsi_at_entry: float
    leverage: int = 10

    # Results
    exit_price: float = None
    exit_time: datetime = None
    exit_reason: str = None
    pnl_pct: float = None


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA, RSI, ATR"""
    df = df.copy()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()

    return df


def process_coin(args) -> List[MRTrade]:
    """Process a single coin"""
    symbol, days = args

    print(f"      [Worker] {symbol}...", flush=True)

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    import urllib3
    urllib3.disable_warnings()

    from data import BybitDataDownloader

    trades = []

    try:
        dl = BybitDataDownloader()

        # Load 5min data for signals
        df_5m = dl.load_or_download(symbol, "5", days + 10)
        if df_5m is None or len(df_5m) < 200:
            return []

        # Add indicators
        df_5m = calculate_indicators(df_5m)

        # Run backtest
        trades = run_backtest(symbol, df_5m)

    except Exception as e:
        print(f"      [Error] {symbol}: {str(e)[:50]}", flush=True)
        return []

    return trades


def run_backtest(symbol: str, df: pd.DataFrame) -> List[MRTrade]:
    """Run mean reversion backtest"""
    trades = []
    active_trade = None

    # Start after enough data for indicators
    start_idx = 50

    for idx in range(start_idx, len(df)):
        candle = df.iloc[idx]
        ts = candle['timestamp']

        # === CHECK ACTIVE TRADE EXIT ===
        if active_trade:
            t = active_trade

            # Check SL hit
            if candle['high'] >= t.sl_price:
                t.exit_price = t.sl_price
                t.exit_reason = 'sl'
            # Check TP hit (price back to EMA20)
            elif candle['low'] <= t.tp_price:
                t.exit_price = t.tp_price
                t.exit_reason = 'tp'

            if t.exit_reason:
                t.exit_time = ts
                gross = (t.entry_price - t.exit_price) / t.entry_price
                fees = MAKER_FEE + TAKER_FEE
                t.pnl_pct = (gross - fees) * 100 * t.leverage
                trades.append(t)
                active_trade = None

        # === LOOK FOR NEW ENTRY ===
        if active_trade:
            continue

        rsi = candle.get('rsi', 50)
        ema20 = candle.get('ema20', candle['close'])
        atr = candle.get('atr', 0)

        if pd.isna(rsi) or pd.isna(ema20) or pd.isna(atr) or atr == 0:
            continue

        # === MEAN REVERSION SHORT CONDITIONS ===
        # 1. RSI overbought
        if rsi <= RSI_OVERBOUGHT:
            continue

        # 2. Price extended above EMA20 by X ATR
        price_extension = (candle['close'] - ema20) / atr
        if price_extension < ATR_EXTENSION:
            continue

        # 3. Candle closes red (reversal signal)
        if candle['close'] >= candle['open']:
            continue  # Not a red candle

        # === CREATE SHORT TRADE ===
        entry = candle['close']

        # SL above recent high
        recent_high = df['high'].iloc[max(0, idx-LOOKBACK_HIGH):idx+1].max()
        sl = recent_high * 1.002  # 0.2% buffer

        # TP at EMA20 (mean reversion target)
        tp = ema20

        # Skip if TP is above entry (no profit potential)
        if tp >= entry:
            continue

        # Calculate leverage based on SL distance
        sl_pct = abs(sl - entry) / entry * 100
        leverage = min(20, max(5, int(2 / sl_pct)))

        active_trade = MRTrade(
            symbol=symbol,
            direction='short',
            entry_price=entry,
            sl_price=sl,
            tp_price=tp,
            entry_time=ts,
            rsi_at_entry=rsi,
            leverage=leverage
        )

    return trades


def run_mean_reversion(num_coins: int = 50, days: int = 30):
    """Run Mean Reversion Short backtest"""
    from config.coins import get_top_n_coins

    print("=" * 80)
    print("MEAN REVERSION SHORT BACKTEST")
    print("=" * 80)
    print(f"Coins: {num_coins} | Days: {days}")
    print(f"RSI Overbought: > {RSI_OVERBOUGHT}")
    print(f"ATR Extension: > {ATR_EXTENSION}x above EMA20")
    print(f"SL: Above {LOOKBACK_HIGH}-bar high | TP: EMA20")
    print(f"Workers: {NUM_WORKERS} | Timeout: {TOTAL_TIMEOUT}s")
    print("=" * 80)

    # Get coins
    coins = get_top_n_coins(num_coins)
    skip = {'USDCUSDT', 'FDUSDUSDT'}
    coins = [c for c in coins if c not in skip]

    print(f"Testing {len(coins)} coins...")

    all_trades = []
    completed = 0

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
                    print(f"    [Skip] {coin}: {str(e)[:30]}", flush=True)
        except Exception as e:
            print(f"\n  Timeout reached - {len(coins) - completed} coins pending")
            executor.shutdown(wait=False, cancel_futures=True)

    print(f"\nCompleted: {completed}/{len(coins)} coins")
    print(f"Total trades: {len(all_trades)}")

    if not all_trades:
        print("\nNo trades found!")
        return

    # === CALCULATE STATS ===
    tp_exits = [t for t in all_trades if t.exit_reason == 'tp']
    sl_exits = [t for t in all_trades if t.exit_reason == 'sl']

    wins = tp_exits
    losses = sl_exits

    total_trades = len(all_trades)
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0

    total_pnl = sum(t.pnl_pct for t in all_trades)
    avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0

    gross_profit = sum(t.pnl_pct for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl_pct for t in losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Max Drawdown
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
    print("RESULTS - MEAN REVERSION SHORTS")
    print("=" * 80)
    print(f"Trades:        {total_trades}")
    print(f"Wins:          {len(wins)} ({win_rate:.1f}%)")
    print(f"Losses:        {len(losses)}")
    print(f"Win Rate:      {win_rate:.1f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Total PnL:     {total_pnl:+.1f}%")
    print(f"Max Drawdown:  {max_dd:.1f}%")
    print(f"$10,000 ->     ${final_equity:,.0f}")
    print("-" * 40)
    print(f"Avg Win:       {avg_win:+.2f}%")
    print(f"Avg Loss:      {avg_loss:+.2f}%")
    print(f"Avg RSI Entry: {np.mean([t.rsi_at_entry for t in all_trades]):.1f}")
    print("=" * 80)

    return all_trades


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--coins', type=int, default=50)
    parser.add_argument('--days', type=int, default=30)
    args = parser.parse_args()

    run_mean_reversion(args.coins, args.days)
