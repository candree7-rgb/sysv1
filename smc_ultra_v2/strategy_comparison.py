"""
SMC Strategy Comparison - PARALLELIZED VERSION with 1MIN EXIT PRECISION
========================================================================
Uses 5min for signals, 1min for precise SL/TP exit checking!
"""

import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pool, cpu_count

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class SetupType(Enum):
    OB_RETEST = "ob_retest"


# Trading costs (REALISTIC)
TAKER_FEE_PCT = 0.055
MAKER_FEE_PCT = 0.02
SLIPPAGE_PCT = 0.02

# Dynamic leverage settings
RISK_PER_TRADE_PCT = 2.0
MAX_LEVERAGE = 50
MIN_LEVERAGE = 5

# Max concurrent trades - SEPARATED by direction for hedging!
MAX_LONGS = int(os.getenv('MAX_LONGS', '2'))
MAX_SHORTS = int(os.getenv('MAX_SHORTS', '2'))

# OB Quality Filters
OB_MIN_STRENGTH = float(os.getenv('OB_MIN_STRENGTH', '0.6'))  # Minimum OB strength (0-1)
OB_MAX_AGE_CANDLES = int(os.getenv('OB_MAX_AGE', '100'))       # Max OB age in candles (~8h for 5min)

# Parallelization
NUM_WORKERS = int(os.getenv('NUM_WORKERS', str(min(8, cpu_count()))))

# Precision mode: use 1min candles for exit checking
USE_1MIN_EXITS = True


@dataclass
class Trade:
    setup: SetupType
    symbol: str
    direction: str
    entry: float
    sl: float
    tp: float
    entry_time: datetime
    leverage: int = 10
    sl_pct: float = 0.0
    exit_time: datetime = None
    exit_price: float = None
    pnl_pct: float = None
    pnl_leveraged: float = None
    result: str = None


def process_single_coin(args) -> List[Trade]:
    """Process a single coin - can be run in parallel

    Uses 5min candles for signal generation (OB detection, EMA trend)
    Uses 1min candles for precise SL/TP exit checking
    """
    symbol, days = args

    # Disable SSL verification for subprocesses (both ssl and requests)
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    import os
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''

    # Import inside function for multiprocessing
    from data import BybitDataDownloader
    from smc import OrderBlockDetector, FVGDetector, LiquidityDetector

    trades = []

    try:
        dl = BybitDataDownloader()
        ob_det = OrderBlockDetector()

        end = datetime.now()
        start = end - timedelta(days=days + 5)

        # Load 5min data for signal generation
        df_5m = dl.load_or_download(symbol, "5", days + 10)
        if df_5m is None or len(df_5m) < 200:
            return []

        # Load 1min data for precise exit checking
        df_1m = None
        if USE_1MIN_EXITS:
            df_1m = dl.load_or_download(symbol, "1", days + 10)
            if df_1m is None or len(df_1m) < 1000:
                # Fall back to 5min exits if 1min not available
                df_1m = None

        # Filter date range for 5min
        mask = (df_5m['timestamp'] >= start) & (df_5m['timestamp'] <= end)
        df_5m = df_5m[mask].reset_index(drop=True)

        if len(df_5m) < 100:
            return []

        # Filter 1min data if available
        if df_1m is not None:
            mask = (df_1m['timestamp'] >= start) & (df_1m['timestamp'] <= end)
            df_1m = df_1m[mask].reset_index(drop=True)

        # Calculate indicators on 5min
        high_low = df_5m['high'] - df_5m['low']
        high_close = abs(df_5m['high'] - df_5m['close'].shift())
        low_close = abs(df_5m['low'] - df_5m['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df_5m['atr'] = tr.rolling(14).mean()
        df_5m['ema20'] = df_5m['close'].ewm(span=20).mean()
        df_5m['ema50'] = df_5m['close'].ewm(span=50).mean()

        # Detect OBs on 5min
        atr = df_5m['atr']
        obs = ob_det.detect(df_5m, atr)

        # Run backtest for this coin
        active_trade = None
        pending_trades = []  # Trades waiting for exit check

        for idx in range(50, len(df_5m)):
            candle = df_5m.iloc[idx]
            ts = candle['timestamp']
            price = candle['close']
            atr_val = candle['atr']

            if pd.isna(atr_val) or atr_val <= 0:
                continue

            # Check and close active trade using 1min precision
            if active_trade:
                if df_1m is not None and USE_1MIN_EXITS:
                    # Use 1min candles for precise exit
                    closed = check_trade_exit_1min(active_trade, df_1m, ts)
                else:
                    # Fallback to 5min
                    closed = check_trade_exit(active_trade, candle)

                if closed:
                    trades.append(active_trade)
                    active_trade = None

            # Skip if we have an active trade
            if active_trade:
                continue

            # Get trend direction
            if candle['close'] > candle['ema20'] > candle['ema50']:
                trend = 'long'
            elif candle['close'] < candle['ema20'] < candle['ema50']:
                trend = 'short'
            else:
                continue

            # Get active OBs (with look-ahead bias fix!)
            active_obs = []
            for ob in obs:
                if ob.timestamp >= ts:
                    continue
                if not ob.is_mitigated:
                    active_obs.append(ob)
                elif ob.mitigation_timestamp is not None and ob.mitigation_timestamp > ts:
                    active_obs.append(ob)

            # Check for OB entry with quality filters
            matching_ob = None
            for ob in active_obs:
                if (trend == 'long' and ob.is_bullish) or (trend == 'short' and not ob.is_bullish):
                    if ob.bottom <= price <= ob.top:
                        # Filter 1: OB Strength
                        if ob.strength < OB_MIN_STRENGTH:
                            continue

                        # Filter 2: OB Age (freshness)
                        ob_age_minutes = (ts - ob.timestamp).total_seconds() / 60
                        ob_age_candles = ob_age_minutes / 5  # 5min candles
                        if ob_age_candles > OB_MAX_AGE_CANDLES:
                            continue

                        matching_ob = ob
                        break

            if matching_ob:
                active_trade = create_trade(symbol, trend, price, atr_val, ts, matching_ob)

        # Close any remaining trade
        if active_trade and len(df_5m) > 0:
            last_candle = df_5m.iloc[-1]
            if df_1m is not None and USE_1MIN_EXITS:
                check_trade_exit_1min(active_trade, df_1m, last_candle['timestamp'])
            else:
                check_trade_exit(active_trade, last_candle)
            if active_trade.exit_time:
                trades.append(active_trade)

    except Exception as e:
        print(f"  Error processing {symbol}: {e}", flush=True)

    return trades


def check_trade_exit_1min(trade: Trade, df_1m: pd.DataFrame, current_5m_ts: datetime) -> bool:
    """Check trade exit using 1min candles for precision.

    Iterates through 1min candles from trade entry to current 5min candle.
    This gives us exact order of SL vs TP hits.
    """
    fee_win = (MAKER_FEE_PCT * 2) + (SLIPPAGE_PCT * 2)
    fee_loss = MAKER_FEE_PCT + TAKER_FEE_PCT + (SLIPPAGE_PCT * 2)

    # Get 1min candles from entry time to current 5min candle
    mask = (df_1m['timestamp'] > trade.entry_time) & (df_1m['timestamp'] <= current_5m_ts)
    candles_to_check = df_1m[mask]

    if len(candles_to_check) == 0:
        return False

    for _, candle in candles_to_check.iterrows():
        if trade.direction == 'long':
            # Check SL first (hit if low <= sl)
            if candle['low'] <= trade.sl:
                trade.exit_price = trade.sl
                trade.result = 'loss'
                gross_pnl = (trade.sl - trade.entry) / trade.entry * 100
                trade.pnl_pct = gross_pnl - fee_loss
                trade.pnl_leveraged = trade.pnl_pct * trade.leverage
                trade.exit_time = candle['timestamp']
                return True
            # Check TP (hit if high >= tp)
            elif candle['high'] >= trade.tp:
                trade.exit_price = trade.tp
                trade.result = 'win'
                gross_pnl = (trade.tp - trade.entry) / trade.entry * 100
                trade.pnl_pct = gross_pnl - fee_win
                trade.pnl_leveraged = trade.pnl_pct * trade.leverage
                trade.exit_time = candle['timestamp']
                return True
        else:  # short
            # Check SL first (hit if high >= sl)
            if candle['high'] >= trade.sl:
                trade.exit_price = trade.sl
                trade.result = 'loss'
                gross_pnl = (trade.entry - trade.sl) / trade.entry * 100
                trade.pnl_pct = gross_pnl - fee_loss
                trade.pnl_leveraged = trade.pnl_pct * trade.leverage
                trade.exit_time = candle['timestamp']
                return True
            # Check TP (hit if low <= tp)
            elif candle['low'] <= trade.tp:
                trade.exit_price = trade.tp
                trade.result = 'win'
                gross_pnl = (trade.entry - trade.tp) / trade.entry * 100
                trade.pnl_pct = gross_pnl - fee_win
                trade.pnl_leveraged = trade.pnl_pct * trade.leverage
                trade.exit_time = candle['timestamp']
                return True

    return False


def create_trade(symbol: str, direction: str, price: float, atr: float, ts: datetime, ob=None) -> Trade:
    """Create a trade with 1:1.5 RR and dynamic leverage

    Entry price: Uses OB edge for realistic limit order fill
    - Long: Entry at OB top (where limit buy would fill)
    - Short: Entry at OB bottom (where limit sell would fill)
    """
    sl_mult = 1.0
    tp_mult = 1.5

    # Use OB edge for precise entry (realistic limit order fill)
    if ob is not None:
        if direction == 'long':
            # Long: Limit buy at OB top (price enters from above)
            entry = ob.top
        else:
            # Short: Limit sell at OB bottom (price enters from below)
            entry = ob.bottom
    else:
        # Fallback to price with slippage
        if direction == 'long':
            entry = price * 1.0003
        else:
            entry = price * 0.9997

    # Calculate SL/TP from entry
    if direction == 'long':
        sl = entry - atr * sl_mult
        tp = entry + atr * tp_mult
    else:
        sl = entry + atr * sl_mult
        tp = entry - atr * tp_mult

    sl_pct = abs(entry - sl) / entry * 100

    if sl_pct > 0:
        calculated_lev = RISK_PER_TRADE_PCT / sl_pct
        leverage = min(int(calculated_lev), MAX_LEVERAGE)
        leverage = max(leverage, MIN_LEVERAGE)
    else:
        leverage = MIN_LEVERAGE

    return Trade(
        setup=SetupType.OB_RETEST,
        symbol=symbol,
        direction=direction,
        entry=entry,
        sl=sl,
        tp=tp,
        entry_time=ts,
        leverage=leverage,
        sl_pct=sl_pct
    )


def check_trade_exit(trade: Trade, candle) -> bool:
    """Check if trade should exit"""
    fee_win = (MAKER_FEE_PCT * 2) + (SLIPPAGE_PCT * 2)
    fee_loss = MAKER_FEE_PCT + TAKER_FEE_PCT + (SLIPPAGE_PCT * 2)

    if trade.direction == 'long':
        if candle['low'] <= trade.sl:
            trade.exit_price = trade.sl
            trade.result = 'loss'
            gross_pnl = (trade.sl - trade.entry) / trade.entry * 100
            trade.pnl_pct = gross_pnl - fee_loss
            trade.pnl_leveraged = trade.pnl_pct * trade.leverage
            trade.exit_time = candle['timestamp']
            return True
        elif candle['high'] >= trade.tp:
            trade.exit_price = trade.tp
            trade.result = 'win'
            gross_pnl = (trade.tp - trade.entry) / trade.entry * 100
            trade.pnl_pct = gross_pnl - fee_win
            trade.pnl_leveraged = trade.pnl_pct * trade.leverage
            trade.exit_time = candle['timestamp']
            return True
    else:
        if candle['high'] >= trade.sl:
            trade.exit_price = trade.sl
            trade.result = 'loss'
            gross_pnl = (trade.entry - trade.sl) / trade.entry * 100
            trade.pnl_pct = gross_pnl - fee_loss
            trade.pnl_leveraged = trade.pnl_pct * trade.leverage
            trade.exit_time = candle['timestamp']
            return True
        elif candle['low'] <= trade.tp:
            trade.exit_price = trade.tp
            trade.result = 'win'
            gross_pnl = (trade.entry - trade.tp) / trade.entry * 100
            trade.pnl_pct = gross_pnl - fee_win
            trade.pnl_leveraged = trade.pnl_pct * trade.leverage
            trade.exit_time = candle['timestamp']
            return True

    return False


def apply_position_limits(trades: List[Trade]) -> List[Trade]:
    """Apply MAX_LONGS/MAX_SHORTS limits across all coins"""
    # Sort by entry time
    sorted_trades = sorted(trades, key=lambda t: t.entry_time)

    filtered_trades = []
    active_longs = []
    active_shorts = []

    for trade in sorted_trades:
        # Remove closed trades
        active_longs = [t for t in active_longs if t.exit_time is None or t.exit_time > trade.entry_time]
        active_shorts = [t for t in active_shorts if t.exit_time is None or t.exit_time > trade.entry_time]

        # Check limits
        if trade.direction == 'long' and len(active_longs) >= MAX_LONGS:
            continue  # Skip this trade
        if trade.direction == 'short' and len(active_shorts) >= MAX_SHORTS:
            continue  # Skip this trade

        # Add trade
        filtered_trades.append(trade)
        if trade.direction == 'long':
            active_longs.append(trade)
        else:
            active_shorts.append(trade)

    return filtered_trades


def print_results(trades: List[Trade]):
    """Print results"""
    print("\n" + "=" * 100)
    print("SMC STRATEGY COMPARISON RESULTS (PARALLELIZED)")
    print("=" * 100)

    if not trades:
        print("NO TRADES")
        return

    winners = [t for t in trades if t.result == 'win']
    losers = [t for t in trades if t.result == 'loss']

    total = len(trades)
    win_count = len(winners)
    loss_count = len(losers)
    win_rate = win_count / total * 100 if total > 0 else 0

    avg_leverage = sum(t.leverage for t in trades) / len(trades)
    avg_win_lev = sum(t.pnl_leveraged for t in winners) / len(winners) if winners else 0
    avg_loss_lev = abs(sum(t.pnl_leveraged for t in losers) / len(losers)) if losers else 0

    gross_win_lev = sum(t.pnl_leveraged for t in winners) if winners else 0
    gross_loss_lev = abs(sum(t.pnl_leveraged for t in losers)) if losers else 1
    pf = gross_win_lev / gross_loss_lev if gross_loss_lev > 0 else gross_win_lev
    total_pnl_lev = gross_win_lev - gross_loss_lev

    print(f"{'Setup':<15} {'Trades':>7} {'Win':>5} {'Loss':>5} {'WR%':>7} {'AvgLev':>7} {'AvgWin':>9} {'AvgLoss':>9} {'TotalPnL':>10}")
    print("-" * 100)
    print(f"{'ob_retest':<15} {total:>7} {win_count:>5} {loss_count:>5} "
          f"{win_rate:>6.1f}% {avg_leverage:>6.1f}x {avg_win_lev:>+8.2f}% {avg_loss_lev:>8.2f}% {total_pnl_lev:>+9.2f}%")

    print("\n" + "=" * 100)
    print(f"BEST SETUP: OB_RETEST")
    print(f"   Win Rate: {win_rate:.1f}%")
    print(f"   Avg Leverage: {avg_leverage:.1f}x")
    print(f"   Profit Factor: {pf:.2f}")
    print(f"   Total PnL (leveraged): {total_pnl_lev:+.2f}%")
    print(f"   Trades: {total}")
    print("=" * 100)

    # Per-coin breakdown
    print(f"\nOB_RETEST - Per Coin Breakdown (Leveraged PnL):")
    print("-" * 60)

    coin_stats = {}
    for trade in trades:
        if trade.symbol not in coin_stats:
            coin_stats[trade.symbol] = {'wins': 0, 'losses': 0, 'pnl': 0, 'avg_lev': []}

        if trade.result == 'win':
            coin_stats[trade.symbol]['wins'] += 1
        else:
            coin_stats[trade.symbol]['losses'] += 1
        coin_stats[trade.symbol]['pnl'] += trade.pnl_leveraged
        coin_stats[trade.symbol]['avg_lev'].append(trade.leverage)

    for symbol, stats in sorted(coin_stats.items(), key=lambda x: -x[1]['pnl']):
        total_coin = stats['wins'] + stats['losses']
        wr = stats['wins'] / total_coin * 100 if total_coin > 0 else 0
        avg_lev = sum(stats['avg_lev']) / len(stats['avg_lev']) if stats['avg_lev'] else 0
        print(f"  {symbol:<12} {total_coin:>3} trades, {wr:>5.1f}% WR, {avg_lev:>4.1f}x lev, {stats['pnl']:>+8.2f}% PnL")

    # Simulate account growth
    print(f"\n{'='*70}")
    print(f"SIMULATED ACCOUNT GROWTH ($10,000 start, {RISK_PER_TRADE_PCT}% risk per trade)")
    print("="*70)

    equity = 10000.0
    peak = 10000.0
    max_dd = 0.0

    for trade in sorted(trades, key=lambda t: t.entry_time):
        pnl_usd = equity * (trade.pnl_leveraged / 100)
        equity += pnl_usd
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100
        max_dd = max(max_dd, dd)

    return_pct = (equity - 10000) / 10000 * 100
    print(f"  {'ob_retest':<15} $10,000 -> ${equity:>10,.2f} ({return_pct:>+7.2f}%)  MaxDD: {max_dd:.1f}%")


def run_comparison(num_coins: int = 200, days: int = 90):
    """Run the strategy comparison - PARALLELIZED"""
    from config.coins import get_top_n_coins

    print(f"Settings: MAX_LONGS={MAX_LONGS}, MAX_SHORTS={MAX_SHORTS}, RISK={RISK_PER_TRADE_PCT}%", flush=True)
    print(f"Filters: OB_STRENGTH>={OB_MIN_STRENGTH}, OB_AGE<={OB_MAX_AGE_CANDLES} candles", flush=True)
    print(f"Using {NUM_WORKERS} parallel workers", flush=True)
    print(f"Exit precision: {'1MIN CANDLES (accurate)' if USE_1MIN_EXITS else '5min candles'}", flush=True)
    print(f"Entry precision: OB EDGE (limit order at OB top/bottom)", flush=True)
    print("NOTE: Look-ahead bias FIXED - results now realistic!", flush=True)

    coins = get_top_n_coins(num_coins)

    # Skip problematic coins
    skip = {'APEUSDT', 'MATICUSDT', 'OCEANUSDT', 'EOSUSDT'}
    coins = [c for c in coins if c not in skip]

    print(f"\nProcessing {len(coins)} coins over {days} days...", flush=True)

    # Prepare arguments for parallel processing
    args = [(coin, days) for coin in coins]

    # Process coins in parallel
    all_trades = []

    with Pool(NUM_WORKERS) as pool:
        results = pool.map(process_single_coin, args)

        for i, coin_trades in enumerate(results):
            if coin_trades:
                print(f"  [{i+1}/{len(coins)}] {coins[i]}: {len(coin_trades)} trades", flush=True)
                all_trades.extend(coin_trades)
            else:
                print(f"  [{i+1}/{len(coins)}] {coins[i]}: SKIP", flush=True)

    print(f"\nTotal raw trades: {len(all_trades)}", flush=True)

    # Apply position limits across all coins
    filtered_trades = apply_position_limits(all_trades)
    print(f"After position limits: {len(filtered_trades)} trades", flush=True)

    # Print results
    print_results(filtered_trades)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--coins', type=int, default=30)
    parser.add_argument('--days', type=int, default=14)
    args = parser.parse_args()

    run_comparison(args.coins, args.days)
