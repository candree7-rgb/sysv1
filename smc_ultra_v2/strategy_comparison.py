"""
SMC Strategy Comparison - PARALLELIZED VERSION
==============================================
8x faster with multiprocessing!
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

# Parallelization
NUM_WORKERS = int(os.getenv('NUM_WORKERS', str(min(8, cpu_count()))))


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
    """Process a single coin - can be run in parallel"""
    symbol, days = args

    # Import inside function for multiprocessing
    from data import BybitDataDownloader
    from smc import OrderBlockDetector, FVGDetector, LiquidityDetector

    trades = []

    try:
        dl = BybitDataDownloader()
        ob_det = OrderBlockDetector()

        end = datetime.now()
        start = end - timedelta(days=days + 5)

        df = dl.load_or_download(symbol, "5", days + 10)
        if df is None or len(df) < 200:
            return []

        # Filter date range
        mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
        df = df[mask].reset_index(drop=True)

        if len(df) < 100:
            return []

        # Calculate indicators
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()

        # Detect OBs
        atr = df['atr']
        obs = ob_det.detect(df, atr)

        # Run backtest for this coin
        active_trade = None

        for idx in range(50, len(df)):
            candle = df.iloc[idx]
            ts = candle['timestamp']
            price = candle['close']
            atr_val = candle['atr']

            if pd.isna(atr_val) or atr_val <= 0:
                continue

            # Check and close active trade
            if active_trade:
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

            # Check for OB entry
            in_ob = False
            for ob in active_obs:
                if (trend == 'long' and ob.is_bullish) or (trend == 'short' and not ob.is_bullish):
                    if ob.bottom <= price <= ob.top:
                        in_ob = True
                        break

            if in_ob:
                active_trade = create_trade(symbol, trend, price, atr_val, ts)

        # Close any remaining trade
        if active_trade and len(df) > 0:
            last_candle = df.iloc[-1]
            check_trade_exit(active_trade, last_candle)
            if active_trade.exit_time:
                trades.append(active_trade)

    except Exception as e:
        print(f"  Error processing {symbol}: {e}", flush=True)

    return trades


def create_trade(symbol: str, direction: str, price: float, atr: float, ts: datetime) -> Trade:
    """Create a trade with 1:1.5 RR and dynamic leverage"""
    sl_mult = 1.0
    tp_mult = 1.5

    if direction == 'long':
        entry = price * 1.0003
        sl = entry - atr * sl_mult
        tp = entry + atr * tp_mult
    else:
        entry = price * 0.9997
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


def run_comparison(num_coins: int = 30, days: int = 14):
    """Run the strategy comparison - PARALLELIZED"""
    from config.coins import get_top_n_coins

    print(f"Settings: MAX_LONGS={MAX_LONGS}, MAX_SHORTS={MAX_SHORTS}, RISK={RISK_PER_TRADE_PCT}%", flush=True)
    print(f"Using {NUM_WORKERS} parallel workers", flush=True)
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
