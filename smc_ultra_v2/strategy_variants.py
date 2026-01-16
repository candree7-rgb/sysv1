"""
SMC Strategy Variants Tester
============================
Tests multiple OB_RETEST variants in parallel to find optimal parameters.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pool, cpu_count
from itertools import product

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

# Position limits
MAX_LONGS = int(os.getenv('MAX_LONGS', '2'))
MAX_SHORTS = int(os.getenv('MAX_SHORTS', '2'))

# Parallelization
NUM_WORKERS = int(os.getenv('NUM_WORKERS', str(min(8, cpu_count()))))

# Precision mode: use 1min candles for exit checking
USE_1MIN_EXITS = True


@dataclass
class VariantConfig:
    """Configuration for a strategy variant"""
    name: str
    ob_min_strength: float
    ob_max_age: int  # candles
    sl_mult: float  # ATR multiplier for SL
    tp_mult: float  # ATR multiplier for TP
    use_ob_entry: bool = True  # True = entry at OB edge, False = entry at close


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
    variant: str = ""


@dataclass
class VariantResult:
    """Results for a strategy variant"""
    name: str
    config: VariantConfig
    trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    max_dd: float
    final_equity: float


# Define variants to test
VARIANTS = [
    # Baseline (current)
    VariantConfig("baseline", 0.6, 100, 1.0, 1.5, False),

    # Entry at OB edge (more realistic)
    VariantConfig("ob_entry", 0.6, 100, 1.0, 1.5, True),

    # Stricter strength filter
    VariantConfig("high_strength", 0.7, 100, 1.0, 1.5, True),
    VariantConfig("very_high_strength", 0.8, 50, 1.0, 1.5, True),

    # Fresher OBs only
    VariantConfig("fresh_ob", 0.6, 50, 1.0, 1.5, True),
    VariantConfig("very_fresh_ob", 0.6, 30, 1.0, 1.5, True),

    # Different RR ratios
    VariantConfig("rr_1_2", 0.6, 100, 1.0, 2.0, True),
    VariantConfig("rr_1_2.5", 0.6, 100, 1.0, 2.5, True),
    VariantConfig("tight_sl", 0.6, 100, 0.7, 1.5, True),

    # Combined strict filters
    VariantConfig("strict_all", 0.75, 40, 1.0, 1.5, True),
    VariantConfig("ultra_strict", 0.8, 30, 0.8, 2.0, True),

    # Relaxed filters (more trades)
    VariantConfig("relaxed", 0.5, 150, 1.0, 1.5, True),
]


def process_coin_for_variant(args) -> List[Trade]:
    """Process a single coin for a specific variant

    Uses 5min candles for signal generation (OB detection, EMA trend)
    Uses 1min candles for precise SL/TP exit checking
    """
    symbol, days, variant = args

    # Disable SSL verification
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''

    from data import BybitDataDownloader
    from smc import OrderBlockDetector

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
                df_1m = None  # Fall back to 5min exits

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

        atr = df_5m['atr']
        obs = ob_det.detect(df_5m, atr)

        active_trade = None

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
                    closed = check_trade_exit_1min(active_trade, df_1m, ts)
                else:
                    closed = check_trade_exit(active_trade, candle)
                if closed:
                    trades.append(active_trade)
                    active_trade = None

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

            # Check for OB entry with variant-specific filters
            matching_ob = None
            for ob in active_obs:
                if (trend == 'long' and ob.is_bullish) or (trend == 'short' and not ob.is_bullish):
                    if ob.bottom <= price <= ob.top:
                        # CRITICAL: Verify limit order would have filled
                        # Long: price must DROP to ob.top (low <= ob.top)
                        # Short: price must RISE to ob.bottom (high >= ob.bottom)
                        if trend == 'long' and candle['low'] > ob.top:
                            continue  # Price never reached our limit buy
                        if trend == 'short' and candle['high'] < ob.bottom:
                            continue  # Price never reached our limit sell

                        # Filter 1: OB Strength (variant-specific)
                        if ob.strength < variant.ob_min_strength:
                            continue

                        # Filter 2: OB Age (variant-specific)
                        ob_age_minutes = (ts - ob.timestamp).total_seconds() / 60
                        ob_age_candles = ob_age_minutes / 5
                        if ob_age_candles > variant.ob_max_age:
                            continue

                        matching_ob = ob
                        break

            if matching_ob:
                active_trade = create_trade_with_variant(
                    symbol, trend, price, atr_val, ts, matching_ob, variant
                )

        # Close remaining trade
        if active_trade and len(df_5m) > 0:
            last_candle = df_5m.iloc[-1]
            if df_1m is not None and USE_1MIN_EXITS:
                check_trade_exit_1min(active_trade, df_1m, last_candle['timestamp'])
            else:
                check_trade_exit(active_trade, last_candle)
            if active_trade.exit_time:
                trades.append(active_trade)

    except Exception as e:
        pass  # Silently skip errors

    return trades


def create_trade_with_variant(
    symbol: str, direction: str, price: float, atr: float,
    ts: datetime, ob, variant: VariantConfig
) -> Trade:
    """Create a trade with variant-specific parameters"""

    # Entry price: at OB edge or at close
    if variant.use_ob_entry:
        if direction == 'long':
            # For long: enter at OB top (limit buy fills here)
            entry = ob.top  # Exact OB edge - maximum accuracy
        else:
            # For short: enter at OB bottom (limit sell fills here)
            entry = ob.bottom  # Exact OB edge - maximum accuracy
    else:
        # Traditional: entry at close with small slippage
        if direction == 'long':
            entry = price * 1.0003
        else:
            entry = price * 0.9997

    # SL/TP with variant-specific multipliers
    if direction == 'long':
        sl = entry - atr * variant.sl_mult
        tp = entry + atr * variant.tp_mult
    else:
        sl = entry + atr * variant.sl_mult
        tp = entry - atr * variant.tp_mult

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
        sl_pct=sl_pct,
        variant=variant.name
    )


def check_trade_exit(trade: Trade, candle) -> bool:
    """Check if trade should exit (5min fallback)"""
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


def apply_position_limits(trades: List[Trade]) -> List[Trade]:
    """Apply MAX_LONGS/MAX_SHORTS limits"""
    sorted_trades = sorted(trades, key=lambda t: t.entry_time)

    filtered_trades = []
    active_longs = []
    active_shorts = []

    for trade in sorted_trades:
        active_longs = [t for t in active_longs if t.exit_time is None or t.exit_time > trade.entry_time]
        active_shorts = [t for t in active_shorts if t.exit_time is None or t.exit_time > trade.entry_time]

        if trade.direction == 'long' and len(active_longs) >= MAX_LONGS:
            continue
        if trade.direction == 'short' and len(active_shorts) >= MAX_SHORTS:
            continue

        filtered_trades.append(trade)
        if trade.direction == 'long':
            active_longs.append(trade)
        else:
            active_shorts.append(trade)

    return filtered_trades


def calculate_variant_results(trades: List[Trade], variant: VariantConfig) -> VariantResult:
    """Calculate results for a variant"""
    if not trades:
        return VariantResult(
            name=variant.name, config=variant,
            trades=0, wins=0, losses=0,
            win_rate=0, profit_factor=0, total_pnl=0,
            max_dd=0, final_equity=10000
        )

    winners = [t for t in trades if t.result == 'win']
    losers = [t for t in trades if t.result == 'loss']

    win_count = len(winners)
    loss_count = len(losers)
    total = win_count + loss_count
    win_rate = win_count / total * 100 if total > 0 else 0

    gross_win = sum(t.pnl_leveraged for t in winners) if winners else 0
    gross_loss = abs(sum(t.pnl_leveraged for t in losers)) if losers else 1
    pf = gross_win / gross_loss if gross_loss > 0 else gross_win
    total_pnl = gross_win - gross_loss

    # Calculate equity curve and max DD
    equity = 10000.0
    peak = 10000.0
    max_dd = 0.0

    for trade in sorted(trades, key=lambda t: t.entry_time):
        pnl_usd = equity * (trade.pnl_leveraged / 100)
        equity += pnl_usd
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100
        max_dd = max(max_dd, dd)

    return VariantResult(
        name=variant.name,
        config=variant,
        trades=total,
        wins=win_count,
        losses=loss_count,
        win_rate=win_rate,
        profit_factor=pf,
        total_pnl=total_pnl,
        max_dd=max_dd,
        final_equity=equity
    )


def run_variant_comparison(num_coins: int = 100, days: int = 90, variants: List[VariantConfig] = None):
    """Run all variants and compare results"""
    from config.coins import get_top_n_coins

    variants = variants or VARIANTS

    print("=" * 100)
    print("SMC STRATEGY VARIANT COMPARISON")
    print("=" * 100)
    print(f"Testing {len(variants)} variants on {num_coins} coins over {days} days")
    print(f"Position limits: MAX_LONGS={MAX_LONGS}, MAX_SHORTS={MAX_SHORTS}")
    print(f"Using {NUM_WORKERS} parallel workers")
    print(f"Exit precision: {'1MIN CANDLES (accurate)' if USE_1MIN_EXITS else '5min candles'}")
    print(f"Entry precision: OB EDGE (limit order at OB top/bottom)")
    print("=" * 100)

    coins = get_top_n_coins(num_coins)
    skip = {'APEUSDT', 'MATICUSDT', 'OCEANUSDT', 'EOSUSDT'}
    coins = [c for c in coins if c not in skip]

    all_results = []

    for variant in variants:
        print(f"\n>>> Testing variant: {variant.name}")
        print(f"    Strength>={variant.ob_min_strength}, Age<={variant.ob_max_age}, "
              f"SL={variant.sl_mult}x, TP={variant.tp_mult}x, OB_Entry={variant.use_ob_entry}")

        # Prepare args for parallel processing
        args = [(coin, days, variant) for coin in coins]

        all_trades = []
        with Pool(NUM_WORKERS) as pool:
            results = pool.map(process_coin_for_variant, args)
            for coin_trades in results:
                if coin_trades:
                    all_trades.extend(coin_trades)

        print(f"    Raw trades: {len(all_trades)}")

        # Apply position limits
        filtered_trades = apply_position_limits(all_trades)
        print(f"    After limits: {len(filtered_trades)}")

        # Calculate results
        result = calculate_variant_results(filtered_trades, variant)
        all_results.append(result)

        print(f"    WR: {result.win_rate:.1f}% | PF: {result.profit_factor:.2f} | "
              f"PnL: {result.total_pnl:+.1f}% | DD: {result.max_dd:.1f}%")

    # Print comparison table
    print("\n" + "=" * 120)
    print("VARIANT COMPARISON RESULTS")
    print("=" * 120)
    print(f"{'Variant':<20} {'Trades':>7} {'Wins':>6} {'Loss':>6} {'WR%':>7} {'PF':>6} "
          f"{'PnL%':>10} {'MaxDD%':>8} {'$10k->':>12}")
    print("-" * 120)

    # Sort by total PnL
    sorted_results = sorted(all_results, key=lambda r: -r.total_pnl)

    for r in sorted_results:
        print(f"{r.name:<20} {r.trades:>7} {r.wins:>6} {r.losses:>6} {r.win_rate:>6.1f}% "
              f"{r.profit_factor:>5.2f} {r.total_pnl:>+9.1f}% {r.max_dd:>7.1f}% ${r.final_equity:>10,.0f}")

    print("=" * 120)

    # Print best variant
    best = sorted_results[0]
    print(f"\n🏆 BEST VARIANT: {best.name}")
    print(f"   Config: Strength>={best.config.ob_min_strength}, Age<={best.config.ob_max_age}, "
          f"SL={best.config.sl_mult}x, TP={best.config.tp_mult}x")
    print(f"   Win Rate: {best.win_rate:.1f}%")
    print(f"   Profit Factor: {best.profit_factor:.2f}")
    print(f"   Total PnL: {best.total_pnl:+.1f}%")
    print(f"   Max Drawdown: {best.max_dd:.1f}%")
    print(f"   $10,000 -> ${best.final_equity:,.0f}")

    return sorted_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--coins', type=int, default=100)
    parser.add_argument('--days', type=int, default=90)
    args = parser.parse_args()

    run_variant_comparison(args.coins, args.days)
