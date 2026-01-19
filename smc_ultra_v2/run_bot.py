#!/usr/bin/env python3
"""
Railway Entry Point
===================
Downloads data and starts the bot automatically.
"""

import os
import sys
import asyncio
import threading
from datetime import datetime

# Suppress pybit WebSocket thread errors (cosmetic, doesn't affect operation)
def _silent_thread_exception(args):
    if 'WebSocketConnectionClosedException' in str(args.exc_type):
        pass  # Ignore WebSocket closed errors in background threads
    else:
        sys.__excepthook__(args.exc_type, args.exc_value, args.exc_traceback)

threading.excepthook = _silent_thread_exception

# Config from env
MODE = os.getenv('BOT_MODE', 'paper')  # paper, live, backtest, scalper_live
NUM_COINS = int(os.getenv('BOT_COINS', '200'))  # Number of coins to trade (200 for better sample)
USE_TESTNET = os.getenv('USE_TESTNET', 'false').lower() == 'true'  # true = testnet.bybit.com, false = bybit.com demo
PAPER_MODE = os.getenv('PAPER_MODE', 'true').lower() == 'true'  # true = log signals only, false = place real orders
MIN_CONFIDENCE = int(os.getenv('MIN_CONFIDENCE', '60'))  # Minimum confidence for trades
# Hedged exposure: separate limits for longs and shorts
MAX_LONGS = int(os.getenv('MAX_LONGS', '2'))    # Max 2 long trades
MAX_SHORTS = int(os.getenv('MAX_SHORTS', '2'))  # Max 2 short trades
BACKTEST_DAYS = int(os.getenv('BACKTEST_DAYS', '90'))  # Days to backtest (90 for better sample)


def download_minimal_data():
    """Download just enough data to start trading"""
    print("=" * 60)
    print("SMC ULTRA V2 - RAILWAY STARTUP")
    print("=" * 60)
    print(f"Mode: {MODE}")
    print(f"Coins: {NUM_COINS}")
    print(f"API: {'testnet.bybit.com' if USE_TESTNET else 'bybit.com (demo)'}")
    print(f"Time: {datetime.utcnow()}")
    print("=" * 60)

    from data import BybitDataDownloader
    from config.coins import get_top_n_coins

    dl = BybitDataDownloader()

    # Get top coins
    coins = get_top_n_coins(NUM_COINS)
    print(f"\nDownloading data for {len(coins)} coins...", flush=True)

    # Download 7 days of 5min data (fast, enough for live)
    # Known problematic coins that often timeout/have no data
    SKIP_COINS = {'APEUSDT', 'MATICUSDT', 'OCEANUSDT', 'EOSUSDT', 'RNDRUSDT', 'FETUSDT', 'AGIXUSDT', 'MKRUSDT', 'FOGOUSDT', 'FHEUSDT'}

    successful = 0
    try:
        for i, symbol in enumerate(coins):
            try:
                # Skip known problematic coins
                if symbol in SKIP_COINS:
                    print(f"  [{i+1}/{len(coins)}] {symbol}... SKIP (known issue)", flush=True)
                    continue

                print(f"  [{i+1}/{len(coins)}] {symbol}...", end="", flush=True)

                df = dl.download_coin(symbol, interval="5", days=7)
                if len(df) > 0:
                    filepath = dl.get_cache_path(symbol, "5", 7)
                    df.to_parquet(filepath)
                    print(f" OK ({len(df)} bars)", flush=True)
                    successful += 1
                else:
                    print(" SKIP (no data)", flush=True)
            except Exception as e:
                print(f" ERROR: {e}", flush=True)

        print(f"\nData download complete! ({successful} coins loaded)", flush=True)
    except Exception as e:
        print(f"\nDownload loop error: {e}", flush=True)
        print(f"Continuing with {successful} coins...", flush=True)


def run_paper_trading():
    """Run paper trading bot"""
    print("\nStarting Paper Trading Bot...", flush=True)
    print(f"Using {'TESTNET (testnet.bybit.com)' if USE_TESTNET else 'DEMO (bybit.com)'}", flush=True)

    from config import config
    config.api.testnet = USE_TESTNET  # False = use bybit.com (demo trading keys work)

    # Check API keys
    if not config.api.api_key:
        print("\nERROR: BYBIT_API_KEY not set!", flush=True)
        print("Set environment variables in Railway dashboard", flush=True)
        sys.exit(1)

    print("[DEBUG] API key found, importing main...", flush=True)

    # Import and run
    from main import run_live_bot
    import argparse

    print("[DEBUG] Starting asyncio loop...", flush=True)

    args = argparse.Namespace(
        coins=NUM_COINS,
        interval_seconds=60,
        paper=True
    )

    asyncio.run(run_live_bot(args))


def run_backtest():
    """Run backtest"""
    print("\n[DEBUG] Running Backtest...", flush=True)

    from datetime import timedelta
    from backtest import BacktestEngine, BacktestConfig
    from config.coins import get_top_n_coins

    print("[DEBUG] Imports done", flush=True)

    coins = get_top_n_coins(NUM_COINS)
    # Filter out skipped coins
    SKIP_COINS = {'APEUSDT', 'MATICUSDT', 'OCEANUSDT', 'EOSUSDT', 'RNDRUSDT', 'FETUSDT', 'AGIXUSDT', 'MKRUSDT', 'FOGOUSDT', 'FHEUSDT'}
    coins = [c for c in coins if c not in SKIP_COINS]

    end = datetime.utcnow()
    start = end - timedelta(days=BACKTEST_DAYS)

    print(f"[DEBUG] Backtest period: {start} to {end}", flush=True)
    print(f"[DEBUG] Coins: {len(coins)}", flush=True)

    bt_config = BacktestConfig(
        symbols=coins[:NUM_COINS],  # Use NUM_COINS from env
        start_date=start,
        end_date=end,
        initial_capital=10000,
        min_confidence=MIN_CONFIDENCE,  # From ENV
        max_trades=MAX_LONGS + MAX_SHORTS  # Total from hedged limits
    )

    print(f"[DEBUG] Config: min_confidence={MIN_CONFIDENCE}, max_longs={MAX_LONGS}, max_shorts={MAX_SHORTS}", flush=True)

    print("[DEBUG] Creating BacktestEngine...", flush=True)
    engine = BacktestEngine(bt_config)

    print("[DEBUG] Running backtest (this may take a few minutes)...", flush=True)
    results = engine.run()

    print("[DEBUG] Saving results...", flush=True)
    engine.save_results()

    print(f"\n{'='*50}", flush=True)
    print(f"BACKTEST COMPLETE!", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"Total Trades: {results.total_trades}", flush=True)
    print(f"Win Rate: {results.win_rate:.1f}%", flush=True)
    print(f"Profit Factor: {results.profit_factor:.2f}", flush=True)
    print(f"Total Return: {results.total_return_pct:.2f}%", flush=True)
    print(f"Max Drawdown: {results.max_drawdown_pct:.2f}%", flush=True)
    print(f"{'='*50}", flush=True)


def run_optimizer():
    """Run strategy optimizer to find best parameters"""
    print("\n" + "=" * 60, flush=True)
    print("STRATEGY OPTIMIZER - Finding Best Parameters", flush=True)
    print("=" * 60, flush=True)

    from optimizer import run_optimization

    # Run optimization
    results = run_optimization(
        num_coins=NUM_COINS,
        days=BACKTEST_DAYS,
        fast=True  # Use fast grid for Railway (full grid takes too long)
    )

    if results:
        best = results[0]
        print(f"\n🏆 BEST STRATEGY: {best['config']}", flush=True)
        print(f"   Win Rate: {best['win_rate']}%", flush=True)
        print(f"   Profit Factor: {best['profit_factor']}", flush=True)


def run_compare():
    """Run SMC strategy comparison"""
    print("\n" + "=" * 60, flush=True)
    print("OB_RETEST BACKTEST (90 days, 100 coins)", flush=True)
    print("=" * 60, flush=True)
    print(f"Testing OB_RETEST with {NUM_COINS} coins over {BACKTEST_DAYS} days", flush=True)

    from strategy_comparison import run_comparison

    run_comparison(
        num_coins=NUM_COINS,
        days=BACKTEST_DAYS
    )


def run_variants():
    """Run multi-variant strategy comparison to find optimal parameters"""
    print("\n" + "=" * 60, flush=True)
    print("STRATEGY VARIANTS COMPARISON", flush=True)
    print("=" * 60, flush=True)
    print(f"Testing 12 variants with {NUM_COINS} coins over {BACKTEST_DAYS} days", flush=True)

    from strategy_variants import run_variant_comparison

    run_variant_comparison(
        num_coins=NUM_COINS,
        days=BACKTEST_DAYS
    )


def run_scalper():
    """Run OB Scalper backtest with 1min precision"""
    print("\n" + "=" * 60, flush=True)
    print("OB SCALPER BACKTEST - 1min Precision", flush=True)
    print("=" * 60, flush=True)
    print(f"Testing {NUM_COINS} coins over {BACKTEST_DAYS} days", flush=True)

    from ob_scalper import run_ob_scalper

    run_ob_scalper(
        num_coins=NUM_COINS,
        days=BACKTEST_DAYS
    )


def run_scalper_live():
    """Run OB Scalper LIVE trading - 1:1 with backtest logic"""
    import time
    from datetime import datetime, timedelta

    # Order management settings
    MAX_ORDER_AGE_MIN = int(os.getenv('MAX_ORDER_AGE_MIN', '30'))  # Cancel unfilled orders after X minutes
    RISK_PER_TRADE_PCT = float(os.getenv('RISK_PER_TRADE_PCT', '2.0'))

    print("\n" + "=" * 60, flush=True)
    print("OB SCALPER LIVE - 1:1 Backtest Logic", flush=True)
    print("=" * 60, flush=True)
    print(f"Mode: {'PAPER (no real orders)' if PAPER_MODE else 'LIVE (real orders!)'}")
    print(f"Network: {'TESTNET' if USE_TESTNET else 'MAINNET'}")
    print(f"Order Expiry: {MAX_ORDER_AGE_MIN} minutes")
    print("=" * 60, flush=True)

    from config import config
    config.api.testnet = USE_TESTNET

    # Check API keys
    if not config.api.api_key:
        print("\nERROR: BYBIT_API_KEY not set!", flush=True)
        sys.exit(1)

    from ob_scalper_live import OBScalperLive, print_signal
    from live.executor import BybitExecutor
    from config.coins import get_top_n_coins

    scanner = OBScalperLive()
    executor = BybitExecutor()

    # Get balance
    balance = executor.get_balance()
    if 'error' in balance:
        print(f"Balance Error: {balance['error']}")
        available = 0
    else:
        available = balance.get('available', 0)
    print(f"Account Balance: ${available:,.2f} USDT")

    coins = get_top_n_coins(NUM_COINS)
    # Filter known problematic coins
    SKIP = {'APEUSDT', 'MATICUSDT', 'OCEANUSDT', 'EOSUSDT', 'FHEUSDT'}
    coins = [c for c in coins if c not in SKIP]

    print(f"Scanning {len(coins)} coins...")
    print(f"Max positions: {MAX_LONGS} longs, {MAX_SHORTS} shorts")
    print("=" * 60)

    # Track pending orders: {order_id: {'symbol': str, 'placed_at': datetime, 'direction': str}}
    pending_orders = {}

    # Main loop
    scan_interval = 60  # seconds
    while True:
        try:
            now = datetime.utcnow()
            print(f"\n[{now.strftime('%H:%M:%S')}] Scanning...", flush=True)

            # === 1. CANCEL EXPIRED ORDERS ===
            if not PAPER_MODE and pending_orders:
                expired_orders = []
                for order_id, info in pending_orders.items():
                    age_min = (now - info['placed_at']).total_seconds() / 60
                    if age_min > MAX_ORDER_AGE_MIN:
                        expired_orders.append((order_id, info))

                for order_id, info in expired_orders:
                    print(f"  [EXPIRE] Cancelling {info['symbol']} order (age: {(now - info['placed_at']).total_seconds()/60:.1f}min)")
                    if executor.cancel_order(info['symbol'], order_id):
                        del pending_orders[order_id]
                        print(f"    Cancelled: {order_id}")
                    else:
                        print(f"    Cancel failed (may be filled)")
                        del pending_orders[order_id]  # Remove anyway

            # === 2. SYNC PENDING ORDERS WITH EXCHANGE ===
            if not PAPER_MODE:
                open_orders = executor.get_open_orders()
                open_order_ids = {o['orderId'] for o in open_orders}
                # Remove orders that are no longer pending (filled or cancelled)
                filled_orders = [oid for oid in pending_orders if oid not in open_order_ids]
                for oid in filled_orders:
                    info = pending_orders.pop(oid)
                    print(f"  [FILLED] {info['symbol']} {info['direction'].upper()} order filled!")

            # === 3. GET CURRENT POSITIONS ===
            positions = executor.get_all_positions()
            long_positions = sum(1 for p in positions if p.side == 'Buy')
            short_positions = sum(1 for p in positions if p.side == 'Sell')
            # Count pending orders as "reserved" slots
            pending_longs = sum(1 for o in pending_orders.values() if o['direction'] == 'long')
            pending_shorts = sum(1 for o in pending_orders.values() if o['direction'] == 'short')

            print(f"  Positions: {long_positions} longs, {short_positions} shorts")
            print(f"  Pending: {pending_longs} long orders, {pending_shorts} short orders")

            # === 4. SCAN FOR NEW SIGNALS ===
            signals = scanner.scan_coins(coins)

            for signal in signals:
                # Check position + pending limits
                total_longs = long_positions + pending_longs
                total_shorts = short_positions + pending_shorts

                if signal.direction == 'long' and total_longs >= MAX_LONGS:
                    print(f"  Skip {signal.symbol} LONG - max longs reached ({total_longs}/{MAX_LONGS})")
                    continue
                if signal.direction == 'short' and total_shorts >= MAX_SHORTS:
                    print(f"  Skip {signal.symbol} SHORT - max shorts reached ({total_shorts}/{MAX_SHORTS})")
                    continue

                # Skip if already have pending order for this symbol
                if any(o['symbol'] == signal.symbol for o in pending_orders.values()):
                    print(f"  Skip {signal.symbol} - already have pending order")
                    continue

                print_signal(signal)

                # === 5. PLACE ORDER ===
                if PAPER_MODE:
                    # Calculate theoretical qty for display
                    sl_pct = abs(signal.entry_price - signal.sl_price) / signal.entry_price * 100
                    risk_usd = available * (RISK_PER_TRADE_PCT / 100)
                    qty_usd = risk_usd / (sl_pct / 100) if sl_pct > 0 else 0
                    qty = qty_usd / signal.entry_price if signal.entry_price > 0 else 0

                    print(f"  [PAPER] Would place {signal.direction.upper()} @ {signal.entry_price:.4f}")
                    print(f"          SL: {signal.sl_price:.4f} ({sl_pct:.2f}%), TP: {signal.tp_price:.4f}")
                    print(f"          Qty: {qty:.4f} (~${qty_usd:.2f}), Lev: {signal.leverage}x")
                else:
                    # Calculate position size
                    balance = executor.get_balance()
                    if 'error' in balance:
                        print(f"  [ERROR] Can't get balance: {balance['error']}")
                        continue

                    equity = balance.get('available', 0)
                    sl_pct = abs(signal.entry_price - signal.sl_price) / signal.entry_price * 100

                    # Risk-based qty calculation
                    risk_usd = equity * (RISK_PER_TRADE_PCT / 100)
                    qty_usd = risk_usd / (sl_pct / 100) if sl_pct > 0 else 0

                    # Apply leverage and convert to coin qty
                    qty = qty_usd / signal.entry_price if signal.entry_price > 0 else 0
                    qty = round(qty, 3)  # Most coins accept 3 decimals

                    if qty <= 0:
                        print(f"  [ERROR] Invalid qty calculated: {qty}")
                        continue

                    # Set leverage first
                    executor.set_leverage(signal.symbol, signal.leverage)

                    # Place limit order with TP/SL
                    try:
                        response = executor.client.place_order(
                            category="linear",
                            symbol=signal.symbol,
                            side='Buy' if signal.direction == 'long' else 'Sell',
                            orderType="Limit",
                            price=str(round(signal.entry_price, 6)),
                            qty=str(qty),
                            timeInForce="PostOnly",
                            reduceOnly=False,
                            takeProfit=str(round(signal.tp_price, 6)),
                            stopLoss=str(round(signal.sl_price, 6)),
                            tpslMode="Full",
                            slOrderType="Market"
                        )

                        if response['retCode'] == 0:
                            order_id = response['result']['orderId']
                            pending_orders[order_id] = {
                                'symbol': signal.symbol,
                                'placed_at': now,
                                'direction': signal.direction
                            }
                            print(f"  [LIVE] Order placed: {order_id}")
                            print(f"         Qty: {qty:.4f} (~${qty_usd:.2f}), Expires in {MAX_ORDER_AGE_MIN}min")
                        else:
                            print(f"  [ERROR] Order failed: {response['retMsg']}")
                    except Exception as e:
                        print(f"  [ERROR] {e}")

            print(f"  Pending orders: {len(pending_orders)}")
            print(f"  Next scan in {scan_interval}s...")
            time.sleep(scan_interval)

        except KeyboardInterrupt:
            print("\nStopping bot...")
            # Cancel all pending orders on exit
            if not PAPER_MODE and pending_orders:
                print("Cancelling pending orders...")
                for order_id, info in pending_orders.items():
                    executor.cancel_order(info['symbol'], order_id)
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(10)


def run_mean_reversion():
    """Run Mean Reversion Short backtest"""
    print("\n" + "=" * 60, flush=True)
    print("MEAN REVERSION SHORT BACKTEST", flush=True)
    print("=" * 60, flush=True)
    print(f"Testing {NUM_COINS} coins over {BACKTEST_DAYS} days", flush=True)

    from mean_reversion_short import run_mean_reversion as mr_backtest

    mr_backtest(
        num_coins=NUM_COINS,
        days=BACKTEST_DAYS
    )


def main():
    print("\n[DEBUG] Starting main()", flush=True)

    # Always download fresh data first
    download_minimal_data()

    print(f"\n[DEBUG] Mode is: {MODE}", flush=True)

    if MODE == 'backtest':
        print("[DEBUG] Calling run_backtest()...", flush=True)
        run_backtest()
    elif MODE == 'optimize':
        print("[DEBUG] Calling run_optimizer()...", flush=True)
        run_optimizer()
    elif MODE == 'compare':
        print("[DEBUG] Calling run_compare()...", flush=True)
        run_compare()
    elif MODE == 'variants':
        print("[DEBUG] Calling run_variants()...", flush=True)
        run_variants()
    elif MODE == 'scalper':
        print("[DEBUG] Calling run_scalper()...", flush=True)
        run_scalper()
    elif MODE == 'scalper_live':
        print("[DEBUG] Calling run_scalper_live()...", flush=True)
        run_scalper_live()
    elif MODE == 'mean_reversion':
        print("[DEBUG] Calling run_mean_reversion()...", flush=True)
        run_mean_reversion()
    elif MODE in ['paper', 'live']:
        print("[DEBUG] Calling run_paper_trading()", flush=True)
        run_paper_trading()
    else:
        print(f"Unknown mode: {MODE}")
        print("Set BOT_MODE to: scalper_live, scalper, paper, live, backtest, optimize, compare, variants, or mean_reversion")
        print("\nRecommended: BOT_MODE=scalper_live (uses ob_scalper strategy)")


if __name__ == '__main__':
    main()
