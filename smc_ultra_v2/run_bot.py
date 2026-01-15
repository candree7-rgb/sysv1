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
MODE = os.getenv('BOT_MODE', 'paper')  # paper, live, backtest
NUM_COINS = int(os.getenv('BOT_COINS', '30'))  # Number of coins to trade
USE_TESTNET = os.getenv('USE_TESTNET', 'false').lower() == 'true'  # true = testnet.bybit.com, false = bybit.com demo
MIN_CONFIDENCE = int(os.getenv('MIN_CONFIDENCE', '60'))  # Minimum confidence for trades
MAX_TRADES = int(os.getenv('MAX_TRADES', '5'))  # Max concurrent trades
BACKTEST_DAYS = int(os.getenv('BACKTEST_DAYS', '7'))  # Days to backtest


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
    SKIP_COINS = {'APEUSDT', 'MATICUSDT', 'OCEANUSDT', 'EOSUSDT', 'RNDRUSDT', 'FETUSDT', 'AGIXUSDT', 'MKRUSDT'}

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
    SKIP_COINS = {'APEUSDT', 'MATICUSDT', 'OCEANUSDT', 'EOSUSDT', 'RNDRUSDT', 'FETUSDT', 'AGIXUSDT', 'MKRUSDT'}
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
        max_trades=MAX_TRADES  # From ENV
    )

    print(f"[DEBUG] Config: min_confidence={MIN_CONFIDENCE}, max_trades={MAX_TRADES}", flush=True)

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
    print("SMC STRATEGY COMPARISON (with Dynamic Leverage)", flush=True)
    print("=" * 60, flush=True)
    print("Testing: SWEEP_FVG, FVG_ONLY", flush=True)

    from strategy_comparison import run_comparison

    run_comparison(
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
    elif MODE in ['paper', 'live']:
        print("[DEBUG] Calling run_paper_trading()", flush=True)
        run_paper_trading()
    else:
        print(f"Unknown mode: {MODE}")
        print("Set BOT_MODE to: paper, live, backtest, optimize, or compare")


if __name__ == '__main__':
    main()
