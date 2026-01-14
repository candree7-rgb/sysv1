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
    print("\nStarting Paper Trading Bot...")
    print(f"Using {'TESTNET (testnet.bybit.com)' if USE_TESTNET else 'DEMO (bybit.com)'}")

    from config import config
    config.api.testnet = USE_TESTNET  # False = use bybit.com (demo trading keys work)

    # Check API keys
    if not config.api.api_key:
        print("\nERROR: BYBIT_API_KEY not set!")
        print("Set environment variables in Railway dashboard")
        sys.exit(1)

    # Import and run
    from main import run_live_bot
    import argparse

    args = argparse.Namespace(
        coins=NUM_COINS,
        interval_seconds=60,
        paper=True
    )

    asyncio.run(run_live_bot(args))


def run_backtest():
    """Run backtest"""
    print("\nRunning Backtest...")

    from datetime import timedelta
    from backtest import BacktestEngine, BacktestConfig
    from config.coins import get_top_n_coins

    coins = get_top_n_coins(NUM_COINS)
    end = datetime.utcnow()
    start = end - timedelta(days=7)

    bt_config = BacktestConfig(
        symbols=coins,
        start_date=start,
        end_date=end,
        initial_capital=10000,
        min_confidence=85,
        max_trades=3
    )

    engine = BacktestEngine(bt_config)
    results = engine.run()
    engine.save_results()

    print(f"\nBacktest Complete!")
    print(f"Win Rate: {results.win_rate}%")
    print(f"Profit Factor: {results.profit_factor}")
    print(f"Total Return: {results.total_return_pct}%")


def main():
    # Always download fresh data first
    download_minimal_data()

    if MODE == 'backtest':
        run_backtest()
    elif MODE in ['paper', 'live']:
        run_paper_trading()
    else:
        print(f"Unknown mode: {MODE}")
        print("Set BOT_MODE to: paper, live, or backtest")


if __name__ == '__main__':
    main()
