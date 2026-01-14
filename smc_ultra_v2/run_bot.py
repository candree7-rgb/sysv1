#!/usr/bin/env python3
"""
Railway Entry Point
===================
Downloads data and starts the bot automatically.
"""

import os
import sys
import asyncio
from datetime import datetime

# Mode from env
MODE = os.getenv('BOT_MODE', 'paper')  # paper, live, backtest


def download_minimal_data():
    """Download just enough data to start trading"""
    print("=" * 60)
    print("SMC ULTRA V2 - RAILWAY STARTUP")
    print("=" * 60)
    print(f"Mode: {MODE}")
    print(f"Time: {datetime.utcnow()}")
    print("=" * 60)

    from data import BybitDataDownloader
    from config.coins import get_top_n_coins

    dl = BybitDataDownloader()

    # Get top coins
    coins = get_top_n_coins(30)  # Top 30 for speed
    print(f"\nDownloading data for {len(coins)} coins...")

    # Download 7 days of 5min data (fast, enough for live)
    for i, symbol in enumerate(coins):
        print(f"  [{i+1}/{len(coins)}] {symbol}...", end="", flush=True)
        try:
            df = dl.download_coin(symbol, interval="5", days=7)
            if len(df) > 0:
                filepath = dl.get_cache_path(symbol, "5", 7)
                df.to_parquet(filepath)
                print(f" OK ({len(df)} bars)")
            else:
                print(" SKIP")
        except Exception as e:
            print(f" ERROR: {e}")

    print("\nData download complete!")


def run_paper_trading():
    """Run paper trading bot"""
    print("\nStarting Paper Trading Bot...")

    from config import config
    config.api.testnet = True

    # Check API keys
    if not config.api.api_key:
        print("\nERROR: BYBIT_API_KEY not set!")
        print("Set environment variables in Railway dashboard")
        sys.exit(1)

    # Import and run
    from main import run_live_bot
    import argparse

    args = argparse.Namespace(
        coins=30,
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

    coins = get_top_n_coins(30)
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
