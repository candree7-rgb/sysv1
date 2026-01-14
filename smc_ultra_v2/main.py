#!/usr/bin/env python3
"""
SMC Ultra V2 - Main Entry Point
================================
Ultimate Smart Money Concepts Trading Bot

Usage:
    # Run backtest
    python main.py backtest --days 90 --coins 50

    # Download data
    python main.py download --coins 100 --days 180

    # Live trading (paper)
    python main.py live --paper

    # Live trading (real)
    python main.py live --real
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import config, get_top_n_coins, TradingMode
from data import download_data, MTFDataLoader
from backtest import BacktestEngine, BacktestConfig
from strategy import SignalGenerator
from ml import MLConfidenceScorer, CoinFilter


def cmd_download(args):
    """Download historical data"""
    print("\n" + "=" * 60)
    print("SMC ULTRA V2 - DATA DOWNLOAD")
    print("=" * 60)

    coins = get_top_n_coins(args.coins)
    print(f"Downloading {len(coins)} coins, {args.days} days, {args.interval}m\n")

    data = download_data(
        symbols=coins,
        interval=args.interval,
        days=args.days
    )

    print(f"\nDownloaded {len(data)} coins successfully")


def cmd_backtest(args):
    """Run backtest"""
    print("\n" + "=" * 60)
    print("SMC ULTRA V2 - BACKTEST")
    print("=" * 60)

    # Date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=args.days)

    # Coins
    coins = get_top_n_coins(args.coins)

    # Config
    bt_config = BacktestConfig(
        symbols=coins,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        min_confidence=args.min_confidence,
        max_trades=args.max_trades,
        risk_per_trade=args.risk,
        max_leverage=args.max_leverage,
        fee_pct=0.075,
        slippage_pct=0.05,
        timeframe=args.timeframe
    )

    # Run
    engine = BacktestEngine(bt_config)
    results = engine.run()

    # Save
    if args.save:
        engine.save_results(args.output)


def cmd_live(args):
    """Run live trading"""
    print("\n" + "=" * 60)
    print("SMC ULTRA V2 - LIVE TRADING")
    print("=" * 60)

    if args.paper:
        print("MODE: PAPER TRADING (Testnet)")
        config.api.testnet = True
    else:
        print("MODE: LIVE TRADING")
        print("\n⚠️  WARNING: REAL MONEY AT RISK ⚠️")
        confirm = input("Type 'YES' to confirm: ")
        if confirm != 'YES':
            print("Aborted.")
            return
        config.api.testnet = False

    # Check API keys
    if not config.api.api_key or not config.api.api_secret:
        print("\nError: API keys not configured!")
        print("Set BYBIT_API_KEY and BYBIT_API_SECRET environment variables")
        return

    asyncio.run(run_live_bot(args))


async def run_live_bot(args):
    """Main live trading loop"""
    from live import BybitExecutor, BybitWebSocket, DataAggregator
    from strategy import SignalGenerator, PositionManager

    # Initialize
    executor = BybitExecutor()
    signal_gen = SignalGenerator()
    position_mgr = PositionManager()

    # Get tradeable coins
    coins = get_top_n_coins(args.coins)

    print(f"\nMonitoring {len(coins)} coins...")

    reconnect_count = 0
    max_reconnects = 10

    while reconnect_count < max_reconnects:
        try:
            # Create fresh WebSocket and aggregator
            ws = BybitWebSocket()
            aggregator = DataAggregator()

            # Setup callbacks
            ws.on_kline(aggregator.add_kline)
            ws.on_ticker(aggregator.add_ticker)

            def on_position_update(data):
                print(f"Position update: {data}")

            ws.on_position(on_position_update)

            # Start WebSocket
            await ws.start()
            ws.subscribe_multiple(coins[:20])  # Subscribe to top 20

            print("WebSocket connected. Starting trading loop...")
            reconnect_count = 0  # Reset on successful connection

            await _trading_loop(executor, ws, aggregator, signal_gen, position_mgr, coins, args)

        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            reconnect_count += 1
            print(f"\nWebSocket error: {e}")
            print(f"Reconnecting ({reconnect_count}/{max_reconnects}) in 10 seconds...")
            try:
                ws.stop()
            except:
                pass
            await asyncio.sleep(10)

    if reconnect_count >= max_reconnects:
        print("Max reconnects reached. Exiting.")


async def _trading_loop(executor, ws, aggregator, signal_gen, position_mgr, coins, args):
    """Inner trading loop"""
    while True:
        try:
            # Check balance
            balance = executor.get_balance()
            if 'error' in balance:
                print(f"Balance error: {balance['error']}")
                await asyncio.sleep(60)
                continue

            print(f"\n[{datetime.utcnow()}] Balance: ${balance['equity']:,.2f}")

            # Scan for signals
            signals = signal_gen.scan_all(coins[:30])

            if signals:
                print(f"Found {len(signals)} signals")
                for sig in signals[:3]:  # Top 3
                    print(f"  {sig.symbol}: {sig.direction} @ {sig.entry_price:.4f} "
                          f"(conf: {sig.confidence}%)")

                # Execute best signal if conditions met
                if len(position_mgr.trades) < config.risk.max_concurrent_trades:
                    best = signals[0]
                    can_trade, reason = position_mgr.can_open_trade(best.symbol)

                    if can_trade:
                        print(f"\nOpening trade: {best.symbol} {best.direction}")
                        result = executor.open_position(best)

                        if result.success:
                            from strategy import Trade
                            import uuid
                            trade = Trade(
                                id=str(uuid.uuid4())[:8],
                                symbol=best.symbol,
                                direction=best.direction,
                                entry_price=best.entry_price,
                                entry_time=datetime.utcnow(),
                                take_profit=best.take_profit,
                                stop_loss=best.stop_loss,
                                current_sl=best.stop_loss,
                                leverage=best.leverage,
                                confidence=best.confidence,
                                factors=best.factors
                            )
                            position_mgr.open_trade(trade)
                            print(f"Trade opened: {result.order_id}")
                        else:
                            print(f"Trade failed: {result.error}")

            # Update positions
            prices = {s: aggregator.get_current_price(s) for s in coins[:30]}
            prices = {k: v for k, v in prices.items() if v is not None}

            closed = position_mgr.update_all(prices, datetime.utcnow())
            for trade in closed:
                print(f"Trade closed: {trade.symbol} "
                      f"PnL: {trade.pnl_pct:.2f}%")

            # Wait
            await asyncio.sleep(args.interval_seconds)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Loop error: {e}")
            await asyncio.sleep(30)


def cmd_train(args):
    """Train ML model"""
    print("\n" + "=" * 60)
    print("SMC ULTRA V2 - ML TRAINING")
    print("=" * 60)

    # Load backtest trades
    import json
    trades_path = Path(args.trades_file)

    if not trades_path.exists():
        print(f"Error: Trades file not found: {trades_path}")
        print("Run backtest first with --save flag")
        return

    with open(trades_path) as f:
        trades_data = json.load(f)

    import pandas as pd
    trades_df = pd.DataFrame(trades_data)
    trades_df['outcome'] = (trades_df['pnl_pct'] > 0).astype(int)
    trades_df['timestamp'] = pd.to_datetime(trades_df['entry_time'])

    print(f"Loaded {len(trades_df)} trades")

    # Train model
    scorer = MLConfidenceScorer()
    metrics = scorer.train(trades_df, save=True)

    print(f"\nTraining complete:")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall: {metrics['recall']:.2%}")

    # Show top features
    if 'feature_importance' in metrics:
        print("\nTop 10 Features:")
        for feat, imp in list(metrics['feature_importance'].items())[:10]:
            print(f"  {feat}: {imp:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="SMC Ultra V2 - Ultimate Smart Money Concepts Trading Bot"
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Download command
    dl_parser = subparsers.add_parser('download', help='Download historical data')
    dl_parser.add_argument('--coins', type=int, default=100, help='Number of coins')
    dl_parser.add_argument('--days', type=int, default=180, help='Days of data')
    dl_parser.add_argument('--interval', default='5', help='Timeframe interval')

    # Backtest command
    bt_parser = subparsers.add_parser('backtest', help='Run backtest')
    bt_parser.add_argument('--coins', type=int, default=50, help='Number of coins')
    bt_parser.add_argument('--days', type=int, default=90, help='Days to backtest')
    bt_parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    bt_parser.add_argument('--min-confidence', type=int, default=85, help='Min confidence')
    bt_parser.add_argument('--max-trades', type=int, default=3, help='Max concurrent trades')
    bt_parser.add_argument('--risk', type=float, default=2.0, help='Risk per trade %')
    bt_parser.add_argument('--max-leverage', type=int, default=50, help='Max leverage')
    bt_parser.add_argument('--timeframe', default='5', help='Primary timeframe')
    bt_parser.add_argument('--save', action='store_true', help='Save results')
    bt_parser.add_argument('--output', default='backtest_results', help='Output directory')

    # Live command
    live_parser = subparsers.add_parser('live', help='Live trading')
    live_parser.add_argument('--paper', action='store_true', help='Paper trading (testnet)')
    live_parser.add_argument('--real', action='store_true', help='Real trading')
    live_parser.add_argument('--coins', type=int, default=30, help='Number of coins to monitor')
    live_parser.add_argument('--interval-seconds', type=int, default=60, help='Scan interval')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train ML model')
    train_parser.add_argument('--trades-file', default='backtest_results/trades.json',
                             help='Path to trades JSON file')

    args = parser.parse_args()

    if args.command == 'download':
        cmd_download(args)
    elif args.command == 'backtest':
        cmd_backtest(args)
    elif args.command == 'live':
        cmd_live(args)
    elif args.command == 'train':
        cmd_train(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
