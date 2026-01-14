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
    print("[DEBUG] run_live_bot started", flush=True)

    from live import BybitExecutor, BybitWebSocket, DataAggregator
    from strategy import SignalGenerator, PositionManager

    print("[DEBUG] Imports done, creating executor...", flush=True)

    # Initialize
    executor = BybitExecutor()
    print("[DEBUG] Executor created", flush=True)

    signal_gen = SignalGenerator()
    print("[DEBUG] SignalGenerator created", flush=True)

    position_mgr = PositionManager()
    print("[DEBUG] PositionManager created", flush=True)

    # Get tradeable coins
    coins = get_top_n_coins(args.coins)

    print(f"\nMonitoring {len(coins)} coins...", flush=True)

    reconnect_count = 0
    max_reconnects = 10

    while reconnect_count < max_reconnects:
        try:
            print("[DEBUG] Creating WebSocket...", flush=True)
            # Create fresh WebSocket and aggregator
            ws = BybitWebSocket()
            aggregator = DataAggregator()

            # Setup callbacks
            ws.on_kline(aggregator.add_kline)
            ws.on_ticker(aggregator.add_ticker)

            def on_position_update(data):
                print(f"Position update: {data}", flush=True)

            ws.on_position(on_position_update)

            # Start WebSocket
            print("[DEBUG] Starting WebSocket...", flush=True)
            await ws.start()
            print("[DEBUG] WebSocket started, subscribing...", flush=True)
            ws.subscribe_multiple(coins[:20])  # Subscribe to top 20

            print("WebSocket connected. Starting trading loop...", flush=True)
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
    print("[DEBUG] Trading loop started", flush=True)
    loop_count = 0

    # Track pending limit orders: {order_id: {'symbol': str, 'placed_at': datetime, 'signal': Signal}}
    pending_orders = {}
    ORDER_EXPIRY_MINUTES = 15  # Cancel unfilled orders after 15 min

    while True:
        loop_count += 1
        print(f"[DEBUG] Loop iteration {loop_count}", flush=True)

        try:
            # Check balance
            print("[DEBUG] Checking balance...", flush=True)
            balance = executor.get_balance()
            print(f"[DEBUG] Balance result: {balance}", flush=True)

            if 'error' in balance:
                print(f"Balance error: {balance['error']}", flush=True)
                await asyncio.sleep(60)
                continue

            print(f"\n[{datetime.utcnow()}] Balance: ${balance['equity']:,.2f}", flush=True)

            # Check and cancel expired pending orders
            now = datetime.utcnow()
            expired_orders = []
            for order_id, order_info in pending_orders.items():
                age_minutes = (now - order_info['placed_at']).total_seconds() / 60
                if age_minutes > ORDER_EXPIRY_MINUTES:
                    print(f"Cancelling expired order: {order_info['symbol']} (age: {age_minutes:.1f}min)", flush=True)
                    if executor.cancel_order(order_info['symbol'], order_id):
                        expired_orders.append(order_id)

            for order_id in expired_orders:
                del pending_orders[order_id]

            # Check if pending orders got filled (became positions)
            filled_orders = []
            for order_id, order_info in pending_orders.items():
                position = executor.get_position(order_info['symbol'])
                if position and position.size > 0:
                    print(f"Order filled: {order_info['symbol']} @ {position.entry_price}", flush=True)
                    filled_orders.append(order_id)

                    # Add to position manager
                    from strategy import Trade
                    import uuid
                    signal = order_info['signal']
                    trade = Trade(
                        id=str(uuid.uuid4())[:8],
                        symbol=signal.symbol,
                        direction=signal.direction,
                        entry_price=position.entry_price,
                        entry_time=datetime.utcnow(),
                        take_profit=signal.take_profit,
                        stop_loss=signal.stop_loss,
                        current_sl=signal.stop_loss,
                        leverage=signal.leverage,
                        confidence=signal.confidence,
                        factors=signal.factors
                    )
                    position_mgr.open_trade(trade)

            for order_id in filled_orders:
                del pending_orders[order_id]

            # Scan for signals
            print("[DEBUG] Scanning for signals...", flush=True)
            signals = signal_gen.scan_all(coins[:30])
            print(f"[DEBUG] Found {len(signals) if signals else 0} signals", flush=True)

            if signals:
                print(f"Found {len(signals)} signals", flush=True)
                for sig in signals[:3]:  # Top 3
                    print(f"  {sig.symbol}: {sig.direction} @ {sig.entry_price:.4f} "
                          f"(conf: {sig.confidence}%)", flush=True)

                # Execute best signal if conditions met
                active_count = len(position_mgr.trades) + len(pending_orders)
                if active_count < config.risk.max_concurrent_trades:
                    best = signals[0]

                    # Check not already pending or in position
                    symbols_in_use = set(o['symbol'] for o in pending_orders.values())
                    symbols_in_use.update(t.symbol for t in position_mgr.trades.values())

                    if best.symbol not in symbols_in_use:
                        can_trade, reason = position_mgr.can_open_trade(best.symbol)

                        if can_trade:
                            print(f"\nPlacing LIMIT order: {best.symbol} {best.direction} @ {best.entry_price:.4f}", flush=True)
                            print(f"  TP: {best.take_profit:.4f} (Limit) | SL: {best.stop_loss:.4f} (Market)", flush=True)
                            result = executor.open_position(best)

                            if result.success:
                                # Track pending order
                                pending_orders[result.order_id] = {
                                    'symbol': best.symbol,
                                    'placed_at': datetime.utcnow(),
                                    'signal': best
                                }
                                print(f"Limit order placed: {result.order_id}", flush=True)
                            else:
                                print(f"Order failed: {result.error}", flush=True)

            # Update positions
            prices = {s: aggregator.get_current_price(s) for s in coins[:30]}
            prices = {k: v for k, v in prices.items() if v is not None}

            closed = position_mgr.update_all(prices, datetime.utcnow())
            for trade in closed:
                print(f"Trade closed: {trade.symbol} "
                      f"PnL: {trade.pnl_pct:.2f}%", flush=True)

            # Status summary
            print(f"[Status] Positions: {len(position_mgr.trades)} | Pending: {len(pending_orders)}", flush=True)

            # Wait
            await asyncio.sleep(args.interval_seconds)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Loop error: {e}", flush=True)
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
