#!/usr/bin/env python3
"""
Railway Entry Point
===================
Downloads data and starts the bot automatically.
"""

import os
import sys
import socket
import asyncio
import threading
from datetime import datetime

# CRITICAL: Set global socket timeout
socket.setdefaulttimeout(20)

# CRITICAL: Monkey-patch requests to ALWAYS use timeout
# This prevents ANY request from hanging forever
import requests
_original_request = requests.Session.request
def _timeout_request(self, method, url, **kwargs):
    if 'timeout' not in kwargs or kwargs['timeout'] is None:
        kwargs['timeout'] = 15  # Force 15s timeout on all requests
    return _original_request(self, method, url, **kwargs)
requests.Session.request = _timeout_request

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
    import signal
    from datetime import datetime, timedelta
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

    # Order management settings
    MAX_ORDER_AGE_MIN = int(os.getenv('MAX_ORDER_AGE_MIN', '30'))  # Cancel unfilled orders after X minutes
    RISK_PER_TRADE_PCT = float(os.getenv('RISK_PER_TRADE_PCT', '2.0'))
    SCAN_TIMEOUT_SEC = int(os.getenv('SCAN_TIMEOUT_SEC', '300'))  # 5 min for full scan (first scan slow)

    print("\n" + "=" * 60, flush=True)
    print("OB SCALPER LIVE - 1:1 Backtest Logic", flush=True)
    print("=" * 60, flush=True)
    print(f"Mode: {'PAPER (no real orders)' if PAPER_MODE else 'LIVE (real orders!)'}", flush=True)
    print(f"Network: {'TESTNET' if USE_TESTNET else 'MAINNET'}", flush=True)
    print(f"Order Expiry: {MAX_ORDER_AGE_MIN} minutes", flush=True)
    print("=" * 60, flush=True)

    print("[INIT] Loading config...", flush=True)
    from config import config
    config.api.testnet = USE_TESTNET

    # Check API keys
    if not config.api.api_key:
        print("\nERROR: BYBIT_API_KEY not set!", flush=True)
        sys.exit(1)
    print("[INIT] Config OK", flush=True)

    print("[INIT] Importing modules...", flush=True)
    from ob_scalper_live import OBScalperLive, print_signal
    print("[INIT] OBScalperLive imported", flush=True)
    from live.executor import BybitExecutor
    print("[INIT] BybitExecutor imported", flush=True)
    from config.coins import get_top_n_coins
    print("[INIT] All imports done", flush=True)

    print("[INIT] Creating scanner...", flush=True)
    scanner = OBScalperLive()
    print("[INIT] Scanner created", flush=True)

    print("[INIT] Creating executor...", flush=True)
    executor = BybitExecutor()
    print("[INIT] Executor created", flush=True)

    # Get balance with detailed debug
    print("Checking account balance...", flush=True)
    try:
        import socket
        socket.setdefaulttimeout(30)  # 30 second timeout for all connections
        balance = executor.get_balance()
        print(f"  Raw balance response: {balance}", flush=True)
    except Exception as e:
        print(f"  Balance check failed: {e}", flush=True)
        balance = {'error': str(e)}
    if 'error' in balance:
        print(f"  Balance Error: {balance['error']}", flush=True)
        print("  Check: API key permissions must include 'Unified Trading'", flush=True)
        print("  Check: Account must be Unified Trading Account (not Standard)", flush=True)
        available = 0
    else:
        available = balance.get('available', 0)
        equity = balance.get('equity', 0)
        print(f"  Equity: ${equity:,.2f}, Available: ${available:,.2f}", flush=True)
    print(f"Account Balance: ${available:,.2f} USDT", flush=True)

    print("[INIT] Getting coin list...", flush=True)
    # Use full 100 coins with rolling scan approach
    live_coin_limit = min(NUM_COINS, 100)
    coins = get_top_n_coins(live_coin_limit)
    print(f"[INIT] Got {len(coins)} coins", flush=True)

    # Filter problematic coins - hardcoded + ENV variable
    SKIP_HARDCODED = {
        'APEUSDT', 'MATICUSDT', 'OCEANUSDT', 'EOSUSDT', 'FHEUSDT',
        'WHITEWHALEUSDT', 'LITUSDT', 'ZKPUSDT',
    }
    # Add coins from ENV: SKIP_COINS="COIN1USDT,COIN2USDT,COIN3USDT"
    skip_env = os.getenv('SKIP_COINS', '')
    SKIP_FROM_ENV = set(c.strip().upper() for c in skip_env.split(',') if c.strip())
    SKIP = SKIP_HARDCODED | SKIP_FROM_ENV

    if SKIP_FROM_ENV:
        print(f"[SKIP] From ENV: {', '.join(SKIP_FROM_ENV)}", flush=True)

    coins = [c for c in coins if c not in SKIP]

    print(f"Scanning {len(coins)} coins...", flush=True)
    print(f"Max positions: {MAX_LONGS} longs, {MAX_SHORTS} shorts", flush=True)
    print("=" * 60, flush=True)

    # Skip preload - let cache build naturally during scans
    # First few scans will be slower but won't hang
    print("\n[STARTUP] Starting scans (HTF cache builds automatically)...", flush=True)

    # Track pending orders: {order_id: {'symbol': str, 'placed_at': datetime, 'direction': str}}
    pending_orders = {}

    # === ROLLING SCAN: Scan coins continuously, one at a time ===
    # 100 coins × 2.5s = 250s per cycle (~4 min) - fits well in 5min OB window
    SCAN_DELAY = 2.5  # seconds between each coin
    coin_index = 0
    last_status_time = time.time()
    STATUS_INTERVAL = 60  # Status update every 60 seconds
    signals_found = 0

    print(f"\n[ROLLING SCAN] {len(coins)} coins × {SCAN_DELAY}s = {len(coins) * SCAN_DELAY / 60:.1f} min cycle", flush=True)

    while True:
        try:
            now = datetime.utcnow()
            symbol = coins[coin_index]

            # === STATUS UPDATE (every 60s) ===
            if time.time() - last_status_time > STATUS_INTERVAL:
                last_status_time = time.time()
                print(f"\n[{now.strftime('%H:%M:%S')}] ── Status ──", flush=True)

                # Sync/cancel orders
                if not PAPER_MODE:
                    # Cancel expired
                    for order_id in list(pending_orders.keys()):
                        info = pending_orders[order_id]
                        if (now - info['placed_at']).total_seconds() / 60 > MAX_ORDER_AGE_MIN:
                            print(f"  [EXPIRE] {info['symbol']}", flush=True)
                            executor.cancel_order(info['symbol'], order_id)
                            del pending_orders[order_id]

                    # Check filled
                    open_orders = executor.get_open_orders()
                    open_ids = {o['orderId'] for o in open_orders}
                    for oid in list(pending_orders.keys()):
                        if oid not in open_ids:
                            info = pending_orders.pop(oid)
                            print(f"  [FILLED] {info['symbol']} {info['direction'].upper()}!", flush=True)

                # Show status
                positions = executor.get_all_positions()
                longs = sum(1 for p in positions if p.side == 'Buy')
                shorts = sum(1 for p in positions if p.side == 'Sell')
                print(f"  Positions: {longs}L/{shorts}S | Pending: {len(pending_orders)}", flush=True)
                print(f"  Cycle: {coin_index}/{len(coins)} | Signals this hour: {signals_found}", flush=True)

            # === SCAN SINGLE COIN ===
            # Global socket timeout (20s) will auto-skip hanging coins
            print(f"  [{coin_index}] {symbol}...", end="", flush=True)

            signal = None
            try:
                signal = scanner.get_signal(symbol)
                print(" OK", flush=True)
            except socket.timeout:
                print(" TIMEOUT!", flush=True)
            except Exception as e:
                err_msg = str(e)[:25]
                if 'timed out' in err_msg.lower():
                    print(" TIMEOUT!", flush=True)
                else:
                    print(f" skip", flush=True)

            if signal:
                # Check position limits
                positions = executor.get_all_positions()
                longs = sum(1 for p in positions if p.side == 'Buy')
                shorts = sum(1 for p in positions if p.side == 'Sell')
                pending_l = sum(1 for o in pending_orders.values() if o['direction'] == 'long')
                pending_s = sum(1 for o in pending_orders.values() if o['direction'] == 'short')

                skip = False
                if signal.direction == 'long' and (longs + pending_l) >= MAX_LONGS:
                    skip = True
                if signal.direction == 'short' and (shorts + pending_s) >= MAX_SHORTS:
                    skip = True
                if any(o['symbol'] == signal.symbol for o in pending_orders.values()):
                    skip = True

                if not skip:
                    signals_found += 1
                    print(f"\n★ {symbol} {signal.direction.upper()} @ {signal.entry_price:.4f}", flush=True)
                    print(f"  SL: {signal.sl_price:.4f} | TP: {signal.tp_price:.4f}", flush=True)

                    if PAPER_MODE:
                        sl_pct = abs(signal.entry_price - signal.sl_price) / signal.entry_price * 100
                        print(f"  [PAPER] SL: {sl_pct:.2f}%, Lev: {signal.leverage}x", flush=True)
                    else:
                        # Place real order
                        balance = executor.get_balance()
                        equity = balance.get('available', 0)
                        sl_pct = abs(signal.entry_price - signal.sl_price) / signal.entry_price * 100

                        # Calculate qty based on risk
                        risk_usd = equity * (RISK_PER_TRADE_PCT / 100)
                        qty_usd = risk_usd / (sl_pct / 100) if sl_pct > 0 else 0

                        # Cap position size to 80% of available margin * leverage
                        max_position_usd = equity * 0.8 * signal.leverage
                        if qty_usd > max_position_usd:
                            qty_usd = max_position_usd
                            print(f"  [CAP] Position capped to ${qty_usd:.0f}", flush=True)

                        qty = qty_usd / signal.entry_price if signal.entry_price > 0 else 0

                        # Round to appropriate precision (most coins: 0-3 decimals)
                        if signal.entry_price > 100:
                            qty = round(qty, 2)  # BTC, ETH etc
                        elif signal.entry_price > 1:
                            qty = round(qty, 1)  # Mid-price coins
                        else:
                            qty = round(qty, 0)  # Low price coins - whole numbers

                        print(f"  [CALC] qty={qty}, notional=${qty*signal.entry_price:.0f}, margin=${qty*signal.entry_price/signal.leverage:.0f}", flush=True)

                        if qty > 0:
                            executor.set_leverage(signal.symbol, signal.leverage)
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
                                    print(f"  [ORDER] {order_id[:8]}... qty={qty:.4f}", flush=True)
                                else:
                                    print(f"  [ERR] {response['retMsg']}", flush=True)
                            except Exception as e:
                                print(f"  [ERR] {str(e)[:50]}", flush=True)

            # Next coin
            coin_index = (coin_index + 1) % len(coins)
            time.sleep(SCAN_DELAY)

        except KeyboardInterrupt:
            print("\nStopping...", flush=True)
            if not PAPER_MODE and pending_orders:
                for order_id, info in pending_orders.items():
                    executor.cancel_order(info['symbol'], order_id)
            break
        except Exception as e:
            print(f"[ERR] {e}", flush=True)
            time.sleep(5)


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
