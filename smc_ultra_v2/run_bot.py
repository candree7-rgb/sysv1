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


def _download_single_coin(symbol, days, result_queue):
    """Download a single coin with timeout support (runs in subprocess)"""
    try:
        from data import BybitDataDownloader
        dl = BybitDataDownloader()
        df = dl.download_coin(symbol, interval="5", days=days)
        if df is not None and len(df) > 0:
            filepath = dl.get_cache_path(symbol, "5", days)
            df.to_parquet(filepath)
            result_queue.put(('ok', len(df)))
        else:
            result_queue.put(('empty', 0))
    except Exception as e:
        result_queue.put(('error', str(e)))


def download_minimal_data():
    """Download just enough data to start trading"""
    import multiprocessing as mp

    print("=" * 60)
    print("SMC ULTRA V2 - RAILWAY STARTUP")
    print("=" * 60)
    print(f"Mode: {MODE}")
    print(f"Coins: {NUM_COINS}")
    print(f"API: {'testnet.bybit.com' if USE_TESTNET else 'bybit.com (demo)'}")
    print(f"Time: {datetime.utcnow()}")
    print("=" * 60)

    from config.coins import get_top_n_coins

    # Get top coins
    coins = get_top_n_coins(NUM_COINS)
    print(f"\nDownloading data for {len(coins)} coins...", flush=True)

    # Download 7 days of 5min data (fast, enough for live)
    # Known problematic coins that often timeout/have no data
    SKIP_COINS = {'APEUSDT', 'MATICUSDT', 'OCEANUSDT', 'EOSUSDT', 'RNDRUSDT', 'FETUSDT', 'AGIXUSDT', 'MKRUSDT', 'FOGOUSDT', 'FHEUSDT', 'SKRUSDT'}

    # Runtime skip: coins that timeout get added here automatically
    runtime_skip = set()
    DOWNLOAD_TIMEOUT = 30  # seconds per coin

    successful = 0
    try:
        for i, symbol in enumerate(coins):
            try:
                # Skip known problematic coins
                if symbol in SKIP_COINS:
                    print(f"  [{i+1}/{len(coins)}] {symbol}... SKIP (known issue)", flush=True)
                    continue

                # Skip coins that timed out previously
                if symbol in runtime_skip:
                    print(f"  [{i+1}/{len(coins)}] {symbol}... SKIP (auto-skip)", flush=True)
                    continue

                print(f"  [{i+1}/{len(coins)}] {symbol}...", end="", flush=True)

                # Use subprocess with timeout (like scanning does)
                result_queue = mp.Queue()
                proc = mp.Process(target=_download_single_coin, args=(symbol, 7, result_queue))
                proc.start()
                proc.join(timeout=DOWNLOAD_TIMEOUT)

                if proc.is_alive():
                    proc.kill()
                    proc.join()
                    print(" TIMEOUT!", flush=True)
                    runtime_skip.add(symbol)  # Auto-skip on timeout
                elif not result_queue.empty():
                    status, data = result_queue.get_nowait()
                    if status == 'ok':
                        print(f" OK ({data} bars)", flush=True)
                        successful += 1
                    elif status == 'empty':
                        print(" SKIP (no data)", flush=True)
                    else:
                        print(f" ERROR: {data}", flush=True)
                else:
                    print(" SKIP (no result)", flush=True)

            except Exception as e:
                print(f" ERROR: {e}", flush=True)

        print(f"\nData download complete! ({successful} coins loaded)", flush=True)
        if runtime_skip:
            print(f"Auto-skipped (timeout): {', '.join(runtime_skip)}", flush=True)
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
    print("[INIT] Coins imported", flush=True)
    from live.trade_logger import get_trade_logger, TradeRecord
    trade_logger = get_trade_logger()
    print("[INIT] Trade logger ready", flush=True)
    from live import telegram_alerts as tg
    if tg.is_enabled():
        print("[INIT] Telegram alerts enabled", flush=True)
    print("[INIT] All imports done", flush=True)

    print("[INIT] Creating scanner...", flush=True)
    scanner = OBScalperLive()
    print("[INIT] Scanner created", flush=True)

    print("[INIT] Creating executor...", flush=True)
    executor = BybitExecutor()
    print("[INIT] Executor created", flush=True)

    # === WEBSOCKET FOR REAL-TIME ORDER MONITORING ===
    # Track trade pairs for SL→BE after TP1
    # Format: {symbol: {'order1': id, 'order2': id, 'entry': price, 'direction': str, 'tp1_filled': bool}}
    trade_pairs = {}

    def on_order_update(message):
        """Handle order fill events - move SL to BE after TP1 + SL safety check"""
        try:
            if 'data' not in message:
                return

            for order_data in message['data']:
                order_id = order_data.get('orderId', '')
                symbol = order_data.get('symbol', '')
                status = order_data.get('orderStatus', '')

                # Only care about filled orders
                if status != 'Filled':
                    continue

                # === SAFETY: Check if position has SL after any fill ===
                # Small delay to let the order settle
                time.sleep(0.5)
                has_sl, current_sl, entry_price, side = executor.check_position_has_sl(symbol)
                if not has_sl and entry_price:
                    print(f"\n  [DANGER] {symbol} has NO SL! Setting emergency SL...", flush=True)
                    executor.set_emergency_sl(symbol, entry_price, side, max_loss_pct=2.0)

                # Check if this is part of a trade pair
                if symbol not in trade_pairs:
                    continue

                pair = trade_pairs[symbol]

                # Check if TP1 order was filled (order1)
                if order_id == pair.get('order1') and not pair.get('tp1_filled'):
                    pair['tp1_filled'] = True
                    entry_price = pair['entry']
                    direction = pair['direction']

                    # Move SL to lock in tiny profit (0.1%) - exactly like backtest
                    if direction == 'long':
                        new_sl = entry_price * 1.001  # 0.1% above entry
                    else:
                        new_sl = entry_price * 0.999  # 0.1% below entry

                    print(f"\n  [TP1 HIT] {symbol} - Moving SL to BE+ ({new_sl:.6f})", flush=True)

                    # Move SL to lock in tiny profit for remaining position
                    try:
                        # Use set_trading_stop to modify position SL
                        response = executor.client.set_trading_stop(
                            category="linear",
                            symbol=symbol,
                            stopLoss=str(round(new_sl, 6)),
                            slTriggerBy="LastPrice",
                            positionIdx=0
                        )
                        if response['retCode'] == 0:
                            print(f"  [BE+ SET] {symbol} SL → {new_sl:.6f} (0.1% locked)", flush=True)

                            # Send TP1 Telegram alert
                            tp1_price = pair.get('tp1_price', entry_price)
                            if direction == 'long':
                                partial_pnl = (tp1_price - entry_price) / entry_price * 100
                            else:
                                partial_pnl = (entry_price - tp1_price) / entry_price * 100
                            tg.send_tp1_hit(symbol, direction, entry_price, tp1_price, partial_pnl)
                        else:
                            print(f"  [WARN] SL modify failed: {response.get('retMsg', 'unknown')}", flush=True)
                    except Exception as e:
                        print(f"  [ERR] SL modify: {str(e)[:50]}", flush=True)

                # Check if TP2 order was filled (order2) - trade complete
                elif order_id == pair.get('order2'):
                    print(f"\n  [TP2 HIT] {symbol} - Trade complete!", flush=True)

                    # Log exit to Supabase
                    if pair.get('db_trade_id'):
                        try:
                            balance = executor.get_balance()
                            equity_now = balance.get('available', 0)

                            # Calculate PnL (approximate)
                            entry = pair['entry']
                            tp2 = pair.get('tp2_price', entry)
                            direction = pair['direction']
                            margin = pair.get('margin_used', 0)

                            if direction == 'long':
                                pnl_pct = (tp2 - entry) / entry * 100
                            else:
                                pnl_pct = (entry - tp2) / entry * 100

                            realized_pnl = margin * (pnl_pct / 100) * pair.get('qty', 0) / margin if margin else 0

                            trade_logger.log_exit(
                                trade_id=pair['db_trade_id'],
                                exit_price=tp2,
                                exit_time=datetime.utcnow(),
                                exit_reason='tp2',
                                realized_pnl=realized_pnl,
                                equity_at_close=equity_now,
                                tp1_hit=pair.get('tp1_filled', False),
                                tp2_hit=True,
                                entry_time=pair.get('entry_time'),
                                margin_used=margin,
                            )

                            # Send Telegram close alert
                            duration_mins = None
                            if pair.get('entry_time'):
                                duration_mins = int((datetime.utcnow() - pair['entry_time']).total_seconds() / 60)
                            tg.send_trade_closed(
                                symbol=symbol,
                                direction=direction,
                                entry_price=entry,
                                exit_price=tp2,
                                pnl_pct=pnl_pct,
                                exit_reason='tp2',
                                tp_hits='2/2' if pair.get('tp1_filled') else '1/2',
                                duration_mins=duration_mins,
                            )
                        except Exception as e:
                            print(f"  [DB ERR] {str(e)[:40]}", flush=True)

                    del trade_pairs[symbol]

        except Exception as e:
            print(f"  [WS ERR] {str(e)[:50]}", flush=True)

    # Start WebSocket in background thread
    if not PAPER_MODE:
        print("[INIT] Starting WebSocket...", flush=True)
        try:
            from live.websocket import BybitWebSocket
            import threading

            ws = BybitWebSocket(testnet=USE_TESTNET)
            ws.on_order(on_order_update)

            def ws_thread():
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(ws.start())
                # Keep running
                while ws.running:
                    time.sleep(1)

            ws_runner = threading.Thread(target=ws_thread, daemon=True)
            ws_runner.start()
            time.sleep(2)  # Give WS time to connect
            print("[INIT] WebSocket started", flush=True)
        except Exception as e:
            print(f"[WARN] WebSocket failed: {e} - continuing without real-time updates", flush=True)
            ws = None

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

    # Skip coins via ENV only - auto-skip handles the rest!
    # ENV: SKIP_COINS="COIN1USDT,COIN2USDT" (optional, for permanent skips)
    skip_env = os.getenv('SKIP_COINS', '')
    SKIP = set(c.strip().upper() for c in skip_env.split(',') if c.strip())

    if SKIP:
        print(f"[SKIP] From ENV: {', '.join(SKIP)}", flush=True)

    coins = [c for c in coins if c not in SKIP]

    print(f"Scanning {len(coins)} coins...", flush=True)
    print(f"Max positions: {MAX_LONGS} longs, {MAX_SHORTS} shorts", flush=True)
    print("=" * 60, flush=True)

    # Skip preload - let cache build naturally during scans
    # First few scans will be slower but won't hang
    print("\n[STARTUP] Starting scans (HTF cache builds automatically)...", flush=True)

    # Track pending orders: {order_id: {'symbol': str, 'placed_at': datetime, 'direction': str}}
    pending_orders = {}

    # === LOAD EXISTING ORDERS ON STARTUP (prevents duplicates after restart) ===
    if not PAPER_MODE:
        print("[STARTUP] Loading existing open orders...", flush=True)
        try:
            existing_orders = executor.get_open_orders()
            for order in existing_orders:
                oid = order.get('orderId', '')
                symbol = order.get('symbol', '')
                side = order.get('side', '')
                direction = 'long' if side == 'Buy' else 'short'
                pending_orders[oid] = {
                    'symbol': symbol,
                    'placed_at': datetime.utcnow(),  # Approximate
                    'direction': direction,
                    'ob_key': f"{symbol}_unknown"  # Can't recover exact OB
                }
            if existing_orders:
                symbols = set(o['symbol'] for o in existing_orders)
                print(f"  [LOADED] {len(existing_orders)} orders for: {', '.join(symbols)}", flush=True)
            else:
                print("  [LOADED] No existing orders", flush=True)
        except Exception as e:
            print(f"  [WARN] Could not load orders: {e}", flush=True)

    # === LOAD EXISTING POSITIONS (skip coins with open positions) ===
    existing_position_symbols = set()
    if not PAPER_MODE:
        print("[STARTUP] Loading existing positions...", flush=True)
        try:
            positions = executor.get_all_positions()
            for pos in positions:
                existing_position_symbols.add(pos.symbol)
            if positions:
                print(f"  [LOADED] Positions: {', '.join(existing_position_symbols)}", flush=True)
            else:
                print("  [LOADED] No existing positions", flush=True)
        except Exception as e:
            print(f"  [WARN] Could not load positions: {e}", flush=True)

    # Send Telegram bot started notification
    tg.send_bot_started(
        equity=available,
        active_positions=len(existing_position_symbols),
        pending_orders=len(pending_orders),
    )

    # === ROLLING SCAN: Scan coins continuously, one at a time ===
    SCAN_DELAY = 2.5  # seconds between each coin
    coin_index = 0
    last_status_time = time.time()
    STATUS_INTERVAL = 60  # Status update every 60 seconds
    signals_found = 0
    runtime_skip = set()  # Coins that timeout get added here automatically
    used_obs = set()  # Track OBs that have been traded (filled) - format: "SYMBOL_obtop_obbottom"

    print(f"\n[ROLLING SCAN] {len(coins)} coins × {SCAN_DELAY}s = {len(coins) * SCAN_DELAY / 60:.1f} min cycle", flush=True)

    while True:
        try:
            now = datetime.utcnow()
            symbol = coins[coin_index]

            # Skip coins that timed out previously
            if symbol in runtime_skip:
                coin_index = (coin_index + 1) % len(coins)
                continue

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
                            # Mark this OB as used so we don't trade it again!
                            if 'ob_key' in info:
                                used_obs.add(info['ob_key'])
                            # Mark trade as filled for exit detection
                            sym = info['symbol']
                            if sym in trade_pairs:
                                trade_pairs[sym]['filled'] = True
                            print(f"  [FILLED] {info['symbol']} {info['direction'].upper()}!", flush=True)

                # Show status
                positions = executor.get_all_positions()
                position_symbols = {p.symbol for p in positions}
                longs = sum(1 for p in positions if p.side == 'Buy')
                shorts = sum(1 for p in positions if p.side == 'Sell')

                # === SL/BE EXIT DETECTION: Check if tracked trades closed ===
                for sym in list(trade_pairs.keys()):
                    pair = trade_pairs[sym]

                    # Skip if orders for this symbol are still pending (not filled yet)
                    if not pair.get('filled', False):
                        # Check if any pending orders exist for this symbol
                        has_pending = any(o['symbol'] == sym for o in pending_orders.values())
                        if has_pending:
                            continue  # Still waiting for fill, not a real exit
                        # If no pending orders and no position, mark as filled (order filled and position exists somewhere)
                        if sym in position_symbols:
                            pair['filled'] = True
                            print(f"  [FILLED] {sym} position detected, tracking for exit", flush=True)
                        continue  # Either way, don't trigger exit yet

                    if sym not in position_symbols:
                        # Position closed (SL or BE+ hit)
                        entry = pair.get('entry', 0)
                        direction = pair.get('direction', 'long')
                        sl_price = pair.get('sl_price', entry)
                        tp1_filled = pair.get('tp1_filled', False)

                        # Determine exit type and approximate PnL
                        if tp1_filled:
                            # TP1 was hit, then BE+ was hit
                            exit_reason = 'be+'
                            exit_price = entry * 1.001 if direction == 'long' else entry * 0.999
                            pnl_pct = 0.1  # Locked 0.1% profit
                            tp_hits = '1/2'
                        else:
                            # Pure SL hit
                            exit_reason = 'sl'
                            exit_price = sl_price
                            if direction == 'long':
                                pnl_pct = (sl_price - entry) / entry * 100
                            else:
                                pnl_pct = (entry - sl_price) / entry * 100
                            tp_hits = '0/2'

                        print(f"  [EXIT] {sym} - {exit_reason.upper()} hit", flush=True)

                        # Log to Supabase
                        if pair.get('db_trade_id'):
                            try:
                                balance = executor.get_balance()
                                equity_now = balance.get('available', 0)
                                margin = pair.get('margin_used', 0)
                                realized_pnl = margin * (pnl_pct / 100) if margin else 0

                                trade_logger.log_exit(
                                    trade_id=pair['db_trade_id'],
                                    exit_price=exit_price,
                                    exit_time=datetime.utcnow(),
                                    exit_reason=exit_reason,
                                    realized_pnl=realized_pnl,
                                    equity_at_close=equity_now,
                                    tp1_hit=tp1_filled,
                                    tp2_hit=False,
                                    entry_time=pair.get('entry_time'),
                                    margin_used=margin,
                                )
                            except Exception as e:
                                print(f"  [DB ERR] {str(e)[:40]}", flush=True)

                        # Send Telegram alert
                        duration_mins = None
                        if pair.get('entry_time'):
                            duration_mins = int((datetime.utcnow() - pair['entry_time']).total_seconds() / 60)
                        tg.send_trade_closed(
                            symbol=sym,
                            direction=direction,
                            entry_price=entry,
                            exit_price=exit_price,
                            pnl_pct=pnl_pct,
                            exit_reason=exit_reason,
                            tp_hits=tp_hits,
                            duration_mins=duration_mins,
                        )

                        del trade_pairs[sym]

                print(f"  Positions: {longs}L/{shorts}S | Pending: {len(pending_orders)}", flush=True)
                print(f"  Cycle: {coin_index}/{len(coins)} | Signals: {signals_found}", flush=True)
                if runtime_skip:
                    print(f"  Auto-skipped: {', '.join(runtime_skip)}", flush=True)

            # === SCAN SINGLE COIN (with process-based timeout) ===
            print(f"  [{coin_index}] {symbol}...", end="", flush=True)

            signal = None
            try:
                # Use multiprocessing for REAL timeout (can kill stuck processes)
                import multiprocessing
                from multiprocessing import Process, Queue

                def scan_worker(sym, queue):
                    try:
                        # Re-create scanner in subprocess
                        from ob_scalper_live import OBScalperLive
                        worker_scanner = OBScalperLive()
                        result = worker_scanner.get_signal(sym)
                        queue.put(('ok', result))
                    except Exception as ex:
                        queue.put(('error', str(ex)[:50]))

                result_queue = Queue()
                proc = Process(target=scan_worker, args=(symbol, result_queue))
                proc.start()
                proc.join(timeout=20)  # 20 second hard timeout

                if proc.is_alive():
                    # Process hung - kill it!
                    proc.terminate()
                    proc.join(timeout=2)
                    if proc.is_alive():
                        proc.kill()  # Force kill
                    print(" TIMEOUT!", flush=True)
                    # Add to runtime skip list
                    runtime_skip.add(symbol)
                elif not result_queue.empty():
                    status, data = result_queue.get_nowait()
                    if status == 'ok':
                        signal = data
                        print(" OK", flush=True)
                    else:
                        print(f" skip", flush=True)
                else:
                    print(" skip", flush=True)

            except Exception as e:
                print(f" err", flush=True)

            if signal:
                # Check position limits
                positions = executor.get_all_positions()
                longs = sum(1 for p in positions if p.side == 'Buy')
                shorts = sum(1 for p in positions if p.side == 'Sell')
                pending_l = sum(1 for o in pending_orders.values() if o['direction'] == 'long')
                pending_s = sum(1 for o in pending_orders.values() if o['direction'] == 'short')

                skip = False
                ob_key = f"{signal.symbol}_{signal.ob_top}_{signal.ob_bottom}"

                if signal.direction == 'long' and (longs + pending_l) >= MAX_LONGS:
                    skip = True
                if signal.direction == 'short' and (shorts + pending_s) >= MAX_SHORTS:
                    skip = True
                if any(o['symbol'] == signal.symbol for o in pending_orders.values()):
                    skip = True
                if ob_key in used_obs:
                    print(f"  [SKIP] {symbol} - OB already traded", flush=True)
                    skip = True
                # Also check if we already have a position for this symbol!
                if any(p.symbol == signal.symbol for p in positions):
                    print(f"  [SKIP] {symbol} - already in position", flush=True)
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

                        # Cap position size - divide by max positions so all can fit
                        max_positions = MAX_LONGS + MAX_SHORTS  # e.g. 2+2=4
                        max_position_usd = (equity * 0.8 / max_positions) * signal.leverage
                        if qty_usd > max_position_usd:
                            qty_usd = max_position_usd
                            print(f"  [CAP] Position capped to ${qty_usd:.0f} (1/{max_positions} of margin)", flush=True)

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
                            # SAFETY: Set isolated margin first (loss limited to this position only)
                            executor.set_isolated_margin(signal.symbol)
                            executor.set_leverage(signal.symbol, signal.leverage)
                            ob_key = f"{signal.symbol}_{signal.ob_top}_{signal.ob_bottom}"
                            side = 'Buy' if signal.direction == 'long' else 'Sell'

                            # === PARTIAL TP: Split into 2 orders (like backtest) ===
                            if signal.use_partial_tp and signal.partial_tp_price:
                                qty1 = qty * signal.partial_size  # First 50%
                                qty2 = qty - qty1  # Remaining 50%

                                # Round quantities
                                if signal.entry_price > 100:
                                    qty1, qty2 = round(qty1, 2), round(qty2, 2)
                                elif signal.entry_price > 1:
                                    qty1, qty2 = round(qty1, 1), round(qty2, 1)
                                else:
                                    qty1, qty2 = round(qty1, 0), round(qty2, 0)

                                print(f"  [PARTIAL] Order1: {qty1} @ TP1={signal.partial_tp_price:.4f}", flush=True)
                                print(f"  [PARTIAL] Order2: {qty2} @ TP2={signal.tp_price:.4f}", flush=True)

                                try:
                                    # Order 1: Partial TP (closes first at 50% of target)
                                    resp1 = executor.client.place_order(
                                        category="linear",
                                        symbol=signal.symbol,
                                        side=side,
                                        orderType="Limit",
                                        price=str(round(signal.entry_price, 6)),
                                        qty=str(qty1),
                                        timeInForce="PostOnly",
                                        reduceOnly=False,
                                        takeProfit=str(round(signal.partial_tp_price, 6)),
                                        stopLoss=str(round(signal.sl_price, 6)),
                                        tpslMode="Full",
                                        slOrderType="Market"
                                    )

                                    # Order 2: Full TP (runs to full target)
                                    resp2 = executor.client.place_order(
                                        category="linear",
                                        symbol=signal.symbol,
                                        side=side,
                                        orderType="Limit",
                                        price=str(round(signal.entry_price, 6)),
                                        qty=str(qty2),
                                        timeInForce="PostOnly",
                                        reduceOnly=False,
                                        takeProfit=str(round(signal.tp_price, 6)),
                                        stopLoss=str(round(signal.sl_price, 6)),
                                        tpslMode="Full",
                                        slOrderType="Market"
                                    )

                                    # Track both orders
                                    if resp1['retCode'] == 0:
                                        oid1 = resp1['result']['orderId']
                                        pending_orders[oid1] = {
                                            'symbol': signal.symbol,
                                            'placed_at': now,
                                            'direction': signal.direction,
                                            'ob_key': ob_key
                                        }
                                        print(f"  [ORDER1] {oid1[:8]}... qty={qty1}", flush=True)
                                    else:
                                        print(f"  [ERR1] {resp1['retMsg']}", flush=True)

                                    if resp2['retCode'] == 0:
                                        oid2 = resp2['result']['orderId']
                                        pending_orders[oid2] = {
                                            'symbol': signal.symbol,
                                            'placed_at': now,
                                            'direction': signal.direction,
                                            'ob_key': ob_key
                                        }
                                        print(f"  [ORDER2] {oid2[:8]}... qty={qty2}", flush=True)

                                        # Register trade pair for WebSocket monitoring (SL→BE after TP1)
                                        if resp1['retCode'] == 0:
                                            # Log to Supabase
                                            trade_record = TradeRecord(
                                                symbol=signal.symbol,
                                                direction=signal.direction,
                                                entry_price=signal.entry_price,
                                                entry_time=now,
                                                qty=qty,
                                                leverage=signal.leverage,
                                                margin_used=qty * signal.entry_price / signal.leverage,
                                                equity_at_entry=equity,
                                                sl_price=signal.sl_price,
                                                tp1_price=signal.partial_tp_price,
                                                tp2_price=signal.tp_price,
                                                order_id_1=oid1,
                                                order_id_2=oid2,
                                                ob_strength=signal.ob_strength,
                                                ob_age_candles=signal.ob_age_candles,
                                                risk_pct=RISK_PER_TRADE_PCT,
                                                hour_utc=now.hour,
                                                day_of_week=now.weekday(),
                                            )
                                            db_trade_id = trade_logger.log_entry(trade_record)

                                            # Send Telegram alert
                                            tg.send_trade_opened(
                                                symbol=signal.symbol,
                                                direction=signal.direction,
                                                entry_price=signal.entry_price,
                                                sl_price=signal.sl_price,
                                                tp1_price=signal.partial_tp_price,
                                                tp2_price=signal.tp_price,
                                                leverage=signal.leverage,
                                                risk_pct=RISK_PER_TRADE_PCT,
                                                ob_strength=signal.ob_strength,
                                                ob_age=signal.ob_age_candles,
                                            )

                                            trade_pairs[signal.symbol] = {
                                                'order1': oid1,  # TP1 order
                                                'order2': oid2,  # TP2 order
                                                'entry': signal.entry_price,
                                                'direction': signal.direction,
                                                'tp1_filled': False,
                                                'tp1_price': signal.partial_tp_price,
                                                'tp2_price': signal.tp_price,
                                                'sl_price': signal.sl_price,
                                                'qty': qty,
                                                'margin_used': qty * signal.entry_price / signal.leverage,
                                                'entry_time': now,
                                                'db_trade_id': db_trade_id,  # For exit logging
                                                'equity_at_entry': equity,
                                            }
                                            print(f"  [TRACK] Registered for SL→BE monitoring", flush=True)
                                    else:
                                        print(f"  [ERR2] {resp2['retMsg']}", flush=True)

                                except Exception as e:
                                    print(f"  [ERR] {str(e)[:50]}", flush=True)

                            else:
                                # === SINGLE ORDER (no partial TP) ===
                                try:
                                    response = executor.client.place_order(
                                        category="linear",
                                        symbol=signal.symbol,
                                        side=side,
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
                                            'direction': signal.direction,
                                            'ob_key': ob_key
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
