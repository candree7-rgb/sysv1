"""
TradingView Webhook Receiver for Railway
=========================================
Receives alerts from TradingView Pine Script strategy
and executes trades via Bybit API.

Usage:
    python webhook_server.py

Environment Variables:
    - BYBIT_API_KEY: Bybit API key
    - BYBIT_API_SECRET: Bybit API secret
    - BYBIT_TESTNET: "true" for testnet, "false" for mainnet
    - WEBHOOK_SECRET: Optional secret for webhook validation
    - PORT: Server port (default: 8080)
"""

import os
import json
import asyncio
import hmac
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass

from flask import Flask, request, jsonify
from pybit.unified_trading import HTTP

from config.settings import config
from live.trade_logger import get_trade_logger, TradeRecord

app = Flask(__name__)

# Configuration
WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET', '')
ORDER_CANCEL_MINUTES = int(os.getenv('ORDER_CANCEL_MINUTES', '30'))
MAX_POSITION_SIZE_PCT = float(os.getenv('MAX_POSITION_SIZE_PCT', '5'))  # Max 5% per trade
DEFAULT_LEVERAGE = int(os.getenv('DEFAULT_LEVERAGE', '10'))

# Global Bybit client
client: Optional[HTTP] = None

# Track pending limit orders for cancellation
pending_orders: Dict[str, Dict[str, Any]] = {}  # order_id -> {symbol, created_at, ...}


def init_client():
    """Initialize Bybit client"""
    global client
    if client is None:
        client = HTTP(
            testnet=config.api.testnet,
            api_key=config.api.api_key,
            api_secret=config.api.api_secret
        )
        print(f"[WEBHOOK] Bybit client initialized (testnet={config.api.testnet})")
    return client


def get_account_equity() -> float:
    """Get current account equity"""
    try:
        result = client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        if result['retCode'] == 0:
            equity = float(result['result']['list'][0]['totalEquity'])
            return equity
    except Exception as e:
        print(f"[ERROR] Failed to get equity: {e}")
    return 0


def calculate_position_size(equity: float, risk_pct: float, entry: float, sl: float, leverage: int) -> float:
    """Calculate position size based on risk"""
    risk_amount = equity * (risk_pct / 100)
    sl_distance_pct = abs(entry - sl) / entry

    # Position value to achieve risk
    position_value = risk_amount / sl_distance_pct

    # Cap at max position size
    max_position_value = equity * (MAX_POSITION_SIZE_PCT / 100) * leverage
    position_value = min(position_value, max_position_value)

    # Convert to quantity
    qty = position_value / entry

    return qty


def get_symbol_info(symbol: str) -> Dict[str, Any]:
    """Get symbol trading rules"""
    try:
        result = client.get_instruments_info(category="linear", symbol=symbol)
        if result['retCode'] == 0 and result['result']['list']:
            info = result['result']['list'][0]
            return {
                'min_qty': float(info['lotSizeFilter']['minOrderQty']),
                'qty_step': float(info['lotSizeFilter']['qtyStep']),
                'tick_size': float(info['priceFilter']['tickSize']),
                'min_notional': float(info.get('lotSizeFilter', {}).get('minNotionalValue', 5))
            }
    except Exception as e:
        print(f"[ERROR] Failed to get symbol info: {e}")
    return {'min_qty': 0.001, 'qty_step': 0.001, 'tick_size': 0.01, 'min_notional': 5}


def round_qty(qty: float, step: float) -> float:
    """Round quantity to valid step"""
    return round(qty / step) * step


def round_price(price: float, tick: float) -> float:
    """Round price to valid tick"""
    return round(price / tick) * tick


def set_leverage(symbol: str, leverage: int):
    """Set leverage for symbol"""
    try:
        client.set_leverage(
            category="linear",
            symbol=symbol,
            buyLeverage=str(leverage),
            sellLeverage=str(leverage)
        )
    except Exception as e:
        # May fail if leverage already set
        pass


def place_limit_order_with_tp_sl(
    symbol: str,
    direction: str,
    qty: float,
    entry: float,
    sl: float,
    tp1: float,
    tp2: float,
    symbol_info: Dict[str, Any]
) -> Optional[str]:
    """
    Place limit order at OB edge with TP/SL.
    Returns order ID if successful.
    """
    try:
        # Round values
        qty = round_qty(qty, symbol_info['qty_step'])
        entry = round_price(entry, symbol_info['tick_size'])
        sl = round_price(sl, symbol_info['tick_size'])
        tp1 = round_price(tp1, symbol_info['tick_size'])
        tp2 = round_price(tp2, symbol_info['tick_size'])

        # Check minimum
        if qty < symbol_info['min_qty']:
            print(f"[WARN] Qty {qty} below minimum {symbol_info['min_qty']}")
            return None

        side = "Buy" if direction == "long" else "Sell"

        # Split into two orders: 50% with TP1, 50% with TP2
        qty1 = round_qty(qty * 0.5, symbol_info['qty_step'])
        qty2 = round_qty(qty - qty1, symbol_info['qty_step'])

        # Order 1: 50% with TP1
        result1 = client.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Limit",
            qty=str(qty1),
            price=str(entry),
            stopLoss=str(sl),
            takeProfit=str(tp1),
            timeInForce="GTC",
            reduceOnly=False
        )

        order_id_1 = None
        if result1['retCode'] == 0:
            order_id_1 = result1['result']['orderId']
            print(f"  [ORDER 1] {side} {qty1} @ {entry}, TP1={tp1}, SL={sl} -> {order_id_1[:8]}...")
        else:
            print(f"  [ERROR] Order 1 failed: {result1['retMsg']}")
            return None

        # Order 2: 50% with TP2
        result2 = client.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Limit",
            qty=str(qty2),
            price=str(entry),
            stopLoss=str(sl),
            takeProfit=str(tp2),
            timeInForce="GTC",
            reduceOnly=False
        )

        order_id_2 = None
        if result2['retCode'] == 0:
            order_id_2 = result2['result']['orderId']
            print(f"  [ORDER 2] {side} {qty2} @ {entry}, TP2={tp2}, SL={sl} -> {order_id_2[:8]}...")
        else:
            print(f"  [ERROR] Order 2 failed: {result2['retMsg']}")

        # Track for cancellation
        if order_id_1:
            pending_orders[order_id_1] = {
                'symbol': symbol,
                'created_at': datetime.utcnow(),
                'paired_order': order_id_2
            }
        if order_id_2:
            pending_orders[order_id_2] = {
                'symbol': symbol,
                'created_at': datetime.utcnow(),
                'paired_order': order_id_1
            }

        return order_id_1

    except Exception as e:
        print(f"[ERROR] Place order failed: {e}")
        return None


def cancel_stale_orders():
    """Cancel limit orders older than ORDER_CANCEL_MINUTES"""
    now = datetime.utcnow()
    to_remove = []

    for order_id, info in pending_orders.items():
        age = (now - info['created_at']).total_seconds() / 60
        if age > ORDER_CANCEL_MINUTES:
            try:
                result = client.cancel_order(
                    category="linear",
                    symbol=info['symbol'],
                    orderId=order_id
                )
                if result['retCode'] == 0:
                    print(f"  [CANCEL] Order {order_id[:8]}... cancelled (age: {age:.0f} min)")
                to_remove.append(order_id)
            except Exception as e:
                # Order may have been filled or already cancelled
                to_remove.append(order_id)

    for order_id in to_remove:
        pending_orders.pop(order_id, None)


def verify_webhook(payload: bytes, signature: str) -> bool:
    """Verify webhook signature (optional)"""
    if not WEBHOOK_SECRET:
        return True  # No secret configured, accept all

    expected = hmac.new(
        WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(signature, expected)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'time': datetime.utcnow().isoformat()})


@app.route('/webhook', methods=['POST'])
def webhook():
    """
    TradingView webhook endpoint.

    Expected JSON payload:
    {
        "action": "entry",
        "symbol": "BTCUSDT",
        "direction": "long" or "short",
        "entry": 50000.00,
        "sl": 49500.00,
        "tp1": 50500.00,
        "tp2": 51000.00,
        "risk_pct": 2.0,
        "timeframe": "5",
        "timestamp": "..."
    }
    """
    try:
        # Verify signature if configured
        if WEBHOOK_SECRET:
            signature = request.headers.get('X-Signature', '')
            if not verify_webhook(request.data, signature):
                return jsonify({'error': 'Invalid signature'}), 401

        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data'}), 400

        print(f"\n[WEBHOOK] Received: {json.dumps(data, indent=2)}")

        action = data.get('action', '')

        if action == 'entry':
            return handle_entry(data)
        elif action == 'close':
            return handle_close(data)
        else:
            return jsonify({'error': f'Unknown action: {action}'}), 400

    except Exception as e:
        print(f"[ERROR] Webhook error: {e}")
        return jsonify({'error': str(e)}), 500


def handle_entry(data: Dict[str, Any]):
    """Handle entry signal from TradingView"""
    symbol = data.get('symbol', '').upper()
    direction = data.get('direction', '').lower()
    entry = float(data.get('entry', 0))
    sl = float(data.get('sl', 0))
    tp1 = float(data.get('tp1', 0))
    tp2 = float(data.get('tp2', 0))
    risk_pct = float(data.get('risk_pct', 2.0))

    # Validate
    if not symbol or not direction or not entry or not sl:
        return jsonify({'error': 'Missing required fields'}), 400

    if direction not in ['long', 'short']:
        return jsonify({'error': 'Invalid direction'}), 400

    # Ensure symbol has USDT suffix for Bybit
    if not symbol.endswith('USDT'):
        symbol = symbol + 'USDT'

    print(f"\n{'='*50}")
    print(f"[SIGNAL] {direction.upper()} {symbol}")
    print(f"  Entry: {entry}")
    print(f"  SL: {sl} ({abs(entry-sl)/entry*100:.2f}%)")
    print(f"  TP1: {tp1}")
    print(f"  TP2: {tp2}")
    print(f"  Risk: {risk_pct}%")

    # Initialize client
    init_client()

    # Cancel stale orders first
    cancel_stale_orders()

    # Get account equity
    equity = get_account_equity()
    if equity <= 0:
        return jsonify({'error': 'Could not get account equity'}), 500

    print(f"  Equity: ${equity:.2f}")

    # Get symbol info
    symbol_info = get_symbol_info(symbol)

    # Set leverage
    set_leverage(symbol, DEFAULT_LEVERAGE)

    # Calculate position size
    qty = calculate_position_size(equity, risk_pct, entry, sl, DEFAULT_LEVERAGE)
    print(f"  Qty: {qty:.6f} (${qty * entry:.2f} notional)")

    # Place order
    order_id = place_limit_order_with_tp_sl(
        symbol=symbol,
        direction=direction,
        qty=qty,
        entry=entry,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        symbol_info=symbol_info
    )

    if order_id:
        # Log to Supabase
        logger = get_trade_logger()
        if logger.enabled:
            trade = TradeRecord(
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                entry_time=datetime.utcnow(),
                qty=qty,
                leverage=DEFAULT_LEVERAGE,
                margin_used=qty * entry / DEFAULT_LEVERAGE,
                equity_at_entry=equity,
                sl_price=sl,
                tp1_price=tp1,
                tp2_price=tp2,
                order_id_1=order_id,
                risk_pct=risk_pct,
                risk_amount=equity * risk_pct / 100
            )
            logger.log_entry(trade)

        return jsonify({
            'status': 'success',
            'order_id': order_id,
            'symbol': symbol,
            'direction': direction,
            'qty': qty,
            'entry': entry
        })
    else:
        return jsonify({'error': 'Failed to place order'}), 500


def handle_close(data: Dict[str, Any]):
    """Handle close signal (manual close via webhook)"""
    symbol = data.get('symbol', '').upper()

    if not symbol.endswith('USDT'):
        symbol = symbol + 'USDT'

    init_client()

    try:
        # Get current position
        result = client.get_positions(category="linear", symbol=symbol)
        if result['retCode'] == 0 and result['result']['list']:
            for pos in result['result']['list']:
                size = float(pos.get('size', 0))
                if size > 0:
                    side = "Sell" if pos['side'] == "Buy" else "Buy"

                    client.place_order(
                        category="linear",
                        symbol=symbol,
                        side=side,
                        orderType="Market",
                        qty=str(size),
                        reduceOnly=True
                    )
                    print(f"[CLOSE] Market closed {symbol} position")

        return jsonify({'status': 'success', 'symbol': symbol})

    except Exception as e:
        print(f"[ERROR] Close position failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/status', methods=['GET'])
def status():
    """Get current bot status"""
    init_client()

    try:
        equity = get_account_equity()

        # Get open positions
        positions = []
        result = client.get_positions(category="linear", settleCoin="USDT")
        if result['retCode'] == 0:
            for pos in result['result']['list']:
                size = float(pos.get('size', 0))
                if size > 0:
                    positions.append({
                        'symbol': pos['symbol'],
                        'side': pos['side'],
                        'size': size,
                        'entry': float(pos.get('avgPrice', 0)),
                        'pnl': float(pos.get('unrealisedPnl', 0))
                    })

        return jsonify({
            'status': 'ok',
            'equity': equity,
            'positions': positions,
            'pending_orders': len(pending_orders),
            'testnet': config.api.testnet
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/orders', methods=['GET'])
def orders():
    """Get pending orders"""
    return jsonify({
        'pending_orders': [
            {
                'order_id': oid,
                'symbol': info['symbol'],
                'age_minutes': (datetime.utcnow() - info['created_at']).total_seconds() / 60
            }
            for oid, info in pending_orders.items()
        ]
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    print(f"\n{'='*50}")
    print(f"SMC Ultra V2 - Webhook Server")
    print(f"{'='*50}")
    print(f"Port: {port}")
    print(f"Testnet: {config.api.testnet}")
    print(f"Order Cancel: {ORDER_CANCEL_MINUTES} minutes")
    print(f"Max Position: {MAX_POSITION_SIZE_PCT}%")
    print(f"Default Leverage: {DEFAULT_LEVERAGE}x")
    print(f"{'='*50}\n")

    # Initialize client on startup
    init_client()

    app.run(host='0.0.0.0', port=port, debug=False)
