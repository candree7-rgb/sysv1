"""
Candle Streamer - WebSocket-based real-time candle data
========================================================
Subscribes to klines for all coins via WebSocket.
Provides instant access to candle data without API calls.

Usage:
    streamer = CandleStreamer(coins=['BTCUSDT', 'ETHUSDT', ...])
    streamer.start()

    # Get candles instantly (no API call!)
    df = streamer.get_candles('BTCUSDT', interval='5', limit=100)
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict

import pandas as pd
import numpy as np

from pybit.unified_trading import WebSocket
from config.settings import config


class CandleStreamer:
    """
    Real-time candle streaming via WebSocket.

    - Subscribes to 5-min klines for OB detection
    - Subscribes to 1H klines for MTF alignment
    - Maintains a rolling buffer of candles in memory
    - No API calls needed during scan cycle!
    """

    def __init__(self, coins: List[str], testnet: bool = None, max_candles: int = 200):
        self.coins = coins
        self.testnet = testnet if testnet is not None else config.api.testnet
        self.max_candles = max_candles

        # Candle storage: {symbol: {interval: [candles]}}
        # Each candle is a dict with timestamp, open, high, low, close, volume
        self._candles: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
        self._last_update: Dict[str, datetime] = {}

        # WebSocket connection
        self.ws = None
        self.running = False
        self._thread = None
        self._lock = threading.Lock()

        # Stats
        self.updates_received = 0
        self.errors = 0

        # Intervals to subscribe (5m for OB, 60m for MTF)
        self.intervals = ['5', '60']  # 5-min and 1-hour

    def start(self):
        """Start WebSocket in background thread"""
        if self.running:
            return

        self.running = True
        self._thread = threading.Thread(target=self._run_ws, daemon=True)
        self._thread.start()

        # Wait for initial connection
        time.sleep(3)
        print(f"[WS CANDLES] Started streaming {len(self.coins)} coins", flush=True)

    def _run_ws(self):
        """Run WebSocket connection (in thread)"""
        try:
            self.ws = WebSocket(
                testnet=self.testnet,
                channel_type="linear"
            )

            # Subscribe to all coins for each interval
            subscribed = 0
            for interval in self.intervals:
                batch_size = 25  # Conservative batch size
                for i in range(0, len(self.coins), batch_size):
                    batch = self.coins[i:i + batch_size]
                    for symbol in batch:
                        try:
                            self.ws.kline_stream(
                                interval=int(interval),
                                symbol=symbol,
                                callback=self._on_kline
                            )
                            subscribed += 1
                        except Exception as e:
                            self.errors += 1

                    # Small delay between batches to avoid rate limits
                    time.sleep(0.3)

            print(f"[WS CANDLES] Subscribed to {subscribed} streams ({len(self.coins)} coins × {len(self.intervals)} intervals)", flush=True)

            # Keep running
            while self.running:
                time.sleep(1)

        except Exception as e:
            print(f"[WS CANDLES] Error: {e}", flush=True)
            self.errors += 1

    def _on_kline(self, message):
        """Handle incoming kline data"""
        try:
            if 'data' not in message:
                return

            topic = message.get('topic', '')
            # Topic format: kline.5.BTCUSDT
            parts = topic.split('.')
            if len(parts) < 3:
                return
            interval = parts[1]
            symbol = parts[-1]

            for data in message['data']:
                candle = {
                    'timestamp': datetime.fromtimestamp(int(data['start']) / 1000),
                    'open': float(data['open']),
                    'high': float(data['high']),
                    'low': float(data['low']),
                    'close': float(data['close']),
                    'volume': float(data['volume']),
                    'confirm': data.get('confirm', False)  # True if candle is closed
                }

                with self._lock:
                    candles = self._candles[symbol][interval]

                    # Update existing candle or append new one
                    if candles and candles[-1]['timestamp'] == candle['timestamp']:
                        # Update in-progress candle
                        candles[-1] = candle
                    else:
                        # New candle
                        candles.append(candle)

                        # Trim to max size
                        if len(candles) > self.max_candles:
                            self._candles[symbol][interval] = candles[-self.max_candles:]

                    self._last_update[symbol] = datetime.utcnow()

                self.updates_received += 1

        except Exception as e:
            self.errors += 1

    def get_candles(self, symbol: str, interval: str = '5', limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Get candle data for a symbol.

        Returns DataFrame with columns: timestamp, open, high, low, close, volume
        Returns None if no data available.
        """
        with self._lock:
            candles = self._candles.get(symbol, {}).get(interval, [])

            if not candles or len(candles) < 10:  # Need minimum data
                return None

            # Get last N candles
            candles = candles[-limit:] if len(candles) > limit else candles.copy()

        # Convert to DataFrame
        df = pd.DataFrame(candles)
        if df.empty:
            return None

        df = df.set_index('timestamp')

        return df

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest close price for a symbol"""
        with self._lock:
            # Try 5m candles first
            candles = self._candles.get(symbol, {}).get('5', [])
            if candles:
                return candles[-1]['close']
        return None

    def has_data(self, symbol: str, interval: str = '5', min_candles: int = 50) -> bool:
        """Check if we have enough data for a symbol"""
        with self._lock:
            return len(self._candles.get(symbol, {}).get(interval, [])) >= min_candles

    def get_stats(self) -> dict:
        """Get streaming stats"""
        with self._lock:
            symbols_with_5m = sum(1 for s in self.coins if len(self._candles.get(s, {}).get('5', [])) > 0)
            symbols_with_1h = sum(1 for s in self.coins if len(self._candles.get(s, {}).get('60', [])) > 0)
            total_candles = sum(
                len(c)
                for sym_data in self._candles.values()
                for c in sym_data.values()
            )

        return {
            'symbols_5m': symbols_with_5m,
            'symbols_1h': symbols_with_1h,
            'total_coins': len(self.coins),
            'total_candles': total_candles,
            'updates_received': self.updates_received,
            'errors': self.errors
        }

    def stop(self):
        """Stop WebSocket connection"""
        self.running = False
        if self.ws:
            try:
                self.ws.exit()
            except:
                pass


# Global instance
_streamer: Optional[CandleStreamer] = None


def get_candle_streamer() -> Optional[CandleStreamer]:
    """Get global candle streamer instance"""
    return _streamer


def start_candle_streamer(coins: List[str], testnet: bool = None) -> CandleStreamer:
    """Start global candle streamer"""
    global _streamer
    if _streamer is None:
        _streamer = CandleStreamer(coins, testnet)
        _streamer.start()
    return _streamer

