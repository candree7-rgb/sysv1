"""
SMC Ultra V2 - WebSocket Handler
================================
Real-time data via WebSocket.

Features:
- Price streaming
- Order updates
- Position updates
- Reconnection handling
"""

import asyncio
import json
from datetime import datetime
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass

from pybit.unified_trading import WebSocket

from config.settings import config


@dataclass
class Kline:
    """Kline/Candlestick data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    interval: str


@dataclass
class Ticker:
    """Real-time ticker data"""
    symbol: str
    last_price: float
    bid: float
    ask: float
    volume_24h: float
    change_24h: float
    timestamp: datetime


class BybitWebSocket:
    """
    WebSocket handler for Bybit.

    Streams:
    - Kline data (candles)
    - Ticker data (prices)
    - Order updates
    - Position updates
    """

    def __init__(self, testnet: bool = None):
        self.testnet = testnet if testnet is not None else config.api.testnet

        self.ws_public = None
        self.ws_private = None

        self.callbacks: Dict[str, List[Callable]] = {
            'kline': [],
            'ticker': [],
            'order': [],
            'position': [],
            'wallet': []
        }

        self.subscribed_symbols: Dict[str, set] = {
            'kline': set(),
            'ticker': set()
        }

        self.running = False

    def on_kline(self, callback: Callable[[Kline], None]):
        """Register callback for kline updates"""
        self.callbacks['kline'].append(callback)

    def on_ticker(self, callback: Callable[[Ticker], None]):
        """Register callback for ticker updates"""
        self.callbacks['ticker'].append(callback)

    def on_order(self, callback: Callable[[Dict], None]):
        """Register callback for order updates"""
        self.callbacks['order'].append(callback)

    def on_position(self, callback: Callable[[Dict], None]):
        """Register callback for position updates"""
        self.callbacks['position'].append(callback)

    async def start(self):
        """Start WebSocket connections"""
        self.running = True

        # Public WebSocket
        self.ws_public = WebSocket(
            testnet=self.testnet,
            channel_type="linear"
        )

        # Private WebSocket (if API keys available)
        if config.api.api_key:
            self.ws_private = WebSocket(
                testnet=self.testnet,
                channel_type="private",
                api_key=config.api.api_key,
                api_secret=config.api.api_secret
            )

            # Subscribe to private channels
            self._subscribe_private()

    def subscribe_kline(self, symbol: str, interval: str = "1"):
        """Subscribe to kline stream"""
        if not self.ws_public:
            return

        topic = f"kline.{interval}.{symbol}"

        def handle_kline(message):
            self._process_kline(message)

        self.ws_public.kline_stream(
            interval=int(interval),
            symbol=symbol,
            callback=handle_kline
        )

        self.subscribed_symbols['kline'].add(symbol)

    def subscribe_ticker(self, symbol: str):
        """Subscribe to ticker stream"""
        if not self.ws_public:
            return

        def handle_ticker(message):
            self._process_ticker(message)

        self.ws_public.ticker_stream(
            symbol=symbol,
            callback=handle_ticker
        )

        self.subscribed_symbols['ticker'].add(symbol)

    def subscribe_multiple(self, symbols: List[str], interval: str = "1"):
        """Subscribe to multiple symbols"""
        for symbol in symbols:
            self.subscribe_kline(symbol, interval)
            self.subscribe_ticker(symbol)

    def _subscribe_private(self):
        """Subscribe to private channels"""
        if not self.ws_private:
            return

        # Order updates
        def handle_order(message):
            for callback in self.callbacks['order']:
                callback(message)

        self.ws_private.order_stream(callback=handle_order)

        # Position updates
        def handle_position(message):
            for callback in self.callbacks['position']:
                callback(message)

        self.ws_private.position_stream(callback=handle_position)

        # Wallet updates
        def handle_wallet(message):
            for callback in self.callbacks['wallet']:
                callback(message)

        self.ws_private.wallet_stream(callback=handle_wallet)

    def _process_kline(self, message):
        """Process kline message"""
        try:
            if 'data' not in message:
                return

            for data in message['data']:
                kline = Kline(
                    timestamp=datetime.fromtimestamp(int(data['start']) / 1000),
                    open=float(data['open']),
                    high=float(data['high']),
                    low=float(data['low']),
                    close=float(data['close']),
                    volume=float(data['volume']),
                    symbol=message.get('topic', '').split('.')[-1],
                    interval=message.get('topic', '').split('.')[1]
                )

                for callback in self.callbacks['kline']:
                    callback(kline)

        except Exception as e:
            print(f"Error processing kline: {e}")

    def _process_ticker(self, message):
        """Process ticker message"""
        try:
            if 'data' not in message:
                return

            data = message['data']

            ticker = Ticker(
                symbol=data.get('symbol', ''),
                last_price=float(data.get('lastPrice', 0)),
                bid=float(data.get('bid1Price', 0)),
                ask=float(data.get('ask1Price', 0)),
                volume_24h=float(data.get('turnover24h', 0)),
                change_24h=float(data.get('price24hPcnt', 0)) * 100,
                timestamp=datetime.utcnow()
            )

            for callback in self.callbacks['ticker']:
                callback(ticker)

        except Exception as e:
            print(f"Error processing ticker: {e}")

    def stop(self):
        """Stop WebSocket connections"""
        self.running = False

        if self.ws_public:
            self.ws_public.exit()
        if self.ws_private:
            self.ws_private.exit()


class DataAggregator:
    """
    Aggregates WebSocket data into OHLCV format.

    Converts streaming data into candlesticks for analysis.
    """

    def __init__(self):
        self.klines: Dict[str, Dict[str, List[Kline]]] = {}  # symbol -> interval -> [klines]
        self.tickers: Dict[str, Ticker] = {}  # symbol -> latest ticker
        self.max_klines = 500  # Keep last 500 klines per symbol/interval

    def add_kline(self, kline: Kline):
        """Add kline to aggregator"""
        if kline.symbol not in self.klines:
            self.klines[kline.symbol] = {}

        if kline.interval not in self.klines[kline.symbol]:
            self.klines[kline.symbol][kline.interval] = []

        klines = self.klines[kline.symbol][kline.interval]

        # Update last kline if same timestamp, else append
        if klines and klines[-1].timestamp == kline.timestamp:
            klines[-1] = kline
        else:
            klines.append(kline)

        # Trim to max size
        if len(klines) > self.max_klines:
            self.klines[kline.symbol][kline.interval] = klines[-self.max_klines:]

    def add_ticker(self, ticker: Ticker):
        """Add ticker to aggregator"""
        self.tickers[ticker.symbol] = ticker

    def get_ohlcv(self, symbol: str, interval: str = "1") -> Optional[list]:
        """Get OHLCV data for symbol"""
        if symbol not in self.klines:
            return None

        if interval not in self.klines[symbol]:
            return None

        return self.klines[symbol][interval]

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        if symbol in self.tickers:
            return self.tickers[symbol].last_price
        return None

    def get_spread(self, symbol: str) -> Optional[float]:
        """Get current spread for symbol"""
        if symbol in self.tickers:
            ticker = self.tickers[symbol]
            if ticker.bid > 0:
                return ((ticker.ask - ticker.bid) / ticker.bid) * 100
        return None
