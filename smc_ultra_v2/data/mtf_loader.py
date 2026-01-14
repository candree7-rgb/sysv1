"""
SMC Ultra V2 - Multi-Timeframe Data Loader
==========================================
Lädt und synchronisiert Daten über mehrere Timeframes
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np

from config.settings import config
from .downloader import BybitDataDownloader


class MTFDataLoader:
    """
    Multi-Timeframe Data Loader

    Lädt und synchronisiert Daten für:
    - HTF (1H): Bias/Trend
    - MTF (15m): Setup/Zone
    - LTF (1m): Precision Entry
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir or config.data_dir)
        self.downloader = BybitDataDownloader(str(self.data_dir))

        # Cache für geladene Daten
        self.cache: Dict[str, Dict[str, pd.DataFrame]] = {}

        # Timeframe config
        self.htf = config.timeframes.htf
        self.mtf = config.timeframes.mtf
        self.ltf = config.timeframes.ltf

    def load_symbol(
        self,
        symbol: str,
        days: int = 30,
        force_download: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Lädt alle Timeframes für einen Coin.

        Returns:
            Dict of {timeframe: DataFrame}
        """
        if symbol in self.cache and not force_download:
            return self.cache[symbol]

        data = {}

        for tf in [self.htf, self.mtf, self.ltf]:
            # Adjust days based on timeframe
            tf_days = self._adjust_days_for_tf(days, tf)

            if force_download:
                df = self.downloader.download_coin(symbol, tf, tf_days)
                if len(df) > 0:
                    filepath = self.downloader.get_cache_path(symbol, tf, tf_days)
                    df.to_parquet(filepath)
            else:
                df = self.downloader.load_or_download(symbol, tf, tf_days)

            if df is not None and len(df) > 0:
                # Add indicators
                df = self._add_base_indicators(df)
                data[tf] = df

        if data:
            self.cache[symbol] = data

        return data

    def _adjust_days_for_tf(self, base_days: int, tf: str) -> int:
        """
        Adjust days to download based on timeframe.
        LTF needs fewer days, HTF needs more for context.
        """
        tf_minutes = int(tf) if tf.isdigit() else 1440  # Default to daily

        if tf_minutes <= 5:
            return min(base_days, 30)  # 1m-5m: max 30 days
        elif tf_minutes <= 15:
            return min(base_days, 60)  # 15m: max 60 days
        elif tf_minutes <= 60:
            return base_days  # 1H: full days
        else:
            return base_days * 2  # 4H+: double for more context

    def _add_base_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add base indicators to dataframe"""
        df = df.copy()

        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_pct'] = (df['atr'] / df['close']) * 100

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # EMAs
        df['ema9'] = df['close'].ewm(span=9).mean()
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['ema200'] = df['close'].ewm(span=200).mean()

        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

        # Volume
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Candle patterns
        df['body'] = df['close'] - df['open']
        df['body_pct'] = abs(df['body']) / df['close'] * 100
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['is_bullish'] = df['close'] > df['open']

        return df

    def load_multiple(
        self,
        symbols: List[str],
        days: int = 30,
        max_workers: int = 5
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Lädt Daten für mehrere Coins parallel.
        """
        result = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.load_symbol, symbol, days): symbol
                for symbol in symbols
            }

            for future in futures:
                symbol = futures[future]
                try:
                    data = future.result()
                    if data:
                        result[symbol] = data
                except Exception as e:
                    print(f"Error loading {symbol}: {e}")

        return result

    def get_aligned_data(
        self,
        symbol: str,
        timestamp: pd.Timestamp = None
    ) -> Optional[Dict[str, pd.Series]]:
        """
        Holt synchronisierte Daten für alle TFs zu einem Zeitpunkt.

        Returns:
            Dict with latest candle for each TF at or before timestamp
        """
        if symbol not in self.cache:
            return None

        timestamp = timestamp or pd.Timestamp.now()
        result = {}

        for tf, df in self.cache[symbol].items():
            # Get the candle at or before timestamp
            mask = df['timestamp'] <= timestamp
            if mask.any():
                result[tf] = df[mask].iloc[-1]

        return result if result else None

    def get_htf_candles(
        self,
        symbol: str,
        n: int = 50
    ) -> Optional[pd.DataFrame]:
        """Get last N HTF candles"""
        if symbol not in self.cache or self.htf not in self.cache[symbol]:
            return None
        return self.cache[symbol][self.htf].tail(n)

    def get_mtf_candles(
        self,
        symbol: str,
        n: int = 100
    ) -> Optional[pd.DataFrame]:
        """Get last N MTF candles"""
        if symbol not in self.cache or self.mtf not in self.cache[symbol]:
            return None
        return self.cache[symbol][self.mtf].tail(n)

    def get_ltf_candles(
        self,
        symbol: str,
        n: int = 200
    ) -> Optional[pd.DataFrame]:
        """Get last N LTF candles"""
        if symbol not in self.cache or self.ltf not in self.cache[symbol]:
            return None
        return self.cache[symbol][self.ltf].tail(n)

    def clear_cache(self, symbol: str = None):
        """Clear cache for symbol or all"""
        if symbol:
            self.cache.pop(symbol, None)
        else:
            self.cache.clear()


class MTFDataSynchronizer:
    """
    Synchronisiert Daten zwischen Timeframes für Backtesting.

    Wichtig: Im Backtest dürfen wir keine "Zukunftsdaten" verwenden.
    """

    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data
        self._build_index()

    def _build_index(self):
        """Build timestamp index for fast lookup"""
        self.indices = {}
        for tf, df in self.data.items():
            self.indices[tf] = df.set_index('timestamp')

    def get_at_time(
        self,
        timestamp: pd.Timestamp,
        lookback_htf: int = 50,
        lookback_mtf: int = 100,
        lookback_ltf: int = 200
    ) -> Dict[str, pd.DataFrame]:
        """
        Holt historische Daten bis zu einem Zeitpunkt.

        WICHTIG: Keine Zukunftsdaten!
        """
        result = {}

        for tf, idx in self.indices.items():
            # Only data up to timestamp
            mask = idx.index <= timestamp
            df = idx[mask]

            # Apply lookback
            if tf == '60' or tf == '240':  # HTF
                df = df.tail(lookback_htf)
            elif tf == '15' or tf == '30':  # MTF
                df = df.tail(lookback_mtf)
            else:  # LTF
                df = df.tail(lookback_ltf)

            if len(df) > 0:
                result[tf] = df.reset_index()

        return result


# Convenience function
def load_mtf_data(
    symbols: List[str],
    days: int = 30
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load multi-timeframe data for symbols.

    Usage:
        data = load_mtf_data(['BTCUSDT', 'ETHUSDT'], days=30)
        btc_htf = data['BTCUSDT']['60']  # 1H data
    """
    loader = MTFDataLoader()
    return loader.load_multiple(symbols, days)
