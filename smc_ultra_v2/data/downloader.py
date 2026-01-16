"""
SMC Ultra V2 - Data Downloader
==============================
Lädt historische Daten von Bybit (KOSTENLOS, kein API Key nötig)
"""

import os
import ssl
import time
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from pathlib import Path

import pandas as pd
import numpy as np

# Disable SSL verification for testing environments
ssl._create_default_https_context = ssl._create_unverified_context
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Patch requests to disable SSL verification globally
import requests
from requests.adapters import HTTPAdapter
old_send = HTTPAdapter.send
def patched_send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
    return old_send(self, request, stream=stream, timeout=timeout, verify=False, cert=cert, proxies=proxies)
HTTPAdapter.send = patched_send

from pybit.unified_trading import HTTP

from config.settings import config


class BybitDataDownloader:
    """
    Lädt historische Klines von Bybit - KOSTENLOS, kein API Key nötig.

    Features:
    - Multi-Timeframe Download
    - Intelligent Caching
    - Incremental Updates
    - Parallel Downloads
    """

    INTERVALS = {
        '1': 1,
        '3': 3,
        '5': 5,
        '15': 15,
        '30': 30,
        '60': 60,
        '120': 120,
        '240': 240,
        '360': 360,
        '720': 720,
        'D': 1440,
        'W': 10080
    }

    def __init__(self, data_dir: str = None):
        # Disable SSL verification for testing
        import requests
        session = requests.Session()
        session.verify = False
        self.client = HTTP()  # Kein API Key für Marktdaten
        # Monkey patch the session to disable SSL
        if hasattr(self.client, '_session'):
            self.client._session.verify = False
        self.data_dir = Path(data_dir or config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting - conservative to avoid 10006 errors
        self.request_delay = 0.5  # 500ms between requests (was 50ms)
        self.last_request_time = 0
        self.rate_limit_backoff = 1  # Exponential backoff multiplier

    def _rate_limit(self):
        """Ensure we don't hit rate limits"""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request_time = time.time()

    def get_top_coins(self, limit: int = 100) -> List[str]:
        """
        Holt Top Coins nach 24h Volumen

        Returns:
            List of symbols sorted by volume
        """
        self._rate_limit()
        response = self.client.get_tickers(category="linear")

        if response['retCode'] != 0:
            raise Exception(f"API Error: {response['retMsg']}")

        tickers = [
            {
                'symbol': t['symbol'],
                'volume': float(t['turnover24h']),
                'price': float(t['lastPrice']),
                'change_pct': float(t.get('price24hPcnt', 0)) * 100
            }
            for t in response['result']['list']
            if t['symbol'].endswith('USDT')
        ]

        tickers.sort(key=lambda x: x['volume'], reverse=True)
        return [t['symbol'] for t in tickers[:limit]]

    def get_coin_info(self, symbol: str) -> Dict:
        """Get current info for a coin"""
        self._rate_limit()
        response = self.client.get_tickers(category="linear", symbol=symbol)

        if response['retCode'] != 0 or not response['result']['list']:
            return None

        t = response['result']['list'][0]
        return {
            'symbol': t['symbol'],
            'price': float(t['lastPrice']),
            'volume_24h': float(t['turnover24h']),
            'change_24h': float(t.get('price24hPcnt', 0)) * 100,
            'high_24h': float(t['highPrice24h']),
            'low_24h': float(t['lowPrice24h']),
            'funding_rate': float(t.get('fundingRate', 0)),
            'open_interest': float(t.get('openInterest', 0))
        }

    def download_coin(
        self,
        symbol: str,
        interval: str = "5",
        days: int = 180,
        end_time: datetime = None
    ) -> pd.DataFrame:
        """
        Lädt historische Daten für einen Coin.

        Bybit Limits:
        - 1000 Kerzen pro Request
        - Keine Rate Limits für Marktdaten

        6 Monate bei 5min = ~52.000 Kerzen = ~52 Requests = ~1 Minute
        """
        end_time = end_time or datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        all_data = []
        current_end = end_time
        request_count = 0

        backoff = 1  # Start with 1 second backoff
        max_retries = 5

        while current_end > start_time:
            self._rate_limit()

            for retry in range(max_retries):
                try:
                    response = self.client.get_kline(
                        category="linear",
                        symbol=symbol,
                        interval=interval,
                        end=int(current_end.timestamp() * 1000),
                        limit=1000
                    )

                    # Check for rate limit error (10006)
                    if response['retCode'] == 10006:
                        wait_time = backoff * (2 ** retry)  # Exponential backoff
                        print(f"  Rate limit for {symbol}, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue  # Retry

                    if response['retCode'] != 0:
                        print(f"  Error for {symbol}: {response['retMsg']}")
                        return pd.DataFrame()  # Give up on this coin

                    klines = response['result']['list']
                    if not klines:
                        break

                    all_data.extend(klines)
                    request_count += 1

                    # Älteste Kerze als neues End
                    oldest_ts = int(klines[-1][0])
                    current_end = datetime.utcfromtimestamp(oldest_ts / 1000)
                    break  # Success, exit retry loop

                except Exception as e:
                    if '10006' in str(e) or 'rate limit' in str(e).lower():
                        wait_time = backoff * (2 ** retry)
                        print(f"  Rate limit exception for {symbol}, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue  # Retry
                    print(f"  Exception for {symbol}: {e}")
                    return pd.DataFrame()  # Give up on this coin
            else:
                # All retries exhausted
                print(f"  Max retries reached for {symbol}, skipping...")
                return pd.DataFrame()

        if not all_data:
            return pd.DataFrame()

        # Zu DataFrame konvertieren
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        # Datentypen konvertieren
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = df[col].astype(float)

        # Sortieren und Duplikate entfernen
        df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

        return df

    def download_mtf(
        self,
        symbol: str,
        intervals: List[str] = None,
        days: int = 180
    ) -> Dict[str, pd.DataFrame]:
        """
        Lädt Multi-Timeframe Daten für einen Coin.

        Returns:
            Dict of {interval: DataFrame}
        """
        intervals = intervals or [
            config.timeframes.htf,
            config.timeframes.mtf,
            config.timeframes.ltf
        ]

        result = {}
        for interval in intervals:
            print(f"  Downloading {symbol} {interval}m...")
            df = self.download_coin(symbol, interval, days)
            if len(df) > 0:
                result[interval] = df

        return result

    def download_all(
        self,
        symbols: List[str],
        interval: str = "5",
        days: int = 180,
        save: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Lädt Daten für alle Coins.

        Args:
            symbols: List of symbols to download
            interval: Timeframe interval
            days: Number of days to download
            save: Whether to save to parquet files

        Returns:
            Dict of {symbol: DataFrame}
        """
        result = {}
        total = len(symbols)

        for i, symbol in enumerate(symbols):
            print(f"[{i+1}/{total}] Downloading {symbol}...")

            df = self.download_coin(symbol, interval, days)

            if len(df) > 0:
                result[symbol] = df
                print(f"  ✓ {len(df)} candles")

                if save:
                    filepath = self.get_cache_path(symbol, interval, days)
                    df.to_parquet(filepath)
            else:
                print(f"  ✗ No data")

        print(f"\nDownloaded {len(result)}/{total} coins")
        return result

    def get_cache_path(self, symbol: str, interval: str, days: int) -> Path:
        """Get cache file path for a symbol"""
        return self.data_dir / f"{symbol}_{interval}m_{days}d.parquet"

    def load_cached(
        self,
        symbol: str,
        interval: str,
        days: int
    ) -> Optional[pd.DataFrame]:
        """
        Lädt gecachte Daten falls vorhanden und aktuell.

        Returns:
            DataFrame or None if not cached
        """
        filepath = self.get_cache_path(symbol, interval, days)

        if not filepath.exists():
            return None

        # Check if cache is recent (< 1 day old)
        mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
        if datetime.now() - mtime > timedelta(days=1):
            return None  # Cache too old

        try:
            return pd.read_parquet(filepath)
        except Exception as e:
            print(f"Error loading cache for {symbol}: {e}")
            return None

    def load_or_download(
        self,
        symbol: str,
        interval: str,
        days: int
    ) -> Optional[pd.DataFrame]:
        """
        Lädt aus Cache oder downloaded falls nötig.
        """
        # Try cache first
        df = self.load_cached(symbol, interval, days)
        if df is not None:
            return df

        # Download
        df = self.download_coin(symbol, interval, days)

        if len(df) > 0:
            # Save to cache
            filepath = self.get_cache_path(symbol, interval, days)
            df.to_parquet(filepath)

        return df if len(df) > 0 else None

    def update_cache(
        self,
        symbol: str,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """
        Aktualisiert gecachte Daten mit neuesten Kerzen.

        Lädt nur die neuen Kerzen seit dem letzten Update.
        """
        # Find existing cache
        cache_files = list(self.data_dir.glob(f"{symbol}_{interval}m_*.parquet"))

        if not cache_files:
            return None

        # Load most recent cache
        cache_file = sorted(cache_files)[-1]
        df = pd.read_parquet(cache_file)

        if len(df) == 0:
            return None

        # Get last timestamp
        last_ts = df['timestamp'].max()

        # Download new data
        new_df = self.download_coin(
            symbol=symbol,
            interval=interval,
            days=2,  # Last 2 days should be enough
            end_time=datetime.utcnow()
        )

        if len(new_df) == 0:
            return df

        # Filter to only new candles
        new_df = new_df[new_df['timestamp'] > last_ts]

        if len(new_df) == 0:
            return df

        # Combine and save
        combined = pd.concat([df, new_df]).drop_duplicates('timestamp').sort_values('timestamp')
        combined.to_parquet(cache_file)

        print(f"Updated {symbol}: +{len(new_df)} new candles")
        return combined


class AsyncBybitDownloader(BybitDataDownloader):
    """
    Async version für parallele Downloads
    """

    async def download_coin_async(
        self,
        symbol: str,
        interval: str = "5",
        days: int = 180
    ) -> pd.DataFrame:
        """Async wrapper for download_coin"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.download_coin(symbol, interval, days)
        )

    async def download_all_async(
        self,
        symbols: List[str],
        interval: str = "5",
        days: int = 180,
        max_concurrent: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Parallele Downloads mit Semaphore für Rate Limiting.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def download_with_semaphore(symbol: str):
            async with semaphore:
                return symbol, await self.download_coin_async(symbol, interval, days)

        tasks = [download_with_semaphore(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        data = {}
        for result in results:
            if isinstance(result, Exception):
                print(f"Error: {result}")
                continue
            symbol, df = result
            if len(df) > 0:
                data[symbol] = df

        return data


# Convenience function
def download_data(
    symbols: List[str] = None,
    interval: str = "5",
    days: int = 180,
    top_n: int = 100
) -> Dict[str, pd.DataFrame]:
    """
    Main function to download data.

    Usage:
        # Download top 100 coins
        data = download_data(top_n=100)

        # Download specific coins
        data = download_data(symbols=['BTCUSDT', 'ETHUSDT'])
    """
    dl = BybitDataDownloader()

    if symbols is None:
        print(f"Fetching top {top_n} coins by volume...")
        symbols = dl.get_top_coins(limit=top_n)
        print(f"Found {len(symbols)} coins")

    print(f"\nDownloading {len(symbols)} coins ({days} days, {interval}m)...")
    print("This may take a while...\n")

    return dl.download_all(symbols, interval, days)


if __name__ == "__main__":
    # Download top 100 coins, 5min, 6 months
    download_data(top_n=100, interval="5", days=180)
