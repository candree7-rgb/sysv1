"""
SMC Ultra V2 - Data Cache Manager
=================================
Intelligentes Caching für schnellen Datenzugriff
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import pickle

import pandas as pd
import numpy as np

from config.settings import config


class CacheManager:
    """
    Manages data caching with:
    - Automatic expiration
    - Compression
    - Metadata tracking
    - Memory + Disk caching
    """

    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir or config.data_dir) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

        # Memory cache for hot data
        self.memory_cache: Dict[str, Any] = {}
        self.memory_cache_size = 0
        self.max_memory_cache_mb = 500  # Max 500MB in memory

    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def _get_cache_key(self, *args) -> str:
        """Generate cache key from arguments"""
        key_str = "|".join(str(a) for a in args)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(
        self,
        key: str,
        max_age_hours: float = 24
    ) -> Optional[Any]:
        """
        Get cached data if exists and not expired.
        """
        # Check memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]

        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if not cache_file.exists():
            return None

        # Check expiration
        if key in self.metadata:
            cached_time = datetime.fromisoformat(self.metadata[key]['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=max_age_hours):
                self.delete(key)
                return None

        # Load from disk
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)

            # Add to memory cache if small enough
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            if self.memory_cache_size + size_mb < self.max_memory_cache_mb:
                self.memory_cache[key] = data
                self.memory_cache_size += size_mb

            return data
        except Exception as e:
            print(f"Error loading cache {key}: {e}")
            return None

    def set(
        self,
        key: str,
        data: Any,
        metadata: Dict = None
    ):
        """
        Cache data to disk and memory.
        """
        cache_file = self.cache_dir / f"{key}.pkl"

        # Save to disk
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving cache {key}: {e}")
            return

        # Update metadata
        self.metadata[key] = {
            'timestamp': datetime.now().isoformat(),
            'size_bytes': cache_file.stat().st_size,
            **(metadata or {})
        }
        self._save_metadata()

        # Add to memory cache if small enough
        size_mb = cache_file.stat().st_size / (1024 * 1024)
        if self.memory_cache_size + size_mb < self.max_memory_cache_mb:
            self.memory_cache[key] = data
            self.memory_cache_size += size_mb

    def delete(self, key: str):
        """Delete cached data"""
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            cache_file.unlink()

        if key in self.metadata:
            del self.metadata[key]
            self._save_metadata()

        if key in self.memory_cache:
            del self.memory_cache[key]

    def clear(self, older_than_hours: float = None):
        """
        Clear cache.
        If older_than_hours is set, only clear old entries.
        """
        if older_than_hours is None:
            # Clear all
            for f in self.cache_dir.glob("*.pkl"):
                f.unlink()
            self.metadata = {}
            self._save_metadata()
            self.memory_cache.clear()
            self.memory_cache_size = 0
        else:
            # Clear old entries
            cutoff = datetime.now() - timedelta(hours=older_than_hours)
            to_delete = []

            for key, meta in self.metadata.items():
                cached_time = datetime.fromisoformat(meta['timestamp'])
                if cached_time < cutoff:
                    to_delete.append(key)

            for key in to_delete:
                self.delete(key)

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_size = sum(
            m.get('size_bytes', 0)
            for m in self.metadata.values()
        )

        return {
            'total_entries': len(self.metadata),
            'total_size_mb': total_size / (1024 * 1024),
            'memory_cache_entries': len(self.memory_cache),
            'memory_cache_size_mb': self.memory_cache_size
        }


class DataFrameCache(CacheManager):
    """
    Specialized cache for DataFrames with Parquet support.
    """

    def get_df(
        self,
        symbol: str,
        interval: str,
        max_age_hours: float = 24
    ) -> Optional[pd.DataFrame]:
        """Get cached DataFrame"""
        key = f"{symbol}_{interval}"

        # Check memory first
        if key in self.memory_cache:
            return self.memory_cache[key]

        # Check parquet file
        parquet_file = self.cache_dir / f"{key}.parquet"
        if not parquet_file.exists():
            return None

        # Check expiration
        if key in self.metadata:
            cached_time = datetime.fromisoformat(self.metadata[key]['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=max_age_hours):
                return None

        try:
            df = pd.read_parquet(parquet_file)
            self.memory_cache[key] = df
            return df
        except Exception as e:
            print(f"Error loading DataFrame cache: {e}")
            return None

    def set_df(
        self,
        symbol: str,
        interval: str,
        df: pd.DataFrame
    ):
        """Cache DataFrame"""
        key = f"{symbol}_{interval}"
        parquet_file = self.cache_dir / f"{key}.parquet"

        try:
            df.to_parquet(parquet_file)

            self.metadata[key] = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'interval': interval,
                'rows': len(df),
                'size_bytes': parquet_file.stat().st_size
            }
            self._save_metadata()

            self.memory_cache[key] = df

        except Exception as e:
            print(f"Error caching DataFrame: {e}")


class IndicatorCache:
    """
    Caches calculated indicators to avoid recalculation.
    """

    def __init__(self):
        self.cache: Dict[str, Dict[str, pd.Series]] = {}

    def get(
        self,
        symbol: str,
        indicator: str,
        params: tuple = None
    ) -> Optional[pd.Series]:
        """Get cached indicator"""
        key = f"{symbol}|{indicator}|{params}"
        return self.cache.get(key)

    def set(
        self,
        symbol: str,
        indicator: str,
        data: pd.Series,
        params: tuple = None
    ):
        """Cache indicator"""
        key = f"{symbol}|{indicator}|{params}"
        self.cache[key] = data

    def invalidate(self, symbol: str = None):
        """Invalidate cache for symbol or all"""
        if symbol:
            to_delete = [k for k in self.cache if k.startswith(symbol)]
            for k in to_delete:
                del self.cache[k]
        else:
            self.cache.clear()


# Global cache instances
cache = CacheManager()
df_cache = DataFrameCache()
indicator_cache = IndicatorCache()
