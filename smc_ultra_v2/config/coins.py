"""
SMC Ultra V2 - Coin Configuration
==================================
Dynamically fetches coins from Bybit Futures API
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

# Cache for dynamic coin list
_DYNAMIC_COINS_CACHE = None


def get_top_n_coins(n: int) -> List[str]:
    """
    Get top N coins by 24h volume from Bybit Futures (linear perpetuals).
    Uses dynamic API fetch to ensure valid symbols.
    """
    return fetch_bybit_futures_symbols(limit=n)


def fetch_bybit_futures_symbols(limit: int = 200) -> List[str]:
    """
    Fetch actual linear perpetual symbols from Bybit API.
    Returns top coins sorted by 24h volume.
    """
    global _DYNAMIC_COINS_CACHE

    if _DYNAMIC_COINS_CACHE is not None:
        return _DYNAMIC_COINS_CACHE[:limit]

    try:
        from pybit.unified_trading import HTTP
        client = HTTP()

        response = client.get_tickers(category="linear")

        if response['retCode'] != 0:
            print(f"Warning: Could not fetch Bybit symbols: {response['retMsg']}")
            return _get_fallback_coins()[:limit]

        # Filter USDT perpetuals and sort by volume
        tickers = [
            {
                'symbol': t['symbol'],
                'volume': float(t['turnover24h'])
            }
            for t in response['result']['list']
            if t['symbol'].endswith('USDT') and not t['symbol'].endswith('USDTUSDT')
        ]

        tickers.sort(key=lambda x: x['volume'], reverse=True)
        _DYNAMIC_COINS_CACHE = [t['symbol'] for t in tickers]

        print(f"Fetched {len(_DYNAMIC_COINS_CACHE)} valid futures symbols from Bybit")
        return _DYNAMIC_COINS_CACHE[:limit]

    except Exception as e:
        print(f"Warning: Could not fetch Bybit symbols: {e}")
        return _get_fallback_coins()[:limit]


def _get_fallback_coins() -> List[str]:
    """Fallback list with correct Bybit Futures symbols"""
    return [
        # Tier 1 - Major coins (always correct)
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
        "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
        "LTCUSDT", "BCHUSDT", "ATOMUSDT", "UNIUSDT", "ETCUSDT",
        "XLMUSDT", "FILUSDT", "TRXUSDT", "NEARUSDT", "APTUSDT",
        # Tier 2
        "ARBUSDT", "OPUSDT", "INJUSDT", "SUIUSDT", "SEIUSDT",
        "TIAUSDT", "IMXUSDT", "RUNEUSDT", "AAVEUSDT", "MKRUSDT",
        "LDOUSDT", "GRTUSDT", "SNXUSDT", "CRVUSDT",
        "WLDUSDT", "STXUSDT", "ALGOUSDT", "VETUSDT",
        "XTZUSDT", "THETAUSDT", "AXSUSDT", "SANDUSDT", "MANAUSDT",
        "GALAUSDT", "DYDXUSDT", "GMXUSDT", "PERPUSDT",
        "ICPUSDT", "HBARUSDT", "QNTUSDT", "EGLDUSDT",
        # Tier 3
        "CFXUSDT", "KAVAUSDT", "ZECUSDT", "NEOUSDT",
        "IOTAUSDT", "ZILUSDT", "ONEUSDT", "CHZUSDT",
        "COMPUSDT", "YFIUSDT", "SUSHIUSDT", "1INCHUSDT", "BATUSDT",
        "ANKRUSDT", "ENJUSDT", "STORJUSDT", "SKLUSDT",
        "MINAUSDT", "BLURUSDT", "WOOUSDT",
        "PENDLEUSDT", "HOOKUSDT",
        "MAGICUSDT", "LQTYUSDT", "SSVUSDT",
        "HIGHUSDT", "ACHUSDT",
        "COTIUSDT", "CELRUSDT", "DENTUSDT",
        "DASHUSDT", "RVNUSDT", "ZENUSDT", "ONTUSDT",
        # Meme coins with correct Bybit Futures symbols (1000x variants)
        "1000PEPEUSDT", "1000SHIBUSDT", "1000FLOKIUSDT", "1000BONKUSDT",
        "WIFUSDT", "MEMEUSDT", "1000LUNCUSDT",
    ]


def get_all_coins() -> List[str]:
    """Get all available futures coins"""
    return fetch_bybit_futures_symbols(limit=500)


# Coins that often lead the market
MARKET_LEADERS = ["BTCUSDT", "ETHUSDT"]

# Coins with high correlation to BTC
HIGH_BTC_CORRELATION = [
    "ETHUSDT", "LTCUSDT", "BCHUSDT", "ETCUSDT", "LINKUSDT"
]
