"""
SMC Ultra V2 - Coin Configuration
==================================
Top 200 Coins und deren spezifische Einstellungen
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CoinConfig:
    """Configuration for a specific coin"""
    symbol: str
    tier: int = 1                    # 1=Top, 2=Mid, 3=Low priority
    max_leverage: int = 50           # Coin-specific max leverage
    min_volume_24h: float = 1000000  # Min 24h volume in USD
    typical_spread_pct: float = 0.05
    volatility_mult: float = 1.0     # Adjustment for volatile coins
    enabled: bool = True


# Tier 1: Most liquid, lowest spread
TIER1_COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT", "LINKUSDT",
    "LTCUSDT", "BCHUSDT", "ATOMUSDT", "UNIUSDT", "ETCUSDT",
    "XLMUSDT", "FILUSDT", "TRXUSDT", "NEARUSDT", "APTUSDT"
]

# Tier 2: Good liquidity
TIER2_COINS = [
    "ARBUSDT", "OPUSDT", "INJUSDT", "SUIUSDT", "SEIUSDT",
    "TIAUSDT", "IMXUSDT", "RUNEUSDT", "AAVEUSDT", "MKRUSDT",
    "LDOUSDT", "GRTUSDT", "SNXUSDT", "CRVUSDT", "RNDRUSDT",
    "FETUSDT", "AGIXUSDT", "OCEANUSDT", "WLDUSDT", "STXUSDT",
    "ALGOUSDT", "VETUSDT", "EOSUSDT", "XTZUSDT", "THETAUSDT",
    "AXSUSDT", "SANDUSDT", "MANAUSDT", "GALAUSDT", "APEUSDT",
    "DYDXUSDT", "GMXUSDT", "PERPUSDT", "LRCUSDT", "ENSUSDT",
    "ICPUSDT", "HBARUSDT", "QNTUSDT", "EGLDUSDT", "FLOWUSDT"
]

# Tier 3: Lower liquidity, wider spreads
TIER3_COINS = [
    "CFXUSDT", "CKBUSDT", "KAVAUSDT", "ZECUSDT", "NEOUSDT",
    "IOTAUSDT", "ZILUSDT", "ONEUSDT", "HOTUSDT", "CHZUSDT",
    "COMPUSDT", "YFIUSDT", "SUSHIUSDT", "1INCHUSDT", "BATUSDT",
    "ANKRUSDT", "ENJUSDT", "STORJUSDT", "SKLUSDT", "CELOUSDT",
    "MINAUSDT", "BLURUSDT", "RLCUSDT", "API3USDT", "WOOUSDT",
    "TOKENUSDT", "ARKMUSDT", "PENDLEUSDT", "CYBERUSDT", "HOOKUSDT",
    "RDNTUSDT", "MAGICUSDT", "LQTYUSDT", "SSVUSDT", "TRUUSDT",
    "HIGHUSDT", "ACHUSDT", "XVSUSDT", "UMAUSDT", "KNCUSDT",
    "COTIUSDT", "REEFUSDT", "CELRUSDT", "DENTUSDT", "SCUSDT",
    "WAVESUSDT", "DASHUSDT", "RVNUSDT", "ZENUSDT", "ONTUSDT"
]

# Additional coins to reach 200
TIER4_COINS = [
    "PEPEUSDT", "SHIBUSDT", "FLOKIUSDT", "BONKUSDT", "WIFUSDT",
    "MEMEUSDT", "PEOPLEUSDT", "LUNCUSDT", "JASMYUSDT", "GLMRUSDT",
    "MOVRUSDT", "GMTUSDT", "TWTUSDT", "MASKUSDT", "BANDUSDT",
    "RSRUSDT", "LITUSDT", "UNFIUSDT", "AUDIOUSDT", "ROSEUSDT",
    "OGNUSDT", "FXSUSDT", "MDTUSDT", "REQUSDT", "BNTUSDT",
    "OXTUSDT", "LPTUSDT", "POLYUSDT", "CTSIUSDT", "PHBUSDT",
    "MTLUSDT", "NKNUSDT", "DGBUSDT", "STMXUSDT", "BELUSDT",
    "XEMUSDT", "IDEXUSDT", "TFUELUSDT", "FORTHUSDT", "ERNUSDT",
    "KLAYUSDT", "IOTXUSDT", "HFTUSDT", "AMBUSDT", "GASUSDT",
    "POWRUSDT", "LEVERUSDT", "EDUUSDT", "OAXUSDT", "XNOUSDT",
    "JOEUSDT", "BICOUSDT", "FLMUSDT", "FRONTUSDT", "COMBOUSDT",
    "MAVUSDT", "MDXUSDT", "ORBSUSDT", "STEEMUSDT", "VIBUSDT",
    "SYNUSDT", "QUICKUSDT", "T USDT", "IDUSDT", "RADUSDT",
    "NMRUSDT", "MLNUSDT", "PNTUSDT", "DREPUSDT", "SUNUSDT",
    "BURGERUSDT", "SLPUSDT", "TLMUSDT", "DARUSDT", "ALPACAUSDT",
    "LINAUSDT", "ATAUSDT", "GTCUSDT", "TORNUSDT", "ALPINEUSDT",
    "AUCTIONUSDT", "BIFIUSDT", "MBLUSDT", "MOBUSDT", "NEXOUSDT"
]


def get_all_coins() -> List[str]:
    """Get all coins sorted by tier"""
    return TIER1_COINS + TIER2_COINS + TIER3_COINS + TIER4_COINS


def get_coins_by_tier(tier: int) -> List[str]:
    """Get coins for a specific tier"""
    tiers = {
        1: TIER1_COINS,
        2: TIER2_COINS,
        3: TIER3_COINS,
        4: TIER4_COINS
    }
    return tiers.get(tier, [])


def get_top_n_coins(n: int) -> List[str]:
    """Get top N coins by priority"""
    all_coins = get_all_coins()
    return all_coins[:n]


# Coins with special handling
VOLATILE_COINS = [
    "PEPEUSDT", "SHIBUSDT", "FLOKIUSDT", "BONKUSDT", "WIFUSDT",
    "MEMEUSDT", "LUNCUSDT"
]

# Coins that often lead the market
MARKET_LEADERS = ["BTCUSDT", "ETHUSDT"]

# Coins with high correlation to BTC
HIGH_BTC_CORRELATION = [
    "ETHUSDT", "LTCUSDT", "BCHUSDT", "ETCUSDT", "LINKUSDT"
]

# Coins with lower BTC correlation (good for diversification)
LOW_BTC_CORRELATION = [
    "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "XRPUSDT"
]


@dataclass
class CoinDatabase:
    """Database of coin configurations"""

    configs: Dict[str, CoinConfig] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize coin configs"""
        # Tier 1
        for symbol in TIER1_COINS:
            self.configs[symbol] = CoinConfig(
                symbol=symbol,
                tier=1,
                max_leverage=75 if symbol in ["BTCUSDT", "ETHUSDT"] else 50,
                min_volume_24h=10000000,
                typical_spread_pct=0.03
            )

        # Tier 2
        for symbol in TIER2_COINS:
            self.configs[symbol] = CoinConfig(
                symbol=symbol,
                tier=2,
                max_leverage=50,
                min_volume_24h=5000000,
                typical_spread_pct=0.05
            )

        # Tier 3
        for symbol in TIER3_COINS:
            self.configs[symbol] = CoinConfig(
                symbol=symbol,
                tier=3,
                max_leverage=30,
                min_volume_24h=1000000,
                typical_spread_pct=0.08
            )

        # Tier 4
        for symbol in TIER4_COINS:
            self.configs[symbol] = CoinConfig(
                symbol=symbol,
                tier=4,
                max_leverage=20,
                min_volume_24h=500000,
                typical_spread_pct=0.1
            )

        # Special handling for volatile coins
        for symbol in VOLATILE_COINS:
            if symbol in self.configs:
                self.configs[symbol].volatility_mult = 1.5
                self.configs[symbol].max_leverage = min(
                    self.configs[symbol].max_leverage, 20
                )

    def get(self, symbol: str) -> Optional[CoinConfig]:
        return self.configs.get(symbol)

    def get_enabled(self) -> List[str]:
        return [s for s, c in self.configs.items() if c.enabled]

    def get_by_tier(self, max_tier: int) -> List[str]:
        return [s for s, c in self.configs.items() if c.tier <= max_tier]


# Global instance
coin_db = CoinDatabase()
