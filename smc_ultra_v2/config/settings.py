"""
SMC Ultra V2 - Configuration Settings
=====================================
Zentrale Konfiguration für das gesamte System
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class TradingMode(Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class RiskLevel(Enum):
    CONSERVATIVE = "conservative"  # 1% risk, lower leverage
    MODERATE = "moderate"          # 2% risk, medium leverage
    AGGRESSIVE = "aggressive"      # 3% risk, higher leverage


@dataclass
class APIConfig:
    """Bybit API Configuration"""
    api_key: str = field(default_factory=lambda: os.getenv("BYBIT_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("BYBIT_API_SECRET", ""))
    testnet: bool = field(default_factory=lambda: os.getenv("BYBIT_TESTNET", "true").lower() == "true")

    @property
    def base_url(self) -> str:
        if self.testnet:
            return "https://api-testnet.bybit.com"
        return "https://api.bybit.com"


@dataclass
class TimeframeConfig:
    """Multi-Timeframe Configuration"""
    htf: str = "60"      # 1H for bias
    mtf: str = "15"      # 15m for setup
    ltf: str = "1"       # 1m for precision entry

    # Alternative aggressive config
    # htf: str = "15"
    # mtf: str = "5"
    # ltf: str = "1"


@dataclass
class RiskConfig:
    """Risk Management Configuration"""
    risk_level: RiskLevel = RiskLevel.MODERATE

    # Per-trade risk (from env or default)
    # $100 Account mit 2% = $2 Risk pro Trade
    max_risk_per_trade_pct: float = float(os.getenv("RISK_PER_TRADE", "2.0"))
    max_daily_risk_pct: float = 6.0
    max_weekly_risk_pct: float = 15.0

    # Leverage (konservativer für kleine Accounts)
    max_leverage: int = int(os.getenv("MAX_LEVERAGE", "30"))
    min_leverage: int = 3

    # Drawdown protection
    max_drawdown_pct: float = 20.0
    drawdown_reduce_leverage_at: float = 10.0
    drawdown_stop_trading_at: float = 15.0

    # Position limits
    max_concurrent_trades: int = 3
    max_trades_per_coin: int = 1
    max_correlation_exposure: float = 0.7  # Max correlation between positions

    def get_risk_multiplier(self) -> float:
        """Returns risk multiplier based on risk level"""
        multipliers = {
            RiskLevel.CONSERVATIVE: 0.5,
            RiskLevel.MODERATE: 1.0,
            RiskLevel.AGGRESSIVE: 1.5
        }
        return multipliers[self.risk_level]


@dataclass
class EntryConfig:
    """Entry Configuration"""
    # Confidence thresholds
    min_confidence: int = 85
    ideal_confidence: int = 92

    # Entry type
    use_limit_orders: bool = True
    limit_order_timeout_minutes: int = 15

    # Slippage
    max_slippage_pct: float = 0.05

    # Filters
    min_volume_ratio: float = 0.5      # vs 20-period average
    max_spread_pct: float = 0.1        # Max spread to enter
    min_atr_pct: float = 0.1           # Min volatility
    max_atr_pct: float = 3.0           # Max volatility


@dataclass
class ExitConfig:
    """Exit/Target Configuration"""
    # RR Target
    target_rr: float = 1.0             # 1:1 Risk Reward

    # Dynamic TP/SL
    min_tp_pct: float = 0.15           # Minimum TP
    max_tp_pct: float = 0.60           # Maximum TP

    # Break-even - DISABLED for pure 1:1 RR
    break_even_at_pct: float = 100.0   # Never trigger (let trades run to TP/SL)

    # Trailing stop - DISABLED for pure 1:1 RR
    trailing_start_pct: float = 100.0  # Never trigger (need full TP wins)
    trailing_offset_pct: float = 30.0  # Irrelevant when disabled

    # Time-based exit - increased for more time
    max_trade_duration_minutes: int = 240  # 4 hours to reach TP

    # Momentum exit - DISABLED for pure 1:1 RR
    enable_momentum_exit: bool = False
    momentum_exit_min_profit_pct: float = 100.0


@dataclass
class RegimeConfig:
    """Market Regime Configuration"""
    # ADX thresholds
    adx_strong_trend: float = 30.0
    adx_weak_trend: float = 20.0

    # Choppiness
    choppiness_threshold: float = 61.8

    # Volatility
    high_volatility_mult: float = 1.8

    # Volume
    low_volume_threshold: float = 0.5

    # Regime-specific adjustments
    regime_settings: Dict = field(default_factory=lambda: {
        'strong_trend_up': {
            'trade': True,
            'direction': 'long_only',
            'min_confidence': 90,
            'tp_multiplier': 1.5,
            'leverage_mult': 1.0
        },
        'strong_trend_down': {
            'trade': True,
            'direction': 'short_only',
            'min_confidence': 90,
            'tp_multiplier': 1.5,
            'leverage_mult': 1.0
        },
        'weak_trend_up': {
            'trade': False,
            'reason': 'weak_trend_disabled'
        },
        'weak_trend_down': {
            'trade': False,
            'reason': 'weak_trend_disabled'
        },
        'ranging': {
            'trade': False,
            'reason': 'ranging_disabled'
        },
        'high_volatility': {
            'trade': False,
            'reason': 'high_volatility_disabled'
        },
        'choppy': {
            'trade': False,
            'reason': 'market_too_choppy'
        },
        'low_volume': {
            'trade': False,
            'reason': 'insufficient_liquidity'
        }
    })


@dataclass
class MLConfig:
    """Machine Learning Configuration"""
    # Model
    model_type: str = "xgboost"
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05

    # Training
    train_test_split: float = 0.2
    min_training_samples: int = 500

    # Recency weighting
    enable_recency_weighting: bool = True
    recency_weights: Dict = field(default_factory=lambda: {
        7: 1.0,      # Last 7 days: weight 1.0
        30: 0.7,     # Last 30 days: weight 0.7
        90: 0.4,     # Last 90 days: weight 0.4
        365: 0.2     # Older: weight 0.2
    })

    # Retraining
    retrain_frequency_days: int = 7
    min_new_trades_for_retrain: int = 50

    # Feature selection
    feature_importance_threshold: float = 0.01


@dataclass
class SessionConfig:
    """Trading Session Configuration"""
    # Session times (UTC)
    sessions: Dict = field(default_factory=lambda: {
        'asia': {'start': 0, 'end': 8, 'weight': 0.7},
        'london': {'start': 8, 'end': 16, 'weight': 1.0},
        'new_york': {'start': 13, 'end': 21, 'weight': 1.0},
        'overlap': {'start': 13, 'end': 16, 'weight': 1.2}  # Best time
    })

    # Day of week (0=Monday)
    day_weights: Dict = field(default_factory=lambda: {
        0: 0.9,   # Monday - often ranging
        1: 1.0,   # Tuesday
        2: 1.0,   # Wednesday
        3: 1.0,   # Thursday
        4: 0.8,   # Friday - early close
        5: 0.5,   # Saturday - low volume
        6: 0.5    # Sunday - low volume
    })


@dataclass
class BacktestConfig:
    """Backtest Configuration"""
    initial_capital: float = 10000.0
    fee_pct: float = 0.075  # Bybit taker fee
    slippage_pct: float = 0.05

    # Data
    data_days: int = 180  # 6 months
    warmup_bars: int = 200  # For indicators

    # Reporting
    save_trades: bool = True
    save_equity_curve: bool = True
    generate_html_report: bool = True


@dataclass
class Config:
    """Main Configuration - Combines all configs"""
    api: APIConfig = field(default_factory=APIConfig)
    timeframes: TimeframeConfig = field(default_factory=TimeframeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    entry: EntryConfig = field(default_factory=EntryConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    sessions: SessionConfig = field(default_factory=SessionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # General
    mode: TradingMode = TradingMode.BACKTEST
    log_level: str = "INFO"
    data_dir: str = "./data/historical"
    model_dir: str = "./models"

    @classmethod
    def load_from_env(cls) -> 'Config':
        """Load configuration from environment variables"""
        config = cls()

        # Override from env
        if os.getenv("TRADING_MODE"):
            config.mode = TradingMode(os.getenv("TRADING_MODE"))

        if os.getenv("RISK_LEVEL"):
            config.risk.risk_level = RiskLevel(os.getenv("RISK_LEVEL"))

        if os.getenv("MIN_CONFIDENCE"):
            config.entry.min_confidence = int(os.getenv("MIN_CONFIDENCE"))

        if os.getenv("MAX_LEVERAGE"):
            config.risk.max_leverage = int(os.getenv("MAX_LEVERAGE"))

        return config


# Global config instance
config = Config()
