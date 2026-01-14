"""
SMC Ultra V2 - Dynamic Trade Manager
====================================
Verwaltet offene Trades mit dynamischem Management.

Features:
- Break-Even bei 30% TP erreicht
- Trailing Stop bei 50% TP erreicht
- Zeit-Exit nach X Minuten
- Momentum-Exit bei Warnsignalen
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

from config.settings import config


class ExitReason(Enum):
    """Reasons for closing a trade"""
    TAKE_PROFIT = "tp"
    STOP_LOSS = "sl"
    TRAILING_STOP = "trailing"
    BREAK_EVEN = "break_even"
    TIME_EXIT = "time"
    MOMENTUM_EXIT = "momentum"
    MANUAL = "manual"
    REGIME_CHANGE = "regime"


@dataclass
class Trade:
    """Represents an active trade"""
    id: str
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    entry_time: datetime

    take_profit: float
    stop_loss: float
    current_sl: float  # Dynamic SL

    leverage: int
    confidence: int
    factors: List[str] = field(default_factory=list)

    # State tracking
    max_profit_price: float = None
    min_profit_price: float = None
    trailing_active: bool = False
    break_even_hit: bool = False

    # Result (when closed)
    exit_price: float = None
    exit_time: datetime = None
    exit_reason: ExitReason = None
    pnl_pct: float = None

    def __post_init__(self):
        if self.max_profit_price is None:
            self.max_profit_price = self.entry_price
        if self.min_profit_price is None:
            self.min_profit_price = self.entry_price

    @property
    def is_open(self) -> bool:
        return self.exit_price is None

    @property
    def unrealized_pnl(self) -> float:
        """Current unrealized PnL in %"""
        if self.direction == 'long':
            return ((self.max_profit_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.min_profit_price) / self.entry_price) * 100


class DynamicTradeManager:
    """
    Dynamic trade management with:
    - Break-Even SL
    - Trailing Stop
    - Time-based exits
    - Momentum-based exits
    """

    def __init__(self):
        self.exit_config = config.exit

        self.be_threshold = self.exit_config.break_even_at_pct / 100
        self.trail_start = self.exit_config.trailing_start_pct / 100
        self.trail_offset = self.exit_config.trailing_offset_pct / 100
        self.max_duration = timedelta(minutes=self.exit_config.max_trade_duration_minutes)

    def update(
        self,
        trade: Trade,
        current_price: float,
        current_time: datetime,
        rsi: float = None,
        volume_ratio: float = None,
        regime_changed: bool = False
    ) -> Optional[ExitReason]:
        """
        Update trade and check for exit conditions.

        Returns:
            ExitReason if trade should be closed, None otherwise
        """
        if not trade.is_open:
            return None

        # Update max/min profit tracking
        if trade.direction == 'long':
            trade.max_profit_price = max(trade.max_profit_price, current_price)
            trade.min_profit_price = min(trade.min_profit_price, current_price)
        else:
            trade.max_profit_price = min(trade.max_profit_price, current_price)
            trade.min_profit_price = max(trade.min_profit_price, current_price)

        # Check exit conditions in order of priority

        # 1. Take Profit hit
        if self._check_tp(trade, current_price):
            return ExitReason.TAKE_PROFIT

        # 2. Stop Loss hit
        if self._check_sl(trade, current_price):
            if trade.trailing_active:
                return ExitReason.TRAILING_STOP
            elif trade.break_even_hit:
                return ExitReason.BREAK_EVEN
            else:
                return ExitReason.STOP_LOSS

        # 3. Update SL (Break-Even and Trailing)
        self._update_sl(trade, current_price)

        # 4. Regime change exit
        if regime_changed and self._calc_progress(trade, current_price) > 0:
            return ExitReason.REGIME_CHANGE

        # 5. Time-based exit
        if self._check_time_exit(trade, current_time, current_price):
            return ExitReason.TIME_EXIT

        # 6. Momentum exit
        if self.exit_config.enable_momentum_exit:
            if self._check_momentum_exit(trade, current_price, rsi, volume_ratio):
                return ExitReason.MOMENTUM_EXIT

        return None

    def _calc_progress(self, trade: Trade, price: float) -> float:
        """
        Calculate progress toward TP as percentage (0-1).
        Negative if in loss.
        """
        if trade.direction == 'long':
            total = trade.take_profit - trade.entry_price
            current = price - trade.entry_price
        else:
            total = trade.entry_price - trade.take_profit
            current = trade.entry_price - price

        return current / total if total != 0 else 0

    def _check_tp(self, trade: Trade, price: float) -> bool:
        """Check if TP was hit"""
        if trade.direction == 'long':
            return price >= trade.take_profit
        return price <= trade.take_profit

    def _check_sl(self, trade: Trade, price: float) -> bool:
        """Check if SL was hit"""
        if trade.direction == 'long':
            return price <= trade.current_sl
        return price >= trade.current_sl

    def _update_sl(self, trade: Trade, price: float):
        """Update SL for break-even and trailing"""
        progress = self._calc_progress(trade, price)

        # Break-Even trigger
        if progress >= self.be_threshold and not trade.break_even_hit:
            trade.break_even_hit = True

            # Move SL to entry + small buffer
            buffer = abs(trade.entry_price - trade.stop_loss) * 0.1

            if trade.direction == 'long':
                new_sl = trade.entry_price + buffer
                trade.current_sl = max(trade.current_sl, new_sl)
            else:
                new_sl = trade.entry_price - buffer
                trade.current_sl = min(trade.current_sl, new_sl)

        # Trailing trigger
        if progress >= self.trail_start:
            trade.trailing_active = True

        # Trailing update
        if trade.trailing_active:
            if trade.direction == 'long':
                profit = trade.max_profit_price - trade.entry_price
                trail_sl = trade.max_profit_price - (profit * self.trail_offset)
                trade.current_sl = max(trade.current_sl, trail_sl)
            else:
                profit = trade.entry_price - trade.max_profit_price
                trail_sl = trade.max_profit_price + (profit * self.trail_offset)
                trade.current_sl = min(trade.current_sl, trail_sl)

    def _check_time_exit(
        self,
        trade: Trade,
        current_time: datetime,
        current_price: float
    ) -> bool:
        """Check time-based exit condition"""
        elapsed = current_time - trade.entry_time

        if elapsed >= self.max_duration:
            progress = self._calc_progress(trade, current_price)
            # Only exit if not too deep in loss
            if progress > -0.3:  # Max 30% of SL distance in loss
                return True

        return False

    def _check_momentum_exit(
        self,
        trade: Trade,
        price: float,
        rsi: float = None,
        volume_ratio: float = None
    ) -> bool:
        """
        Check momentum-based exit.

        Exit when momentum fades and we're in profit.
        """
        progress = self._calc_progress(trade, price)

        # Only consider if significantly in profit
        if progress < self.exit_config.momentum_exit_min_profit_pct / 100:
            return False

        warnings = 0

        # RSI warning
        if rsi is not None:
            if trade.direction == 'long' and rsi > 80:
                warnings += 1
            elif trade.direction == 'short' and rsi < 20:
                warnings += 1

        # Retracement warning
        if trade.direction == 'long':
            max_profit = trade.max_profit_price - trade.entry_price
            current_profit = price - trade.entry_price
            if max_profit > 0:
                retracement = 1 - (current_profit / max_profit)
                if retracement > 0.4:  # Gave back 40%
                    warnings += 1
        else:
            max_profit = trade.entry_price - trade.max_profit_price
            current_profit = trade.entry_price - price
            if max_profit > 0:
                retracement = 1 - (current_profit / max_profit)
                if retracement > 0.4:
                    warnings += 1

        # Volume warning
        if volume_ratio is not None and volume_ratio < 0.5:
            warnings += 1

        return warnings >= 2

    def close_trade(
        self,
        trade: Trade,
        exit_price: float,
        exit_time: datetime,
        reason: ExitReason
    ) -> Trade:
        """
        Close a trade and calculate PnL.
        """
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.exit_reason = reason

        # Calculate PnL
        if trade.direction == 'long':
            pnl_pct = ((exit_price - trade.entry_price) / trade.entry_price) * 100
        else:
            pnl_pct = ((trade.entry_price - exit_price) / trade.entry_price) * 100

        # Apply leverage
        trade.pnl_pct = pnl_pct * trade.leverage

        return trade


class PositionManager:
    """
    Manages multiple positions with risk controls.
    """

    def __init__(self):
        self.risk_config = config.risk
        self.trades: Dict[str, Trade] = {}  # symbol -> Trade
        self.closed_trades: List[Trade] = []

        # Risk tracking
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.current_drawdown = 0.0
        self.peak_equity = 0.0

    def can_open_trade(self, symbol: str) -> Tuple[bool, str]:
        """Check if we can open a new trade"""
        # Max concurrent trades
        if len(self.trades) >= self.risk_config.max_concurrent_trades:
            return False, "max_trades_reached"

        # Already have trade on this symbol
        if symbol in self.trades:
            return False, "already_trading_symbol"

        # Daily risk limit
        if abs(self.daily_pnl) >= self.risk_config.max_daily_risk_pct:
            return False, "daily_risk_limit"

        # Drawdown limit
        if self.current_drawdown >= self.risk_config.drawdown_stop_trading_at:
            return False, "drawdown_limit"

        return True, "ok"

    def open_trade(self, trade: Trade) -> bool:
        """Add trade to active positions"""
        can_open, reason = self.can_open_trade(trade.symbol)
        if not can_open:
            print(f"Cannot open trade: {reason}")
            return False

        self.trades[trade.symbol] = trade
        return True

    def close_trade(
        self,
        symbol: str,
        exit_price: float,
        exit_time: datetime,
        reason: ExitReason
    ) -> Optional[Trade]:
        """Close a trade"""
        if symbol not in self.trades:
            return None

        trade = self.trades[symbol]

        # Close using trade manager
        manager = DynamicTradeManager()
        closed_trade = manager.close_trade(trade, exit_price, exit_time, reason)

        # Update risk tracking
        self.daily_pnl += closed_trade.pnl_pct
        self.weekly_pnl += closed_trade.pnl_pct

        # Move to closed
        del self.trades[symbol]
        self.closed_trades.append(closed_trade)

        return closed_trade

    def update_all(
        self,
        prices: Dict[str, float],
        current_time: datetime,
        indicators: Dict[str, Dict] = None
    ) -> List[Trade]:
        """
        Update all active trades.

        Returns:
            List of trades that were closed
        """
        closed = []
        manager = DynamicTradeManager()

        for symbol, trade in list(self.trades.items()):
            if symbol not in prices:
                continue

            price = prices[symbol]
            rsi = indicators.get(symbol, {}).get('rsi') if indicators else None
            vol_ratio = indicators.get(symbol, {}).get('volume_ratio') if indicators else None

            exit_reason = manager.update(
                trade, price, current_time,
                rsi=rsi, volume_ratio=vol_ratio
            )

            if exit_reason:
                closed_trade = self.close_trade(symbol, price, current_time, exit_reason)
                if closed_trade:
                    closed.append(closed_trade)

        return closed

    def get_stats(self) -> Dict:
        """Get position manager statistics"""
        if not self.closed_trades:
            return {'trades': 0}

        winners = [t for t in self.closed_trades if t.pnl_pct > 0]
        losers = [t for t in self.closed_trades if t.pnl_pct <= 0]

        return {
            'trades': len(self.closed_trades),
            'open_trades': len(self.trades),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(self.closed_trades) * 100,
            'avg_win': np.mean([t.pnl_pct for t in winners]) if winners else 0,
            'avg_loss': np.mean([t.pnl_pct for t in losers]) if losers else 0,
            'total_pnl': sum(t.pnl_pct for t in self.closed_trades),
            'daily_pnl': self.daily_pnl,
            'current_drawdown': self.current_drawdown
        }
