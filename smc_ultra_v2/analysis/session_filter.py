"""
SMC Ultra V2 - Trading Session Filter
=====================================
Filtert Trades basierend auf Trading-Sessions und Tageszeiten.

Beste Zeiten:
- London/NY Overlap (13:00-16:00 UTC): Höchste Liquidität
- London Session (08:00-16:00 UTC): Gute Bewegungen
- NY Session (13:00-21:00 UTC): Gute Bewegungen
- Asia Session (00:00-08:00 UTC): Oft Ranging
"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

import pandas as pd

from config.settings import config


class TradingSession(Enum):
    ASIA = "asia"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP = "overlap"  # London/NY overlap
    OFF_HOURS = "off_hours"


@dataclass
class SessionInfo:
    """Information about current session"""
    session: TradingSession
    weight: float  # 0-1.2, higher = better
    hours_until_close: float
    is_optimal: bool
    day_weight: float
    combined_weight: float


class SessionFilter:
    """
    Filters trades based on trading sessions.

    Uses session and day-of-week weights from config.
    """

    def __init__(self):
        self.session_config = config.sessions.sessions
        self.day_weights = config.sessions.day_weights

    def get_current_session(self, timestamp: datetime = None) -> SessionInfo:
        """
        Get current trading session information.

        Args:
            timestamp: Time to check (default: now UTC)

        Returns:
            SessionInfo with all session details
        """
        timestamp = timestamp or datetime.utcnow()
        hour = timestamp.hour
        day = timestamp.weekday()  # 0 = Monday

        # Determine session
        session = self._get_session(hour)
        session_weight = self._get_session_weight(session)
        day_weight = self.day_weights.get(day, 1.0)

        # Hours until session close
        hours_until_close = self._calc_hours_until_close(hour, session)

        # Is this optimal trading time?
        is_optimal = (
            session in [TradingSession.OVERLAP, TradingSession.LONDON, TradingSession.NEW_YORK] and
            day_weight >= 0.9 and
            hours_until_close >= 1  # At least 1 hour left
        )

        combined_weight = session_weight * day_weight

        return SessionInfo(
            session=session,
            weight=session_weight,
            hours_until_close=hours_until_close,
            is_optimal=is_optimal,
            day_weight=day_weight,
            combined_weight=combined_weight
        )

    def should_trade(self, timestamp: datetime = None) -> Tuple[bool, str]:
        """
        Check if we should trade at this time.

        Returns:
            (should_trade, reason)
        """
        info = self.get_current_session(timestamp)

        # Weekend check
        if info.day_weight < 0.6:
            return False, "weekend_low_volume"

        # Off hours check
        if info.session == TradingSession.OFF_HOURS:
            return False, "off_hours"

        # Session close check
        if info.hours_until_close < 0.5:
            return False, "session_closing"

        # Combined weight check
        if info.combined_weight < 0.6:
            return False, "low_session_weight"

        return True, "ok"

    def get_confidence_adjustment(self, timestamp: datetime = None) -> float:
        """
        Get confidence adjustment factor based on session.

        Returns:
            Multiplier for confidence (0.7 - 1.2)
        """
        info = self.get_current_session(timestamp)

        if info.session == TradingSession.OVERLAP:
            return 1.1  # Boost during overlap
        elif info.session in [TradingSession.LONDON, TradingSession.NEW_YORK]:
            return 1.0  # Normal
        elif info.session == TradingSession.ASIA:
            return 0.9  # Slight reduction
        else:
            return 0.7  # Off hours

    def get_leverage_adjustment(self, timestamp: datetime = None) -> float:
        """
        Get leverage adjustment factor based on session.

        Lower leverage during low liquidity periods.
        """
        info = self.get_current_session(timestamp)

        if info.session == TradingSession.OVERLAP:
            return 1.0  # Full leverage
        elif info.session in [TradingSession.LONDON, TradingSession.NEW_YORK]:
            return 0.9  # Slightly reduced
        elif info.session == TradingSession.ASIA:
            return 0.7  # More reduced
        else:
            return 0.5  # Half leverage

    def _get_session(self, hour: int) -> TradingSession:
        """Determine session from hour (UTC)"""
        # Overlap: 13-16 UTC
        if 13 <= hour < 16:
            return TradingSession.OVERLAP

        # London: 8-16 UTC
        if 8 <= hour < 16:
            return TradingSession.LONDON

        # New York: 13-21 UTC (excluding overlap)
        if 16 <= hour < 21:
            return TradingSession.NEW_YORK

        # Asia: 0-8 UTC
        if 0 <= hour < 8:
            return TradingSession.ASIA

        # Off hours: 21-00 UTC
        return TradingSession.OFF_HOURS

    def _get_session_weight(self, session: TradingSession) -> float:
        """Get weight for session"""
        weights = {
            TradingSession.OVERLAP: 1.2,
            TradingSession.LONDON: 1.0,
            TradingSession.NEW_YORK: 1.0,
            TradingSession.ASIA: 0.7,
            TradingSession.OFF_HOURS: 0.5
        }
        return weights.get(session, 0.5)

    def _calc_hours_until_close(self, hour: int, session: TradingSession) -> float:
        """Calculate hours until session close"""
        close_hours = {
            TradingSession.OVERLAP: 16,
            TradingSession.LONDON: 16,
            TradingSession.NEW_YORK: 21,
            TradingSession.ASIA: 8,
            TradingSession.OFF_HOURS: 24  # Next day
        }

        close = close_hours.get(session, 24)

        if close > hour:
            return close - hour
        else:
            return (24 - hour) + close


class EventFilter:
    """
    Filters trades around major economic events.

    Avoids trading during:
    - FOMC meetings
    - CPI releases
    - NFP
    - Major crypto events (halving, etc.)
    """

    # Major recurring events (simplified)
    AVOID_PERIODS = {
        'fomc': {'day_of_month': [1, 15], 'hour': 18, 'duration_hours': 4},
        'cpi': {'day_of_month': [10, 11, 12, 13], 'hour': 12, 'duration_hours': 2},
        'nfp': {'weekday': 4, 'week_of_month': 1, 'hour': 12, 'duration_hours': 2}  # First Friday
    }

    def should_avoid(self, timestamp: datetime = None) -> Tuple[bool, str]:
        """
        Check if we should avoid trading due to events.

        Returns:
            (should_avoid, reason)
        """
        timestamp = timestamp or datetime.utcnow()

        # Check each event type
        for event_name, event_config in self.AVOID_PERIODS.items():
            if self._is_during_event(timestamp, event_config):
                return True, f"near_{event_name}"

        return False, "ok"

    def _is_during_event(self, timestamp: datetime, event_config: Dict) -> bool:
        """Check if timestamp is during event"""
        hour = timestamp.hour
        day = timestamp.day
        weekday = timestamp.weekday()

        # Check day of month events
        if 'day_of_month' in event_config:
            if day in event_config['day_of_month']:
                event_hour = event_config.get('hour', 12)
                duration = event_config.get('duration_hours', 2)
                if event_hour - 1 <= hour <= event_hour + duration:
                    return True

        # Check weekday events (like NFP)
        if 'weekday' in event_config:
            if weekday == event_config['weekday']:
                # Check week of month
                week = (day - 1) // 7 + 1
                if week == event_config.get('week_of_month', 1):
                    event_hour = event_config.get('hour', 12)
                    duration = event_config.get('duration_hours', 2)
                    if event_hour - 1 <= hour <= event_hour + duration:
                        return True

        return False


class CombinedFilter:
    """Combines session and event filters"""

    def __init__(self):
        self.session_filter = SessionFilter()
        self.event_filter = EventFilter()

    def analyze(self, timestamp: datetime = None) -> Dict:
        """
        Complete analysis of trading conditions.

        Returns:
            Dict with all filter results and adjustments
        """
        timestamp = timestamp or datetime.utcnow()

        session_info = self.session_filter.get_current_session(timestamp)
        should_trade, session_reason = self.session_filter.should_trade(timestamp)
        avoid_event, event_reason = self.event_filter.should_avoid(timestamp)

        final_should_trade = should_trade and not avoid_event

        return {
            'should_trade': final_should_trade,
            'session': session_info.session.value,
            'session_weight': session_info.combined_weight,
            'is_optimal': session_info.is_optimal,
            'hours_until_close': session_info.hours_until_close,
            'confidence_adjustment': self.session_filter.get_confidence_adjustment(timestamp),
            'leverage_adjustment': self.session_filter.get_leverage_adjustment(timestamp),
            'avoid_event': avoid_event,
            'reason': event_reason if avoid_event else session_reason
        }


# Convenience instance
session_filter = CombinedFilter()
