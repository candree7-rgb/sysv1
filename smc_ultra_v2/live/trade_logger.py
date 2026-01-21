"""
Trade Logger - Supabase Integration
====================================
Logs all trades to Supabase for tracking and ML analysis.
"""

import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

# Supabase client
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("[WARN] supabase not installed - trade logging disabled")


@dataclass
class TradeRecord:
    """Complete trade record for logging"""
    # === BASICS ===
    symbol: str
    direction: str  # 'long' / 'short'

    # === ENTRY ===
    entry_price: float
    entry_time: datetime
    qty: float
    leverage: int
    margin_used: float
    equity_at_entry: float

    # === TP/SL ===
    sl_price: float
    tp1_price: float
    tp2_price: float

    # === ORDER IDs ===
    order_id_1: str = None
    order_id_2: str = None

    # === ML FEATURES (filled at entry) ===
    ob_strength: float = None
    ob_age_candles: int = None
    ob_size_pct: float = None
    mtf_1h_aligned: bool = None
    mtf_4h_aligned: bool = None
    mtf_daily_aligned: bool = None
    alignment_score: int = None
    atr_pct: float = None
    rsi_14: float = None
    volume_ratio: float = None
    hour_utc: int = None
    day_of_week: int = None
    is_asian_session: bool = None
    is_london_session: bool = None
    is_ny_session: bool = None
    price_change_1h: float = None
    price_change_4h: float = None
    price_change_24h: float = None
    btc_price: float = None
    btc_change_24h: float = None
    funding_rate: float = None
    risk_pct: float = None
    risk_amount: float = None


class TradeLogger:
    """
    Logs trades to Supabase.

    Usage:
        logger = TradeLogger()
        trade_id = logger.log_entry(trade_record)
        logger.log_exit(trade_id, exit_data)
    """

    def __init__(self):
        self.client: Optional[Client] = None
        self.enabled = False

        if not SUPABASE_AVAILABLE:
            print("[TradeLogger] Supabase not available", flush=True)
            return

        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_KEY')

        if not url or not key:
            print("[TradeLogger] SUPABASE_URL or SUPABASE_KEY not set", flush=True)
            return

        try:
            self.client = create_client(url, key)
            self.enabled = True
            print("[TradeLogger] Connected to Supabase", flush=True)
        except Exception as e:
            print(f"[TradeLogger] Failed to connect: {e}", flush=True)

    def log_entry(self, trade: TradeRecord) -> Optional[str]:
        """
        Log trade entry. Returns trade ID for later update.
        """
        if not self.enabled:
            return None

        try:
            # Determine session
            hour = trade.entry_time.hour
            is_asian = 0 <= hour < 8
            is_london = 8 <= hour < 16
            is_ny = 13 <= hour < 21

            data = {
                'symbol': trade.symbol,
                'direction': trade.direction,
                'entry_price': float(trade.entry_price),
                'entry_time': trade.entry_time.isoformat(),
                'qty': float(trade.qty),
                'leverage': trade.leverage,
                'margin_used': float(trade.margin_used),
                'equity_at_entry': float(trade.equity_at_entry),
                'sl_price': float(trade.sl_price),
                'tp1_price': float(trade.tp1_price),
                'tp2_price': float(trade.tp2_price),
                'order_id_1': trade.order_id_1,
                'order_id_2': trade.order_id_2,

                # ML Features
                'ob_strength': trade.ob_strength,
                'ob_age_candles': trade.ob_age_candles,
                'mtf_1h_aligned': trade.mtf_1h_aligned,
                'mtf_4h_aligned': trade.mtf_4h_aligned,
                'mtf_daily_aligned': trade.mtf_daily_aligned,
                'hour_utc': hour,
                'day_of_week': trade.entry_time.weekday(),
                'is_asian_session': is_asian,
                'is_london_session': is_london,
                'is_ny_session': is_ny,
                'risk_pct': trade.risk_pct,
                'risk_amount': trade.risk_amount,
            }

            # Remove None values
            data = {k: v for k, v in data.items() if v is not None}

            result = self.client.table('trades').insert(data).execute()

            if result.data:
                trade_id = result.data[0]['id']
                print(f"  [DB] Trade logged: {trade_id[:8]}...", flush=True)
                return trade_id

        except Exception as e:
            print(f"  [DB ERR] Log entry: {str(e)[:50]}", flush=True)

        return None

    def log_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        realized_pnl: float,
        equity_at_close: float,
        tp1_hit: bool = False,
        tp2_hit: bool = False,
        entry_fee: float = 0,
        exit_fee: float = 0,
        funding_fee: float = 0,
        max_favorable_move: float = None,
        max_adverse_move: float = None,
        entry_time: datetime = None,
        margin_used: float = None
    ) -> bool:
        """
        Update trade with exit data.
        """
        if not self.enabled or not trade_id:
            return False

        try:
            # Calculate duration
            duration_minutes = None
            if entry_time:
                duration_minutes = int((exit_time - entry_time).total_seconds() / 60)

            # Calculate fees and net PnL
            total_fees = entry_fee + exit_fee
            net_pnl = realized_pnl - total_fees - funding_fee

            # Calculate PnL percentages
            pnl_pct = None
            pnl_pct_equity = None
            if margin_used and margin_used > 0:
                pnl_pct = (realized_pnl / margin_used) * 100
            if equity_at_close and equity_at_close > 0:
                pnl_pct_equity = (realized_pnl / equity_at_close) * 100

            # Calculate R-multiple (if we have risk info)
            r_multiple = None

            data = {
                'exit_price': float(exit_price),
                'exit_time': exit_time.isoformat(),
                'exit_reason': exit_reason,
                'duration_minutes': duration_minutes,
                'realized_pnl': float(realized_pnl),
                'pnl_pct': pnl_pct,
                'pnl_pct_equity': pnl_pct_equity,
                'equity_at_close': float(equity_at_close),
                'is_win': realized_pnl > 0,
                'tp1_hit': tp1_hit,
                'tp2_hit': tp2_hit,
                'entry_fee': float(entry_fee),
                'exit_fee': float(exit_fee),
                'total_fees': float(total_fees),
                'funding_fee': float(funding_fee),
                'net_pnl': float(net_pnl),
                'max_favorable_move': max_favorable_move,
                'max_adverse_move': max_adverse_move,
            }

            # Remove None values
            data = {k: v for k, v in data.items() if v is not None}

            result = self.client.table('trades').update(data).eq('id', trade_id).execute()

            if result.data:
                pnl_str = f"+${realized_pnl:.2f}" if realized_pnl > 0 else f"-${abs(realized_pnl):.2f}"
                print(f"  [DB] Exit logged: {exit_reason} {pnl_str}", flush=True)
                return True

        except Exception as e:
            print(f"  [DB ERR] Log exit: {str(e)[:50]}", flush=True)

        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get trading statistics"""
        if not self.enabled:
            return {}

        try:
            # Get all completed trades
            result = self.client.table('trades')\
                .select('*')\
                .not_.is_('exit_time', 'null')\
                .execute()

            trades = result.data
            if not trades:
                return {'total_trades': 0}

            wins = [t for t in trades if t.get('is_win')]
            losses = [t for t in trades if not t.get('is_win')]

            total_pnl = sum(t.get('net_pnl', 0) or t.get('realized_pnl', 0) for t in trades)
            total_fees = sum(t.get('total_fees', 0) for t in trades)

            return {
                'total_trades': len(trades),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': len(wins) / len(trades) * 100 if trades else 0,
                'total_pnl': total_pnl,
                'total_fees': total_fees,
                'avg_win': sum(t.get('realized_pnl', 0) for t in wins) / len(wins) if wins else 0,
                'avg_loss': sum(t.get('realized_pnl', 0) for t in losses) / len(losses) if losses else 0,
            }

        except Exception as e:
            print(f"  [DB ERR] Get stats: {e}", flush=True)
            return {}

    def get_equity_curve(self) -> list:
        """Get equity curve data points"""
        if not self.enabled:
            return []

        try:
            result = self.client.table('trades')\
                .select('exit_time, equity_at_close, net_pnl, pnl_pct_equity')\
                .not_.is_('exit_time', 'null')\
                .order('exit_time')\
                .execute()

            return result.data

        except Exception as e:
            print(f"  [DB ERR] Get equity curve: {e}", flush=True)
            return []


# Global instance
_logger: Optional[TradeLogger] = None

def get_trade_logger() -> TradeLogger:
    """Get or create global trade logger instance"""
    global _logger
    if _logger is None:
        _logger = TradeLogger()
    return _logger
