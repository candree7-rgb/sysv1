"""
SMC Strategy Variants Tester
============================
Tests multiple OB_RETEST variants in parallel to find optimal parameters.
"""

import os
import sys
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from multiprocessing import cpu_count
from itertools import product

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Timeout settings for downloads
COIN_TIMEOUT = int(os.getenv('COIN_TIMEOUT', '45'))  # seconds per coin
COIN_DELAY = float(os.getenv('COIN_DELAY', '0.3'))   # delay between coins


class SetupType(Enum):
    OB_RETEST = "ob_retest"


# Trading costs (REALISTIC)
TAKER_FEE_PCT = 0.055
MAKER_FEE_PCT = 0.02
SLIPPAGE_PCT = 0.02

# Dynamic leverage settings
RISK_PER_TRADE_PCT = 2.0
MAX_LEVERAGE = 50
MIN_LEVERAGE = 5

# Position limits
MAX_LONGS = int(os.getenv('MAX_LONGS', '2'))
MAX_SHORTS = int(os.getenv('MAX_SHORTS', '2'))

# Parallelization
NUM_WORKERS = int(os.getenv('NUM_WORKERS', str(min(8, cpu_count()))))

# Precision mode: use 1min candles for exit checking
# Set USE_1MIN_EXITS=false for faster backtest (uses OHLC heuristic instead)
USE_1MIN_EXITS = os.getenv('USE_1MIN_EXITS', 'true').lower() == 'true'


@dataclass
class VariantConfig:
    """Configuration for a strategy variant"""
    name: str
    ob_min_strength: float
    ob_max_age: int  # candles
    sl_mult: float  # ATR multiplier for SL
    tp_mult: float  # ATR multiplier for TP
    use_ob_entry: bool = True  # True = entry at OB edge, False = entry at close
    # New filters
    use_mtf_alignment: bool = False  # Require 1H trend alignment
    use_volume_spike: bool = False  # Require above-average volume
    use_liquidity_sweep: bool = False  # Require recent liquidity sweep
    use_fvg_confluence: bool = False  # Require FVG overlapping with OB


@dataclass
class Trade:
    setup: SetupType
    symbol: str
    direction: str
    entry: float
    sl: float
    tp: float
    entry_time: datetime
    leverage: int = 10
    sl_pct: float = 0.0
    exit_time: datetime = None
    exit_price: float = None
    pnl_pct: float = None
    pnl_leveraged: float = None
    result: str = None
    variant: str = ""
    # Dynamic SL tracking for BE/Trailing
    current_sl: float = None  # Dynamic SL (starts as original SL)
    break_even_hit: bool = False
    trailing_active: bool = False
    max_profit_price: float = None  # Peak price for trailing
    exit_reason: str = None  # 'sl', 'tp', 'be', 'trailing'

    def __post_init__(self):
        if self.current_sl is None:
            self.current_sl = self.sl
        if self.max_profit_price is None:
            self.max_profit_price = self.entry


# Dynamic Trade Management Config (from settings.py)
BE_THRESHOLD = 0.60      # 60% toward TP → move SL to break-even
TRAIL_START = 0.75       # 75% toward TP → start trailing
TRAIL_OFFSET = 0.25      # Trail 25% behind peak


@dataclass
class VariantResult:
    """Results for a strategy variant"""
    name: str
    config: VariantConfig
    trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    max_dd: float
    final_equity: float
    # Detailed stats
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_leverage: float = 0.0
    avg_rr: float = 0.0  # Risk/Reward achieved
    # Exit reason breakdown
    exit_tp: int = 0       # Full TP hit
    exit_sl: int = 0       # Full SL hit (loss)
    exit_be: int = 0       # Break-even exit
    exit_trailing: int = 0  # Trailing stop exit


# Winner config with MTF Alignment (FINAL STRATEGY)
WINNER_CONFIG = {
    "ob_min_strength": 0.8,
    "ob_max_age": 50,
    "sl_mult": 1.0,
    "tp_mult": 1.5,
    "use_ob_entry": True,
    "use_mtf_alignment": True,  # 1H trend must align
}

# Define variants to test
VARIANTS = [
    # Final strategy: Winner with MTF Alignment
    VariantConfig(
        "winner_mtf", **WINNER_CONFIG,
        use_volume_spike=False, use_liquidity_sweep=False, use_fvg_confluence=False
    ),
]


def process_coin_for_variant(args) -> List[Trade]:
    """Process a single coin for a specific variant

    Uses 5min candles for signal generation (OB detection, EMA trend)
    Uses 1min candles for precise SL/TP exit checking
    Uses 1H candles for MTF alignment filter
    """
    symbol, days, variant = args

    # Disable SSL verification
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''

    from data import BybitDataDownloader
    from smc import OrderBlockDetector, FVGDetector, LiquidityDetector

    trades = []

    try:
        dl = BybitDataDownloader()
        ob_det = OrderBlockDetector()
        fvg_det = FVGDetector(min_size_pct=0.1)
        liq_det = LiquidityDetector()

        end = datetime.now()
        start = end - timedelta(days=days + 5)

        # Load 5min data for signal generation
        df_5m = dl.load_or_download(symbol, "5", days + 10)
        if df_5m is None or len(df_5m) < 200:
            return []

        # Load 1min data for precise exit checking
        df_1m = None
        if USE_1MIN_EXITS:
            df_1m = dl.load_or_download(symbol, "1", days + 10)
            if df_1m is None or len(df_1m) < 1000:
                df_1m = None  # Fall back to 5min exits

        # Load 1H data for MTF alignment (if needed)
        df_1h = None
        if variant.use_mtf_alignment:
            df_1h = dl.load_or_download(symbol, "60", days + 10)
            if df_1h is not None and len(df_1h) > 50:
                mask = (df_1h['timestamp'] >= start) & (df_1h['timestamp'] <= end)
                df_1h = df_1h[mask].reset_index(drop=True)
                df_1h['ema20'] = df_1h['close'].ewm(span=20).mean()
                df_1h['ema50'] = df_1h['close'].ewm(span=50).mean()

        mask = (df_5m['timestamp'] >= start) & (df_5m['timestamp'] <= end)
        df_5m = df_5m[mask].reset_index(drop=True)

        if len(df_5m) < 100:
            return []

        # Filter 1min data if available
        if df_1m is not None:
            mask = (df_1m['timestamp'] >= start) & (df_1m['timestamp'] <= end)
            df_1m = df_1m[mask].reset_index(drop=True)

        # Calculate indicators on 5min
        high_low = df_5m['high'] - df_5m['low']
        high_close = abs(df_5m['high'] - df_5m['close'].shift())
        low_close = abs(df_5m['low'] - df_5m['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df_5m['atr'] = tr.rolling(14).mean()
        df_5m['ema20'] = df_5m['close'].ewm(span=20).mean()
        df_5m['ema50'] = df_5m['close'].ewm(span=50).mean()

        # Volume ratio for volume spike filter
        if 'volume' in df_5m.columns:
            df_5m['vol_avg'] = df_5m['volume'].rolling(20).mean()
            df_5m['vol_ratio'] = df_5m['volume'] / df_5m['vol_avg']
        else:
            df_5m['vol_ratio'] = 1.0

        atr = df_5m['atr']
        obs = ob_det.detect(df_5m, atr)

        # Detect FVGs if needed
        fvgs = []
        if variant.use_fvg_confluence:
            fvgs = fvg_det.detect(df_5m)

        # Detect liquidity sweeps if needed
        sweeps = []
        if variant.use_liquidity_sweep:
            sweeps = liq_det.find_sweeps(df_5m, lookback_bars=20)

        active_trade = None
        pending_signal = None  # Store signal from previous candle for next-bar entry

        for idx in range(50, len(df_5m) - 1):  # -1 because we need idx+1 for exit checks
            candle = df_5m.iloc[idx]
            next_candle = df_5m.iloc[idx + 1]  # Next candle for exit verification
            ts = candle['timestamp']
            next_ts = next_candle['timestamp']
            price = candle['close']
            atr_val = candle['atr']

            if pd.isna(atr_val) or atr_val <= 0:
                continue

            # Check and close active trade using 1min precision
            if active_trade:
                if df_1m is not None and USE_1MIN_EXITS:
                    closed = check_trade_exit_1min(active_trade, df_1m, next_ts)
                else:
                    closed = check_trade_exit(active_trade, next_candle)
                if closed:
                    trades.append(active_trade)
                    active_trade = None

            if active_trade:
                continue

            # =====================================================
            # NEXT-BAR ENTRY: Check if pending signal fills
            # Signal was detected at idx-1, fill checked on idx (current candle)
            # =====================================================
            if pending_signal:
                sig = pending_signal
                pending_signal = None  # Clear it

                # Check if CURRENT candle fills our limit order (placed after previous candle close)
                # Long: price must DROP to ob.top (low <= ob.top)
                # Short: price must RISE to ob.bottom (high >= ob.bottom)
                filled = False
                if sig['trend'] == 'long' and candle['low'] <= sig['ob'].top:
                    filled = True
                elif sig['trend'] == 'short' and candle['high'] >= sig['ob'].bottom:
                    filled = True

                if filled:
                    # Create trade with current candle timestamp (realistic fill time)
                    active_trade = create_trade_with_variant(
                        symbol, sig['trend'], sig['price'], sig['atr'],
                        ts, sig['ob'], variant  # Use ts (current candle) as entry time
                    )
                    continue  # Don't look for new signals this candle

            # Get trend direction
            if candle['close'] > candle['ema20'] > candle['ema50']:
                trend = 'long'
            elif candle['close'] < candle['ema20'] < candle['ema50']:
                trend = 'short'
            else:
                continue

            # Get active OBs (with look-ahead bias fix!)
            active_obs = []
            for ob in obs:
                if ob.timestamp >= ts:
                    continue
                if not ob.is_mitigated:
                    active_obs.append(ob)
                elif ob.mitigation_timestamp is not None and ob.mitigation_timestamp > ts:
                    active_obs.append(ob)

            # =====================================================
            # NEW FILTERS (applied before OB check)
            # =====================================================

            # Filter: MTF Alignment (1H trend must match 5min trend)
            if variant.use_mtf_alignment and df_1h is not None:
                # Find the 1H candle that contains this 5min timestamp
                h1_match = df_1h[df_1h['timestamp'] <= ts].tail(1)
                if len(h1_match) > 0:
                    h1_candle = h1_match.iloc[0]
                    h1_ema20 = h1_candle.get('ema20', 0)
                    h1_ema50 = h1_candle.get('ema50', 0)
                    h1_close = h1_candle['close']

                    # Check 1H trend alignment
                    if trend == 'long' and not (h1_close > h1_ema20 > h1_ema50):
                        continue  # 1H not bullish, skip
                    if trend == 'short' and not (h1_close < h1_ema20 < h1_ema50):
                        continue  # 1H not bearish, skip

            # Filter: Volume Spike (volume > 1.5x average)
            if variant.use_volume_spike:
                vol_ratio = candle.get('vol_ratio', 1.0)
                if vol_ratio < 1.5:
                    continue  # Volume not high enough

            # Filter: Recent Liquidity Sweep
            if variant.use_liquidity_sweep:
                recent_sweep = liq_det.has_recent_sweep(sweeps, ts, lookback_bars=10)
                if trend == 'long' and not recent_sweep.get('bullish_sweep', False):
                    continue  # No recent bullish sweep
                if trend == 'short' and not recent_sweep.get('bearish_sweep', False):
                    continue  # No recent bearish sweep

            # =====================================================
            # Check for OB signal (fill verified on NEXT candle)
            # =====================================================
            matching_ob = None
            for ob in active_obs:
                if (trend == 'long' and ob.is_bullish) or (trend == 'short' and not ob.is_bullish):
                    if ob.bottom <= price <= ob.top:
                        # NOTE: We NO LONGER check fill here - that happens next candle!

                        # Filter 1: OB Strength (variant-specific)
                        if ob.strength < variant.ob_min_strength:
                            continue

                        # Filter 2: OB Age (variant-specific)
                        ob_age_minutes = (ts - ob.timestamp).total_seconds() / 60
                        ob_age_candles = ob_age_minutes / 5
                        if ob_age_candles > variant.ob_max_age:
                            continue

                        # Filter: FVG + OB Confluence
                        if variant.use_fvg_confluence:
                            has_fvg_overlap = False
                            for fvg in fvgs:
                                if fvg.is_filled:
                                    continue
                                if fvg.timestamp >= ts:
                                    continue
                                # Check if FVG overlaps with OB
                                if fvg.is_bullish == ob.is_bullish:
                                    # Check zone overlap
                                    overlap_top = min(ob.top, fvg.top)
                                    overlap_bottom = max(ob.bottom, fvg.bottom)
                                    if overlap_top > overlap_bottom:
                                        has_fvg_overlap = True
                                        break
                            if not has_fvg_overlap:
                                continue  # No FVG confluence

                        matching_ob = ob
                        break

            # Store signal for next-bar fill check (instead of immediate entry)
            if matching_ob:
                pending_signal = {
                    'trend': trend,
                    'price': price,
                    'atr': atr_val,
                    'ob': matching_ob,
                    'signal_ts': ts
                }

        # Close remaining trade
        if active_trade and len(df_5m) > 0:
            last_candle = df_5m.iloc[-1]
            if df_1m is not None and USE_1MIN_EXITS:
                check_trade_exit_1min(active_trade, df_1m, last_candle['timestamp'])
            else:
                check_trade_exit(active_trade, last_candle)
            if active_trade.exit_time:
                trades.append(active_trade)

    except Exception as e:
        pass  # Silently skip errors

    return trades


def create_trade_with_variant(
    symbol: str, direction: str, price: float, atr: float,
    ts: datetime, ob, variant: VariantConfig
) -> Trade:
    """Create a trade with variant-specific parameters"""

    # Entry price: at OB edge or at close
    if variant.use_ob_entry:
        if direction == 'long':
            # For long: enter at OB top (limit buy fills here)
            entry = ob.top  # Exact OB edge - maximum accuracy
        else:
            # For short: enter at OB bottom (limit sell fills here)
            entry = ob.bottom  # Exact OB edge - maximum accuracy
    else:
        # Traditional: entry at close with small slippage
        if direction == 'long':
            entry = price * 1.0003
        else:
            entry = price * 0.9997

    # SL/TP with variant-specific multipliers
    if direction == 'long':
        sl = entry - atr * variant.sl_mult
        tp = entry + atr * variant.tp_mult
    else:
        sl = entry + atr * variant.sl_mult
        tp = entry - atr * variant.tp_mult

    sl_pct = abs(entry - sl) / entry * 100

    if sl_pct > 0:
        calculated_lev = RISK_PER_TRADE_PCT / sl_pct
        leverage = min(int(calculated_lev), MAX_LEVERAGE)
        leverage = max(leverage, MIN_LEVERAGE)
    else:
        leverage = MIN_LEVERAGE

    return Trade(
        setup=SetupType.OB_RETEST,
        symbol=symbol,
        direction=direction,
        entry=entry,
        sl=sl,
        tp=tp,
        entry_time=ts,
        leverage=leverage,
        sl_pct=sl_pct,
        variant=variant.name
    )


def update_dynamic_sl(trade: Trade, candle) -> None:
    """Update trade's dynamic SL based on price progress (BE and Trailing).

    Called BEFORE checking exit to update SL levels.
    Uses OHLC heuristic: for longs, assume price hits high before checking SL.
    """
    is_bullish = candle['close'] > candle['open']

    if trade.direction == 'long':
        # Update max profit price (peak)
        # For bullish candle: assume high is reached
        # For bearish candle: assume high is reached first, then drops
        trade.max_profit_price = max(trade.max_profit_price, candle['high'])

        # Calculate progress toward TP (0 = entry, 1 = TP)
        total_distance = trade.tp - trade.entry
        if total_distance > 0:
            current_progress = (trade.max_profit_price - trade.entry) / total_distance
        else:
            current_progress = 0

        # Break-Even: 60% toward TP → move SL to entry + small buffer
        if current_progress >= BE_THRESHOLD and not trade.break_even_hit:
            trade.break_even_hit = True
            # Buffer = 10% of original SL distance (ensures small profit)
            buffer = abs(trade.entry - trade.sl) * 0.1
            new_sl = trade.entry + buffer
            trade.current_sl = max(trade.current_sl, new_sl)

        # Trailing: 75% toward TP → start trailing
        if current_progress >= TRAIL_START:
            trade.trailing_active = True

        # Update trailing SL if active
        if trade.trailing_active:
            profit = trade.max_profit_price - trade.entry
            trail_sl = trade.max_profit_price - (profit * TRAIL_OFFSET)
            trade.current_sl = max(trade.current_sl, trail_sl)

    else:  # short
        # Update max profit price (peak = lowest price for shorts)
        trade.max_profit_price = min(trade.max_profit_price, candle['low'])

        # Calculate progress toward TP
        total_distance = trade.entry - trade.tp
        if total_distance > 0:
            current_progress = (trade.entry - trade.max_profit_price) / total_distance
        else:
            current_progress = 0

        # Break-Even
        if current_progress >= BE_THRESHOLD and not trade.break_even_hit:
            trade.break_even_hit = True
            buffer = abs(trade.sl - trade.entry) * 0.1
            new_sl = trade.entry - buffer
            trade.current_sl = min(trade.current_sl, new_sl)

        # Trailing
        if current_progress >= TRAIL_START:
            trade.trailing_active = True

        if trade.trailing_active:
            profit = trade.entry - trade.max_profit_price
            trail_sl = trade.max_profit_price + (profit * TRAIL_OFFSET)
            trade.current_sl = min(trade.current_sl, trail_sl)


def check_trade_exit(trade: Trade, candle) -> bool:
    """Check if trade should exit (5min with OHLC heuristic + dynamic BE/Trailing SL).

    OHLC Heuristic for determining SL vs TP order when both could hit:
    - Bullish candle (close > open): Price went Low → High, so Low hit first
    - Bearish candle (close < open): Price went High → Low, so High hit first

    Dynamic SL Management:
    - First updates BE/Trailing levels based on price progress
    - Then checks exit against current_sl (not original sl)
    """
    # First: Update dynamic SL (BE and Trailing)
    update_dynamic_sl(trade, candle)

    fee_win = (MAKER_FEE_PCT * 2) + (SLIPPAGE_PCT * 2)
    fee_loss = MAKER_FEE_PCT + TAKER_FEE_PCT + (SLIPPAGE_PCT * 2)

    is_bullish = candle['close'] > candle['open']

    if trade.direction == 'long':
        # Use current_sl (dynamic) instead of original sl
        sl_hit = candle['low'] <= trade.current_sl
        tp_hit = candle['high'] >= trade.tp

        if sl_hit and tp_hit:
            # Both could hit - use OHLC heuristic
            # Bullish: Low first → SL hit first
            # Bearish: High first → TP hit first
            if is_bullish:
                result = 'sl_exit'
            else:
                result = 'tp'
        elif sl_hit:
            result = 'sl_exit'
        elif tp_hit:
            result = 'tp'
        else:
            return False

        if result == 'sl_exit':
            trade.exit_price = trade.current_sl
            gross_pnl = (trade.current_sl - trade.entry) / trade.entry * 100
            trade.pnl_pct = gross_pnl - fee_loss

            # Determine exit reason and result based on PnL
            if trade.trailing_active:
                trade.exit_reason = 'trailing'
                trade.result = 'win' if gross_pnl > 0 else 'loss'
            elif trade.break_even_hit:
                trade.exit_reason = 'be'
                trade.result = 'win' if gross_pnl > 0 else 'loss'
            else:
                trade.exit_reason = 'sl'
                trade.result = 'loss'
        else:  # TP hit
            trade.exit_price = trade.tp
            trade.result = 'win'
            trade.exit_reason = 'tp'
            gross_pnl = (trade.tp - trade.entry) / trade.entry * 100
            trade.pnl_pct = gross_pnl - fee_win

        trade.pnl_leveraged = trade.pnl_pct * trade.leverage
        trade.exit_time = candle['timestamp']
        return True

    else:  # short
        # Use current_sl (dynamic) instead of original sl
        sl_hit = candle['high'] >= trade.current_sl
        tp_hit = candle['low'] <= trade.tp

        if sl_hit and tp_hit:
            # Both could hit - use OHLC heuristic
            # Bullish: Low first → TP hit first (for short)
            # Bearish: High first → SL hit first (for short)
            if is_bullish:
                result = 'tp'
            else:
                result = 'sl_exit'
        elif sl_hit:
            result = 'sl_exit'
        elif tp_hit:
            result = 'tp'
        else:
            return False

        if result == 'sl_exit':
            trade.exit_price = trade.current_sl
            gross_pnl = (trade.entry - trade.current_sl) / trade.entry * 100
            trade.pnl_pct = gross_pnl - fee_loss

            # Determine exit reason and result based on PnL
            if trade.trailing_active:
                trade.exit_reason = 'trailing'
                trade.result = 'win' if gross_pnl > 0 else 'loss'
            elif trade.break_even_hit:
                trade.exit_reason = 'be'
                trade.result = 'win' if gross_pnl > 0 else 'loss'
            else:
                trade.exit_reason = 'sl'
                trade.result = 'loss'
        else:  # TP hit
            trade.exit_price = trade.tp
            trade.result = 'win'
            trade.exit_reason = 'tp'
            gross_pnl = (trade.entry - trade.tp) / trade.entry * 100
            trade.pnl_pct = gross_pnl - fee_win

        trade.pnl_leveraged = trade.pnl_pct * trade.leverage
        trade.exit_time = candle['timestamp']
        return True

    return False


def check_trade_exit_1min(trade: Trade, df_1m: pd.DataFrame, current_5m_ts: datetime) -> bool:
    """Check trade exit using 1min candles for precision + dynamic BE/Trailing SL.

    Iterates through 1min candles from trade entry to current 5min candle.
    This gives us exact order of SL vs TP hits.

    IMPORTANT: We start checking 1 candle AFTER entry to be conservative.
    This accounts for the fact that the limit order fills somewhere within
    the entry 5min candle, not at its start.
    """
    fee_win = (MAKER_FEE_PCT * 2) + (SLIPPAGE_PCT * 2)
    fee_loss = MAKER_FEE_PCT + TAKER_FEE_PCT + (SLIPPAGE_PCT * 2)

    # Get 1min candles from entry time to current 5min candle
    # Use >= entry_time + 5min to ensure we only check AFTER the entry candle
    # This is conservative: assumes fill happens at END of entry candle
    entry_candle_end = trade.entry_time + pd.Timedelta(minutes=5)
    mask = (df_1m['timestamp'] >= entry_candle_end) & (df_1m['timestamp'] <= current_5m_ts)
    candles_to_check = df_1m[mask]

    if len(candles_to_check) == 0:
        return False

    for _, candle in candles_to_check.iterrows():
        # Update dynamic SL (BE and Trailing) for each 1min candle
        update_dynamic_sl(trade, candle)

        if trade.direction == 'long':
            # Check SL first (hit if low <= current_sl)
            if candle['low'] <= trade.current_sl:
                trade.exit_price = trade.current_sl
                gross_pnl = (trade.current_sl - trade.entry) / trade.entry * 100
                trade.pnl_pct = gross_pnl - fee_loss
                trade.pnl_leveraged = trade.pnl_pct * trade.leverage
                trade.exit_time = candle['timestamp']

                if trade.trailing_active:
                    trade.exit_reason = 'trailing'
                    trade.result = 'win' if gross_pnl > 0 else 'loss'
                elif trade.break_even_hit:
                    trade.exit_reason = 'be'
                    trade.result = 'win' if gross_pnl > 0 else 'loss'
                else:
                    trade.exit_reason = 'sl'
                    trade.result = 'loss'
                return True
            # Check TP (hit if high >= tp)
            elif candle['high'] >= trade.tp:
                trade.exit_price = trade.tp
                trade.result = 'win'
                trade.exit_reason = 'tp'
                gross_pnl = (trade.tp - trade.entry) / trade.entry * 100
                trade.pnl_pct = gross_pnl - fee_win
                trade.pnl_leveraged = trade.pnl_pct * trade.leverage
                trade.exit_time = candle['timestamp']
                return True
        else:  # short
            # Check SL first (hit if high >= current_sl)
            if candle['high'] >= trade.current_sl:
                trade.exit_price = trade.current_sl
                gross_pnl = (trade.entry - trade.current_sl) / trade.entry * 100
                trade.pnl_pct = gross_pnl - fee_loss
                trade.pnl_leveraged = trade.pnl_pct * trade.leverage
                trade.exit_time = candle['timestamp']

                if trade.trailing_active:
                    trade.exit_reason = 'trailing'
                    trade.result = 'win' if gross_pnl > 0 else 'loss'
                elif trade.break_even_hit:
                    trade.exit_reason = 'be'
                    trade.result = 'win' if gross_pnl > 0 else 'loss'
                else:
                    trade.exit_reason = 'sl'
                    trade.result = 'loss'
                return True
            # Check TP (hit if low <= tp)
            elif candle['low'] <= trade.tp:
                trade.exit_price = trade.tp
                trade.result = 'win'
                trade.exit_reason = 'tp'
                gross_pnl = (trade.entry - trade.tp) / trade.entry * 100
                trade.pnl_pct = gross_pnl - fee_win
                trade.pnl_leveraged = trade.pnl_pct * trade.leverage
                trade.exit_time = candle['timestamp']
                return True

    return False


def apply_position_limits(trades: List[Trade]) -> List[Trade]:
    """Apply MAX_LONGS/MAX_SHORTS limits"""
    sorted_trades = sorted(trades, key=lambda t: t.entry_time)

    filtered_trades = []
    active_longs = []
    active_shorts = []

    for trade in sorted_trades:
        active_longs = [t for t in active_longs if t.exit_time is None or t.exit_time > trade.entry_time]
        active_shorts = [t for t in active_shorts if t.exit_time is None or t.exit_time > trade.entry_time]

        if trade.direction == 'long' and len(active_longs) >= MAX_LONGS:
            continue
        if trade.direction == 'short' and len(active_shorts) >= MAX_SHORTS:
            continue

        filtered_trades.append(trade)
        if trade.direction == 'long':
            active_longs.append(trade)
        else:
            active_shorts.append(trade)

    return filtered_trades


def calculate_variant_results(trades: List[Trade], variant: VariantConfig) -> VariantResult:
    """Calculate results for a variant"""
    if not trades:
        return VariantResult(
            name=variant.name, config=variant,
            trades=0, wins=0, losses=0,
            win_rate=0, profit_factor=0, total_pnl=0,
            max_dd=0, final_equity=10000
        )

    winners = [t for t in trades if t.result == 'win']
    losers = [t for t in trades if t.result == 'loss']

    win_count = len(winners)
    loss_count = len(losers)
    total = win_count + loss_count
    win_rate = win_count / total * 100 if total > 0 else 0

    gross_win = sum(t.pnl_leveraged for t in winners) if winners else 0
    gross_loss = abs(sum(t.pnl_leveraged for t in losers)) if losers else 1
    pf = gross_win / gross_loss if gross_loss > 0 else gross_win
    total_pnl = gross_win - gross_loss

    # Calculate detailed stats
    avg_win = (gross_win / win_count) if win_count > 0 else 0
    avg_loss = (gross_loss / loss_count) if loss_count > 0 else 0
    avg_leverage = sum(t.leverage for t in trades) / total if total > 0 else 0
    avg_rr = avg_win / avg_loss if avg_loss > 0 else avg_win

    # Count exit reasons
    exit_tp = sum(1 for t in trades if t.exit_reason == 'tp')
    exit_sl = sum(1 for t in trades if t.exit_reason == 'sl')
    exit_be = sum(1 for t in trades if t.exit_reason == 'be')
    exit_trailing = sum(1 for t in trades if t.exit_reason == 'trailing')

    # Calculate equity curve and max DD
    equity = 10000.0
    peak = 10000.0
    max_dd = 0.0

    for trade in sorted(trades, key=lambda t: t.entry_time):
        pnl_usd = equity * (trade.pnl_leveraged / 100)
        equity += pnl_usd
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100
        max_dd = max(max_dd, dd)

    return VariantResult(
        name=variant.name,
        config=variant,
        trades=total,
        wins=win_count,
        losses=loss_count,
        win_rate=win_rate,
        profit_factor=pf,
        total_pnl=total_pnl,
        max_dd=max_dd,
        final_equity=equity,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_leverage=avg_leverage,
        avg_rr=avg_rr,
        exit_tp=exit_tp,
        exit_sl=exit_sl,
        exit_be=exit_be,
        exit_trailing=exit_trailing
    )


def run_variant_comparison(num_coins: int = 100, days: int = 90, variants: List[VariantConfig] = None):
    """Run all variants and compare results"""
    from config.coins import get_top_n_coins

    variants = variants or VARIANTS

    print("=" * 100)
    print("SMC STRATEGY VARIANT COMPARISON")
    print("=" * 100)
    print(f"Testing {len(variants)} variants on {num_coins} coins over {days} days")
    print(f"Position limits: MAX_LONGS={MAX_LONGS}, MAX_SHORTS={MAX_SHORTS}")
    print(f"Using {NUM_WORKERS} parallel workers")
    print(f"Exit precision: {'1MIN CANDLES (accurate)' if USE_1MIN_EXITS else '5min candles'}")
    print(f"Entry precision: OB EDGE (limit order at OB top/bottom)")
    print(f"Dynamic SL: BE at {int(BE_THRESHOLD*100)}%, Trailing at {int(TRAIL_START*100)}% ({int(TRAIL_OFFSET*100)}% offset)")
    print(f"Rate limit protection: {COIN_TIMEOUT}s timeout, {COIN_DELAY}s delay between coins")
    print("=" * 100)

    coins = get_top_n_coins(num_coins)
    # Default skip list (known problematic coins)
    skip = {'APEUSDT', 'MATICUSDT', 'OCEANUSDT', 'EOSUSDT', 'FOGOUSDT', 'FHEUSDT', 'LITUSDT', 'WHITEWHALEUSDT', 'STABLEUSDT'}
    # Add extra coins from env: SKIP_COINS=COIN1,COIN2,COIN3
    extra_skip = os.getenv('SKIP_COINS', '')
    if extra_skip:
        skip.update(c.strip().upper() for c in extra_skip.split(',') if c.strip())
        print(f"Extra skip coins from env: {extra_skip}")
    coins = [c for c in coins if c not in skip]

    all_results = []

    for variant in variants:
        print(f"\n>>> Testing variant: {variant.name}")
        filters_str = []
        if variant.use_mtf_alignment:
            filters_str.append("MTF")
        if variant.use_volume_spike:
            filters_str.append("VOL")
        if variant.use_liquidity_sweep:
            filters_str.append("LIQ")
        if variant.use_fvg_confluence:
            filters_str.append("FVG")
        filters_display = "+".join(filters_str) if filters_str else "none"
        print(f"    Strength>={variant.ob_min_strength}, Age<={variant.ob_max_age}, "
              f"Filters=[{filters_display}]")

        # Prepare args for parallel processing
        args = [(coin, days, variant) for coin in coins]

        all_trades = []
        skipped_coins = []
        completed_count = 0

        # Simple sequential processing with timeout per coin
        # More reliable than complex batch processing
        print(f"    Processing {len(coins)} coins (max {COIN_TIMEOUT}s per coin)...")

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit all at once
            future_to_coin = {executor.submit(process_coin_for_variant, arg): arg[0] for arg in args}

            from concurrent.futures import as_completed
            try:
                # Use as_completed with a total timeout
                total_timeout = len(coins) * 15  # 15s average per coin should be plenty
                for future in as_completed(future_to_coin, timeout=total_timeout):
                    coin = future_to_coin[future]
                    try:
                        coin_trades = future.result(timeout=5)
                        if coin_trades:
                            all_trades.extend(coin_trades)
                        completed_count += 1
                        # Print every coin for live feedback
                        print(f"    [{completed_count}/{len(coins)}] {coin} ✓", flush=True)
                    except Exception as e:
                        skipped_coins.append(coin)
                        print(f"    [{completed_count}/{len(coins)}] {coin} ✗ {str(e)[:30]}", flush=True)
            except TimeoutError:
                # Total timeout exceeded - collect what we have
                pending_coins = [future_to_coin[f] for f in future_to_coin if not f.done()]
                skipped_coins.extend(pending_coins[:10])  # Just note first 10
                print(f"    ⚠️ Total timeout - {len(pending_coins)} coins still pending", flush=True)

        print(f"    Completed: {completed_count}/{len(coins)} coins")
        if skipped_coins:
            print(f"    ⚠️ Skipped/Errors: {len(skipped_coins)} coins")
        print(f"    Raw trades: {len(all_trades)}")

        # Apply position limits
        filtered_trades = apply_position_limits(all_trades)
        print(f"    After limits: {len(filtered_trades)}")

        # Calculate results
        result = calculate_variant_results(filtered_trades, variant)
        all_results.append(result)

        print(f"    WR: {result.win_rate:.1f}% | PF: {result.profit_factor:.2f} | "
              f"PnL: {result.total_pnl:+.1f}% | DD: {result.max_dd:.1f}%")

    # Print comparison table
    print("\n" + "=" * 120)
    print("VARIANT COMPARISON RESULTS")
    print("=" * 120)
    print(f"{'Variant':<20} {'Trades':>7} {'Wins':>6} {'Loss':>6} {'WR%':>7} {'PF':>6} "
          f"{'PnL%':>10} {'MaxDD%':>8} {'$10k->':>12}")
    print("-" * 120)

    # Sort by total PnL
    sorted_results = sorted(all_results, key=lambda r: -r.total_pnl)

    for r in sorted_results:
        print(f"{r.name:<20} {r.trades:>7} {r.wins:>6} {r.losses:>6} {r.win_rate:>6.1f}% "
              f"{r.profit_factor:>5.2f} {r.total_pnl:>+9.1f}% {r.max_dd:>7.1f}% ${r.final_equity:>10,.0f}")

    print("=" * 120)

    # Print best variant
    best = sorted_results[0]
    filters_str = []
    if best.config.use_mtf_alignment:
        filters_str.append("MTF_Alignment")
    if best.config.use_volume_spike:
        filters_str.append("Volume_Spike")
    if best.config.use_liquidity_sweep:
        filters_str.append("Liquidity_Sweep")
    if best.config.use_fvg_confluence:
        filters_str.append("FVG_OB_Confluence")
    filters_display = ", ".join(filters_str) if filters_str else "Base OB filters only"

    print(f"\n🏆 BEST VARIANT: {best.name}")
    print(f"   Config: Strength>={best.config.ob_min_strength}, Age<={best.config.ob_max_age}")
    print(f"   Filters: {filters_display}")
    print(f"   Win Rate: {best.win_rate:.1f}%")
    print(f"   Profit Factor: {best.profit_factor:.2f}")
    print(f"   Total PnL: {best.total_pnl:+.1f}%")
    print(f"   Max Drawdown: {best.max_dd:.1f}%")
    print(f"   $10,000 -> ${best.final_equity:,.0f}")
    print(f"\n   📊 DETAILED STATS:")
    print(f"   Avg Win:  +{best.avg_win:.2f}% (leveraged)")
    print(f"   Avg Loss: -{best.avg_loss:.2f}% (leveraged)")
    print(f"   Avg R:R:  {best.avg_rr:.2f}")
    print(f"   Avg Leverage: {best.avg_leverage:.1f}x")
    print(f"\n   🎯 EXIT BREAKDOWN:")
    print(f"   TP Hits:      {best.exit_tp:>3} ({best.exit_tp/best.trades*100:.1f}%)" if best.trades > 0 else "   TP Hits:        0")
    print(f"   SL Hits:      {best.exit_sl:>3} ({best.exit_sl/best.trades*100:.1f}%)" if best.trades > 0 else "   SL Hits:        0")
    print(f"   Break-Even:   {best.exit_be:>3} ({best.exit_be/best.trades*100:.1f}%)" if best.trades > 0 else "   Break-Even:     0")
    print(f"   Trailing:     {best.exit_trailing:>3} ({best.exit_trailing/best.trades*100:.1f}%)" if best.trades > 0 else "   Trailing:       0")

    return sorted_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--coins', type=int, default=100)
    parser.add_argument('--days', type=int, default=90)
    args = parser.parse_args()

    run_variant_comparison(args.coins, args.days)
