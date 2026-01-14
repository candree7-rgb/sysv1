"""
SMC Ultra V2 - Backtesting Engine
=================================
Vollständiges Backtesting mit realistischer Simulation.

Features:
- Multi-Timeframe Backtesting
- Realistic Fees and Slippage
- Dynamic Position Sizing
- Detailed Analytics
- ML Training Data Generation
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import uuid

import pandas as pd
import numpy as np

from config.settings import config
from config.coins import get_top_n_coins

from data import BybitDataDownloader, MTFDataLoader
from analysis import RegimeDetector, MTFAnalyzer
from smc import OrderBlockDetector, FVGDetector, LiquidityDetector
from strategy import SignalGenerator, Signal, DynamicTradeManager, Trade, ExitReason
from ml import FeatureExtractor


@dataclass
class BacktestConfig:
    """Backtest configuration"""
    symbols: List[str]
    start_date: datetime
    end_date: datetime

    # Capital
    initial_capital: float = 10000.0

    # Risk
    min_confidence: int = 85
    max_trades: int = 3
    risk_per_trade: float = 2.0
    max_leverage: int = 50

    # Costs
    fee_pct: float = 0.075  # Taker fee
    slippage_pct: float = 0.05

    # Data
    timeframe: str = "5"  # Primary timeframe for simulation
    warmup_bars: int = 200


@dataclass
class BacktestTrade:
    """Record of a backtest trade"""
    id: str
    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    take_profit: float
    stop_loss: float
    leverage: int
    confidence: int
    factors: List[str]
    exit_reason: str
    pnl_pct: float
    pnl_usd: float
    duration_minutes: float
    features: Dict = None  # For ML training


@dataclass
class BacktestResult:
    """Complete backtest results"""
    # Summary
    total_trades: int
    winners: int
    losers: int
    win_rate: float
    profit_factor: float

    # Returns
    total_return_pct: float
    total_return_usd: float
    avg_win_pct: float
    avg_loss_pct: float
    best_trade_pct: float
    worst_trade_pct: float

    # Risk
    max_drawdown_pct: float
    avg_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float

    # Trade stats
    avg_trade_duration_min: float
    trades_per_day: float

    # By category
    by_exit_reason: Dict
    by_confidence: Dict
    by_symbol: Dict
    by_regime: Dict

    # Data
    trades: List[BacktestTrade]
    equity_curve: pd.DataFrame


class BacktestEngine:
    """
    Main backtesting engine.

    Simulates trading with realistic conditions.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.bt_config = config

        # Components
        self.downloader = BybitDataDownloader()
        self.mtf_loader = MTFDataLoader()
        self.signal_generator = SignalGenerator()
        self.trade_manager = DynamicTradeManager()
        self.feature_extractor = FeatureExtractor()

        # SMC Detectors
        self.ob_detector = OrderBlockDetector()
        self.fvg_detector = FVGDetector()
        self.liq_detector = LiquidityDetector()
        self.regime_detector = RegimeDetector()

        # State
        self.data: Dict[str, pd.DataFrame] = {}
        self.equity = config.initial_capital
        self.peak_equity = config.initial_capital
        self.active_trades: Dict[str, Trade] = {}
        self.closed_trades: List[BacktestTrade] = []
        self.equity_history: List[Dict] = []
        self._signal_log_count = 0  # For debug logging
        self._debug_counts = {}  # For tracking filter reasons

    def run(self) -> BacktestResult:
        """
        Run the backtest.

        Returns:
            BacktestResult with all metrics
        """
        print("=" * 60)
        print("SMC ULTRA V2 BACKTEST")
        print("=" * 60)
        print(f"Symbols: {len(self.bt_config.symbols)}")
        print(f"Period: {self.bt_config.start_date} to {self.bt_config.end_date}")
        print(f"Capital: ${self.bt_config.initial_capital:,.2f}")
        print("=" * 60)

        # 1. Load data
        print("\n[1/5] Loading data...")
        self._load_data()

        # 2. Calculate indicators
        print("[2/5] Calculating indicators...")
        self._calc_indicators()

        # 3. Detect SMC structures
        print("[3/5] Detecting SMC structures...")
        self._detect_smc()

        # 4. Run simulation
        print("[4/5] Running simulation...")
        self._simulate()

        # 5. Calculate results
        print("[5/5] Calculating results...")
        results = self._calc_results()

        self._print_results(results)

        return results

    def _load_data(self):
        """Load historical data for all symbols"""
        days = (self.bt_config.end_date - self.bt_config.start_date).days + 30

        for i, symbol in enumerate(self.bt_config.symbols):
            print(f"  [{i+1}/{len(self.bt_config.symbols)}] {symbol}...", end="")

            df = self.downloader.load_or_download(
                symbol, self.bt_config.timeframe, days
            )

            if df is not None and len(df) > self.bt_config.warmup_bars:
                # Filter date range
                mask = (df['timestamp'] >= self.bt_config.start_date) & \
                       (df['timestamp'] <= self.bt_config.end_date)
                self.data[symbol] = df[mask].reset_index(drop=True)
                print(f" {len(self.data[symbol])} bars")
            else:
                print(" SKIP")

        print(f"\n  Loaded {len(self.data)} symbols")

    def _calc_indicators(self):
        """Calculate technical indicators"""
        for symbol, df in self.data.items():
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
            df['ema20'] = df['close'].ewm(span=20).mean()
            df['ema50'] = df['close'].ewm(span=50).mean()
            df['ema200'] = df['close'].ewm(span=200).mean()

            # Volume
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']

            self.data[symbol] = df

    def _detect_smc(self):
        """Detect SMC structures for all data"""
        # Create list copy to avoid modifying dict during iteration
        symbols_data = [(s, df) for s, df in self.data.items() if not s.endswith(('_obs', '_fvgs', '_sweeps'))]

        for symbol, df in symbols_data:
            print(f"  Detecting SMC for {symbol}...", flush=True)
            atr = df['atr']
            self.data[symbol + '_obs'] = self.ob_detector.detect(df, atr)
            self.data[symbol + '_fvgs'] = self.fvg_detector.detect(df)
            self.data[symbol + '_sweeps'] = self.liq_detector.find_sweeps(df)

    def _simulate(self):
        """Run the main simulation loop"""
        # Get all unique timestamps
        all_timestamps = set()
        for symbol, df in self.data.items():
            if not symbol.endswith(('_obs', '_fvgs', '_sweeps')):
                all_timestamps.update(df['timestamp'].tolist())

        timestamps = sorted(all_timestamps)
        total = len(timestamps)

        print(f"  Simulating {total} bars...")

        for i, ts in enumerate(timestamps):
            if i % 500 == 0:
                print(f"    Progress: {i}/{total} ({i/total*100:.1f}%)", flush=True)

            # 1. Update active trades
            self._update_trades(ts)

            # 2. Check for new signals
            if len(self.active_trades) < self.bt_config.max_trades:
                self._check_signals(ts)

            # 3. Track equity
            self.equity_history.append({
                'timestamp': ts,
                'equity': self.equity,
                'drawdown': self._calc_drawdown()
            })

        print(f"  Completed: {len(self.closed_trades)} trades", flush=True)

        # Debug: show filter reasons
        print(f"\n  Signal Filter Stats:", flush=True)
        for reason, count in sorted(self._debug_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count:,}", flush=True)

    def _update_trades(self, ts: datetime):
        """Update all active trades"""
        for symbol in list(self.active_trades.keys()):
            trade = self.active_trades[symbol]

            if symbol not in self.data:
                continue

            df = self.data[symbol]
            current = df[df['timestamp'] == ts]

            if len(current) == 0:
                continue

            candle = current.iloc[0]
            price = candle['close']

            # Check intra-bar TP/SL
            exit_reason = self._check_intra_bar(trade, candle)

            if not exit_reason:
                exit_reason = self.trade_manager.update(
                    trade, price, ts,
                    rsi=candle.get('rsi'),
                    volume_ratio=candle.get('volume_ratio')
                )

            if exit_reason:
                self._close_trade(trade, candle, exit_reason, ts)

    def _check_intra_bar(self, trade: Trade, candle) -> Optional[ExitReason]:
        """Check if SL or TP was hit within the candle"""
        if trade.direction == 'long':
            # Check SL first (worst case)
            if candle['low'] <= trade.current_sl:
                return ExitReason.STOP_LOSS
            if candle['high'] >= trade.take_profit:
                return ExitReason.TAKE_PROFIT
        else:
            if candle['high'] >= trade.current_sl:
                return ExitReason.STOP_LOSS
            if candle['low'] <= trade.take_profit:
                return ExitReason.TAKE_PROFIT

        return None

    def _check_signals(self, ts: datetime):
        """Check for new trading signals"""
        best_signal = None
        best_score = 0
        signals_found = 0

        for symbol, df in self.data.items():
            if symbol.endswith(('_obs', '_fvgs', '_sweeps')):
                continue
            if symbol in self.active_trades:
                continue

            current = df[df['timestamp'] == ts]
            if len(current) == 0:
                continue

            candle = current.iloc[0]
            idx = current.index[0]

            # Need enough data for analysis
            if idx < self.bt_config.warmup_bars:
                continue

            # Analyze
            signal = self._analyze_bar(symbol, df, idx, ts)

            if signal and signal.should_trade:
                signals_found += 1
                if signal.confidence > best_score:
                    best_signal = signal
                    best_score = signal.confidence

        if best_signal:
            if best_signal.confidence >= self.bt_config.min_confidence:
                print(f"  [TRADE] {best_signal.symbol} {best_signal.direction} @ {best_signal.entry_price:.2f} (conf: {best_signal.confidence}%)", flush=True)
                self._open_trade(best_signal, ts)
            elif signals_found > 0 and self._signal_log_count < 10:
                # Log some rejected signals for debugging
                print(f"  [SKIP] {best_signal.symbol} conf={best_signal.confidence}% < min={self.bt_config.min_confidence}%", flush=True)
                self._signal_log_count += 1

    def _analyze_bar(
        self,
        symbol: str,
        df: pd.DataFrame,
        idx: int,
        ts: datetime
    ) -> Optional[Signal]:
        """Analyze a single bar for signals"""
        candle = df.iloc[idx]
        price = candle['close']

        # Get historical data up to this point
        hist = df.iloc[:idx+1].tail(100)

        if len(hist) < 50:
            self._debug_counts['hist_too_short'] = self._debug_counts.get('hist_too_short', 0) + 1
            return None

        # Regime check
        regime = self.regime_detector.detect(hist)
        # Track regime distribution for debugging
        regime_key = f'regime_{regime.regime.value}'
        self._debug_counts[regime_key] = self._debug_counts.get(regime_key, 0) + 1
        if not regime.should_trade:
            self._debug_counts['regime_no_trade'] = self._debug_counts.get('regime_no_trade', 0) + 1
            return None

        # Get SMC structures
        obs = self.data.get(symbol + '_obs', [])
        fvgs = self.data.get(symbol + '_fvgs', [])
        sweeps = self.data.get(symbol + '_sweeps', [])

        # Filter to structures before this timestamp
        active_obs = [ob for ob in obs if ob.timestamp < ts and not ob.is_mitigated]
        active_fvgs = [fvg for fvg in fvgs if fvg.timestamp < ts and not fvg.is_filled]
        recent_sweeps = [s for s in sweeps if ts - timedelta(hours=1) <= s.timestamp <= ts]

        # Simple confluence scoring
        score = 50

        # HTF bias from EMAs
        if candle['close'] > candle['ema20'] > candle['ema50']:
            direction = 'long'
            score += 15
        elif candle['close'] < candle['ema20'] < candle['ema50']:
            direction = 'short'
            score += 15
        else:
            self._debug_counts['no_ema_direction'] = self._debug_counts.get('no_ema_direction', 0) + 1
            return None  # No clear direction

        # ============================================
        # SMC SCORING - Based on real trading value
        # ============================================
        # Sweep = #1 (Smart Money aktiv)
        # OB = #2 (Institutional Orders)
        # FVG = #3 (Imbalance, weniger zuverlässig)
        # RSI = Timing-Hilfe

        # 1. REQUIRED: Liquidity Sweep (wichtigstes Signal!)
        has_sweep = False
        for sweep in recent_sweeps:
            if (direction == 'long' and sweep.is_bullish) or \
               (direction == 'short' and not sweep.is_bullish):
                has_sweep = True
                score += 30  # Höchster Wert!
                break

        if not has_sweep:
            self._debug_counts['no_sweep'] = self._debug_counts.get('no_sweep', 0) + 1
            return None

        # 2. REQUIRED: Near Order Block (sehr wichtig nach Sweep!)
        near_ob = False
        for ob in active_obs:
            dist = abs(price - ob.mid) / price * 100
            if dist < 0.5:  # Within 0.5%
                if (direction == 'long' and ob.is_bullish) or \
                   (direction == 'short' and not ob.is_bullish):
                    score += 25  # Zweithöchster Wert
                    near_ob = True
                    break

        # OB ist Pflicht - FVG allein reicht nicht!
        if not near_ob:
            self._debug_counts['no_ob'] = self._debug_counts.get('no_ob', 0) + 1
            return None

        # 3. BONUS: In FVG (extra Confluence)
        in_fvg = False
        for fvg in active_fvgs:
            if fvg.bottom <= price <= fvg.top:
                if (direction == 'long' and fvg.is_bullish) or \
                   (direction == 'short' and not fvg.is_bullish):
                    score += 10
                    in_fvg = True
                    break

        # 4. BONUS: RSI extreme (besseres Timing)
        rsi = candle.get('rsi', 50)
        if direction == 'long' and rsi < 35:
            score += 10
        elif direction == 'short' and rsi > 65:
            score += 10

        # 5. BONUS: FVG + RSI combo
        if in_fvg and ((direction == 'long' and rsi < 35) or (direction == 'short' and rsi > 65)):
            score += 5

        # Regime adjustment
        score = int(score * regime.leverage_multiplier)

        # Track score distribution
        if score >= 80:
            self._debug_counts['score_80+'] = self._debug_counts.get('score_80+', 0) + 1
        elif score >= 70:
            self._debug_counts['score_70-79'] = self._debug_counts.get('score_70-79', 0) + 1
        elif score >= 60:
            self._debug_counts['score_60-69'] = self._debug_counts.get('score_60-69', 0) + 1
        else:
            self._debug_counts['score_below_60'] = self._debug_counts.get('score_below_60', 0) + 1

        if score < self.bt_config.min_confidence:
            self._debug_counts['low_score'] = self._debug_counts.get('low_score', 0) + 1
            return None

        # Calculate targets - 1:1 RR
        atr = candle['atr']
        if direction == 'long':
            entry = price * (1 + self.bt_config.slippage_pct / 100)
            sl = entry - atr * 1.5   # SL distance
            tp = entry + atr * 1.5   # 1:1 RR
        else:
            entry = price * (1 - self.bt_config.slippage_pct / 100)
            sl = entry + atr * 1.5
            tp = entry - atr * 1.5

        sl_pct = abs(entry - sl) / entry * 100
        tp_pct = abs(tp - entry) / entry * 100

        # Leverage
        max_lev_for_risk = self.bt_config.risk_per_trade / sl_pct * 100
        leverage = min(int(max_lev_for_risk), self.bt_config.max_leverage)
        leverage = max(leverage, 5)

        factors = []
        if near_ob:
            factors.append('ob')
        factors.append(f'regime_{regime.regime.value}')

        return Signal(
            symbol=symbol,
            direction=direction,
            entry_price=entry,
            take_profit=tp,
            stop_loss=sl,
            confidence=score,
            should_trade=True,
            regime=regime.regime.value,
            htf_bias=direction,
            pattern='ema_cross',
            factors=factors,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            effective_rr=tp_pct / sl_pct if sl_pct > 0 else 1,
            leverage=leverage,
            timestamp=ts
        )

    def _open_trade(self, signal: Signal, ts: datetime):
        """Open a new trade"""
        trade = Trade(
            id=str(uuid.uuid4())[:8],
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            entry_time=ts,
            take_profit=signal.take_profit,
            stop_loss=signal.stop_loss,
            current_sl=signal.stop_loss,
            leverage=signal.leverage,
            confidence=signal.confidence,
            factors=signal.factors
        )

        # Deduct entry fee
        trade_size = self.equity * 0.01  # 1% of equity
        fee = trade_size * (self.bt_config.fee_pct / 100) * signal.leverage
        self.equity -= fee

        self.active_trades[signal.symbol] = trade

    def _close_trade(
        self,
        trade: Trade,
        candle: pd.Series,
        reason: ExitReason,
        ts: datetime
    ):
        """Close a trade"""
        # Determine exit price
        if reason == ExitReason.TAKE_PROFIT:
            exit_price = trade.take_profit
        elif reason in [ExitReason.STOP_LOSS, ExitReason.TRAILING_STOP, ExitReason.BREAK_EVEN]:
            exit_price = trade.current_sl
        else:
            exit_price = candle['close']

        # Apply slippage to exit
        if trade.direction == 'long':
            exit_price *= (1 - self.bt_config.slippage_pct / 200)
        else:
            exit_price *= (1 + self.bt_config.slippage_pct / 200)

        # Calculate PnL
        if trade.direction == 'long':
            pnl_pct = ((exit_price - trade.entry_price) / trade.entry_price) * 100
        else:
            pnl_pct = ((trade.entry_price - exit_price) / trade.entry_price) * 100

        pnl_pct_leveraged = pnl_pct * trade.leverage

        # Update equity
        trade_size = self.equity * 0.01
        pnl_usd = trade_size * pnl_pct_leveraged / 100
        self.equity += pnl_usd

        # Exit fee
        fee = trade_size * (self.bt_config.fee_pct / 100) * trade.leverage
        self.equity -= fee

        # Update peak equity
        self.peak_equity = max(self.peak_equity, self.equity)

        # Record trade
        duration = (ts - trade.entry_time).total_seconds() / 60

        bt_trade = BacktestTrade(
            id=trade.id,
            symbol=trade.symbol,
            direction=trade.direction,
            entry_time=trade.entry_time,
            entry_price=trade.entry_price,
            exit_time=ts,
            exit_price=exit_price,
            take_profit=trade.take_profit,
            stop_loss=trade.stop_loss,
            leverage=trade.leverage,
            confidence=trade.confidence,
            factors=trade.factors,
            exit_reason=reason.value,
            pnl_pct=pnl_pct_leveraged,
            pnl_usd=pnl_usd,
            duration_minutes=duration
        )

        self.closed_trades.append(bt_trade)
        del self.active_trades[trade.symbol]

    def _calc_drawdown(self) -> float:
        """Calculate current drawdown"""
        if self.peak_equity == 0:
            return 0
        return ((self.peak_equity - self.equity) / self.peak_equity) * 100

    def _calc_results(self) -> BacktestResult:
        """Calculate all backtest metrics"""
        if not self.closed_trades:
            return self._empty_result()

        trades_df = pd.DataFrame([
            {
                'symbol': t.symbol,
                'direction': t.direction,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'pnl_pct': t.pnl_pct,
                'pnl_usd': t.pnl_usd,
                'confidence': t.confidence,
                'exit_reason': t.exit_reason,
                'duration': t.duration_minutes,
                'leverage': t.leverage
            }
            for t in self.closed_trades
        ])

        winners = trades_df[trades_df['pnl_pct'] > 0]
        losers = trades_df[trades_df['pnl_pct'] <= 0]

        # Basic stats
        total_trades = len(trades_df)
        win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0

        # Returns
        total_return_pct = ((self.equity - self.bt_config.initial_capital) /
                           self.bt_config.initial_capital) * 100
        total_return_usd = self.equity - self.bt_config.initial_capital

        avg_win = winners['pnl_pct'].mean() if len(winners) > 0 else 0
        avg_loss = abs(losers['pnl_pct'].mean()) if len(losers) > 0 else 0

        # Profit factor
        gross_profit = winners['pnl_pct'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['pnl_pct'].sum()) if len(losers) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

        # Drawdown
        equity_df = pd.DataFrame(self.equity_history)
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['dd'] = (equity_df['peak'] - equity_df['equity']) / equity_df['peak'] * 100
        max_dd = equity_df['dd'].max()
        avg_dd = equity_df['dd'].mean()

        # Sharpe/Sortino (simplified)
        if len(trades_df) > 1:
            returns = trades_df['pnl_pct'].values
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            downside = returns[returns < 0]
            sortino = np.mean(returns) / np.std(downside) * np.sqrt(252) if len(downside) > 0 and np.std(downside) > 0 else 0
        else:
            sharpe = 0
            sortino = 0

        # By category
        by_exit = trades_df.groupby('exit_reason').agg({
            'pnl_pct': ['count', 'mean', 'sum']
        }).to_dict()

        by_confidence = {}
        for low, high in [(95, 100), (90, 94), (85, 89), (80, 84)]:
            subset = trades_df[(trades_df['confidence'] >= low) & (trades_df['confidence'] <= high)]
            if len(subset) > 0:
                by_confidence[f'{low}-{high}'] = {
                    'count': len(subset),
                    'win_rate': len(subset[subset['pnl_pct'] > 0]) / len(subset) * 100,
                    'avg_pnl': subset['pnl_pct'].mean()
                }

        by_symbol = trades_df.groupby('symbol').agg({
            'pnl_pct': ['count', 'mean', 'sum']
        }).to_dict()

        # Trading days
        days = (self.bt_config.end_date - self.bt_config.start_date).days
        trades_per_day = total_trades / days if days > 0 else 0

        return BacktestResult(
            total_trades=total_trades,
            winners=len(winners),
            losers=len(losers),
            win_rate=round(win_rate, 2),
            profit_factor=round(profit_factor, 2),
            total_return_pct=round(total_return_pct, 2),
            total_return_usd=round(total_return_usd, 2),
            avg_win_pct=round(avg_win, 2),
            avg_loss_pct=round(avg_loss, 2),
            best_trade_pct=round(trades_df['pnl_pct'].max(), 2),
            worst_trade_pct=round(trades_df['pnl_pct'].min(), 2),
            max_drawdown_pct=round(max_dd, 2),
            avg_drawdown_pct=round(avg_dd, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            avg_trade_duration_min=round(trades_df['duration'].mean(), 1),
            trades_per_day=round(trades_per_day, 2),
            by_exit_reason=by_exit,
            by_confidence=by_confidence,
            by_symbol=by_symbol,
            by_regime={},
            trades=self.closed_trades,
            equity_curve=equity_df
        )

    def _empty_result(self) -> BacktestResult:
        """Return empty result"""
        return BacktestResult(
            total_trades=0, winners=0, losers=0,
            win_rate=0, profit_factor=0,
            total_return_pct=0, total_return_usd=0,
            avg_win_pct=0, avg_loss_pct=0,
            best_trade_pct=0, worst_trade_pct=0,
            max_drawdown_pct=0, avg_drawdown_pct=0,
            sharpe_ratio=0, sortino_ratio=0,
            avg_trade_duration_min=0, trades_per_day=0,
            by_exit_reason={}, by_confidence={},
            by_symbol={}, by_regime={},
            trades=[], equity_curve=pd.DataFrame()
        )

    def _print_results(self, r: BacktestResult):
        """Print backtest results"""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        print(f"\n{'PERFORMANCE':^60}")
        print("-" * 60)
        print(f"Total Trades:      {r.total_trades}")
        print(f"Win Rate:          {r.win_rate}%")
        print(f"Profit Factor:     {r.profit_factor}")
        print(f"Total Return:      {r.total_return_pct}% (${r.total_return_usd:,.2f})")

        print(f"\n{'TRADE STATS':^60}")
        print("-" * 60)
        print(f"Winners:           {r.winners}")
        print(f"Losers:            {r.losers}")
        print(f"Avg Win:           {r.avg_win_pct}%")
        print(f"Avg Loss:          {r.avg_loss_pct}%")
        print(f"Best Trade:        {r.best_trade_pct}%")
        print(f"Worst Trade:       {r.worst_trade_pct}%")

        print(f"\n{'RISK METRICS':^60}")
        print("-" * 60)
        print(f"Max Drawdown:      {r.max_drawdown_pct}%")
        print(f"Avg Drawdown:      {r.avg_drawdown_pct}%")
        print(f"Sharpe Ratio:      {r.sharpe_ratio}")
        print(f"Sortino Ratio:     {r.sortino_ratio}")

        print(f"\n{'ACTIVITY':^60}")
        print("-" * 60)
        print(f"Avg Duration:      {r.avg_trade_duration_min} min")
        print(f"Trades/Day:        {r.trades_per_day}")

        if r.by_confidence:
            print(f"\n{'BY CONFIDENCE':^60}")
            print("-" * 60)
            for bucket, stats in r.by_confidence.items():
                print(f"  {bucket}: {stats['count']} trades, "
                      f"{stats['win_rate']:.1f}% WR, "
                      f"{stats['avg_pnl']:.2f}% avg")

        print("\n" + "=" * 60)

    def save_results(self, path: str = None):
        """Save backtest results to file"""
        if not self.closed_trades:
            return

        path = Path(path or "backtest_results")
        path.mkdir(parents=True, exist_ok=True)

        # Save trades
        trades_data = [
            {
                'id': t.id,
                'symbol': t.symbol,
                'direction': t.direction,
                'entry_time': t.entry_time.isoformat(),
                'exit_time': t.exit_time.isoformat(),
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl_pct': t.pnl_pct,
                'pnl_usd': t.pnl_usd,
                'confidence': t.confidence,
                'exit_reason': t.exit_reason,
                'leverage': t.leverage,
                'factors': t.factors
            }
            for t in self.closed_trades
        ]

        with open(path / "trades.json", 'w') as f:
            json.dump(trades_data, f, indent=2)

        # Save equity curve
        eq_df = pd.DataFrame(self.equity_history)
        eq_df.to_csv(path / "equity_curve.csv", index=False)

        print(f"Results saved to {path}")
