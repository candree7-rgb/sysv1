"""
SMC Ultra V2 - Strategy Optimizer
=================================
Grid Search über alle Parameter-Kombinationen um die beste Strategie zu finden.

Usage:
    python optimizer.py

Testet automatisch:
- Verschiedene SL/TP Kombinationen
- Verschiedene Filter (Sweep, RSI, Trend, etc.)
- Verschiedene Volatility Thresholds
- Verschiedene Confidence Levels
"""

import os
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from itertools import product
import json

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.coins import get_top_n_coins


@dataclass
class StrategyConfig:
    """Eine Strategie-Konfiguration zum Testen"""
    name: str
    sl_atr: float           # SL in ATR
    tp_atr: float           # TP in ATR
    require_sweep: bool     # Sweep required?
    require_high_vol: bool  # High volatility required?
    vol_threshold: float    # ATR multiplier for high vol
    require_volume: bool    # Volume > avg required?
    rsi_filter: str         # 'none', 'moderate', 'extreme'
    trend_filter: str       # 'none', 'weak_ok', 'strong_only'
    min_confidence: int     # Minimum score


# Alle zu testenden Kombinationen
PARAM_GRID = {
    'sl_atr': [0.5, 0.8, 1.0, 1.2, 1.5],
    'tp_atr': [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
    'require_sweep': [True, False],
    'require_high_vol': [True, False],
    'vol_threshold': [1.1, 1.2, 1.3],
    'require_volume': [True, False],
    'rsi_filter': ['none', 'moderate', 'extreme'],
    'trend_filter': ['none', 'weak_ok', 'strong_only'],
    'min_confidence': [70, 80, 85, 90],
}

# Reduziertes Grid für schnelleren Test
PARAM_GRID_FAST = {
    'sl_atr': [0.5, 0.8, 1.2],
    'tp_atr': [0.8, 1.0, 1.5],
    'require_sweep': [True, False],
    'require_high_vol': [True, False],
    'vol_threshold': [1.2],
    'require_volume': [True],
    'rsi_filter': ['none', 'moderate'],
    'trend_filter': ['none', 'strong_only'],
    'min_confidence': [80, 85],
}


def generate_configs(fast: bool = True) -> List[StrategyConfig]:
    """Generiere alle Strategie-Kombinationen"""
    grid = PARAM_GRID_FAST if fast else PARAM_GRID

    configs = []
    keys = list(grid.keys())

    for values in product(*grid.values()):
        params = dict(zip(keys, values))

        # Skip invalid combinations
        if params['tp_atr'] < params['sl_atr']:
            continue  # TP should be >= SL for reasonable RR

        name = f"SL{params['sl_atr']}_TP{params['tp_atr']}"
        if params['require_sweep']:
            name += "_SWEEP"
        if params['require_high_vol']:
            name += f"_VOL{params['vol_threshold']}"
        if params['trend_filter'] != 'none':
            name += f"_{params['trend_filter'].upper()}"
        if params['rsi_filter'] != 'none':
            name += f"_RSI{params['rsi_filter']}"

        configs.append(StrategyConfig(name=name, **params))

    return configs


class FlexibleBacktester:
    """Backtester mit konfigurierbaren Parametern"""

    def __init__(self, config: StrategyConfig, symbols: List[str], days: int = 7):
        self.config = config
        self.symbols = symbols
        self.days = days

        # Lazy imports
        from data import BybitDataDownloader
        from analysis import RegimeDetector
        from smc import OrderBlockDetector, FVGDetector, LiquidityDetector

        self.downloader = BybitDataDownloader()
        self.regime_detector = RegimeDetector()
        self.ob_detector = OrderBlockDetector()
        self.fvg_detector = FVGDetector()
        self.liq_detector = LiquidityDetector()

        self.data = {}
        self.trades = []

    def load_data(self):
        """Load data for all symbols"""
        import pandas as pd

        end = datetime.now()
        start = end - timedelta(days=self.days + 5)

        for symbol in self.symbols:
            df = self.downloader.load_or_download(symbol, "5", self.days + 10)
            if df is None or len(df) < 200:
                continue

            # Filter date range
            mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
            df = df[mask].reset_index(drop=True)

            if len(df) < 100:
                continue

            # Calculate indicators
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

            # Volume
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']

            self.data[symbol] = df

            # Detect SMC
            atr = df['atr']
            self.data[symbol + '_obs'] = self.ob_detector.detect(df, atr)
            self.data[symbol + '_fvgs'] = self.fvg_detector.detect(df)
            self.data[symbol + '_sweeps'] = self.liq_detector.find_sweeps(df)

    def run(self) -> Dict[str, Any]:
        """Run backtest with this config"""
        self.trades = []

        for symbol, df in self.data.items():
            if symbol.endswith(('_obs', '_fvgs', '_sweeps')):
                continue

            self._backtest_symbol(symbol, df)

        return self._calc_results()

    def _backtest_symbol(self, symbol: str, df):
        """Backtest single symbol"""
        import pandas as pd

        obs = self.data.get(symbol + '_obs', [])
        fvgs = self.data.get(symbol + '_fvgs', [])
        sweeps = self.data.get(symbol + '_sweeps', [])

        active_trade = None

        for idx in range(50, len(df)):
            candle = df.iloc[idx]
            ts = candle['timestamp']
            price = candle['close']

            # Check active trade
            if active_trade:
                # Check SL/TP
                if active_trade['direction'] == 'long':
                    if candle['low'] <= active_trade['sl']:
                        self._close_trade(active_trade, active_trade['sl'], 'sl')
                        active_trade = None
                    elif candle['high'] >= active_trade['tp']:
                        self._close_trade(active_trade, active_trade['tp'], 'tp')
                        active_trade = None
                else:
                    if candle['high'] >= active_trade['sl']:
                        self._close_trade(active_trade, active_trade['sl'], 'sl')
                        active_trade = None
                    elif candle['low'] <= active_trade['tp']:
                        self._close_trade(active_trade, active_trade['tp'], 'tp')
                        active_trade = None
                continue

            # Check for new signal
            signal = self._check_signal(symbol, df, idx, ts, obs, fvgs, sweeps)
            if signal:
                active_trade = signal

    def _check_signal(self, symbol, df, idx, ts, obs, fvgs, sweeps) -> Dict:
        """Check for signal based on config"""
        candle = df.iloc[idx]
        price = candle['close']
        hist = df.iloc[:idx+1].tail(50)

        if len(hist) < 30:
            return None

        # Direction from EMAs
        if candle['close'] > candle['ema20'] > candle['ema50']:
            direction = 'long'
        elif candle['close'] < candle['ema20'] < candle['ema50']:
            direction = 'short'
        else:
            return None

        score = 50

        # HIGH VOLATILITY CHECK
        if self.config.require_high_vol:
            atr_pct = candle.get('atr_pct', 0)
            recent_atr = hist['atr_pct'].mean() if 'atr_pct' in hist.columns else atr_pct
            if atr_pct < recent_atr * self.config.vol_threshold:
                return None
            score += 20

        # VOLUME CHECK
        if self.config.require_volume:
            vol_ratio = candle.get('volume_ratio', 1.0)
            if vol_ratio < 1.0:
                return None
            if vol_ratio > 1.5:
                score += 15

        # REGIME/TREND CHECK
        regime = self.regime_detector.detect(hist)

        if self.config.trend_filter == 'strong_only':
            if regime.regime.value not in ('strong_trend_up', 'strong_trend_down'):
                return None
            score += 25
        elif self.config.trend_filter == 'weak_ok':
            if regime.regime.value == 'ranging':
                return None
            if 'strong' in regime.regime.value:
                score += 20
            else:
                score += 10

        # SWEEP CHECK
        recent_sweeps = [s for s in sweeps if ts - timedelta(hours=1) <= s.timestamp <= ts]
        has_sweep = False
        for sweep in recent_sweeps:
            if (direction == 'long' and sweep.is_bullish) or \
               (direction == 'short' and not sweep.is_bullish):
                has_sweep = True
                score += 25
                break

        if self.config.require_sweep and not has_sweep:
            return None

        # RSI CHECK
        rsi = candle.get('rsi', 50)
        if self.config.rsi_filter == 'extreme':
            if direction == 'long' and rsi > 35:
                return None
            if direction == 'short' and rsi < 65:
                return None
            score += 20
        elif self.config.rsi_filter == 'moderate':
            if direction == 'long' and rsi < 45:
                score += 15
            elif direction == 'short' and rsi > 55:
                score += 15

        # Apply regime multiplier
        score = int(score * regime.leverage_multiplier)

        if score < self.config.min_confidence:
            return None

        # Calculate SL/TP
        atr = candle['atr']
        if direction == 'long':
            entry = price * 1.0005  # Small slippage
            sl = entry - atr * self.config.sl_atr
            tp = entry + atr * self.config.tp_atr
        else:
            entry = price * 0.9995
            sl = entry + atr * self.config.sl_atr
            tp = entry - atr * self.config.tp_atr

        return {
            'symbol': symbol,
            'direction': direction,
            'entry': entry,
            'sl': sl,
            'tp': tp,
            'score': score,
            'timestamp': ts
        }

    def _close_trade(self, trade: Dict, exit_price: float, reason: str):
        """Close a trade"""
        if trade['direction'] == 'long':
            pnl_pct = (exit_price - trade['entry']) / trade['entry'] * 100
        else:
            pnl_pct = (trade['entry'] - exit_price) / trade['entry'] * 100

        self.trades.append({
            'symbol': trade['symbol'],
            'direction': trade['direction'],
            'pnl_pct': pnl_pct,
            'reason': reason
        })

    def _calc_results(self) -> Dict[str, Any]:
        """Calculate backtest results"""
        if not self.trades:
            return {
                'config': self.config.name,
                'trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_pnl': 0
            }

        winners = [t for t in self.trades if t['pnl_pct'] > 0]
        losers = [t for t in self.trades if t['pnl_pct'] <= 0]

        win_rate = len(winners) / len(self.trades) * 100
        avg_win = sum(t['pnl_pct'] for t in winners) / len(winners) if winners else 0
        avg_loss = abs(sum(t['pnl_pct'] for t in losers) / len(losers)) if losers else 0

        gross_profit = sum(t['pnl_pct'] for t in winners) if winners else 0
        gross_loss = abs(sum(t['pnl_pct'] for t in losers)) if losers else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

        total_pnl = sum(t['pnl_pct'] for t in self.trades)

        return {
            'config': self.config.name,
            'trades': len(self.trades),
            'win_rate': round(win_rate, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'total_pnl': round(total_pnl, 2),
            'winners': len(winners),
            'losers': len(losers)
        }


def run_optimization(num_coins: int = 30, days: int = 7, fast: bool = True):
    """Run full optimization"""
    print("=" * 70)
    print("SMC ULTRA V2 - STRATEGY OPTIMIZER")
    print("=" * 70)

    configs = generate_configs(fast=fast)
    print(f"\nTesting {len(configs)} strategy configurations...")
    print(f"Coins: {num_coins}, Days: {days}")
    print("=" * 70)

    coins = get_top_n_coins(num_coins)
    results = []

    # Load data once (shared across configs)
    print("\n[1/2] Loading data...")
    from data import BybitDataDownloader
    downloader = BybitDataDownloader()

    for i, config in enumerate(configs):
        print(f"\n[2/2] Testing {i+1}/{len(configs)}: {config.name[:50]}...")

        try:
            bt = FlexibleBacktester(config, coins, days)
            bt.load_data()
            result = bt.run()
            results.append(result)

            # Print progress
            if result['trades'] > 0:
                print(f"       Trades: {result['trades']}, WR: {result['win_rate']}%, "
                      f"PF: {result['profit_factor']}, PnL: {result['total_pnl']}%")
        except Exception as e:
            print(f"       ERROR: {e}")
            continue

    # Sort by win rate (primary) and profit factor (secondary)
    results = [r for r in results if r['trades'] >= 5]  # Min 5 trades
    results.sort(key=lambda x: (x['win_rate'], x['profit_factor']), reverse=True)

    # Print results
    print("\n" + "=" * 70)
    print("TOP 20 STRATEGIES (by Win Rate)")
    print("=" * 70)
    print(f"{'Config':<45} {'Trades':>6} {'WR%':>6} {'AvgW':>6} {'AvgL':>6} {'PF':>5} {'PnL%':>7}")
    print("-" * 70)

    for r in results[:20]:
        print(f"{r['config'][:45]:<45} {r['trades']:>6} {r['win_rate']:>6.1f} "
              f"{r['avg_win']:>6.1f} {r['avg_loss']:>6.1f} {r['profit_factor']:>5.1f} {r['total_pnl']:>7.1f}")

    # Best strategy
    if results:
        best = results[0]
        print("\n" + "=" * 70)
        print("🏆 BEST STRATEGY FOUND:")
        print("=" * 70)
        print(f"Config:        {best['config']}")
        print(f"Win Rate:      {best['win_rate']}%")
        print(f"Profit Factor: {best['profit_factor']}")
        print(f"Avg Win:       {best['avg_win']}%")
        print(f"Avg Loss:      {best['avg_loss']}%")
        print(f"Total Trades:  {best['trades']}")
        print(f"Total PnL:     {best['total_pnl']}%")
        print("=" * 70)

    # Save results
    with open('optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to optimization_results.json")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='SMC Strategy Optimizer')
    parser.add_argument('--coins', type=int, default=30, help='Number of coins')
    parser.add_argument('--days', type=int, default=7, help='Backtest days')
    parser.add_argument('--full', action='store_true', help='Run full grid (slow)')

    args = parser.parse_args()

    run_optimization(
        num_coins=args.coins,
        days=args.days,
        fast=not args.full
    )
