"""
SMC Strategy Comparison
=======================
Testet die ECHTEN SMC Setups gegeneinander um das beste zu finden.

Setups:
1. SWEEP_OB: Liquidity Sweep + Order Block Retest
2. SWEEP_FVG: Liquidity Sweep + FVG Fill
3. SWEEP_OB_FVG: Triple Confluence (beste aber selten)
4. OB_ONLY: Nur Order Block (Baseline)
5. FVG_ONLY: Nur FVG Fill (Baseline)
"""

import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class SetupType(Enum):
    SWEEP_OB = "sweep_ob"           # Sweep + Order Block
    SWEEP_FVG = "sweep_fvg"         # Sweep + FVG
    SWEEP_OB_FVG = "sweep_ob_fvg"   # Triple Confluence
    OB_ONLY = "ob_only"             # Just OB
    FVG_ONLY = "fvg_only"           # Just FVG


@dataclass
class Trade:
    setup: SetupType
    symbol: str
    direction: str
    entry: float
    sl: float
    tp: float
    entry_time: datetime
    exit_time: datetime = None
    exit_price: float = None
    pnl_pct: float = None
    result: str = None  # 'win', 'loss'


class SMCStrategyTester:
    """Tests specific SMC setups"""

    def __init__(self, symbols: List[str], days: int = 14):
        self.symbols = symbols
        self.days = days
        self.data = {}
        self.trades_by_setup: Dict[SetupType, List[Trade]] = {s: [] for s in SetupType}

    def load_data(self):
        """Load and prepare data"""
        import pandas as pd
        from data import BybitDataDownloader
        from smc import OrderBlockDetector, FVGDetector, LiquidityDetector

        print("Loading data...", flush=True)

        dl = BybitDataDownloader()
        ob_det = OrderBlockDetector()
        fvg_det = FVGDetector()
        liq_det = LiquidityDetector()

        end = datetime.now()
        start = end - timedelta(days=self.days + 5)

        for i, symbol in enumerate(self.symbols):
            print(f"  [{i+1}/{len(self.symbols)}] {symbol}...", end="", flush=True)

            df = dl.load_or_download(symbol, "5", self.days + 10)
            if df is None or len(df) < 200:
                print(" SKIP", flush=True)
                continue

            # Filter date range
            mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
            df = df[mask].reset_index(drop=True)

            if len(df) < 100:
                print(" SKIP (short)", flush=True)
                continue

            # Calculate indicators
            # ATR
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(14).mean()

            # EMAs for trend
            df['ema20'] = df['close'].ewm(span=20).mean()
            df['ema50'] = df['close'].ewm(span=50).mean()

            self.data[symbol] = df

            # Detect SMC structures
            atr = df['atr']
            self.data[f"{symbol}_obs"] = ob_det.detect(df, atr)
            self.data[f"{symbol}_fvgs"] = fvg_det.detect(df)
            self.data[f"{symbol}_sweeps"] = liq_det.find_sweeps(df)

            print(f" OK ({len(df)} bars)", flush=True)

        print(f"\nLoaded {len([k for k in self.data.keys() if not k.endswith(('_obs', '_fvgs', '_sweeps'))])} symbols")

    def run_all_strategies(self):
        """Run all strategy variations"""
        print("\nTesting strategies...", flush=True)

        for symbol in [k for k in self.data.keys() if not k.endswith(('_obs', '_fvgs', '_sweeps'))]:
            df = self.data[symbol]
            obs = self.data.get(f"{symbol}_obs", [])
            fvgs = self.data.get(f"{symbol}_fvgs", [])
            sweeps = self.data.get(f"{symbol}_sweeps", [])

            self._test_symbol(symbol, df, obs, fvgs, sweeps)

        self._print_results()

    def _test_symbol(self, symbol: str, df, obs: list, fvgs: list, sweeps: list):
        """Test all setups on one symbol"""

        active_trades = {s: None for s in SetupType}

        for idx in range(50, len(df)):
            candle = df.iloc[idx]
            ts = candle['timestamp']
            price = candle['close']
            atr = candle['atr']

            if pd.isna(atr) or atr <= 0:
                continue

            # Check and close active trades
            for setup_type in SetupType:
                trade = active_trades[setup_type]
                if trade:
                    closed = self._check_trade_exit(trade, candle)
                    if closed:
                        self.trades_by_setup[setup_type].append(trade)
                        active_trades[setup_type] = None

            # Get trend direction
            if candle['close'] > candle['ema20'] > candle['ema50']:
                trend = 'long'
            elif candle['close'] < candle['ema20'] < candle['ema50']:
                trend = 'short'
            else:
                continue  # No clear trend

            # Get recent structures
            recent_sweeps = [s for s in sweeps if ts - timedelta(hours=2) <= s.timestamp <= ts]
            active_obs = [ob for ob in obs if ob.timestamp < ts and not ob.is_mitigated]
            active_fvgs = [fvg for fvg in fvgs if fvg.timestamp < ts and not fvg.is_filled]

            # Check for sweep in our direction
            has_sweep = False
            for sweep in recent_sweeps:
                if (trend == 'long' and sweep.is_bullish) or (trend == 'short' and not sweep.is_bullish):
                    has_sweep = True
                    break

            # Check for OB nearby
            near_ob = False
            for ob in active_obs:
                dist = abs(price - ob.mid) / price * 100
                if dist < 1.5:  # Within 1.5%
                    if (trend == 'long' and ob.is_bullish) or (trend == 'short' and not ob.is_bullish):
                        near_ob = True
                        break

            # Check for FVG
            in_fvg = False
            for fvg in active_fvgs:
                if fvg.bottom <= price <= fvg.top:
                    if (trend == 'long' and fvg.is_bullish) or (trend == 'short' and not fvg.is_bullish):
                        in_fvg = True
                        break

            # === SETUP DETECTION ===

            # 1. SWEEP + OB (Classic SMC)
            if has_sweep and near_ob and active_trades[SetupType.SWEEP_OB] is None:
                active_trades[SetupType.SWEEP_OB] = self._create_trade(
                    SetupType.SWEEP_OB, symbol, trend, price, atr, ts
                )

            # 2. SWEEP + FVG
            if has_sweep and in_fvg and active_trades[SetupType.SWEEP_FVG] is None:
                active_trades[SetupType.SWEEP_FVG] = self._create_trade(
                    SetupType.SWEEP_FVG, symbol, trend, price, atr, ts
                )

            # 3. SWEEP + OB + FVG (Triple)
            if has_sweep and near_ob and in_fvg and active_trades[SetupType.SWEEP_OB_FVG] is None:
                active_trades[SetupType.SWEEP_OB_FVG] = self._create_trade(
                    SetupType.SWEEP_OB_FVG, symbol, trend, price, atr, ts
                )

            # 4. OB Only (Baseline)
            if near_ob and active_trades[SetupType.OB_ONLY] is None:
                active_trades[SetupType.OB_ONLY] = self._create_trade(
                    SetupType.OB_ONLY, symbol, trend, price, atr, ts
                )

            # 5. FVG Only (Baseline)
            if in_fvg and active_trades[SetupType.FVG_ONLY] is None:
                active_trades[SetupType.FVG_ONLY] = self._create_trade(
                    SetupType.FVG_ONLY, symbol, trend, price, atr, ts
                )

    def _create_trade(self, setup: SetupType, symbol: str, direction: str,
                      price: float, atr: float, ts: datetime) -> Trade:
        """Create a trade with 1:1.5 RR"""
        sl_mult = 1.0
        tp_mult = 1.5  # 1:1.5 RR

        if direction == 'long':
            entry = price * 1.0003  # Small slippage
            sl = entry - atr * sl_mult
            tp = entry + atr * tp_mult
        else:
            entry = price * 0.9997
            sl = entry + atr * sl_mult
            tp = entry - atr * tp_mult

        return Trade(
            setup=setup,
            symbol=symbol,
            direction=direction,
            entry=entry,
            sl=sl,
            tp=tp,
            entry_time=ts
        )

    def _check_trade_exit(self, trade: Trade, candle) -> bool:
        """Check if trade should exit"""
        if trade.direction == 'long':
            if candle['low'] <= trade.sl:
                trade.exit_price = trade.sl
                trade.result = 'loss'
                trade.pnl_pct = (trade.sl - trade.entry) / trade.entry * 100
                trade.exit_time = candle['timestamp']
                return True
            elif candle['high'] >= trade.tp:
                trade.exit_price = trade.tp
                trade.result = 'win'
                trade.pnl_pct = (trade.tp - trade.entry) / trade.entry * 100
                trade.exit_time = candle['timestamp']
                return True
        else:
            if candle['high'] >= trade.sl:
                trade.exit_price = trade.sl
                trade.result = 'loss'
                trade.pnl_pct = (trade.entry - trade.sl) / trade.entry * 100
                trade.exit_time = candle['timestamp']
                return True
            elif candle['low'] <= trade.tp:
                trade.exit_price = trade.tp
                trade.result = 'win'
                trade.pnl_pct = (trade.entry - trade.tp) / trade.entry * 100
                trade.exit_time = candle['timestamp']
                return True

        return False

    def _print_results(self):
        """Print comparison results"""
        print("\n" + "=" * 80)
        print("SMC STRATEGY COMPARISON RESULTS")
        print("=" * 80)
        print(f"{'Setup':<20} {'Trades':>8} {'Winners':>8} {'Losers':>8} {'WR%':>8} {'AvgWin':>8} {'AvgLoss':>8} {'PF':>8}")
        print("-" * 80)

        results = []

        for setup_type in SetupType:
            trades = self.trades_by_setup[setup_type]

            if not trades:
                print(f"{setup_type.value:<20} {'NO TRADES':>8}")
                continue

            winners = [t for t in trades if t.result == 'win']
            losers = [t for t in trades if t.result == 'loss']

            total = len(trades)
            win_count = len(winners)
            loss_count = len(losers)
            win_rate = win_count / total * 100 if total > 0 else 0

            avg_win = sum(t.pnl_pct for t in winners) / len(winners) if winners else 0
            avg_loss = abs(sum(t.pnl_pct for t in losers) / len(losers)) if losers else 0

            gross_win = sum(t.pnl_pct for t in winners) if winners else 0
            gross_loss = abs(sum(t.pnl_pct for t in losers)) if losers else 1
            pf = gross_win / gross_loss if gross_loss > 0 else gross_win

            print(f"{setup_type.value:<20} {total:>8} {win_count:>8} {loss_count:>8} "
                  f"{win_rate:>7.1f}% {avg_win:>7.2f}% {avg_loss:>7.2f}% {pf:>7.2f}")

            results.append({
                'setup': setup_type.value,
                'trades': total,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': pf,
                'total_pnl': gross_win - gross_loss
            })

        # Sort by win rate
        results.sort(key=lambda x: x['win_rate'], reverse=True)

        if results:
            best = results[0]
            print("\n" + "=" * 80)
            print(f"🏆 BEST SETUP: {best['setup'].upper()}")
            print(f"   Win Rate: {best['win_rate']:.1f}%")
            print(f"   Profit Factor: {best['profit_factor']:.2f}")
            print(f"   Trades: {best['trades']}")
            print("=" * 80)

        # Per-coin breakdown for best setup
        if results:
            best_setup = SetupType(results[0]['setup'])
            print(f"\n{results[0]['setup'].upper()} - Per Coin Breakdown:")
            print("-" * 50)

            coin_stats = {}
            for trade in self.trades_by_setup[best_setup]:
                if trade.symbol not in coin_stats:
                    coin_stats[trade.symbol] = {'wins': 0, 'losses': 0, 'pnl': 0}

                if trade.result == 'win':
                    coin_stats[trade.symbol]['wins'] += 1
                else:
                    coin_stats[trade.symbol]['losses'] += 1
                coin_stats[trade.symbol]['pnl'] += trade.pnl_pct

            for symbol, stats in sorted(coin_stats.items(), key=lambda x: -x[1]['pnl']):
                total = stats['wins'] + stats['losses']
                wr = stats['wins'] / total * 100 if total > 0 else 0
                print(f"  {symbol:<12} {total:>3} trades, {wr:>5.1f}% WR, {stats['pnl']:>+6.2f}% PnL")


def run_comparison(num_coins: int = 30, days: int = 14):
    """Run the strategy comparison"""
    from config.coins import get_top_n_coins

    coins = get_top_n_coins(num_coins)

    # Skip problematic coins
    skip = {'APEUSDT', 'MATICUSDT', 'OCEANUSDT', 'EOSUSDT'}
    coins = [c for c in coins if c not in skip]

    tester = SMCStrategyTester(coins, days)
    tester.load_data()
    tester.run_all_strategies()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--coins', type=int, default=30)
    parser.add_argument('--days', type=int, default=14)
    args = parser.parse_args()

    run_comparison(args.coins, args.days)
