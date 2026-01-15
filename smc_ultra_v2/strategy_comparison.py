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
    SWEEP_FVG = "sweep_fvg"         # Sweep + FVG (best combo)
    FVG_ONLY = "fvg_only"           # Just FVG (highest WR)


# Trading costs (REALISTIC)
TAKER_FEE_PCT = 0.055    # Bybit taker fee per side
MAKER_FEE_PCT = 0.02     # Bybit maker fee per side
SLIPPAGE_PCT = 0.02      # Realistic slippage

# Dynamic leverage settings
RISK_PER_TRADE_PCT = 2.0  # Risk 2% of account per trade
MAX_LEVERAGE = 50         # Cap leverage at 50x
MIN_LEVERAGE = 5          # Minimum leverage


@dataclass
class Trade:
    setup: SetupType
    symbol: str
    direction: str
    entry: float
    sl: float
    tp: float
    entry_time: datetime
    leverage: int = 10           # Dynamic leverage
    sl_pct: float = 0.0          # SL distance in %
    exit_time: datetime = None
    exit_price: float = None
    pnl_pct: float = None        # RAW pnl (without leverage)
    pnl_leveraged: float = None  # Leveraged pnl
    result: str = None           # 'win', 'loss'


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

            # === SETUP DETECTION (FVG strategies only) ===

            # 1. SWEEP + FVG (Premium Setup)
            if has_sweep and in_fvg and active_trades[SetupType.SWEEP_FVG] is None:
                active_trades[SetupType.SWEEP_FVG] = self._create_trade(
                    SetupType.SWEEP_FVG, symbol, trend, price, atr, ts
                )

            # 2. FVG Only (High WR Setup)
            if in_fvg and active_trades[SetupType.FVG_ONLY] is None:
                active_trades[SetupType.FVG_ONLY] = self._create_trade(
                    SetupType.FVG_ONLY, symbol, trend, price, atr, ts
                )

    def _create_trade(self, setup: SetupType, symbol: str, direction: str,
                      price: float, atr: float, ts: datetime) -> Trade:
        """Create a trade with 1:1.5 RR and dynamic leverage"""
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

        # Calculate dynamic leverage based on SL distance
        sl_pct = abs(entry - sl) / entry * 100

        # Leverage formula: risk_per_trade / sl_pct
        # Example: 2% risk / 0.5% SL = 4x leverage (to risk 2% of account)
        if sl_pct > 0:
            calculated_lev = RISK_PER_TRADE_PCT / sl_pct
            leverage = min(int(calculated_lev), MAX_LEVERAGE)
            leverage = max(leverage, MIN_LEVERAGE)
        else:
            leverage = MIN_LEVERAGE

        return Trade(
            setup=setup,
            symbol=symbol,
            direction=direction,
            entry=entry,
            sl=sl,
            tp=tp,
            entry_time=ts,
            leverage=leverage,
            sl_pct=sl_pct
        )

    def _check_trade_exit(self, trade: Trade, candle) -> bool:
        """Check if trade should exit - WITH REALISTIC FEES & LEVERAGE!"""
        # Fee structure:
        # - Entry: Limit order = MAKER_FEE (0.02%)
        # - TP exit: Limit order = MAKER_FEE (0.02%)
        # - SL exit: Market order = TAKER_FEE (0.055%)
        # Note: Fees apply to leveraged position size
        fee_win = (MAKER_FEE_PCT * 2) + (SLIPPAGE_PCT * 2)    # ~0.08% for TP hit
        fee_loss = MAKER_FEE_PCT + TAKER_FEE_PCT + (SLIPPAGE_PCT * 2)  # ~0.115% for SL hit

        if trade.direction == 'long':
            if candle['low'] <= trade.sl:
                trade.exit_price = trade.sl
                trade.result = 'loss'
                gross_pnl = (trade.sl - trade.entry) / trade.entry * 100
                trade.pnl_pct = gross_pnl - fee_loss  # Raw PnL after fees
                trade.pnl_leveraged = trade.pnl_pct * trade.leverage  # Leveraged PnL
                trade.exit_time = candle['timestamp']
                return True
            elif candle['high'] >= trade.tp:
                trade.exit_price = trade.tp
                trade.result = 'win'
                gross_pnl = (trade.tp - trade.entry) / trade.entry * 100
                trade.pnl_pct = gross_pnl - fee_win
                trade.pnl_leveraged = trade.pnl_pct * trade.leverage
                trade.exit_time = candle['timestamp']
                return True
        else:
            if candle['high'] >= trade.sl:
                trade.exit_price = trade.sl
                trade.result = 'loss'
                gross_pnl = (trade.entry - trade.sl) / trade.entry * 100
                trade.pnl_pct = gross_pnl - fee_loss
                trade.pnl_leveraged = trade.pnl_pct * trade.leverage
                trade.exit_time = candle['timestamp']
                return True
            elif candle['low'] <= trade.tp:
                trade.exit_price = trade.tp
                trade.result = 'win'
                gross_pnl = (trade.entry - trade.tp) / trade.entry * 100
                trade.pnl_pct = gross_pnl - fee_win
                trade.pnl_leveraged = trade.pnl_pct * trade.leverage
                trade.exit_time = candle['timestamp']
                return True

        return False

    def _print_results(self):
        """Print comparison results with leverage stats"""
        print("\n" + "=" * 100)
        print("SMC STRATEGY COMPARISON RESULTS (with Dynamic Leverage)")
        print("=" * 100)
        print(f"{'Setup':<15} {'Trades':>7} {'Win':>5} {'Loss':>5} {'WR%':>7} {'AvgLev':>7} {'AvgWin':>9} {'AvgLoss':>9} {'TotalPnL':>10}")
        print("-" * 100)

        results = []

        for setup_type in SetupType:
            trades = self.trades_by_setup[setup_type]

            if not trades:
                print(f"{setup_type.value:<15} {'NO TRADES':>7}")
                continue

            winners = [t for t in trades if t.result == 'win']
            losers = [t for t in trades if t.result == 'loss']

            total = len(trades)
            win_count = len(winners)
            loss_count = len(losers)
            win_rate = win_count / total * 100 if total > 0 else 0

            # Average leverage used
            avg_leverage = sum(t.leverage for t in trades) / len(trades) if trades else 0

            # Leveraged PnL stats
            avg_win_lev = sum(t.pnl_leveraged for t in winners) / len(winners) if winners else 0
            avg_loss_lev = abs(sum(t.pnl_leveraged for t in losers) / len(losers)) if losers else 0

            gross_win_lev = sum(t.pnl_leveraged for t in winners) if winners else 0
            gross_loss_lev = abs(sum(t.pnl_leveraged for t in losers)) if losers else 1
            pf = gross_win_lev / gross_loss_lev if gross_loss_lev > 0 else gross_win_lev

            total_pnl_lev = gross_win_lev - gross_loss_lev

            print(f"{setup_type.value:<15} {total:>7} {win_count:>5} {loss_count:>5} "
                  f"{win_rate:>6.1f}% {avg_leverage:>6.1f}x {avg_win_lev:>+8.2f}% {avg_loss_lev:>8.2f}% {total_pnl_lev:>+9.2f}%")

            results.append({
                'setup': setup_type.value,
                'trades': total,
                'win_rate': win_rate,
                'avg_leverage': avg_leverage,
                'avg_win': avg_win_lev,
                'avg_loss': avg_loss_lev,
                'profit_factor': pf,
                'total_pnl': total_pnl_lev
            })

        # Sort by total PnL (with leverage)
        results.sort(key=lambda x: x['total_pnl'], reverse=True)

        if results:
            best = results[0]
            print("\n" + "=" * 100)
            print(f"BEST SETUP: {best['setup'].upper()}")
            print(f"   Win Rate: {best['win_rate']:.1f}%")
            print(f"   Avg Leverage: {best['avg_leverage']:.1f}x")
            print(f"   Profit Factor: {best['profit_factor']:.2f}")
            print(f"   Total PnL (leveraged): {best['total_pnl']:+.2f}%")
            print(f"   Trades: {best['trades']}")
            print("=" * 100)

        # Per-coin breakdown for best setup
        if results:
            best_setup = SetupType(results[0]['setup'])
            print(f"\n{results[0]['setup'].upper()} - Per Coin Breakdown (Leveraged PnL):")
            print("-" * 60)

            coin_stats = {}
            for trade in self.trades_by_setup[best_setup]:
                if trade.symbol not in coin_stats:
                    coin_stats[trade.symbol] = {'wins': 0, 'losses': 0, 'pnl': 0, 'avg_lev': []}

                if trade.result == 'win':
                    coin_stats[trade.symbol]['wins'] += 1
                else:
                    coin_stats[trade.symbol]['losses'] += 1
                coin_stats[trade.symbol]['pnl'] += trade.pnl_leveraged
                coin_stats[trade.symbol]['avg_lev'].append(trade.leverage)

            for symbol, stats in sorted(coin_stats.items(), key=lambda x: -x[1]['pnl']):
                total = stats['wins'] + stats['losses']
                wr = stats['wins'] / total * 100 if total > 0 else 0
                avg_lev = sum(stats['avg_lev']) / len(stats['avg_lev']) if stats['avg_lev'] else 0
                print(f"  {symbol:<12} {total:>3} trades, {wr:>5.1f}% WR, {avg_lev:>4.1f}x lev, {stats['pnl']:>+8.2f}% PnL")

        # Simulate account growth with $10,000
        # The leverage is calculated as: leverage = RISK_PER_TRADE / sl_pct
        # This means pnl_leveraged already represents the % of account risked/gained
        # Win: ~+3% of account (2% risk × 1.5 RR)
        # Loss: ~-2% of account (the risk)
        if results:
            print(f"\n{'='*70}")
            print(f"SIMULATED ACCOUNT GROWTH ($10,000 start, {RISK_PER_TRADE_PCT}% risk per trade)")
            print("="*70)

            for setup_type in SetupType:
                trades = self.trades_by_setup[setup_type]
                if not trades:
                    continue

                equity = 10000.0
                peak = 10000.0
                max_dd = 0.0

                for trade in sorted(trades, key=lambda t: t.entry_time):
                    # pnl_leveraged IS the account % change
                    # Because leverage = risk / sl_pct, the math works out:
                    # - Loss: sl_pct × leverage = sl_pct × (risk/sl_pct) = risk = ~2%
                    # - Win: tp_pct × leverage = tp_pct × (risk/sl_pct) = risk × RR = ~3%
                    pnl_usd = equity * (trade.pnl_leveraged / 100)
                    equity += pnl_usd
                    peak = max(peak, equity)
                    dd = (peak - equity) / peak * 100
                    max_dd = max(max_dd, dd)

                return_pct = (equity - 10000) / 10000 * 100
                print(f"  {setup_type.value:<15} $10,000 -> ${equity:>10,.2f} ({return_pct:>+7.2f}%)  MaxDD: {max_dd:.1f}%")


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
