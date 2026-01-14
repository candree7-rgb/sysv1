# SMC Ultra V2 - Ultimate Smart Money Concepts Trading Bot

Vollautomatischer Krypto-Trading-Bot mit:
- Smart Money Concepts (Order Blocks, FVG, Liquidity Sweeps)
- Multi-Timeframe Analyse
- Machine Learning Confidence Scoring
- Dynamisches Trade Management
- Automatische Regime-Erkennung

## Quick Start

```bash
# 1. Dependencies installieren
pip install -r requirements.txt

# 2. Daten herunterladen (Top 100 Coins, 6 Monate)
python main.py download --coins 100 --days 180

# 3. Backtest ausführen
python main.py backtest --days 90 --coins 50 --save

# 4. ML Model trainieren (optional)
python main.py train --trades-file backtest_results/trades.json

# 5. Paper Trading starten
python main.py live --paper
```

## Projektstruktur

```
smc_ultra_v2/
├── config/          # Konfiguration
├── data/            # Data Layer (Download, Cache)
├── analysis/        # Regime Detection, MTF Analysis
├── smc/             # SMC Detection (OB, FVG, Sweeps)
├── ml/              # Machine Learning
├── strategy/        # Signal Generation, Trade Management
├── backtest/        # Backtesting Engine
├── live/            # Live Trading (Bybit)
└── main.py          # Entry Point
```

## Features

### Multi-Timeframe Analyse
- HTF (1H): Trend-Richtung und Bias
- MTF (15m): Entry Zones (OB, FVG)
- LTF (1m): Precision Entry

### SMC Konzepte
- Order Blocks mit Strength-Rating
- Fair Value Gaps mit Fill-Tracking
- Liquidity Sweeps und Inducement
- Market Structure (BOS, CHoCH)

### Machine Learning
- XGBoost Confidence Scoring
- Recency Weighting (neuere Trades höher gewichtet)
- Feature Engineering mit 50+ Features
- Coin Pre-Filter für beste Tradability

### Dynamisches Management
- Break-Even bei 30% TP
- Trailing Stop bei 50% TP
- Zeit-basierter Exit
- Momentum-Exit bei Warnsignalen

### Regime Detection
- Trending (Long/Short only)
- Ranging (Mean Reversion)
- Choppy (Kein Trading!)
- High Volatility (Reduzierter Hebel)

## Konfiguration

Kopiere `.env.example` zu `.env` und passe an:

```bash
BYBIT_API_KEY=dein_api_key
BYBIT_API_SECRET=dein_api_secret
BYBIT_TESTNET=true

MIN_CONFIDENCE=85
MAX_LEVERAGE=50
RISK_LEVEL=moderate
```

## Erwartete Performance

| Metrik | Ziel |
|--------|------|
| Win Rate | 65-75% |
| Profit Factor | 1.5-2.5 |
| Max Drawdown | <20% |
| RR Ratio | ~1:1 |

## Wichtig

- Immer erst im Paper Trading testen!
- Nur mit Geld traden, das du verlieren kannst
- Keine Garantie für Gewinne
- Vergangene Performance ≠ zukünftige Ergebnisse
