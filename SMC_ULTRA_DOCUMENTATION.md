# SMC Ultra Trading System - Dokumentation

## Übersicht

Du hast zwei Pine Scripts erhalten:

1. **smc_ultra_system.pine** - Indikator-Version
   - Zeigt alle SMC-Elemente visuell an
   - Dashboard mit Echtzeit-Scores
   - Für manuelles Trading und visuelle Analyse

2. **smc_ultra_strategy.pine** - Strategie-Version
   - Für automatisches Backtesting
   - Performance-Statistiken
   - Optimierbare Parameter

---

## Installation in TradingView

1. Gehe zu TradingView → Pine Editor
2. Lösche den Standard-Code
3. Kopiere den gesamten Code eines Scripts
4. Klicke "Zur Chart hinzufügen"
5. Für die Strategie-Version: Gehe zu "Strategie-Tester" Tab für Ergebnisse

---

## Strategie-Logik

### Confluence Scoring System

Das System berechnet einen Score von 0-100% basierend auf:

| Faktor | Punkte | Beschreibung |
|--------|--------|--------------|
| HTF Alignment | +25 | Higher Timeframe Trend stimmt überein |
| HTF Neutral | +10 | HTF ist neutral (kein Gegentrend) |
| Liquidity Sweep | +20 | Preis hat Liquidität genommen und reversed |
| Order Block | +20 | Preis ist an einem unmitigated OB |
| Fair Value Gap | +15 | Preis ist in einem ungefüllten FVG |
| MTF Oversold/Overbought | +10 | RSI auf Medium TF extrem |
| Market Structure | +10 | Structure stimmt mit Trade-Richtung |

**Minimum für Signal: 90% (einstellbar)**

### Entry-Bedingungen

**Long Entry:**
- Bull Score >= Minimum Confidence
- Aktuelle Kerze ist bullish (close > open)
- Volatilität im akzeptablen Bereich

**Short Entry:**
- Bear Score >= Minimum Confidence
- Aktuelle Kerze ist bearish (close < open)
- Volatilität im akzeptablen Bereich

### Trade Management

```
Entry → Position eröffnet
   │
   ├─ Bei 50% des TP erreicht:
   │     └─ Trailing Stop aktiviert
   │     └─ SL wird nachgezogen
   │
   ├─ Max Trade Duration erreicht:
   │     └─ Position wird geschlossen
   │
   └─ TP oder SL getroffen:
         └─ Trade beendet
```

---

## Empfohlene Einstellungen pro Asset-Typ

### Bitcoin (BTC)
```
Minimum Confidence: 90%
ATR Length: 14
ATR Multiplier TP: 1.5
ATR Multiplier SL: 1.0
Higher Timeframe: 4H
Medium Timeframe: 15M
Max Trade Duration: 60 bars (auf 5M = 5 Stunden)
```

### Ethereum (ETH)
```
Minimum Confidence: 88%
ATR Length: 14
ATR Multiplier TP: 1.8
ATR Multiplier SL: 1.0
Higher Timeframe: 4H
Medium Timeframe: 15M
Max Trade Duration: 50 bars
```

### Altcoins (SOL, AVAX, etc.)
```
Minimum Confidence: 85%
ATR Length: 10
ATR Multiplier TP: 2.0
ATR Multiplier SL: 1.2
Higher Timeframe: 1H
Medium Timeframe: 5M
Max Trade Duration: 40 bars
```

### High-Volatility Altcoins (DOGE, SHIB, Memes)
```
Minimum Confidence: 92%
ATR Length: 8
ATR Multiplier TP: 2.5
ATR Multiplier SL: 1.5
Higher Timeframe: 1H
Medium Timeframe: 5M
Max Trade Duration: 30 bars
```

---

## Backtesting Anleitung

### Schritt 1: Grundtest
1. Lade die Strategie-Version
2. Wähle einen Coin (z.B. BTCUSDT.P auf Bybit)
3. Wähle Timeframe: 5M oder 15M
4. Schaue die Ergebnisse im Strategy Tester

### Schritt 2: Parameter Optimierung
1. Rechtsklick auf den Indikator → "Einstellungen"
2. Teste verschiedene Confidence Levels:
   - 85% = mehr Trades, niedrigere Winrate
   - 90% = ausgewogen
   - 95% = wenige Trades, höhere Winrate

### Schritt 3: Verschiedene Marktphasen testen
- **Bull Market:** Q4 2023 - Q1 2024
- **Crash:** April 2024
- **Range:** Sommer 2024
- **Recovery:** Q4 2024

Ziel: Parameter finden, die in ALLEN Phasen profitabel sind.

---

## Erwartete Metriken

Basierend auf typischen SMC-Strategien:

| Metrik | Realistisch | Sehr gut | Exzellent |
|--------|-------------|----------|-----------|
| Win Rate | 55-60% | 60-67% | 67-72% |
| Profit Factor | 1.3-1.6 | 1.6-2.0 | 2.0+ |
| Avg Win:Loss | 1:1 | 1.2:1 | 1.5:1 |
| Max Drawdown | 15-20% | 10-15% | <10% |

**Hinweis:** Diese Zahlen sind OHNE Hebel. Mit Hebel skalieren Gewinne UND Verluste.

---

## Nächste Schritte für Bot-Entwicklung

Nach erfolgreichem Backtesting brauchst du:

### 1. Python Bot Framework
```
/crypto_bot
├── config/
│   ├── settings.py      # API Keys, Einstellungen
│   └── coins.json       # Coin-spezifische Parameter
├── core/
│   ├── data_fetcher.py  # Kurs-Daten von Exchange
│   ├── smc_detector.py  # Order Blocks, FVG, Sweeps
│   ├── scorer.py        # Confluence Scoring
│   └── executor.py      # Trade Execution
├── ml/
│   ├── feature_eng.py   # Feature Engineering
│   ├── trainer.py       # Model Training
│   └── predictor.py     # Live Predictions
├── database/
│   └── supabase.py      # Logging & Analytics
└── main.py              # Entry Point
```

### 2. Supabase Schema
```sql
-- Trades Table
CREATE TABLE trades (
    id UUID PRIMARY KEY,
    coin VARCHAR(20),
    direction VARCHAR(5),
    entry_price DECIMAL,
    exit_price DECIMAL,
    entry_time TIMESTAMP,
    exit_time TIMESTAMP,
    pnl_percent DECIMAL,
    confidence_score INT,
    htf_bias VARCHAR(10),
    ob_present BOOLEAN,
    fvg_present BOOLEAN,
    sweep_present BOOLEAN,
    exit_reason VARCHAR(20)
);

-- Signals Table (auch nicht genommene)
CREATE TABLE signals (
    id UUID PRIMARY KEY,
    coin VARCHAR(20),
    timestamp TIMESTAMP,
    direction VARCHAR(5),
    confidence_score INT,
    was_taken BOOLEAN,
    factors JSONB
);
```

### 3. ML Pipeline
1. Sammle 1000+ Trades (auch simulierte)
2. Features: alle Confluence-Faktoren + Markt-Kontext
3. Target: Trade Outcome (Win/Loss)
4. Model: XGBoost oder LightGBM (gut für tabellarische Daten)
5. Output: Probability 0-100%

---

## Wichtige Warnungen

⚠️ **Slippage & Fees**
- Die Strategie rechnet mit 0.075% Gebühren pro Trade
- Bei kleinen Moves können Fees die Edge eliminieren
- Teste mit höheren Fee-Einstellungen (0.1%) für konservative Schätzung

⚠️ **Overfitting**
- Wenn eine Einstellung nur auf einem Coin funktioniert = Overfitting
- Teste IMMER auf mehreren Coins und Zeiträumen

⚠️ **Live Trading Unterschiede**
- Backtests sind immer optimistischer als Live
- Rechne mit 20-30% schlechterer Performance live
- Starte mit kleinen Positionen

⚠️ **High Leverage**
- 50x+ Hebel = eine schlechte Trade-Serie kann dich liquidieren
- Empfehlung: Starte mit 10-20x, auch wenn Backtests mehr erlauben

---

## Support & Weiterentwicklung

Wenn du bereit bist für den Python-Bot, sag Bescheid. Ich kann:

1. Das komplette Bot-Framework bauen
2. Supabase Integration einrichten
3. ML-Pipeline implementieren
4. Railway Deployment konfigurieren

Aber erst: Validiere die Strategie mit dem Pine Script Backtester!
