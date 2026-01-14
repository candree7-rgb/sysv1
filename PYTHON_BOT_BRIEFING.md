# SMC Ultra Trading Bot - Entwicklungs-Briefing

## Projekt-Übersicht

Ziel: Automatisierter Krypto-Trading-Bot basierend auf Smart Money Concepts (SMC) mit Machine Learning Confidence Scoring.

**Tech Stack:**
- Python 3.11+
- Bybit API (primär), Binance (optional)
- Supabase (PostgreSQL) für Logging & Analytics
- Railway für Deployment
- Optional: TensorFlow/XGBoost für ML

**Basis-Strategie:** Bereits implementiert in Pine Script (siehe `smc_ultra_system.pine` und `smc_ultra_strategy.pine`)

---

## Teil 1: Strategie-Logik (bereits definiert)

### 1.1 Smart Money Concepts Detection

#### Order Blocks (OB)
```python
# Bullish Order Block:
# - Letzter bearisher Candle VOR einem starken bullischen Impuls
# - Impuls = Kerze mit (close - open) > 1.5 * ATR
# - OB Zone = High und Low dieser bearischen Kerze

# Bearish Order Block:
# - Letzter bullischer Candle VOR einem starken bearischen Impuls
# - Inverse Logik

# OB Mitigation:
# - Bullish OB ist "mitigated" wenn Preis 50% in die Zone eindringt
# - Danach ist der OB nicht mehr gültig für neue Trades
```

#### Fair Value Gaps (FVG)
```python
# Bullish FVG:
# - Lücke zwischen Candle[2].high und Candle[0].low
# - Bedingung: low > high[2]
# - Mindestgröße: 0.1% des Preises

# Bearish FVG:
# - Lücke zwischen Candle[2].low und Candle[0].high
# - Bedingung: high < low[2]

# FVG Fill:
# - Bullish FVG gefüllt wenn Preis in die Zone fällt
# - Danach nicht mehr gültig
```

#### Liquidity Sweeps
```python
# Bullish Sweep:
# - Preis unterschreitet ein Recent Swing Low
# - Schließt dann ÜBER diesem Level
# - Kerze ist bullish (close > open)
# = Zeichen dass Liquidität genommen wurde und Reversal kommt

# Bearish Sweep:
# - Preis überschreitet ein Recent Swing High
# - Schließt dann UNTER diesem Level
# - Kerze ist bearish
```

### 1.2 Multi-Timeframe Analysis

```python
# Higher Timeframe (HTF) = 4H oder 1H
# - Bestimmt den BIAS (Bullish/Bearish/Neutral)
# - Bullish: Close > EMA20 > EMA50
# - Bearish: Close < EMA20 < EMA50
# - Neutral: Alles andere

# Medium Timeframe (MTF) = 15M oder 5M
# - RSI für Oversold/Overbought
# - Setup Confirmation

# Lower Timeframe (LTF) = 1M oder 5M
# - Entry Trigger
# - Präzises Timing
```

### 1.3 Confluence Scoring System

```python
def calculate_bullish_score(data):
    score = 0
    
    # HTF Alignment
    if htf_bullish:
        score += 25
    elif htf_neutral:
        score += 10
    # htf_bearish = 0 Punkte
    
    # Liquidity Sweep
    if bullish_sweep_detected:
        score += 20
    
    # Near Order Block
    if price_at_bullish_ob:
        score += 20
    
    # In Fair Value Gap
    if price_in_bullish_fvg:
        score += 15
    
    # MTF Oversold
    if mtf_rsi < 30:
        score += 10
    
    # Market Structure
    if is_bullish_structure:
        score += 10
    
    return min(score, 100)

# Minimum Score für Entry: 90% (konfigurierbar)
```

### 1.4 Dynamic Targets (ATR-basiert)

```python
def calculate_targets(current_price, atr, coin_config):
    """
    Berechnet dynamische TP/SL basierend auf Volatilität
    """
    # Target und Stop basierend auf ATR
    target_move = atr * coin_config['atr_multiplier_tp']  # z.B. 1.5
    stop_move = atr * coin_config['atr_multiplier_sl']    # z.B. 1.0
    
    # Berechne benötigten Hebel für Ziel-Profit
    target_move_percent = (target_move / current_price) * 100
    target_profit_percent = 10  # z.B. 10% Gewinn pro Trade
    
    required_leverage = target_profit_percent / target_move_percent
    practical_leverage = min(required_leverage, coin_config['max_leverage'])
    
    # Runde auf praktische Werte
    if practical_leverage <= 10:
        practical_leverage = round(practical_leverage)
    elif practical_leverage <= 25:
        practical_leverage = round(practical_leverage / 5) * 5
    else:
        practical_leverage = round(practical_leverage / 10) * 10
    
    return {
        'take_profit': target_move,
        'stop_loss': stop_move,
        'leverage': practical_leverage
    }
```

### 1.5 Trade Management

```python
class TradeManager:
    def __init__(self, trade):
        self.entry_price = trade.entry_price
        self.take_profit = trade.take_profit
        self.stop_loss = trade.stop_loss
        self.trailing_activated = False
        self.entry_time = trade.entry_time
        self.max_duration_minutes = 60  # Konfigurierbar
    
    def update(self, current_price, current_time):
        """
        Wird bei jedem Tick aufgerufen
        """
        profit_percent = self.calculate_profit_percent(current_price)
        target_percent = self.calculate_target_percent()
        progress = profit_percent / target_percent * 100
        
        # Bei 50% des Targets: Trailing Stop aktivieren
        if progress >= 50 and not self.trailing_activated:
            self.trailing_activated = True
            self.stop_loss = self.entry_price  # Break Even
        
        # Trailing Stop nachziehen
        if self.trailing_activated:
            new_sl = self.calculate_trailing_sl(current_price)
            if self.is_long and new_sl > self.stop_loss:
                self.stop_loss = new_sl
            elif not self.is_long and new_sl < self.stop_loss:
                self.stop_loss = new_sl
        
        # Zeit-basierter Exit
        duration = (current_time - self.entry_time).total_minutes()
        if duration >= self.max_duration_minutes:
            if profit_percent > 0:
                return 'CLOSE_PROFIT'
            elif profit_percent > -0.3 * target_percent:
                return 'CLOSE_SMALL_LOSS'
        
        # Check TP/SL
        if self.check_take_profit(current_price):
            return 'TAKE_PROFIT'
        if self.check_stop_loss(current_price):
            return 'STOP_LOSS'
        
        return 'HOLD'
```

---

## Teil 2: Bot-Architektur

### 2.1 Projektstruktur

```
smc_ultra_bot/
├── config/
│   ├── __init__.py
│   ├── settings.py          # Umgebungsvariablen, API Keys
│   ├── coins.py              # Coin-spezifische Parameter
│   └── logging_config.py     # Logging Setup
│
├── core/
│   ├── __init__.py
│   ├── data_fetcher.py       # Kurs-Daten von Exchange
│   ├── indicators.py         # ATR, RSI, EMA Berechnungen
│   ├── smc_detector.py       # Order Blocks, FVG, Sweeps
│   ├── mtf_analyzer.py       # Multi-Timeframe Analysis
│   ├── scorer.py             # Confluence Scoring
│   ├── signal_generator.py   # Signal-Generierung
│   └── trade_manager.py      # Position Management
│
├── exchange/
│   ├── __init__.py
│   ├── base.py               # Abstract Base Class
│   ├── bybit.py              # Bybit Implementation
│   └── binance.py            # Binance Implementation (optional)
│
├── ml/
│   ├── __init__.py
│   ├── features.py           # Feature Engineering
│   ├── trainer.py            # Model Training
│   ├── predictor.py          # Live Predictions
│   └── models/               # Gespeicherte Models
│
├── database/
│   ├── __init__.py
│   ├── supabase_client.py    # Supabase Connection
│   ├── models.py             # Datenbank-Modelle
│   └── queries.py            # SQL Queries
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py            # Utility Functions
│   └── notifications.py      # Discord/Telegram Alerts
│
├── tests/
│   ├── __init__.py
│   ├── test_smc_detector.py
│   ├── test_scorer.py
│   └── test_backtester.py
│
├── main.py                   # Entry Point
├── backtester.py             # Backtesting Engine
├── requirements.txt
├── Dockerfile
├── railway.toml
└── README.md
```

### 2.2 Kern-Module Spezifikationen

#### data_fetcher.py
```python
class DataFetcher:
    """
    Holt Kurs-Daten von der Exchange
    """
    def __init__(self, exchange_client):
        self.exchange = exchange_client
        self.cache = {}
    
    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """
        Returns DataFrame mit columns: timestamp, open, high, low, close, volume
        """
        pass
    
    async def get_multi_timeframe_data(self, symbol: str) -> dict:
        """
        Holt Daten für alle benötigten Timeframes gleichzeitig
        Returns: {
            '4h': DataFrame,
            '15m': DataFrame,
            '5m': DataFrame,
            '1m': DataFrame
        }
        """
        pass
    
    async def stream_realtime(self, symbols: list, callback):
        """
        WebSocket Stream für Echtzeit-Daten
        """
        pass
```

#### smc_detector.py
```python
class SMCDetector:
    """
    Erkennt Smart Money Concepts
    """
    def __init__(self, config):
        self.ob_lookback = config.get('ob_lookback', 10)
        self.fvg_min_size = config.get('fvg_min_size', 0.001)
        self.swing_length = config.get('swing_length', 5)
    
    def detect_order_blocks(self, df: pd.DataFrame) -> list[OrderBlock]:
        """
        Findet alle aktiven Order Blocks
        """
        pass
    
    def detect_fair_value_gaps(self, df: pd.DataFrame) -> list[FVG]:
        """
        Findet alle ungefüllten FVGs
        """
        pass
    
    def detect_liquidity_sweeps(self, df: pd.DataFrame) -> dict:
        """
        Returns: {'bullish': bool, 'bearish': bool}
        """
        pass
    
    def detect_market_structure(self, df: pd.DataFrame) -> str:
        """
        Returns: 'bullish', 'bearish', or 'neutral'
        """
        pass
    
    def get_swing_points(self, df: pd.DataFrame) -> dict:
        """
        Returns: {'swing_highs': [...], 'swing_lows': [...]}
        """
        pass
```

#### scorer.py
```python
class ConfluenceScorer:
    """
    Berechnet Confluence Scores
    """
    def __init__(self, weights: dict = None):
        self.weights = weights or {
            'htf_alignment': 25,
            'htf_neutral': 10,
            'liquidity_sweep': 20,
            'order_block': 20,
            'fair_value_gap': 15,
            'mtf_extreme': 10,
            'market_structure': 10
        }
    
    def calculate_score(self, analysis: dict) -> dict:
        """
        Input: {
            'htf_bias': 'bullish'/'bearish'/'neutral',
            'bullish_sweep': bool,
            'bearish_sweep': bool,
            'near_bullish_ob': bool,
            'near_bearish_ob': bool,
            'in_bullish_fvg': bool,
            'in_bearish_fvg': bool,
            'mtf_rsi': float,
            'market_structure': 'bullish'/'bearish'/'neutral'
        }
        
        Output: {
            'bullish_score': int (0-100),
            'bearish_score': int (0-100),
            'factors': {
                'bullish': ['htf_alignment', 'order_block', ...],
                'bearish': [...]
            }
        }
        """
        pass
```

#### signal_generator.py
```python
class SignalGenerator:
    """
    Generiert Trading-Signale
    """
    def __init__(self, min_confidence: int = 90):
        self.min_confidence = min_confidence
        self.active_signals = {}
    
    def generate_signals(self, coins_data: dict) -> list[Signal]:
        """
        Analysiert alle Coins und generiert Signale
        
        Returns list of Signal objects:
        - symbol
        - direction ('long'/'short')
        - confidence
        - entry_price
        - take_profit
        - stop_loss
        - leverage
        - factors (welche Confluence-Faktoren)
        """
        pass
    
    def rank_signals(self, signals: list[Signal]) -> list[Signal]:
        """
        Sortiert Signale nach Confidence
        """
        pass
    
    def filter_correlated(self, signals: list[Signal]) -> list[Signal]:
        """
        Entfernt korrelierte Signale (z.B. nicht 3x Long auf ähnliche Coins)
        """
        pass
```

### 2.3 Exchange Integration

#### bybit.py
```python
from pybit.unified_trading import HTTP, WebSocket

class BybitClient:
    """
    Bybit API Wrapper
    """
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.client = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret
        )
        self.ws = None
    
    # === Market Data ===
    async def get_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        pass
    
    async def get_ticker(self, symbol: str) -> dict:
        pass
    
    async def get_orderbook(self, symbol: str, limit: int = 25) -> dict:
        pass
    
    # === Trading ===
    async def place_order(
        self,
        symbol: str,
        side: str,  # 'Buy' or 'Sell'
        order_type: str,  # 'Market' or 'Limit'
        qty: float,
        price: float = None,
        take_profit: float = None,
        stop_loss: float = None,
        leverage: int = None
    ) -> dict:
        """
        Platziert Order mit TP/SL
        """
        pass
    
    async def modify_position(
        self,
        symbol: str,
        take_profit: float = None,
        stop_loss: float = None
    ) -> dict:
        """
        Modifiziert TP/SL einer offenen Position
        """
        pass
    
    async def close_position(self, symbol: str) -> dict:
        """
        Schließt Position zum Market-Preis
        """
        pass
    
    async def get_position(self, symbol: str) -> dict:
        pass
    
    async def get_open_positions(self) -> list:
        pass
    
    # === Account ===
    async def get_balance(self) -> dict:
        pass
    
    async def set_leverage(self, symbol: str, leverage: int) -> dict:
        pass
    
    # === WebSocket ===
    async def subscribe_klines(self, symbols: list, interval: str, callback):
        pass
    
    async def subscribe_trades(self, symbols: list, callback):
        pass
```

### 2.4 Order-Ausführung Logik

```python
class OrderExecutor:
    """
    Führt Trades aus mit Hybrid Order-Logik
    """
    def __init__(self, exchange_client, config):
        self.exchange = exchange_client
        self.limit_timeout_seconds = config.get('limit_timeout', 120)  # 2 Minuten
        self.use_limit_entry = config.get('use_limit_entry', True)
    
    async def execute_entry(self, signal: Signal) -> Trade:
        """
        Entry-Logik:
        1. Versuche Limit Order im OTE-Bereich
        2. Wenn nach Timeout nicht gefüllt, entscheide:
           - Preis besser geworden → Market Order
           - Preis schlechter → Cancel
        """
        if self.use_limit_entry:
            # Berechne Limit-Preis (z.B. 0.1% besser als Signal-Preis)
            limit_price = self.calculate_limit_price(signal)
            
            order = await self.exchange.place_order(
                symbol=signal.symbol,
                side='Buy' if signal.direction == 'long' else 'Sell',
                order_type='Limit',
                qty=self.calculate_qty(signal),
                price=limit_price,
                take_profit=signal.take_profit,
                stop_loss=signal.stop_loss,
                leverage=signal.leverage
            )
            
            # Warte auf Fill oder Timeout
            filled = await self.wait_for_fill(order['orderId'], self.limit_timeout_seconds)
            
            if not filled:
                current_price = await self.exchange.get_ticker(signal.symbol)
                
                # Preis ist in unsere Richtung gelaufen → Market Order
                if self.price_moved_favorably(signal, current_price):
                    await self.exchange.cancel_order(order['orderId'])
                    order = await self.exchange.place_order(
                        symbol=signal.symbol,
                        side='Buy' if signal.direction == 'long' else 'Sell',
                        order_type='Market',
                        qty=self.calculate_qty(signal),
                        take_profit=signal.take_profit,
                        stop_loss=signal.stop_loss,
                        leverage=signal.leverage
                    )
                else:
                    # Signal verpasst, cancel
                    await self.exchange.cancel_order(order['orderId'])
                    return None
            
            return self.create_trade_from_order(order, signal)
        
        else:
            # Direkt Market Order
            order = await self.exchange.place_order(
                symbol=signal.symbol,
                side='Buy' if signal.direction == 'long' else 'Sell',
                order_type='Market',
                qty=self.calculate_qty(signal),
                take_profit=signal.take_profit,
                stop_loss=signal.stop_loss,
                leverage=signal.leverage
            )
            return self.create_trade_from_order(order, signal)
    
    async def execute_exit(self, trade: Trade, reason: str) -> dict:
        """
        Exit-Logik: Immer Market für Sicherheit
        """
        result = await self.exchange.close_position(trade.symbol)
        return {
            'trade_id': trade.id,
            'exit_price': result['avgPrice'],
            'exit_time': datetime.utcnow(),
            'reason': reason
        }
```

---

## Teil 3: Datenbank Schema (Supabase)

### 3.1 Tables

```sql
-- Trades Tabelle
CREATE TABLE trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Trade Info
    symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(5) NOT NULL,  -- 'long' or 'short'
    
    -- Entry
    entry_price DECIMAL(20, 8) NOT NULL,
    entry_time TIMESTAMP WITH TIME ZONE NOT NULL,
    entry_order_type VARCHAR(10),  -- 'market' or 'limit'
    
    -- Exit
    exit_price DECIMAL(20, 8),
    exit_time TIMESTAMP WITH TIME ZONE,
    exit_reason VARCHAR(20),  -- 'take_profit', 'stop_loss', 'trailing', 'time_exit', 'manual'
    
    -- Position
    leverage INT NOT NULL,
    position_size DECIMAL(20, 8) NOT NULL,
    
    -- Targets
    take_profit_price DECIMAL(20, 8) NOT NULL,
    stop_loss_price DECIMAL(20, 8) NOT NULL,
    
    -- Results
    pnl_usd DECIMAL(20, 8),
    pnl_percent DECIMAL(10, 4),
    fees_usd DECIMAL(20, 8),
    
    -- Confluence Factors
    confidence_score INT NOT NULL,
    htf_bias VARCHAR(10),
    had_order_block BOOLEAN,
    had_fvg BOOLEAN,
    had_liquidity_sweep BOOLEAN,
    mtf_rsi DECIMAL(5, 2),
    market_structure VARCHAR(10),
    
    -- Meta
    duration_minutes INT,
    max_drawdown_percent DECIMAL(10, 4),
    max_profit_percent DECIMAL(10, 4)
);

-- Index für schnelle Abfragen
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_created_at ON trades(created_at);
CREATE INDEX idx_trades_direction ON trades(direction);

-- Signals Tabelle (alle generierten Signale, auch nicht genommene)
CREATE TABLE signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(5) NOT NULL,
    confidence_score INT NOT NULL,
    
    -- War Signal genommen?
    was_executed BOOLEAN DEFAULT FALSE,
    trade_id UUID REFERENCES trades(id),
    
    -- Wenn nicht genommen, warum?
    skip_reason VARCHAR(50),  -- 'below_threshold', 'correlated', 'max_positions', etc.
    
    -- Confluence Details
    factors JSONB NOT NULL,
    
    -- Market Context
    btc_price DECIMAL(20, 8),
    btc_change_1h DECIMAL(10, 4),
    market_volatility DECIMAL(10, 4)
);

-- Daily Stats Tabelle
CREATE TABLE daily_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    date DATE NOT NULL UNIQUE,
    
    total_trades INT DEFAULT 0,
    winning_trades INT DEFAULT 0,
    losing_trades INT DEFAULT 0,
    
    total_pnl_usd DECIMAL(20, 8) DEFAULT 0,
    total_pnl_percent DECIMAL(10, 4) DEFAULT 0,
    
    best_trade_pnl DECIMAL(20, 8),
    worst_trade_pnl DECIMAL(20, 8),
    
    avg_confidence INT,
    avg_duration_minutes INT,
    
    signals_generated INT DEFAULT 0,
    signals_executed INT DEFAULT 0
);

-- Coin Performance Tabelle
CREATE TABLE coin_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) NOT NULL,
    
    total_trades INT DEFAULT 0,
    win_rate DECIMAL(5, 2),
    avg_pnl_percent DECIMAL(10, 4),
    profit_factor DECIMAL(10, 4),
    
    best_timeframe VARCHAR(10),
    best_session VARCHAR(20),
    
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(symbol)
);

-- ML Training Data
CREATE TABLE ml_features (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trade_id UUID REFERENCES trades(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Features (für ML Training)
    features JSONB NOT NULL,
    
    -- Label
    outcome VARCHAR(10),  -- 'win' or 'loss'
    pnl_percent DECIMAL(10, 4)
);
```

### 3.2 Views für Analytics

```sql
-- Win Rate by Confidence Level
CREATE VIEW win_rate_by_confidence AS
SELECT 
    CASE 
        WHEN confidence_score >= 95 THEN '95-100'
        WHEN confidence_score >= 90 THEN '90-94'
        WHEN confidence_score >= 85 THEN '85-89'
        ELSE '80-84'
    END as confidence_bucket,
    COUNT(*) as total_trades,
    COUNT(*) FILTER (WHERE pnl_percent > 0) as wins,
    ROUND(COUNT(*) FILTER (WHERE pnl_percent > 0)::DECIMAL / COUNT(*) * 100, 2) as win_rate,
    ROUND(AVG(pnl_percent), 4) as avg_pnl
FROM trades
WHERE exit_price IS NOT NULL
GROUP BY confidence_bucket
ORDER BY confidence_bucket DESC;

-- Performance by Factor
CREATE VIEW performance_by_factor AS
SELECT 
    'Order Block' as factor,
    COUNT(*) FILTER (WHERE had_order_block = true) as trades_with,
    ROUND(AVG(pnl_percent) FILTER (WHERE had_order_block = true), 4) as avg_pnl_with,
    ROUND(AVG(pnl_percent) FILTER (WHERE had_order_block = false), 4) as avg_pnl_without
FROM trades
WHERE exit_price IS NOT NULL
UNION ALL
SELECT 
    'FVG' as factor,
    COUNT(*) FILTER (WHERE had_fvg = true),
    ROUND(AVG(pnl_percent) FILTER (WHERE had_fvg = true), 4),
    ROUND(AVG(pnl_percent) FILTER (WHERE had_fvg = false), 4)
FROM trades
WHERE exit_price IS NOT NULL
UNION ALL
SELECT 
    'Liquidity Sweep' as factor,
    COUNT(*) FILTER (WHERE had_liquidity_sweep = true),
    ROUND(AVG(pnl_percent) FILTER (WHERE had_liquidity_sweep = true), 4),
    ROUND(AVG(pnl_percent) FILTER (WHERE had_liquidity_sweep = false), 4)
FROM trades
WHERE exit_price IS NOT NULL;
```

---

## Teil 4: ML Pipeline

### 4.1 Feature Engineering

```python
class FeatureEngineer:
    """
    Erstellt Features für ML Model
    """
    def extract_features(self, signal: Signal, market_data: dict) -> dict:
        """
        Extrahiert alle relevanten Features
        """
        return {
            # Confluence Scores
            'confidence_score': signal.confidence,
            'htf_alignment_score': self._htf_score(signal),
            
            # Binary Factors
            'has_order_block': int(signal.factors.get('order_block', False)),
            'has_fvg': int(signal.factors.get('fvg', False)),
            'has_liquidity_sweep': int(signal.factors.get('sweep', False)),
            
            # Market Context
            'volatility_ratio': market_data['volatility_ratio'],
            'atr_percentile': market_data['atr_percentile'],  # ATR vs letzte 100 Perioden
            'btc_correlation': market_data['btc_correlation'],
            'btc_change_1h': market_data['btc_change_1h'],
            'btc_change_4h': market_data['btc_change_4h'],
            
            # Momentum
            'rsi_14': market_data['rsi'],
            'rsi_divergence': self._check_divergence(market_data),
            
            # Volume
            'volume_ratio': market_data['volume'] / market_data['volume_sma'],
            
            # Time Features
            'hour_of_day': market_data['hour'],
            'day_of_week': market_data['day_of_week'],
            'is_weekend': int(market_data['day_of_week'] >= 5),
            
            # Price Position
            'distance_from_daily_high': (market_data['daily_high'] - market_data['price']) / market_data['price'],
            'distance_from_daily_low': (market_data['price'] - market_data['daily_low']) / market_data['price'],
            
            # Recent Performance (Meta-Feature)
            'recent_win_rate_this_coin': self._get_recent_win_rate(signal.symbol),
            'recent_win_rate_overall': self._get_recent_win_rate_overall()
        }
```

### 4.2 Model Training

```python
class MLTrainer:
    """
    Trainiert ML Modelle auf historischen Daten
    """
    def __init__(self, supabase_client):
        self.db = supabase_client
        self.model = None
    
    def train(self, min_samples: int = 500):
        """
        Trainiert XGBoost Model auf historischen Trades
        """
        # Lade Training Data
        data = self.db.get_training_data(min_samples)
        
        X = pd.DataFrame([d['features'] for d in data])
        y = pd.Series([1 if d['outcome'] == 'win' else 0 for d in data])
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Train Model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='binary:logistic'
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Feature Importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'accuracy': accuracy,
            'feature_importance': importance.to_dict('records')
        }
    
    def predict_probability(self, features: dict) -> float:
        """
        Returns Win-Probability 0-100
        """
        if self.model is None:
            return None
        
        X = pd.DataFrame([features])
        proba = self.model.predict_proba(X)[0][1]
        return round(proba * 100, 2)
```

---

## Teil 5: Main Loop & Deployment

### 5.1 Main Application

```python
# main.py
import asyncio
from core.data_fetcher import DataFetcher
from core.signal_generator import SignalGenerator
from core.trade_manager import TradeManager
from exchange.bybit import BybitClient
from database.supabase_client import SupabaseClient
from ml.predictor import MLPredictor
from config.settings import Settings
from config.coins import COIN_CONFIGS

class SMCUltraBot:
    def __init__(self):
        self.settings = Settings()
        self.exchange = BybitClient(
            api_key=self.settings.BYBIT_API_KEY,
            api_secret=self.settings.BYBIT_API_SECRET,
            testnet=self.settings.USE_TESTNET
        )
        self.db = SupabaseClient(self.settings.SUPABASE_URL, self.settings.SUPABASE_KEY)
        self.data_fetcher = DataFetcher(self.exchange)
        self.signal_generator = SignalGenerator(min_confidence=self.settings.MIN_CONFIDENCE)
        self.trade_manager = TradeManager(self.exchange, self.db)
        self.ml_predictor = MLPredictor()
        
        self.active_trades = {}
        self.running = False
    
    async def run(self):
        """
        Main Loop
        """
        self.running = True
        print("🚀 SMC Ultra Bot gestartet")
        
        while self.running:
            try:
                # 1. Fetch Data für alle Coins
                market_data = await self.fetch_all_coins_data()
                
                # 2. Manage bestehende Trades
                await self.manage_active_trades(market_data)
                
                # 3. Generiere neue Signale (wenn Kapazität frei)
                if len(self.active_trades) < self.settings.MAX_CONCURRENT_TRADES:
                    signals = await self.generate_and_filter_signals(market_data)
                    
                    if signals:
                        # Nehme bestes Signal
                        best_signal = signals[0]
                        
                        # ML Confidence Check
                        ml_confidence = self.ml_predictor.predict(best_signal)
                        if ml_confidence and ml_confidence < self.settings.ML_MIN_CONFIDENCE:
                            print(f"⚠️ Signal gefiltert durch ML: {ml_confidence}% < {self.settings.ML_MIN_CONFIDENCE}%")
                            continue
                        
                        # Execute Trade
                        trade = await self.trade_manager.execute_entry(best_signal)
                        if trade:
                            self.active_trades[trade.symbol] = trade
                            await self.db.log_trade(trade)
                            print(f"✅ Trade eröffnet: {trade.symbol} {trade.direction} @ {trade.entry_price}")
                
                # 4. Warte bis nächster Zyklus
                await asyncio.sleep(self.settings.LOOP_INTERVAL_SECONDS)
                
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                await asyncio.sleep(10)
    
    async def fetch_all_coins_data(self) -> dict:
        """
        Holt Daten für alle konfigurierten Coins
        """
        tasks = []
        for symbol in COIN_CONFIGS.keys():
            tasks.append(self.data_fetcher.get_multi_timeframe_data(symbol))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            symbol: data 
            for symbol, data in zip(COIN_CONFIGS.keys(), results)
            if not isinstance(data, Exception)
        }
    
    async def generate_and_filter_signals(self, market_data: dict) -> list:
        """
        Generiert Signale und filtert nach Correlation
        """
        # Generiere alle Signale
        all_signals = self.signal_generator.generate_signals(market_data)
        
        # Filtere unter Minimum Confidence
        signals = [s for s in all_signals if s.confidence >= self.settings.MIN_CONFIDENCE]
        
        # Filtere korrelierte Signale
        signals = self.signal_generator.filter_correlated(signals)
        
        # Sortiere nach Confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        # Log alle Signale (für ML Training)
        for signal in all_signals:
            await self.db.log_signal(signal, was_executed=(signal in signals[:1]))
        
        return signals
    
    async def manage_active_trades(self, market_data: dict):
        """
        Managed alle aktiven Trades
        """
        for symbol, trade in list(self.active_trades.items()):
            if symbol not in market_data:
                continue
            
            current_price = market_data[symbol]['1m']['close'].iloc[-1]
            action = self.trade_manager.update(trade, current_price)
            
            if action in ['TAKE_PROFIT', 'STOP_LOSS', 'TRAILING_STOP', 'TIME_EXIT']:
                result = await self.trade_manager.execute_exit(trade, action)
                await self.db.update_trade_exit(trade.id, result)
                del self.active_trades[symbol]
                print(f"📤 Trade geschlossen: {symbol} - {action} - PnL: {result['pnl_percent']}%")
            
            elif action == 'UPDATE_SL':
                await self.exchange.modify_position(symbol, stop_loss=trade.stop_loss)
    
    def stop(self):
        self.running = False

if __name__ == "__main__":
    bot = SMCUltraBot()
    asyncio.run(bot.run())
```

### 5.2 Railway Deployment

```toml
# railway.toml
[build]
builder = "dockerfile"

[deploy]
startCommand = "python main.py"
healthcheckPath = "/health"
healthcheckTimeout = 30
restartPolicyType = "always"

[[services]]
name = "smc-ultra-bot"
```

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run
CMD ["python", "main.py"]
```

```txt
# requirements.txt
pybit==5.6.2
pandas==2.1.0
numpy==1.24.0
ta==0.10.2
python-dotenv==1.0.0
supabase==2.0.0
xgboost==2.0.0
scikit-learn==1.3.0
aiohttp==3.8.5
asyncio==3.4.3
```

---

## Teil 6: Coin-Konfigurationen

```python
# config/coins.py

COIN_CONFIGS = {
    # === Large Caps ===
    'BTCUSDT': {
        'name': 'Bitcoin',
        'category': 'large_cap',
        'min_confidence': 90,
        'max_leverage': 30,
        'atr_multiplier_tp': 1.5,
        'atr_multiplier_sl': 1.0,
        'htf_timeframe': '4h',
        'mtf_timeframe': '15m',
        'ltf_timeframe': '5m',
        'max_trade_duration_minutes': 120,
        'min_volume_usd': 100_000_000,
    },
    'ETHUSDT': {
        'name': 'Ethereum',
        'category': 'large_cap',
        'min_confidence': 88,
        'max_leverage': 35,
        'atr_multiplier_tp': 1.8,
        'atr_multiplier_sl': 1.0,
        'htf_timeframe': '4h',
        'mtf_timeframe': '15m',
        'ltf_timeframe': '5m',
        'max_trade_duration_minutes': 90,
        'min_volume_usd': 50_000_000,
    },
    
    # === Mid Caps ===
    'SOLUSDT': {
        'name': 'Solana',
        'category': 'mid_cap',
        'min_confidence': 85,
        'max_leverage': 25,
        'atr_multiplier_tp': 2.0,
        'atr_multiplier_sl': 1.2,
        'htf_timeframe': '1h',
        'mtf_timeframe': '15m',
        'ltf_timeframe': '5m',
        'max_trade_duration_minutes': 60,
        'min_volume_usd': 20_000_000,
    },
    'AVAXUSDT': {
        'name': 'Avalanche',
        'category': 'mid_cap',
        'min_confidence': 85,
        'max_leverage': 25,
        'atr_multiplier_tp': 2.0,
        'atr_multiplier_sl': 1.2,
        'htf_timeframe': '1h',
        'mtf_timeframe': '15m',
        'ltf_timeframe': '5m',
        'max_trade_duration_minutes': 60,
        'min_volume_usd': 10_000_000,
    },
    
    # === Altcoins ===
    'DOGEUSDT': {
        'name': 'Dogecoin',
        'category': 'altcoin',
        'min_confidence': 92,
        'max_leverage': 15,
        'atr_multiplier_tp': 2.5,
        'atr_multiplier_sl': 1.5,
        'htf_timeframe': '1h',
        'mtf_timeframe': '5m',
        'ltf_timeframe': '1m',
        'max_trade_duration_minutes': 30,
        'min_volume_usd': 5_000_000,
    },
    
    # Weitere Coins nach gleichem Schema...
}

# Coins dynamisch laden basierend auf Bybit Top-Volumen
async def get_top_coins_by_volume(exchange_client, limit=50):
    """
    Holt die Top Coins nach 24h Volumen und merged mit COIN_CONFIGS
    """
    pass
```

---

## Teil 7: Checkliste für Implementierung

### Phase 1: Core Bot (Woche 1-2)
- [ ] Projektstruktur aufsetzen
- [ ] Bybit API Integration
- [ ] SMC Detection implementieren
- [ ] Confluence Scoring implementieren
- [ ] Basis Signal-Generierung
- [ ] Simple Trade Execution (Market Orders)

### Phase 2: Trade Management (Woche 2-3)
- [ ] Trailing Stop Logic
- [ ] Zeit-basierter Exit
- [ ] Dynamische TP/SL
- [ ] Position Sizing

### Phase 3: Database & Logging (Woche 3)
- [ ] Supabase Setup
- [ ] Trade Logging
- [ ] Signal Logging
- [ ] Performance Analytics

### Phase 4: ML Integration (Woche 4-5)
- [ ] Feature Engineering
- [ ] Data Collection (min. 500 Trades)
- [ ] Model Training
- [ ] Live Prediction Integration

### Phase 5: Deployment & Testing (Woche 5-6)
- [ ] Railway Setup
- [ ] Testnet Testing
- [ ] Paper Trading (1 Woche)
- [ ] Live Trading mit kleinem Kapital

---

## Kontakt & Support

Bei Fragen zur Implementierung:
1. Überprüfe die Pine Script Logik als Referenz
2. Teste Module isoliert mit Unit Tests
3. Starte im Testnet bevor du live gehst

**Wichtig:** Dieses Briefing enthält die komplette Logik. Ein Entwickler sollte damit in der Lage sein, den Bot vollständig zu implementieren.
