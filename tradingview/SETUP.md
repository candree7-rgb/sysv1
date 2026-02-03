# TradingView + Railway Webhook Setup

## Overview
1. TradingView runs the Pine Script strategy (signal generation)
2. When conditions are met, TradingView sends webhook to Railway
3. Railway receives webhook and places limit orders on Bybit

## Step 1: Add Strategy to TradingView

1. Open TradingView (tradingview.com)
2. Go to Pine Editor (bottom panel)
3. Copy contents of `smc_ob_strategy.pine`
4. Click "Add to Chart"
5. Set chart to **5 minute** timeframe

## Step 2: Configure Strategy Settings

Click the gear icon on the strategy:

### Strategy Settings
- **Trade Direction**: Both (or Long/Short only)
- **Risk % per Trade**: 2.0 (matches your backtest)
- **Risk:Reward Ratio**: 2.0
- **SL Buffer %**: 0.5 (extra buffer for SL)
- **Max OB Age**: 30 bars

### Partial Take Profit
- **Use Partial TP**: ✓ (enabled)
- **Partial TP Size %**: 50
- **Partial TP at % of Full TP**: 50
- **Move SL to BE after Partial**: ✓ (enabled)

### MTF Trend Filter
- **Use 1H Trend Filter**: ✓ (enabled)
- **Use 4H Trend Filter**: ☐ (optional)
- **Use Daily Trend Filter**: ☐ (optional)

## Step 3: Create Alert for Webhook

1. Right-click on chart → "Add Alert"
2. **Condition**: Select "SMC Order Block Strategy"
3. **Alert type**: "Any alert() function call"
4. **Alert actions**: Check "Webhook URL"
5. **Webhook URL**: `https://your-railway-app.railway.app/webhook`
6. **Alert name**: "SMC OB Webhook"
7. Click "Create"

## Step 4: Deploy Webhook Server to Railway

### Option A: Railway CLI
```bash
cd smc_ultra_v2
railway login
railway init
railway up
```

### Option B: GitHub Deploy
1. Push to GitHub
2. Connect repo to Railway
3. Set root directory to `smc_ultra_v2`
4. Add environment variables

### Environment Variables (Railway)
```
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
BYBIT_TESTNET=false
PORT=8080
ORDER_CANCEL_MINUTES=30
MAX_POSITION_SIZE_PCT=5
DEFAULT_LEVERAGE=10
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

## Step 5: Test the Setup

1. Open browser: `https://your-railway-app.railway.app/health`
   - Should return: `{"status": "ok", "time": "..."}`

2. Check status: `https://your-railway-app.railway.app/status`
   - Shows equity, positions, pending orders

3. Wait for TradingView alert or test manually:
```bash
curl -X POST https://your-railway-app.railway.app/webhook \
  -H "Content-Type: application/json" \
  -d '{"action":"entry","symbol":"BTCUSDT","direction":"long","entry":50000,"sl":49500,"tp1":50500,"tp2":51000,"risk_pct":2}'
```

## Webhook JSON Format

TradingView sends this JSON when alert fires:
```json
{
  "action": "entry",
  "symbol": "BTCUSDT",
  "direction": "long",
  "entry": 50000.00,
  "sl": 49500.00,
  "tp1": 50500.00,
  "tp2": 51000.00,
  "risk_pct": 2.0,
  "timeframe": "5",
  "timestamp": "1706000000000"
}
```

## How Orders Work

1. **Webhook received** → Calculate position size based on risk %
2. **Two limit orders placed** at OB edge:
   - Order 1: 50% qty with TP1, SL
   - Order 2: 50% qty with TP2, SL
3. **Auto-cancel** after 30 minutes if not filled
4. **Logged to Supabase** for tracking

## Multi-Coin Setup

To monitor multiple coins:
1. Add the strategy to each coin's chart
2. Create alert for each chart
3. All alerts use the same webhook URL
4. Server handles all coins automatically

**Tip**: Use TradingView Premium for more alerts, or create a watchlist and rotate alerts.

## Troubleshooting

### Alert not firing
- Check strategy conditions match chart
- Ensure 5-min timeframe
- Verify MTF filters aren't blocking signals

### Webhook not received
- Check Railway logs: `railway logs`
- Verify webhook URL is correct
- Test with curl manually

### Order not placed
- Check Bybit API keys are valid
- Ensure sufficient balance
- Check symbol exists on Bybit

### Position size too small
- Increase risk % or equity
- Check symbol minimum order size
