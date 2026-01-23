import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.SUPABASE_URL || ''
const supabaseKey = process.env.SUPABASE_KEY || ''

export const supabase = createClient(supabaseUrl, supabaseKey)

export interface Trade {
  id: string
  symbol: string
  direction: string
  entry_price: number
  exit_price: number | null
  entry_time: string
  exit_time: string | null
  qty: number
  leverage: number
  margin_used: number
  equity_at_entry: number
  equity_at_close: number | null
  sl_price: number
  tp1_price: number
  tp2_price: number
  realized_pnl: number | null
  pnl_pct: number | null
  pnl_pct_equity: number | null
  is_win: boolean | null
  tp1_hit: boolean | null
  tp2_hit: boolean | null
  exit_reason: string | null
  duration_minutes: number | null
  ob_strength: number | null
  ob_age_candles: number | null
  hour_utc: number | null
  day_of_week: number | null
  is_asian_session: boolean | null
  is_london_session: boolean | null
  is_ny_session: boolean | null
  risk_pct: number | null
  total_fees: number | null
  net_pnl: number | null
}

export interface Stats {
  total_trades: number
  wins: number
  losses: number
  breakeven: number
  win_rate: number
  total_pnl: number
  total_pnl_pct: number
  avg_pnl: number
  avg_pnl_pct: number
  avg_win: number
  avg_win_pct: number
  avg_loss: number
  avg_loss_pct: number
  best_trade: number
  worst_trade: number
  profit_factor: number
  tp1_rate: number
  tp2_rate: number
  sl_rate: number
  avg_duration: number
}

export interface EquityPoint {
  date: string
  equity: number
  daily_pnl: number
}

export interface TPDistribution {
  level: string
  count: number
  percentage: number
}
