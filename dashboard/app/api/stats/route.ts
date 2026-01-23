import { NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const daysParam = searchParams.get('days')
    const days = daysParam ? parseInt(daysParam) : undefined
    const fromParam = searchParams.get('from')
    const toParam = searchParams.get('to')

    // Build query for completed trades
    let query = supabase
      .from('trades')
      .select('*')
      .not('exit_time', 'is', null)

    // Filter by custom date range or days
    if (fromParam && toParam) {
      query = query.gte('exit_time', `${fromParam}T00:00:00`)
      query = query.lte('exit_time', `${toParam}T23:59:59`)
    } else if (days) {
      const fromDate = new Date()
      fromDate.setDate(fromDate.getDate() - days)
      query = query.gte('exit_time', fromDate.toISOString())
    }

    const { data: trades, error } = await query.order('exit_time', { ascending: false })

    if (error) throw error
    if (!trades || trades.length === 0) {
      return NextResponse.json({
        total_trades: 0,
        wins: 0,
        losses: 0,
        breakeven: 0,
        win_rate: 0,
        total_pnl: 0,
        total_pnl_pct: 0,
        avg_pnl: 0,
        avg_pnl_pct: 0,
        avg_win: 0,
        avg_win_pct: 0,
        avg_loss: 0,
        avg_loss_pct: 0,
        best_trade: 0,
        worst_trade: 0,
        profit_factor: 0,
        tp1_rate: 0,
        tp2_rate: 0,
        sl_rate: 0,
        avg_duration: 0,
      })
    }

    // Calculate stats
    const wins = trades.filter(t => t.realized_pnl > 0)
    const losses = trades.filter(t => t.realized_pnl < 0 && !t.tp1_hit)
    const breakeven = trades.filter(t => t.realized_pnl <= 0 && t.tp1_hit)

    const totalPnl = trades.reduce((sum, t) => sum + (t.realized_pnl || 0), 0)
    const totalPnlPct = trades.reduce((sum, t) => sum + (t.pnl_pct_equity || 0), 0)

    const avgWin = wins.length > 0
      ? wins.reduce((sum, t) => sum + (t.realized_pnl || 0), 0) / wins.length
      : 0
    const avgWinPct = wins.length > 0
      ? wins.reduce((sum, t) => sum + (t.pnl_pct_equity || 0), 0) / wins.length
      : 0

    const avgLoss = losses.length > 0
      ? losses.reduce((sum, t) => sum + (t.realized_pnl || 0), 0) / losses.length
      : 0
    const avgLossPct = losses.length > 0
      ? losses.reduce((sum, t) => sum + (t.pnl_pct_equity || 0), 0) / losses.length
      : 0

    const grossProfit = wins.reduce((sum, t) => sum + (t.realized_pnl || 0), 0)
    const grossLoss = Math.abs(losses.reduce((sum, t) => sum + (t.realized_pnl || 0), 0))
    const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? Infinity : 0

    const tp1Hits = trades.filter(t => t.tp1_hit).length
    const tp2Hits = trades.filter(t => t.tp2_hit).length
    const slHits = trades.filter(t => t.exit_reason === 'sl').length

    const durations = trades.filter(t => t.duration_minutes).map(t => t.duration_minutes)
    const avgDuration = durations.length > 0
      ? durations.reduce((sum, d) => sum + d, 0) / durations.length
      : 0

    const pnls = trades.map(t => t.realized_pnl || 0)

    // Session stats
    const asianTrades = trades.filter(t => t.is_asian_session)
    const londonTrades = trades.filter(t => t.is_london_session)
    const nyTrades = trades.filter(t => t.is_ny_session)

    const sessions = [
      {
        session: 'Asian',
        trades: asianTrades.length,
        wins: asianTrades.filter(t => t.realized_pnl > 0).length,
        pnl: asianTrades.reduce((sum, t) => sum + (t.realized_pnl || 0), 0),
        win_rate: asianTrades.length > 0
          ? (asianTrades.filter(t => t.realized_pnl > 0).length / asianTrades.length) * 100
          : 0,
      },
      {
        session: 'London',
        trades: londonTrades.length,
        wins: londonTrades.filter(t => t.realized_pnl > 0).length,
        pnl: londonTrades.reduce((sum, t) => sum + (t.realized_pnl || 0), 0),
        win_rate: londonTrades.length > 0
          ? (londonTrades.filter(t => t.realized_pnl > 0).length / londonTrades.length) * 100
          : 0,
      },
      {
        session: 'New York',
        trades: nyTrades.length,
        wins: nyTrades.filter(t => t.realized_pnl > 0).length,
        pnl: nyTrades.reduce((sum, t) => sum + (t.realized_pnl || 0), 0),
        win_rate: nyTrades.length > 0
          ? (nyTrades.filter(t => t.realized_pnl > 0).length / nyTrades.length) * 100
          : 0,
      },
    ]

    return NextResponse.json({
      total_trades: trades.length,
      wins: wins.length,
      losses: losses.length,
      breakeven: breakeven.length,
      win_rate: (wins.length / trades.length) * 100,
      total_pnl: totalPnl,
      total_pnl_pct: totalPnlPct,
      avg_pnl: totalPnl / trades.length,
      avg_pnl_pct: totalPnlPct / trades.length,
      avg_win: avgWin,
      avg_win_pct: avgWinPct,
      avg_loss: avgLoss,
      avg_loss_pct: avgLossPct,
      best_trade: Math.max(...pnls),
      worst_trade: Math.min(...pnls),
      profit_factor: profitFactor,
      tp1_rate: (tp1Hits / trades.length) * 100,
      tp2_rate: (tp2Hits / trades.length) * 100,
      sl_rate: (slHits / trades.length) * 100,
      avg_duration: avgDuration,
      sessions,
    })
  } catch (error) {
    console.error('Failed to fetch stats:', error)
    return NextResponse.json({ error: 'Failed to fetch stats' }, { status: 500 })
  }
}

export const dynamic = 'force-dynamic'
export const revalidate = 0
