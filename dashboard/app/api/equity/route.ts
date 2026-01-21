import { NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'
import { format, subDays, parseISO } from 'date-fns'

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
      .select('exit_time, equity_at_close, realized_pnl, pnl_pct_equity')
      .not('exit_time', 'is', null)
      .order('exit_time', { ascending: true })

    // Filter by custom date range or days
    if (fromParam && toParam) {
      query = query.gte('exit_time', `${fromParam}T00:00:00`)
      query = query.lte('exit_time', `${toParam}T23:59:59`)
    } else if (days) {
      const fromDate = new Date()
      fromDate.setDate(fromDate.getDate() - days)
      query = query.gte('exit_time', fromDate.toISOString())
    }

    const { data: trades, error } = await query

    if (error) throw error
    if (!trades || trades.length === 0) {
      return NextResponse.json([])
    }

    // Group by date and calculate daily equity
    const dailyData: Record<string, { equity: number; pnl: number; count: number }> = {}

    trades.forEach((trade) => {
      const date = format(parseISO(trade.exit_time), 'yyyy-MM-dd')

      if (!dailyData[date]) {
        dailyData[date] = { equity: 0, pnl: 0, count: 0 }
      }

      dailyData[date].equity = trade.equity_at_close || dailyData[date].equity
      dailyData[date].pnl += trade.realized_pnl || 0
      dailyData[date].count++
    })

    // Convert to array and fill gaps
    const dates = Object.keys(dailyData).sort()
    const result = dates.map(date => ({
      date,
      equity: dailyData[date].equity,
      daily_pnl: dailyData[date].pnl,
    }))

    return NextResponse.json(result)
  } catch (error) {
    console.error('Failed to fetch equity:', error)
    return NextResponse.json({ error: 'Failed to fetch equity' }, { status: 500 })
  }
}

export const dynamic = 'force-dynamic'
export const revalidate = 0
