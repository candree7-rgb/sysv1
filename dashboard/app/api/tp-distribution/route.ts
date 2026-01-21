import { NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const daysParam = searchParams.get('days')
    const days = daysParam ? parseInt(daysParam) : undefined

    // Build query
    let query = supabase
      .from('trades')
      .select('exit_reason, tp1_hit, tp2_hit')
      .not('exit_time', 'is', null)

    // Filter by date if days specified
    if (days) {
      const fromDate = new Date()
      fromDate.setDate(fromDate.getDate() - days)
      query = query.gte('exit_time', fromDate.toISOString())
    }

    const { data: trades, error } = await query

    if (error) throw error
    if (!trades || trades.length === 0) {
      return NextResponse.json([])
    }

    // Calculate distribution
    const total = trades.length

    // TP2 (Full TP) = both TPs hit
    const tp2Full = trades.filter(t => t.tp2_hit).length

    // TP1 + BE = TP1 hit but not TP2 (exited at breakeven after TP1)
    const tp1Be = trades.filter(t => t.tp1_hit && !t.tp2_hit && t.exit_reason !== 'sl').length

    // Stop Loss = pure SL, no TPs hit
    const stopLoss = trades.filter(t => t.exit_reason === 'sl' && !t.tp1_hit).length

    // Break Even = exited at BE without hitting any TP (rare)
    const breakEven = trades.filter(t => !t.tp1_hit && !t.tp2_hit && t.exit_reason !== 'sl').length

    const distribution = [
      {
        level: 'TP2 (Full)',
        count: tp2Full,
        percentage: (tp2Full / total) * 100,
      },
      {
        level: 'TP1 + BE',
        count: tp1Be,
        percentage: (tp1Be / total) * 100,
      },
      {
        level: 'Stop Loss',
        count: stopLoss,
        percentage: (stopLoss / total) * 100,
      },
      {
        level: 'Break Even',
        count: breakEven,
        percentage: (breakEven / total) * 100,
      },
    ].filter(d => d.count > 0) // Only show categories with data

    return NextResponse.json(distribution)
  } catch (error) {
    console.error('Failed to fetch TP distribution:', error)
    return NextResponse.json({ error: 'Failed to fetch TP distribution' }, { status: 500 })
  }
}

export const dynamic = 'force-dynamic'
export const revalidate = 0
