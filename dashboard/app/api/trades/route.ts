import { NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const limit = parseInt(searchParams.get('limit') || '50')
    const daysParam = searchParams.get('days')
    const days = daysParam ? parseInt(daysParam) : undefined
    const fromParam = searchParams.get('from')
    const toParam = searchParams.get('to')

    // Build query
    let query = supabase
      .from('trades')
      .select('*')
      .not('exit_time', 'is', null)
      .order('exit_time', { ascending: false })
      .limit(limit)

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

    return NextResponse.json(trades || [])
  } catch (error) {
    console.error('Failed to fetch trades:', error)
    return NextResponse.json({ error: 'Failed to fetch trades' }, { status: 500 })
  }
}

export const dynamic = 'force-dynamic'
export const revalidate = 0
