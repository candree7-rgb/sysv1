'use client'

import { useEffect, useState } from 'react'
import { Trade } from '@/lib/supabase'
import { formatCurrency, formatDate, formatDuration, cn } from '@/lib/utils'
import { TimeRange, TIME_RANGES } from './time-range-selector'

interface TradesTableProps {
  timeRange: TimeRange
  customDateRange?: { from: string; to: string } | null
}

export default function TradesTable({ timeRange, customDateRange }: TradesTableProps) {
  const [trades, setTrades] = useState<Trade[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchTrades() {
      try {
        const params = new URLSearchParams({ limit: '50' })

        if (timeRange === 'CUSTOM' && customDateRange) {
          params.append('from', customDateRange.from)
          params.append('to', customDateRange.to)
        } else {
          const range = TIME_RANGES.find(r => r.value === timeRange)
          if (range?.days) params.append('days', range.days.toString())
        }

        const res = await fetch(`/api/trades?${params.toString()}`)
        const data = await res.json()
        setTrades(data)
      } catch (error) {
        console.error('Failed to fetch trades:', error)
      } finally {
        setLoading(false)
      }
    }

    setLoading(true)
    fetchTrades()
    const interval = setInterval(fetchTrades, 30000)
    return () => clearInterval(interval)
  }, [timeRange, customDateRange])

  if (loading) {
    return (
      <div className="bg-card border border-border rounded-lg p-6">
        <div className="h-8 bg-muted rounded w-1/4 mb-4 animate-pulse"></div>
        <div className="space-y-2">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-16 bg-muted rounded animate-pulse"></div>
          ))}
        </div>
      </div>
    )
  }

  if (trades.length === 0) {
    return (
      <div className="bg-card border border-border rounded-lg p-6">
        <h2 className="text-xl font-bold mb-4">Trade History</h2>
        <div className="text-center text-muted-foreground py-8">
          No trades found for this period
        </div>
      </div>
    )
  }

  return (
    <div className="bg-card border border-border rounded-lg overflow-hidden">
      <div className="p-6 pb-4">
        <h2 className="text-xl font-bold">Trade History</h2>
        <p className="text-sm text-muted-foreground mt-1">Last {trades.length} trades</p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="border-y border-border bg-muted/30">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground uppercase">Symbol</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground uppercase">Time</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground uppercase">Side</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground uppercase">Entry</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground uppercase">Duration</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground uppercase">P&L</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground uppercase">P&L %</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground uppercase">Exit</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground uppercase">TPs</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/50">
            {trades.map((trade) => (
              <tr key={trade.id} className="hover:bg-muted/20 transition-colors">
                <td className="px-4 py-4">
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-semibold">
                      {trade.symbol.replace('USDT', '')}
                    </span>
                    <span className="px-1.5 py-0.5 rounded text-xs font-semibold bg-muted/80 text-muted-foreground">
                      {trade.leverage}x
                    </span>
                  </div>
                </td>

                <td className="px-4 py-4 text-sm text-muted-foreground">
                  {trade.exit_time ? formatDate(trade.exit_time) : '-'}
                </td>

                <td className="px-4 py-4">
                  <span className={cn(
                    'px-2 py-1 rounded text-xs font-semibold',
                    trade.direction === 'long'
                      ? 'bg-success/20 text-success'
                      : 'bg-danger/20 text-danger'
                  )}>
                    {trade.direction.toUpperCase()}
                  </span>
                </td>

                <td className="px-4 py-4 font-mono text-sm">
                  ${trade.entry_price?.toFixed(4)}
                </td>

                <td className="px-4 py-4 text-sm text-muted-foreground">
                  {formatDuration(trade.duration_minutes)}
                </td>

                <td className="px-4 py-4">
                  <span className={cn(
                    'font-semibold',
                    (trade.realized_pnl || 0) >= 0 ? 'text-success' : 'text-danger'
                  )}>
                    {(trade.realized_pnl || 0) >= 0 ? '+' : ''}
                    {formatCurrency(trade.realized_pnl || 0)}
                  </span>
                </td>

                <td className="px-4 py-4">
                  <span className={cn(
                    'font-semibold text-sm',
                    (trade.pnl_pct_equity || 0) >= 0 ? 'text-success' : 'text-danger'
                  )}>
                    {(trade.pnl_pct_equity || 0) >= 0 ? '+' : ''}
                    {(trade.pnl_pct_equity || 0).toFixed(2)}%
                  </span>
                </td>

                <td className="px-4 py-4 text-sm">
                  <span className={cn(
                    'px-2 py-1 rounded text-xs',
                    trade.exit_reason === 'tp2' ? 'bg-success/20 text-success' :
                    trade.exit_reason === 'tp1' || trade.exit_reason === 'be+' ? 'bg-primary/20 text-primary' :
                    'bg-danger/20 text-danger'
                  )}>
                    {trade.exit_reason?.toUpperCase() || '-'}
                  </span>
                </td>

                <td className="px-4 py-4 text-sm text-muted-foreground">
                  {trade.tp1_hit ? '1' : '0'}/{trade.tp2_hit ? '2' : trade.tp1_hit ? '1' : '0'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
