'use client'

import { useEffect, useState } from 'react'
import { Stats } from '@/lib/supabase'
import { formatCurrency } from '@/lib/utils'
import { TimeRange, TIME_RANGES } from './time-range-selector'

interface StatsCardsProps {
  timeRange: TimeRange
}

export default function StatsCards({ timeRange }: StatsCardsProps) {
  const [stats, setStats] = useState<Stats | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchStats() {
      try {
        const range = TIME_RANGES.find(r => r.value === timeRange)
        const days = range?.days
        const params = days ? `?days=${days}` : ''

        const res = await fetch(`/api/stats${params}`)
        const data = await res.json()
        setStats(data)
      } catch (error) {
        console.error('Failed to fetch stats:', error)
      } finally {
        setLoading(false)
      }
    }

    setLoading(true)
    fetchStats()
    const interval = setInterval(fetchStats, 30000)
    return () => clearInterval(interval)
  }, [timeRange])

  if (loading) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
        {[...Array(10)].map((_, i) => (
          <div key={i} className="bg-card border border-border rounded-lg p-4 animate-pulse">
            <div className="h-4 bg-muted rounded w-1/2 mb-2"></div>
            <div className="h-8 bg-muted rounded w-3/4"></div>
          </div>
        ))}
      </div>
    )
  }

  if (!stats || stats.total_trades === 0) {
    return (
      <div className="bg-card border border-border rounded-lg p-6 text-center">
        <p className="text-muted-foreground">No trade data available for this period</p>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
      {/* Total Trades */}
      <StatCard
        label="Total Trades"
        value={stats.total_trades.toString()}
        subValue={`${stats.wins}W / ${stats.breakeven}BE / ${stats.losses}L`}
      />

      {/* Win Rate */}
      <StatCard
        label="Win Rate"
        value={`${stats.win_rate.toFixed(1)}%`}
        variant={stats.win_rate >= 50 ? 'success' : 'danger'}
        subValue="Wins / Total"
      />

      {/* Total PnL */}
      <StatCard
        label="Total PnL"
        value={formatCurrency(stats.total_pnl)}
        variant={stats.total_pnl >= 0 ? 'success' : 'danger'}
        subValue={`${stats.total_pnl_pct >= 0 ? '+' : ''}${stats.total_pnl_pct.toFixed(2)}% Equity`}
      />

      {/* Profit Factor */}
      <StatCard
        label="Profit Factor"
        value={stats.profit_factor === Infinity ? '∞' : stats.profit_factor.toFixed(2)}
        variant={stats.profit_factor >= 1.5 ? 'success' : stats.profit_factor >= 1 ? 'default' : 'danger'}
        subValue="Gross Win / Loss"
      />

      {/* Avg PnL */}
      <StatCard
        label="Avg PnL"
        value={`${stats.avg_pnl_pct >= 0 ? '+' : ''}${stats.avg_pnl_pct.toFixed(2)}%`}
        variant={stats.avg_pnl >= 0 ? 'success' : 'danger'}
        subValue={formatCurrency(stats.avg_pnl)}
      />

      {/* Avg Win */}
      <StatCard
        label="Avg Win"
        value={`+${stats.avg_win_pct.toFixed(2)}%`}
        valueColor="text-success"
        subValue={formatCurrency(stats.avg_win)}
      />

      {/* Avg Loss */}
      <StatCard
        label="Avg Loss"
        value={`${stats.avg_loss_pct.toFixed(2)}%`}
        valueColor="text-danger"
        subValue={formatCurrency(stats.avg_loss)}
      />

      {/* Best Trade */}
      <StatCard
        label="Best Trade"
        value={formatCurrency(stats.best_trade)}
        valueColor="text-success"
      />

      {/* Worst Trade */}
      <StatCard
        label="Worst Trade"
        value={formatCurrency(stats.worst_trade)}
        valueColor="text-danger"
      />

      {/* TP Rates */}
      <StatCard
        label="TP Hit Rates"
        value={`${stats.tp1_rate.toFixed(0)}% / ${stats.tp2_rate.toFixed(0)}%`}
        subValue="TP1 / TP2"
      />

      {/* SL Rate */}
      <StatCard
        label="Stop Loss Rate"
        value={`${stats.sl_rate.toFixed(1)}%`}
        valueColor={stats.sl_rate < 30 ? 'text-success' : 'text-danger'}
        subValue="Pure SL exits"
      />

      {/* Avg Duration */}
      <StatCard
        label="Avg Duration"
        value={stats.avg_duration > 60 ? `${(stats.avg_duration / 60).toFixed(1)}h` : `${stats.avg_duration.toFixed(0)}m`}
        subValue="Per trade"
      />
    </div>
  )
}

interface StatCardProps {
  label: string
  value: string
  subValue?: string
  variant?: 'default' | 'success' | 'danger'
  valueColor?: string
}

function StatCard({ label, value, subValue, variant = 'default', valueColor }: StatCardProps) {
  let borderClass = 'border-border'
  let textClass = valueColor || 'text-foreground'

  if (variant === 'success') {
    borderClass = 'border-success/30'
    textClass = valueColor || 'text-success'
  } else if (variant === 'danger') {
    borderClass = 'border-danger/30'
    textClass = valueColor || 'text-danger'
  }

  return (
    <div className={`bg-card border ${borderClass} rounded-lg p-4`}>
      <div className="text-sm text-muted-foreground mb-1">{label}</div>
      <div className={`text-2xl font-bold ${textClass}`}>{value}</div>
      {subValue && <div className="text-xs text-muted-foreground mt-1">{subValue}</div>}
    </div>
  )
}
