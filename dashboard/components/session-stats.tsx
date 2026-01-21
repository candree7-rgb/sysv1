'use client'

import { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { TimeRange, TIME_RANGES } from './time-range-selector'
import { formatCurrency } from '@/lib/utils'

interface SessionStatsProps {
  timeRange: TimeRange
}

interface SessionData {
  session: string
  trades: number
  wins: number
  pnl: number
  win_rate: number
}

export default function SessionStats({ timeRange }: SessionStatsProps) {
  const [data, setData] = useState<SessionData[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchData() {
      try {
        const range = TIME_RANGES.find(r => r.value === timeRange)
        const days = range?.days
        const params = days ? `?days=${days}` : ''

        const res = await fetch(`/api/stats${params}`)
        const stats = await res.json()

        // Create session data from stats
        if (stats.sessions) {
          setData(stats.sessions)
        }
      } catch (error) {
        console.error('Failed to fetch session stats:', error)
      } finally {
        setLoading(false)
      }
    }

    setLoading(true)
    fetchData()
    const interval = setInterval(fetchData, 60000)
    return () => clearInterval(interval)
  }, [timeRange])

  if (loading) {
    return (
      <div className="bg-card border border-border rounded-lg p-6">
        <div className="h-8 bg-muted rounded w-1/3 mb-4 animate-pulse"></div>
        <div className="h-48 bg-muted rounded animate-pulse"></div>
      </div>
    )
  }

  if (data.length === 0) {
    return (
      <div className="bg-card border border-border rounded-lg p-6">
        <h2 className="text-xl font-bold mb-4">Session Performance</h2>
        <div className="h-48 flex items-center justify-center text-muted-foreground">
          No session data available
        </div>
      </div>
    )
  }

  return (
    <div className="bg-card border border-border rounded-lg p-6">
      <h2 className="text-xl font-bold mb-2">Session Performance</h2>
      <p className="text-sm text-muted-foreground mb-4">
        Performance by trading session (UTC)
      </p>

      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" stroke="#333" opacity={0.1} />
          <XAxis type="number" stroke="#888" fontSize={12} tickFormatter={(v) => `$${v}`} />
          <YAxis type="category" dataKey="session" stroke="#888" fontSize={12} width={80} />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="pnl" radius={[0, 4, 4, 0]}>
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.pnl >= 0 ? '#22c55e' : '#ef4444'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t border-border">
        {data.map((d) => (
          <div key={d.session} className="text-center">
            <div className="text-xs text-muted-foreground mb-1">{d.session}</div>
            <div className={`text-lg font-bold ${d.pnl >= 0 ? 'text-success' : 'text-danger'}`}>
              {d.win_rate.toFixed(0)}%
            </div>
            <div className="text-xs text-muted-foreground">{d.trades} trades</div>
          </div>
        ))}
      </div>
    </div>
  )
}

function CustomTooltip({ active, payload }: any) {
  if (!active || !payload || !payload[0]) return null

  const data = payload[0].payload
  return (
    <div className="bg-card border border-border rounded-lg p-3 shadow-lg">
      <p className="text-sm font-semibold mb-1">{data.session}</p>
      <p className="text-sm">Trades: <span className="font-bold">{data.trades}</span></p>
      <p className="text-sm">Win Rate: <span className="font-bold">{data.win_rate.toFixed(1)}%</span></p>
      <p className={`text-sm ${data.pnl >= 0 ? 'text-success' : 'text-danger'}`}>
        PnL: <span className="font-bold">{formatCurrency(data.pnl)}</span>
      </p>
    </div>
  )
}
