'use client'

import { useState } from 'react'
import StatsCards from '@/components/stats-cards'
import EquityChart from '@/components/equity-chart'
import TradesTable from '@/components/trades-table'
import TPDistributionChart from '@/components/tp-distribution'
import TimeRangeSelector, { TimeRange } from '@/components/time-range-selector'

export default function Dashboard() {
  const [timeRange, setTimeRange] = useState<TimeRange>('1M')

  return (
    <main className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <h1 className="text-2xl font-bold text-foreground">
                OB Scalper Dashboard
              </h1>
              <p className="text-sm text-muted-foreground">
                Real-time trading performance
              </p>
            </div>
            <TimeRangeSelector selected={timeRange} onSelect={setTimeRange} />
          </div>
        </div>
      </header>

      {/* Content */}
      <div className="container mx-auto px-4 py-6 space-y-6">
        {/* Stats Cards */}
        <section>
          <StatsCards timeRange={timeRange} />
        </section>

        {/* Charts Row */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <EquityChart timeRange={timeRange} />
          </div>
          <div>
            <TPDistributionChart timeRange={timeRange} />
          </div>
        </section>

        {/* Trades Table */}
        <section>
          <TradesTable timeRange={timeRange} />
        </section>
      </div>

      {/* Footer */}
      <footer className="border-t border-border py-4 mt-8">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          OB Scalper Bot • Auto-refreshes every 30s
        </div>
      </footer>
    </main>
  )
}
