"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { TrendingUp, Activity, DollarSign, Target } from "lucide-react"

export default function PerformanceSection() {
  const [stats, setStats] = useState({
    totalPnL: 47832.45,
    winRate: 84.2,
    activeUsers: 2847,
    totalTrades: 15634,
  })

  // Simulate live updates
  useEffect(() => {
    const interval = setInterval(() => {
      setStats((prev) => ({
        ...prev,
        totalPnL: prev.totalPnL + (Math.random() - 0.3) * 50,
        winRate: Math.max(80, Math.min(90, prev.winRate + (Math.random() - 0.5) * 0.5)),
      }))
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  return (
    <section className="py-16 bg-muted/30">
      <div className="container mx-auto px-4">
        <div className="text-center mb-12">
          <Badge className="mb-4 bg-green-500/10 text-green-500 border-green-500/20">Live Performance</Badge>
          <h2 className="text-3xl font-bold mb-4">Real-Time Trading Results</h2>
          <p className="text-muted-foreground">Watch SniprX in action with live MT5 integration</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto">
          <Card className="glassmorphism">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center">
                <DollarSign className="w-4 h-4 mr-2 text-green-500" />
                Community P&L
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-500">${stats.totalPnL.toLocaleString()}</div>
              <div className="flex items-center text-sm text-muted-foreground mt-1">
                <TrendingUp className="w-4 h-4 mr-1" />
                +18.4% this month
              </div>
            </CardContent>
          </Card>

          <Card className="glassmorphism">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center">
                <Target className="w-4 h-4 mr-2 text-primary" />
                Win Rate
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.winRate.toFixed(1)}%</div>
              <div className="flex items-center text-sm text-muted-foreground mt-1">
                <TrendingUp className="w-4 h-4 mr-1" />
                Above industry avg
              </div>
            </CardContent>
          </Card>

          <Card className="glassmorphism">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center">
                <Activity className="w-4 h-4 mr-2 text-primary" />
                Active Users
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.activeUsers.toLocaleString()}</div>
              <div className="flex items-center text-sm text-muted-foreground mt-1">
                <TrendingUp className="w-4 h-4 mr-1" />
                Growing daily
              </div>
            </CardContent>
          </Card>

          <Card className="glassmorphism">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center">
                <Activity className="w-4 h-4 mr-2 text-primary" />
                Total Trades
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.totalTrades.toLocaleString()}</div>
              <div className="flex items-center text-sm text-muted-foreground mt-1">
                <Activity className="w-4 h-4 mr-1" />
                Executed today
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  )
}
