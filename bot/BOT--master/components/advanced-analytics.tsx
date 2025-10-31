"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { BarChart3, TrendingUp, Clock, Target, Calendar } from "lucide-react"

interface StrategyPerformance {
  strategy: string
  pnl: number
  trades: number
  winRate: number
  avgWin: number
  avgLoss: number
  sharpeRatio: number
  maxDrawdown: number
  recoveryTime: number
}

interface TimeframeAnalysis {
  timeframe: string
  winRate: number
  avgPnL: number
  trades: number
  bestHour: string
}

interface HeatMapData {
  hour: number
  day: string
  pnl: number
  trades: number
}

export default function AdvancedAnalytics() {
  const [selectedPeriod, setSelectedPeriod] = useState("7D")
  const [selectedMetric, setSelectedMetric] = useState("pnl")

  const [strategyData, setStrategyData] = useState<StrategyPerformance[]>([
    {
      strategy: "MMXM",
      pnl: 1247.89,
      trades: 45,
      winRate: 87.5,
      avgWin: 67.34,
      avgLoss: -31.45,
      sharpeRatio: 2.34,
      maxDrawdown: 4.2,
      recoveryTime: 2.5,
    },
    {
      strategy: "OTE",
      pnl: 834.56,
      trades: 32,
      winRate: 82.1,
      avgWin: 54.23,
      avgLoss: -28.67,
      sharpeRatio: 1.98,
      maxDrawdown: 3.8,
      recoveryTime: 1.8,
    },
    {
      strategy: "Judas Swing",
      pnl: 567.23,
      trades: 28,
      winRate: 79.8,
      avgWin: 45.67,
      avgLoss: -25.34,
      sharpeRatio: 1.76,
      maxDrawdown: 5.1,
      recoveryTime: 3.2,
    },
  ])

  const [timeframeData, setTimeframeData] = useState<TimeframeAnalysis[]>([
    { timeframe: "1M", winRate: 85.2, avgPnL: 23.45, trades: 156, bestHour: "09:00-10:00" },
    { timeframe: "5M", winRate: 82.7, avgPnL: 45.67, trades: 89, bestHour: "14:00-15:00" },
    { timeframe: "15M", winRate: 79.3, avgPnL: 78.23, trades: 45, bestHour: "08:00-09:00" },
    { timeframe: "1H", winRate: 76.8, avgPnL: 134.56, trades: 23, bestHour: "13:00-14:00" },
  ])

  const [heatMapData, setHeatMapData] = useState<HeatMapData[]>([])

  // Generate heat map data
  useEffect(() => {
    const days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    const hours = Array.from({ length: 24 }, (_, i) => i)

    const data: HeatMapData[] = []
    days.forEach((day) => {
      hours.forEach((hour) => {
        // Simulate higher activity during London/NY sessions
        const isLondonSession = hour >= 8 && hour <= 17
        const isNYSession = hour >= 13 && hour <= 22
        const isOverlap = hour >= 13 && hour <= 17

        let basePnL = Math.random() * 100 - 30
        if (isOverlap) basePnL += 50
        else if (isLondonSession || isNYSession) basePnL += 20

        data.push({
          hour,
          day,
          pnl: basePnL,
          trades: Math.floor(Math.random() * 10) + (isOverlap ? 5 : 0),
        })
      })
    })

    setHeatMapData(data)
  }, [])

  const periods = ["1D", "7D", "1M", "3M", "1Y"]
  const metrics = [
    { value: "pnl", label: "P&L" },
    { value: "winRate", label: "Win Rate" },
    { value: "sharpe", label: "Sharpe Ratio" },
    { value: "drawdown", label: "Max Drawdown" },
  ]

  const getHeatMapColor = (pnl: number) => {
    if (pnl > 50) return "bg-green-500"
    if (pnl > 20) return "bg-green-400"
    if (pnl > 0) return "bg-green-300"
    if (pnl > -20) return "bg-red-300"
    if (pnl > -50) return "bg-red-400"
    return "bg-red-500"
  }

  return (
    <div className="space-y-6">
      {/* Controls */}
      <Card className="glassmorphism border-primary/20">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center">
              <BarChart3 className="w-5 h-5 mr-2 text-primary" />
              Advanced Analytics
            </CardTitle>
            <div className="flex items-center space-x-4">
              <Select value={selectedPeriod} onValueChange={setSelectedPeriod}>
                <SelectTrigger className="w-24">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {periods.map((period) => (
                    <SelectItem key={period} value={period}>
                      {period}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select value={selectedMetric} onValueChange={setSelectedMetric}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {metrics.map((metric) => (
                    <SelectItem key={metric.value} value={metric.value}>
                      {metric.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Strategy Performance Breakdown */}
      <Card className="glassmorphism border-primary/20">
        <CardHeader>
          <CardTitle className="flex items-center">
            <Target className="w-5 h-5 mr-2 text-primary" />
            Strategy Performance Breakdown
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {strategyData.map((strategy, index) => (
              <div key={index} className="border border-border/40 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <Badge className="gradient-primary text-black">{strategy.strategy}</Badge>
                    <span className="font-semibold">{strategy.trades} trades</span>
                  </div>
                  <div className={`text-lg font-bold ${strategy.pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                    ${strategy.pnl.toFixed(2)}
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Win Rate:</span>
                    <div className="font-bold text-green-500">{strategy.winRate}%</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Avg Win:</span>
                    <div className="font-bold text-green-500">${strategy.avgWin}</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Avg Loss:</span>
                    <div className="font-bold text-red-500">${strategy.avgLoss}</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Sharpe:</span>
                    <div className="font-bold text-primary">{strategy.sharpeRatio}</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Max DD:</span>
                    <div className="font-bold text-red-500">{strategy.maxDrawdown}%</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Recovery:</span>
                    <div className="font-bold">{strategy.recoveryTime}d</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">R:R Ratio:</span>
                    <div className="font-bold text-primary">
                      {(strategy.avgWin / Math.abs(strategy.avgLoss)).toFixed(2)}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Timeframe Analysis */}
      <Card className="glassmorphism border-primary/20">
        <CardHeader>
          <CardTitle className="flex items-center">
            <Clock className="w-5 h-5 mr-2 text-primary" />
            Win/Loss Ratio by Timeframe
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {timeframeData.map((tf, index) => (
              <div key={index} className="border border-border/40 rounded-lg p-4 text-center">
                <div className="text-lg font-bold mb-2">{tf.timeframe}</div>
                <div className="space-y-2">
                  <div>
                    <span className="text-sm text-muted-foreground">Win Rate:</span>
                    <div className="font-bold text-green-500">{tf.winRate}%</div>
                  </div>
                  <div>
                    <span className="text-sm text-muted-foreground">Avg P&L:</span>
                    <div className="font-bold">${tf.avgPnL}</div>
                  </div>
                  <div>
                    <span className="text-sm text-muted-foreground">Trades:</span>
                    <div className="font-bold">{tf.trades}</div>
                  </div>
                  <div>
                    <span className="text-sm text-muted-foreground">Best Hour:</span>
                    <div className="font-bold text-primary text-xs">{tf.bestHour}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Heat Map of Profitable Trading Hours */}
      <Card className="glassmorphism border-primary/20">
        <CardHeader>
          <CardTitle className="flex items-center">
            <Calendar className="w-5 h-5 mr-2 text-primary" />
            Heat Map: Profitable Trading Hours
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between text-sm">
              <span>Hours (GMT)</span>
              <div className="flex items-center space-x-2">
                <span>Loss</span>
                <div className="flex space-x-1">
                  <div className="w-3 h-3 bg-red-500 rounded"></div>
                  <div className="w-3 h-3 bg-red-400 rounded"></div>
                  <div className="w-3 h-3 bg-red-300 rounded"></div>
                  <div className="w-3 h-3 bg-green-300 rounded"></div>
                  <div className="w-3 h-3 bg-green-400 rounded"></div>
                  <div className="w-3 h-3 bg-green-500 rounded"></div>
                </div>
                <span>Profit</span>
              </div>
            </div>

            <div className="overflow-x-auto">
              <div className="grid grid-cols-25 gap-1 min-w-max">
                {/* Hour headers */}
                <div></div>
                {Array.from({ length: 24 }, (_, i) => (
                  <div key={i} className="text-xs text-center p-1 font-mono">
                    {i.toString().padStart(2, "0")}
                  </div>
                ))}

                {/* Heat map rows */}
                {["Mon", "Tue", "Wed", "Thu", "Fri"].map((day) => (
                  <div key={day} className="contents">
                    <div className="text-xs p-1 font-medium">{day}</div>
                    {Array.from({ length: 24 }, (_, hour) => {
                      const data = heatMapData.find((d) => d.day === day && d.hour === hour)
                      return (
                        <div
                          key={hour}
                          className={`w-6 h-6 rounded ${getHeatMapColor(data?.pnl || 0)} opacity-80 hover:opacity-100 cursor-pointer transition-opacity`}
                          title={`${day} ${hour}:00 - P&L: $${data?.pnl.toFixed(2)} (${data?.trades} trades)`}
                        />
                      )
                    })}
                  </div>
                ))}
              </div>
            </div>

            <div className="text-xs text-muted-foreground">
              💡 Darker green = Higher profits | Red = Losses | Best performance during London/NY overlap (13:00-17:00
              GMT)
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Drawdown Analysis */}
      <Card className="glassmorphism border-primary/20">
        <CardHeader>
          <CardTitle className="flex items-center">
            <TrendingUp className="w-5 h-5 mr-2 text-primary" />
            Drawdown Analysis & Recovery Time
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64 bg-muted/20 rounded-lg flex items-center justify-center mb-4">
            <p className="text-muted-foreground">Drawdown Recovery Chart Visualization</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
            <div className="p-4 border border-border/40 rounded-lg">
              <div className="text-2xl font-bold text-red-500">-4.2%</div>
              <div className="text-sm text-muted-foreground">Max Drawdown</div>
            </div>
            <div className="p-4 border border-border/40 rounded-lg">
              <div className="text-2xl font-bold text-primary">2.5 days</div>
              <div className="text-sm text-muted-foreground">Avg Recovery Time</div>
            </div>
            <div className="p-4 border border-border/40 rounded-lg">
              <div className="text-2xl font-bold text-green-500">1.8</div>
              <div className="text-sm text-muted-foreground">Recovery Factor</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
