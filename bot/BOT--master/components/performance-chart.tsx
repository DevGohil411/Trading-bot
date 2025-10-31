"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { TrendingUp, Calendar } from "lucide-react"

export default function PerformanceChart() {
  const [timeRange, setTimeRange] = useState("7D")
  const [chartData, setChartData] = useState([
    { day: "Mon", pnl: 156.32, trades: 8 },
    { day: "Tue", pnl: 289.67, trades: 12 },
    { day: "Wed", pnl: -45.23, trades: 6 },
    { day: "Thu", pnl: 423.89, trades: 15 },
    { day: "Fri", pnl: 204.67, trades: 11 },
    { day: "Sat", pnl: 178.45, trades: 7 },
    { day: "Sun", pnl: 92.34, trades: 4 },
  ])

  const timeRanges = ["1D", "7D", "1M", "3M", "1Y"]

  // Simulate data updates
  useEffect(() => {
    const interval = setInterval(() => {
      setChartData((prev) =>
        prev.map((item, index) => ({
          ...item,
          pnl: index === prev.length - 1 ? item.pnl + (Math.random() - 0.3) * 30 : item.pnl,
        })),
      )
    }, 8000)

    return () => clearInterval(interval)
  }, [])

  const maxPnL = Math.max(...chartData.map((d) => d.pnl))
  const minPnL = Math.min(...chartData.map((d) => d.pnl))
  const range = maxPnL - minPnL

  return (
    <Card className="glassmorphism border-primary/20">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center">
            <Calendar className="w-5 h-5 mr-2 text-primary" />
            Performance Chart
          </CardTitle>
          <div className="flex space-x-1">
            {timeRanges.map((range) => (
              <Button
                key={range}
                size="sm"
                variant={timeRange === range ? "default" : "outline"}
                onClick={() => setTimeRange(range)}
                className={timeRange === range ? "gradient-primary text-black" : ""}
              >
                {range}
              </Button>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-64 flex items-end space-x-2">
          {chartData.map((item, index) => {
            const height = range > 0 ? Math.max(10, ((item.pnl - minPnL) / range) * 200) : 50
            const isPositive = item.pnl >= 0
            return (
              <div key={index} className="flex-1 flex flex-col items-center">
                <div className="text-xs text-muted-foreground mb-2">${item.pnl.toFixed(0)}</div>
                <div
                  className={`w-full rounded-t-lg ${
                    isPositive
                      ? "bg-gradient-to-t from-green-500 to-green-400"
                      : "bg-gradient-to-t from-red-500 to-red-400"
                  } transition-all duration-500 hover:opacity-80`}
                  style={{ height: `${height}px` }}
                />
                <div className="text-xs text-muted-foreground mt-2">{item.day}</div>
                <div className="text-xs text-muted-foreground">{item.trades} trades</div>
              </div>
            )
          })}
        </div>
        <div className="mt-4 flex items-center justify-between text-sm">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-green-500 rounded"></div>
              <span className="text-muted-foreground">Profit Days</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-red-500 rounded"></div>
              <span className="text-muted-foreground">Loss Days</span>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <TrendingUp className="w-4 h-4 text-green-500" />
            <span className="text-green-500 font-medium">
              +{(chartData.reduce((sum, d) => sum + d.pnl, 0) / chartData.length / 10).toFixed(1)}%
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
