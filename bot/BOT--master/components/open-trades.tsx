"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { TrendingUp, TrendingDown, X, Activity } from "lucide-react"

interface Trade {
  id: string
  symbol: string
  type: "BUY" | "SELL"
  lots: number
  openPrice: number
  currentPrice: number
  pnl: number
  openTime: string
  strategy: string
}

export default function OpenTrades() {
  const [trades, setTrades] = useState<Trade[]>([
    {
      id: "1",
      symbol: "XAUUSD",
      type: "BUY",
      lots: 0.1,
      openPrice: 2043.45,
      currentPrice: 2045.67,
      pnl: 22.2,
      openTime: "10:30",
      strategy: "MMXM",
    },
    {
      id: "2",
      symbol: "XAUUSD",
      type: "SELL",
      lots: 0.05,
      openPrice: 2046.12,
      currentPrice: 2045.67,
      pnl: 2.25,
      openTime: "11:15",
      strategy: "Judas",
    },
    {
      id: "3",
      symbol: "BTCUSD",
      type: "BUY",
      lots: 0.02,
      openPrice: 43250.89,
      currentPrice: 43387.45,
      pnl: 27.31,
      openTime: "12:45",
      strategy: "OTE",
    },
  ])

  // Simulate real-time price updates
  useEffect(() => {
    const interval = setInterval(() => {
      setTrades((prevTrades) =>
        prevTrades.map((trade) => {
          const priceChange = (Math.random() - 0.5) * (trade.symbol === "BTCUSD" ? 50 : 1)
          const newPrice = trade.currentPrice + priceChange
          const priceDiff = newPrice - trade.openPrice
          const multiplier = trade.symbol === "BTCUSD" ? 1 : 100
          const newPnl =
            trade.type === "BUY" ? priceDiff * trade.lots * multiplier : -priceDiff * trade.lots * multiplier

          return {
            ...trade,
            currentPrice: newPrice,
            pnl: newPnl,
          }
        }),
      )
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  const totalPnL = trades.reduce((sum, trade) => sum + trade.pnl, 0)

  return (
    <Card className="glassmorphism">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center">
            <Activity className="w-5 h-5 mr-2 text-primary" />
            Open Trades
          </CardTitle>
          <Badge
            className={`${totalPnL >= 0 ? "bg-green-500/10 text-green-500 border-green-500/20" : "bg-red-500/10 text-red-500 border-red-500/20"}`}
          >
            Total: ${totalPnL.toFixed(2)}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {trades.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">No open trades</div>
        ) : (
          trades.map((trade) => (
            <div key={trade.id} className="border rounded-lg p-4 space-y-3 glassmorphism">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Badge variant={trade.type === "BUY" ? "default" : "destructive"}>{trade.type}</Badge>
                  <span className="font-semibold">{trade.symbol}</span>
                  <span className="text-sm text-muted-foreground">{trade.lots} lots</span>
                  <Badge variant="outline" className="text-xs">
                    {trade.strategy}
                  </Badge>
                </div>
                <Button size="sm" variant="ghost" className="h-6 w-6 p-0">
                  <X className="h-4 w-4" />
                </Button>
              </div>

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-muted-foreground">Open:</span>
                  <div className="font-mono">{trade.openPrice.toFixed(2)}</div>
                </div>
                <div>
                  <span className="text-muted-foreground">Current:</span>
                  <div className="font-mono">{trade.currentPrice.toFixed(2)}</div>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  {trade.pnl >= 0 ? (
                    <TrendingUp className="w-4 h-4 text-green-500" />
                  ) : (
                    <TrendingDown className="w-4 h-4 text-red-500" />
                  )}
                  <span className={`font-semibold ${trade.pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                    ${trade.pnl.toFixed(2)}
                  </span>
                </div>
                <span className="text-xs text-muted-foreground">{trade.openTime}</span>
              </div>
            </div>
          ))
        )}
      </CardContent>
    </Card>
  )
}
