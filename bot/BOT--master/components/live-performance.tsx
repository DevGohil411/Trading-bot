"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { TrendingUp, Activity, DollarSign } from "lucide-react"

export default function LivePerformance() {
  const [botActive, setBotActive] = useState(true)
  const [pnl, setPnl] = useState(2847.32)
  const [winRate, setWinRate] = useState(78.4)
  const [openTrades, setOpenTrades] = useState(3)

  // Simulate live updates
  useEffect(() => {
    const interval = setInterval(() => {
      if (botActive) {
        setPnl((prev) => prev + (Math.random() - 0.5) * 10)
        setWinRate((prev) => Math.max(70, Math.min(85, prev + (Math.random() - 0.5) * 0.5)))
      }
    }, 3000)

    return () => clearInterval(interval)
  }, [botActive])

  return (
    <section className="py-16 bg-muted/30">
      <div className="container mx-auto px-4">
        <div className="text-center mb-12">
          <Badge className="mb-4 bg-green-500/10 text-green-500 border-green-500/20">Live Performance</Badge>
          <h2 className="text-3xl font-bold mb-4">Real-Time Trading Results</h2>
          <p className="text-muted-foreground">Watch SniprX in action with live MT5 integration</p>
        </div>

        <div className="max-w-4xl mx-auto">
          {/* Bot Control */}
          <Card className="glassmorphism mb-8">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className={`w-3 h-3 rounded-full ${botActive ? "bg-green-500" : "bg-red-500"} animate-pulse`} />
                  <div>
                    <h3 className="font-semibold">Bot Status</h3>
                    <p className="text-sm text-muted-foreground">
                      {botActive ? "Active - Monitoring Markets" : "Inactive - Standby Mode"}
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  <span className="text-sm font-medium">OFF</span>
                  <Switch
                    checked={botActive}
                    onCheckedChange={setBotActive}
                    className="data-[state=checked]:bg-primary"
                  />
                  <span className="text-sm font-medium">ON</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Performance Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="glassmorphism">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium flex items-center">
                  <DollarSign className="w-4 h-4 mr-2 text-primary" />
                  Total P&L
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-green-500">${pnl.toFixed(2)}</div>
                <div className="flex items-center text-sm text-muted-foreground mt-1">
                  <TrendingUp className="w-4 h-4 mr-1" />
                  +12.4% this month
                </div>
              </CardContent>
            </Card>

            <Card className="glassmorphism">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium flex items-center">
                  <Activity className="w-4 h-4 mr-2 text-primary" />
                  Win Rate
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{winRate.toFixed(1)}%</div>
                <div className="flex items-center text-sm text-muted-foreground mt-1">
                  <TrendingUp className="w-4 h-4 mr-1" />
                  Above average
                </div>
              </CardContent>
            </Card>

            <Card className="glassmorphism">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium flex items-center">
                  <Activity className="w-4 h-4 mr-2 text-primary" />
                  Open Trades
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{openTrades}</div>
                <div className="flex items-center text-sm text-muted-foreground mt-1">
                  <Activity className="w-4 h-4 mr-1" />
                  XAUUSD positions
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="text-center mt-8">
            <Button className="gold-gradient text-black hover:opacity-90">Connect Your MT5 Account</Button>
          </div>
        </div>
      </div>
    </section>
  )
}
