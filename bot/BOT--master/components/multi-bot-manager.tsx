"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Bot, Plus, Settings, Activity, Target, Trash2 } from "lucide-react"

interface TradingBot {
  id: string
  name: string
  strategy: string
  symbol: string
  timeframe: string
  status: "active" | "inactive" | "paused"
  pnl: number
  trades: number
  winRate: number
  lotSize: number
  riskLevel: number
  lastSignal: string
}

export default function MultiBotManager() {
  const [bots, setBots] = useState<TradingBot[]>([
    {
      id: "bot-1",
      name: "Gold Hunter",
      strategy: "MMXM",
      symbol: "XAUUSD",
      timeframe: "1M",
      status: "active",
      pnl: 847.32,
      trades: 23,
      winRate: 87.5,
      lotSize: 0.1,
      riskLevel: 2,
      lastSignal: "BUY @ 2045.67",
    },
    {
      id: "bot-2",
      name: "EUR Sniper",
      strategy: "OTE",
      symbol: "EURUSD",
      timeframe: "15M",
      status: "active",
      pnl: 234.56,
      trades: 12,
      winRate: 83.3,
      lotSize: 0.2,
      riskLevel: 1.5,
      lastSignal: "SELL @ 1.0892",
    },
    {
      id: "bot-3",
      name: "BTC Judas",
      strategy: "Judas Swing",
      symbol: "BTCUSD",
      timeframe: "5M",
      status: "paused",
      pnl: -45.23,
      trades: 8,
      winRate: 62.5,
      lotSize: 0.05,
      riskLevel: 3,
      lastSignal: "Waiting...",
    },
  ])

  const [showAddBot, setShowAddBot] = useState(false)
  const [newBot, setNewBot] = useState({
    name: "",
    strategy: "",
    symbol: "",
    timeframe: "",
    lotSize: 0.1,
    riskLevel: 2,
  })

  const [portfolioStats, setPortfolioStats] = useState({
    totalPnL: 0,
    totalTrades: 0,
    avgWinRate: 0,
    activeBotsCount: 0,
    portfolioRisk: 0,
  })

  // Calculate portfolio stats
  useEffect(() => {
    const totalPnL = bots.reduce((sum, bot) => sum + bot.pnl, 0)
    const totalTrades = bots.reduce((sum, bot) => sum + bot.trades, 0)
    const avgWinRate = bots.length > 0 ? bots.reduce((sum, bot) => sum + bot.winRate, 0) / bots.length : 0
    const activeBotsCount = bots.filter((bot) => bot.status === "active").length
    const portfolioRisk = bots.reduce((sum, bot) => sum + (bot.status === "active" ? bot.riskLevel : 0), 0)

    setPortfolioStats({
      totalPnL,
      totalTrades,
      avgWinRate,
      activeBotsCount,
      portfolioRisk,
    })
  }, [bots])

  const strategies = ["MMXM", "OTE", "Judas Swing", "AMD", "MSB Retest", "Order Block + CHOCH", "MMC Combo"]
  const symbols = ["XAUUSD", "USDJPY", "BTCUSD", "ETHUSD", "SPX500"]
  const timeframes = ["1M", "5M", "15M", "30M", "1H", "4H", "1D"]

  const toggleBotStatus = (botId: string) => {
    setBots((prev) =>
      prev.map((bot) => (bot.id === botId ? { ...bot, status: bot.status === "active" ? "paused" : "active" } : bot)),
    )
  }

  const addNewBot = () => {
    const bot: TradingBot = {
      id: `bot-${Date.now()}`,
      name: newBot.name || `${newBot.strategy} Bot`,
      strategy: newBot.strategy,
      symbol: newBot.symbol,
      timeframe: newBot.timeframe,
      status: "inactive",
      pnl: 0,
      trades: 0,
      winRate: 0,
      lotSize: newBot.lotSize,
      riskLevel: newBot.riskLevel,
      lastSignal: "Initializing...",
    }

    setBots((prev) => [...prev, bot])
    setNewBot({ name: "", strategy: "", symbol: "", timeframe: "", lotSize: 0.1, riskLevel: 2 })
    setShowAddBot(false)
  }

  const deleteBot = (botId: string) => {
    setBots((prev) => prev.filter((bot) => bot.id !== botId))
  }

  return (
    <div className="space-y-6">
      {/* Portfolio Overview */}
      <Card className="glassmorphism border-primary/20">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center">
              <Target className="w-5 h-5 mr-2 text-primary" />
              Portfolio Overview
            </CardTitle>
            <Dialog open={showAddBot} onOpenChange={setShowAddBot}>
              <DialogTrigger asChild>
                <Button className="gradient-primary text-black">
                  <Plus className="w-4 h-4 mr-2" />
                  Add Bot
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Create New Trading Bot</DialogTitle>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <Label>Bot Name</Label>
                    <Input
                      placeholder="e.g., Gold Hunter Pro"
                      value={newBot.name}
                      onChange={(e) => setNewBot((prev) => ({ ...prev, name: e.target.value }))}
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label>Strategy</Label>
                      <Select
                        value={newBot.strategy}
                        onValueChange={(value) => setNewBot((prev) => ({ ...prev, strategy: value }))}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select strategy" />
                        </SelectTrigger>
                        <SelectContent>
                          {strategies.map((strategy) => (
                            <SelectItem key={strategy} value={strategy}>
                              {strategy}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div>
                      <Label>Symbol</Label>
                      <Select
                        value={newBot.symbol}
                        onValueChange={(value) => setNewBot((prev) => ({ ...prev, symbol: value }))}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select symbol" />
                        </SelectTrigger>
                        <SelectContent>
                          {symbols.map((symbol) => (
                            <SelectItem key={symbol} value={symbol}>
                              {symbol}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <Label>Timeframe</Label>
                      <Select
                        value={newBot.timeframe}
                        onValueChange={(value) => setNewBot((prev) => ({ ...prev, timeframe: value }))}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="TF" />
                        </SelectTrigger>
                        <SelectContent>
                          {timeframes.map((tf) => (
                            <SelectItem key={tf} value={tf}>
                              {tf}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div>
                      <Label>Lot Size</Label>
                      <Input
                        type="number"
                        step="0.01"
                        value={newBot.lotSize}
                        onChange={(e) => setNewBot((prev) => ({ ...prev, lotSize: Number(e.target.value) }))}
                      />
                    </div>

                    <div>
                      <Label>Risk Level</Label>
                      <Input
                        type="number"
                        step="0.5"
                        value={newBot.riskLevel}
                        onChange={(e) => setNewBot((prev) => ({ ...prev, riskLevel: Number(e.target.value) }))}
                      />
                    </div>
                  </div>

                  <Button
                    onClick={addNewBot}
                    disabled={!newBot.strategy || !newBot.symbol || !newBot.timeframe}
                    className="w-full gradient-primary text-black"
                  >
                    Create Bot
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="text-center">
              <div className={`text-2xl font-bold ${portfolioStats.totalPnL >= 0 ? "text-green-500" : "text-red-500"}`}>
                ${portfolioStats.totalPnL.toFixed(2)}
              </div>
              <div className="text-sm text-muted-foreground">Total P&L</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">{portfolioStats.activeBotsCount}</div>
              <div className="text-sm text-muted-foreground">Active Bots</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold">{portfolioStats.totalTrades}</div>
              <div className="text-sm text-muted-foreground">Total Trades</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-500">{portfolioStats.avgWinRate.toFixed(1)}%</div>
              <div className="text-sm text-muted-foreground">Avg Win Rate</div>
            </div>
            <div className="text-center">
              <div
                className={`text-2xl font-bold ${portfolioStats.portfolioRisk > 10 ? "text-red-500" : "text-green-500"}`}
              >
                {portfolioStats.portfolioRisk.toFixed(1)}%
              </div>
              <div className="text-sm text-muted-foreground">Portfolio Risk</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Individual Bots */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {bots.map((bot) => (
          <Card key={bot.id} className="glassmorphism border-primary/20">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <div
                    className={`w-3 h-3 rounded-full ${
                      bot.status === "active"
                        ? "bg-green-500"
                        : bot.status === "paused"
                          ? "bg-yellow-500"
                          : "bg-red-500"
                    } animate-pulse`}
                  />
                  <CardTitle className="text-lg">{bot.name}</CardTitle>
                </div>
                <div className="flex items-center space-x-2">
                  <Button size="sm" variant="ghost">
                    <Settings className="w-4 h-4" />
                  </Button>
                  <Button size="sm" variant="ghost" onClick={() => deleteBot(bot.id)}>
                    <Trash2 className="w-4 h-4 text-red-500" />
                  </Button>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <Badge variant="outline">{bot.strategy}</Badge>
                <Badge variant="outline">{bot.symbol}</Badge>
                <Badge variant="outline">{bot.timeframe}</Badge>
              </div>
            </CardHeader>

            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-muted-foreground">P&L:</span>
                  <div className={`font-bold ${bot.pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                    ${bot.pnl.toFixed(2)}
                  </div>
                </div>
                <div>
                  <span className="text-muted-foreground">Win Rate:</span>
                  <div className="font-bold">{bot.winRate.toFixed(1)}%</div>
                </div>
                <div>
                  <span className="text-muted-foreground">Trades:</span>
                  <div className="font-bold">{bot.trades}</div>
                </div>
                <div>
                  <span className="text-muted-foreground">Risk:</span>
                  <div className="font-bold">{bot.riskLevel}%</div>
                </div>
              </div>

              <div className="space-y-2">
                <div className="text-sm">
                  <span className="text-muted-foreground">Last Signal:</span>
                  <div className="font-mono text-xs">{bot.lastSignal}</div>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <Badge
                  className={
                    bot.status === "active"
                      ? "bg-green-500/10 text-green-500 border-green-500/20"
                      : bot.status === "paused"
                        ? "bg-yellow-500/10 text-yellow-500 border-yellow-500/20"
                        : "bg-red-500/10 text-red-500 border-red-500/20"
                  }
                >
                  {bot.status === "active" ? (
                    <>
                      <Activity className="w-3 h-3 mr-1" />
                      Active
                    </>
                  ) : bot.status === "paused" ? (
                    <>
                      <Bot className="w-3 h-3 mr-1" />
                      Paused
                    </>
                  ) : (
                    <>
                      <Bot className="w-3 h-3 mr-1" />
                      Inactive
                    </>
                  )}
                </Badge>

                <Switch
                  checked={bot.status === "active"}
                  onCheckedChange={() => toggleBotStatus(bot.id)}
                  className="data-[state=checked]:bg-primary"
                />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
