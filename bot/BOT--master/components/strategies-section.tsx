"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Search, Brain, Clock, Target } from "lucide-react"

const strategies = [
  {
    id: 1,
    name: "Judas Swing Setup",
    icon: "📉",
    type: "Scalping",
    focus: "XAUUSD 1M / 5M",
    description: "Detects fake breakouts and captures early reversals using liquidity sweeps and session-based traps.",
    concepts: "Liquidity sweeps, session traps, fake breakouts",
    filters: "Volume spike, news filter, ML reversal detection",
    tags: ["Gold", "Scalping", "Advanced"],
    winRate: 87.3,
    avgRR: 2.4,
    maxDD: 4.2,
    timeframe: "1M-5M",
    sessions: "London/NY Open",
  },
  {
    id: 2,
    name: "Optimal Trade Entry (OTE)",
    icon: "⏳",
    type: "Swing",
    focus: "XAUUSD 5M / 15M",
    description: "Fibonacci-based sniper entry after institutional retracement aligned with OBs & CHOCH.",
    concepts: "Fibonacci retracement, OB alignment, CHOCH",
    filters: "HTF trend filter, volume confirmation, news avoidance",
    tags: ["Gold", "Fibonacci", "Intermediate"],
    winRate: 82.1,
    avgRR: 3.1,
    maxDD: 3.8,
    timeframe: "5M-15M",
    sessions: "All Sessions",
  },
  {
    id: 3,
    name: "Order Block + CHOCH Combo",
    icon: "🧱",
    type: "Structure",
    focus: "XAUUSD 15M / 1H",
    description: "Identifies higher timeframe OBs and confirms entry with low-TF CHOCH.",
    concepts: "Order blocks, CHOCH, structure breaks",
    filters: "HTF bias, volume profile, session timing",
    tags: ["Gold", "Structure", "Intermediate"],
    winRate: 79.8,
    avgRR: 2.8,
    maxDD: 5.1,
    timeframe: "15M-1H",
    sessions: "London/NY",
  },
  {
    id: 4,
    name: "AMD (Accumulation–Manipulation–Distribution)",
    icon: "🔄",
    type: "Cycle",
    focus: "XAUUSD 1H / 4H",
    description: "Trades based on market cycle logic, ideal for high/low volatility zones.",
    concepts: "Market cycles, accumulation zones, distribution",
    filters: "Volatility zones, institutional flow, cycle timing",
    tags: ["Gold", "Swing", "Advanced"],
    winRate: 76.4,
    avgRR: 4.2,
    maxDD: 6.3,
    timeframe: "1H-4H",
    sessions: "All Sessions",
  },
  {
    id: 5,
    name: "MSB Retest",
    icon: "📊",
    type: "Structure",
    focus: "XAUUSD 5M / 15M",
    description: "Enters on structure break and retest into OB/FVG zones with trend alignment.",
    concepts: "Market structure breaks, retests, OB/FVG zones",
    filters: "Trend alignment, volume confirmation, retest quality",
    tags: ["Gold", "Structure", "Beginner"],
    winRate: 81.7,
    avgRR: 2.6,
    maxDD: 4.7,
    timeframe: "5M-15M",
    sessions: "London/NY",
  },
  {
    id: 6,
    name: "MMXM (Market Maker X Manipulation)",
    icon: "🎯",
    type: "Manipulation",
    focus: "XAUUSD 1M / 5M",
    description: "Uses volume spikes + manipulation zones to enter high-probability reversal trades.",
    concepts: "Market maker manipulation, liquidity grabs, reversals",
    filters: "Volume spikes, manipulation zones, ML confirmation",
    tags: ["Gold", "Scalping", "Advanced"],
    winRate: 85.2,
    avgRR: 2.9,
    maxDD: 5.8,
    timeframe: "1M-5M",
    sessions: "London/NY Open",
  },
  {
    id: 7,
    name: "MMC Combo",
    icon: "⚡",
    type: "Combo",
    focus: "XAUUSD 5M / 15M",
    description:
      "Combines smart money concepts (OB + CHOCH + liquidity grab + engulfing) for gold volatility sniper trades.",
    concepts: "Smart money concepts, OB + CHOCH + liquidity + engulfing",
    filters: "Volatility filter, news avoidance, session timing",
    tags: ["Gold", "Combo", "Advanced"],
    winRate: 83.9,
    avgRR: 3.4,
    maxDD: 4.9,
    timeframe: "5M-15M",
    sessions: "All Sessions",
  },
]

const filterTags = [
  "All",
  "Gold",
  "Scalping",
  "Beginner",
  "Intermediate",
  "Advanced",
  "Structure",
  "Fibonacci",
  "Swing",
  "Combo",
]

export default function StrategiesSection() {
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedTag, setSelectedTag] = useState("All")

  const filteredStrategies = strategies.filter((strategy) => {
    const matchesSearch =
      strategy.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      strategy.concepts.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesTag = selectedTag === "All" || strategy.tags.includes(selectedTag)
    return matchesSearch && matchesTag
  })

  return (
    <div className="space-y-8">
      {/* Search and Filters */}
      <div className="space-y-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
          <Input
            placeholder="Search strategies..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10"
          />
        </div>

        <div className="flex flex-wrap gap-2">
          {filterTags.map((tag) => (
            <Button
              key={tag}
              size="sm"
              variant={selectedTag === tag ? "default" : "outline"}
              onClick={() => setSelectedTag(tag)}
              className={selectedTag === tag ? "gradient-primary text-black" : ""}
            >
              {tag}
            </Button>
          ))}
        </div>
      </div>

      {/* Strategies Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {filteredStrategies.map((strategy) => (
          <div key={strategy.id} className="flip-card h-80">
            <div className="flip-card-inner">
              {/* Front of Card */}
              <Card className="flip-card-front glassmorphism border-primary/20">
                <CardHeader className="text-center">
                  <div className="text-4xl mb-2">{strategy.icon}</div>
                  <CardTitle className="text-lg">{strategy.name}</CardTitle>
                  <Badge variant="outline" className="w-fit mx-auto">
                    {strategy.type}
                  </Badge>
                </CardHeader>
                <CardContent className="text-center space-y-4">
                  <p className="text-sm text-muted-foreground">{strategy.description}</p>
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div>
                      <div className="font-bold text-green-500">{strategy.winRate}%</div>
                      <div className="text-muted-foreground">Win Rate</div>
                    </div>
                    <div>
                      <div className="font-bold text-primary">{strategy.avgRR}</div>
                      <div className="text-muted-foreground">Avg RR</div>
                    </div>
                    <div>
                      <div className="font-bold text-red-500">{strategy.maxDD}%</div>
                      <div className="text-muted-foreground">Max DD</div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Back of Card */}
              <Card className="flip-card-back glassmorphism border-primary/20">
                <CardHeader>
                  <CardTitle className="text-lg flex items-center">
                    <Brain className="w-5 h-5 mr-2 text-primary" />
                    Strategy Details
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 text-sm">
                  <div>
                    <span className="font-medium flex items-center">
                      <Clock className="w-4 h-4 mr-1" />
                      Timeframe:
                    </span>
                    <p className="text-muted-foreground">{strategy.timeframe}</p>
                  </div>
                  <div>
                    <span className="font-medium flex items-center">
                      <Target className="w-4 h-4 mr-1" />
                      Sessions:
                    </span>
                    <p className="text-muted-foreground">{strategy.sessions}</p>
                  </div>
                  <div>
                    <span className="font-medium">Core Concepts:</span>
                    <p className="text-muted-foreground">{strategy.concepts}</p>
                  </div>
                  <div>
                    <span className="font-medium">Smart Filters:</span>
                    <p className="text-muted-foreground">{strategy.filters}</p>
                  </div>

                  <Dialog>
                    <DialogTrigger asChild>
                      <Button size="sm" className="w-full gradient-primary text-black">
                        View Backtest
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="max-w-2xl">
                      <DialogHeader>
                        <DialogTitle className="flex items-center">
                          {strategy.icon} {strategy.name} - Backtest Results
                        </DialogTitle>
                      </DialogHeader>
                      <div className="space-y-4">
                        <div className="grid grid-cols-3 gap-4 text-center">
                          <div className="p-4 bg-green-500/10 rounded-lg">
                            <div className="text-2xl font-bold text-green-500">{strategy.winRate}%</div>
                            <div className="text-sm text-muted-foreground">Win Rate</div>
                          </div>
                          <div className="p-4 bg-primary/10 rounded-lg">
                            <div className="text-2xl font-bold text-primary">{strategy.avgRR}</div>
                            <div className="text-sm text-muted-foreground">Risk:Reward</div>
                          </div>
                          <div className="p-4 bg-red-500/10 rounded-lg">
                            <div className="text-2xl font-bold text-red-500">{strategy.maxDD}%</div>
                            <div className="text-sm text-muted-foreground">Max Drawdown</div>
                          </div>
                        </div>
                        <div className="h-64 bg-muted/20 rounded-lg flex items-center justify-center">
                          <p className="text-muted-foreground">Backtest Chart Visualization</p>
                        </div>
                      </div>
                    </DialogContent>
                  </Dialog>
                </CardContent>
              </Card>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
