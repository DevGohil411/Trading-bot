"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Brain, TrendingUp, AlertTriangle, Target, Zap, Activity } from "lucide-react"

interface MarketSentiment {
  symbol: string
  sentiment: "bullish" | "bearish" | "neutral"
  confidence: number
  factors: string[]
}

interface AIInsight {
  type: "entry_timing" | "risk_warning" | "strategy_switch" | "market_condition"
  title: string
  description: string
  confidence: number
  action?: string
  priority: "high" | "medium" | "low"
}

interface PerformanceForecast {
  strategy: string
  expectedWinRate: number
  expectedPnL: number
  riskLevel: "low" | "medium" | "high"
  recommendation: string
}

export default function AIInsightsPanel() {
  const [marketSentiment, setMarketSentiment] = useState<MarketSentiment[]>([
    {
      symbol: "XAUUSD",
      sentiment: "bullish",
      confidence: 87,
      factors: ["DXY weakness", "Inflation concerns", "Technical breakout"],
    },
    {
      symbol: "BTCUSD",
      sentiment: "neutral",
      confidence: 65,
      factors: ["Mixed institutional flows", "Regulatory uncertainty", "Technical consolidation"],
    },
  ])

  const [aiInsights, setAIInsights] = useState<AIInsight[]>([
    {
      type: "entry_timing",
      title: "Optimal Entry Window Detected",
      description:
        "MMXM strategy shows 92% probability of successful entry in next 15 minutes during London session overlap.",
      confidence: 92,
      action: "Prepare for XAUUSD long entry",
      priority: "high",
    },
    {
      type: "risk_warning",
      title: "High Volatility Alert",
      description: "NFP release in 2 hours. Consider reducing position sizes or enabling news filter.",
      confidence: 88,
      action: "Enable news filter",
      priority: "high",
    },
    {
      type: "strategy_switch",
      title: "Strategy Performance Shift",
      description: "OTE strategy showing declining performance. Consider switching to MMXM for better results.",
      confidence: 76,
      action: "Review strategy allocation",
      priority: "medium",
    },
    {
      type: "market_condition",
      title: "Market Regime Change",
      description:
        "Volatility patterns suggest transition from trending to ranging market. Adjust strategies accordingly.",
      confidence: 82,
      priority: "medium",
    },
  ])

  const [performanceForecasts, setPerformanceForecasts] = useState<PerformanceForecast[]>([
    {
      strategy: "MMXM",
      expectedWinRate: 89.2,
      expectedPnL: 234.56,
      riskLevel: "medium",
      recommendation: "Increase allocation - optimal conditions",
    },
    {
      strategy: "OTE",
      expectedWinRate: 76.8,
      expectedPnL: 145.23,
      riskLevel: "low",
      recommendation: "Maintain current settings",
    },
    {
      strategy: "Judas Swing",
      expectedWinRate: 82.4,
      expectedPnL: 189.45,
      riskLevel: "high",
      recommendation: "Reduce risk per trade",
    },
  ])

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      // Update market sentiment
      setMarketSentiment((prev) =>
        prev.map((sentiment) => ({
          ...sentiment,
          confidence: Math.max(60, Math.min(95, sentiment.confidence + (Math.random() - 0.5) * 10)),
        })),
      )

      // Occasionally add new insights
      if (Math.random() < 0.3) {
        const newInsights = [
          {
            type: "entry_timing" as const,
            title: "New Signal Detected",
            description: "AI detected high-probability setup forming on XAUUSD 15M timeframe.",
            confidence: Math.floor(Math.random() * 20) + 75,
            action: "Monitor for entry",
            priority: "medium" as const,
          },
        ]

        setAIInsights((prev) => [newInsights[0], ...prev.slice(0, 3)])
      }
    }, 15000)

    return () => clearInterval(interval)
  }, [])

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case "bullish":
        return "text-green-500"
      case "bearish":
        return "text-red-500"
      default:
        return "text-yellow-500"
    }
  }

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case "bullish":
        return "📈"
      case "bearish":
        return "📉"
      default:
        return "➡️"
    }
  }

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case "high":
        return "border-red-500/20 bg-red-500/10"
      case "medium":
        return "border-yellow-500/20 bg-yellow-500/10"
      default:
        return "border-blue-500/20 bg-blue-500/10"
    }
  }

  const getInsightIcon = (type: string) => {
    switch (type) {
      case "entry_timing":
        return <Target className="w-4 h-4 text-green-500" />
      case "risk_warning":
        return <AlertTriangle className="w-4 h-4 text-red-500" />
      case "strategy_switch":
        return <Zap className="w-4 h-4 text-blue-500" />
      default:
        return <Activity className="w-4 h-4 text-primary" />
    }
  }

  return (
    <div className="space-y-6">
      {/* Market Sentiment Analysis */}
      <Card className="glassmorphism border-primary/20">
        <CardHeader>
          <CardTitle className="flex items-center">
            <Brain className="w-5 h-5 mr-2 text-primary" />
            AI Market Sentiment Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {marketSentiment.map((sentiment, index) => (
              <div key={index} className="border border-border/40 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    <span className="font-semibold">{sentiment.symbol}</span>
                    <span className="text-lg">{getSentimentIcon(sentiment.sentiment)}</span>
                  </div>
                  <Badge className={`${getSentimentColor(sentiment.sentiment)} bg-transparent border-current`}>
                    {sentiment.sentiment.toUpperCase()}
                  </Badge>
                </div>

                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Confidence</span>
                      <span>{sentiment.confidence}%</span>
                    </div>
                    <Progress value={sentiment.confidence} className="h-2" />
                  </div>

                  <div>
                    <div className="text-sm text-muted-foreground mb-2">Key Factors:</div>
                    <div className="space-y-1">
                      {sentiment.factors.map((factor, i) => (
                        <div key={i} className="text-xs bg-muted/20 rounded px-2 py-1">
                          {factor}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* AI Insights & Recommendations */}
      <Card className="glassmorphism border-primary/20">
        <CardHeader>
          <CardTitle className="flex items-center">
            <Zap className="w-5 h-5 mr-2 text-primary" />
            AI Insights & Recommendations
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {aiInsights.map((insight, index) => (
              <div key={index} className={`border rounded-lg p-4 ${getPriorityColor(insight.priority)}`}>
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    {getInsightIcon(insight.type)}
                    <h4 className="font-semibold">{insight.title}</h4>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Badge variant="outline" className="text-xs">
                      {insight.confidence}% confidence
                    </Badge>
                    <Badge
                      className={
                        insight.priority === "high"
                          ? "bg-red-500/10 text-red-500 border-red-500/20"
                          : insight.priority === "medium"
                            ? "bg-yellow-500/10 text-yellow-500 border-yellow-500/20"
                            : "bg-blue-500/10 text-blue-500 border-blue-500/20"
                      }
                    >
                      {insight.priority}
                    </Badge>
                  </div>
                </div>

                <p className="text-sm text-muted-foreground mb-3">{insight.description}</p>

                {insight.action && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Recommended Action: {insight.action}</span>
                    <Button size="sm" variant="outline">
                      Apply
                    </Button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Strategy Performance Forecasting */}
      <Card className="glassmorphism border-primary/20">
        <CardHeader>
          <CardTitle className="flex items-center">
            <TrendingUp className="w-5 h-5 mr-2 text-primary" />
            Strategy Performance Forecasting
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {performanceForecasts.map((forecast, index) => (
              <div key={index} className="border border-border/40 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    <Badge className="gradient-primary text-black">{forecast.strategy}</Badge>
                    <Badge
                      className={
                        forecast.riskLevel === "high"
                          ? "bg-red-500/10 text-red-500 border-red-500/20"
                          : forecast.riskLevel === "medium"
                            ? "bg-yellow-500/10 text-yellow-500 border-yellow-500/20"
                            : "bg-green-500/10 text-green-500 border-green-500/20"
                      }
                    >
                      {forecast.riskLevel} risk
                    </Badge>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-muted-foreground">Expected P&L</div>
                    <div className="font-bold text-green-500">${forecast.expectedPnL}</div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 mb-3">
                  <div>
                    <div className="text-sm text-muted-foreground">Expected Win Rate</div>
                    <div className="font-bold">{forecast.expectedWinRate}%</div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground">Risk Level</div>
                    <div className="font-bold capitalize">{forecast.riskLevel}</div>
                  </div>
                </div>

                <div className="bg-muted/20 rounded p-3">
                  <div className="text-sm font-medium mb-1">AI Recommendation:</div>
                  <div className="text-sm text-muted-foreground">{forecast.recommendation}</div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Auto-Strategy Switching */}
      <Card className="glassmorphism border-primary/20">
        <CardHeader>
          <CardTitle className="flex items-center">
            <Activity className="w-5 h-5 mr-2 text-primary" />
            Auto-Strategy Switching
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 border border-border/40 rounded-lg">
              <div>
                <h4 className="font-semibold">Market Condition Detection</h4>
                <p className="text-sm text-muted-foreground">
                  AI monitors market conditions and automatically switches strategies for optimal performance
                </p>
              </div>
              <Badge className="bg-green-500/10 text-green-500 border-green-500/20">Active</Badge>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 border border-border/40 rounded-lg">
                <div className="text-sm text-muted-foreground mb-1">Current Market Regime</div>
                <div className="font-bold text-primary">Trending Market</div>
                <div className="text-xs text-muted-foreground mt-1">Optimal for MMXM & OTE strategies</div>
              </div>

              <div className="p-4 border border-border/40 rounded-lg">
                <div className="text-sm text-muted-foreground mb-1">Next Switch Probability</div>
                <div className="font-bold">23% (4 hours)</div>
                <div className="text-xs text-muted-foreground mt-1">To ranging market conditions</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
