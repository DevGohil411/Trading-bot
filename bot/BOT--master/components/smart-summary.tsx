"use client"

import { useState, useEffect } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { TrendingUp, Zap } from "lucide-react"
import { useMt5AccountData } from "@/hooks/use-mt5-account-data"

export default function SmartSummary() {
  const { data, loading } = useMt5AccountData()
  const [isVisible, setIsVisible] = useState(true)

  if (!isVisible || loading) return null
  const account = data && data.length > 0 ? data[0] : null

  return (
    <Card className="glassmorphism mb-8 border-primary/20 animate-success">
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 gradient-primary rounded-2xl flex items-center justify-center">
              <TrendingUp className="w-6 h-6 text-black" />
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-1">Smart Summary</h3>
              {account && (
                <div className="text-xs text-muted-foreground mb-1">
                  <span className="font-bold">{account.name}</span> | {account.login} • {account.server}
                </div>
              )}
              <p className="text-muted-foreground">
                📊 Your bot earned <span className="text-primary font-bold">${account ? account.balance.toFixed(2) : "-"}</span> this week using <span className="text-primary font-medium">MMXM Strategy</span> with{" "}
                <span className="text-green-500 font-bold">{account ? (account.winRate || 0).toFixed(0) : "-"}%</span> win rate.
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <Badge className="bg-green-500/10 text-green-500 border-green-500/20">
              <Zap className="w-3 h-3 mr-1" />
              Active
            </Badge>
            <button
              onClick={() => setIsVisible(false)}
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              ×
            </button>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
