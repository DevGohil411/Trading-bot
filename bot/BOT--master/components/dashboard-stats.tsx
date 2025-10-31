"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { TrendingUp, TrendingDown, DollarSign, Activity, Target, Clock } from "lucide-react"
import { useMt5AccountData } from "@/hooks/use-mt5-account-data"

export default function DashboardStats() {
  const { data, loading } = useMt5AccountData();
  if (loading || !data || data.length === 0) return null;
  const account = data[0];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      <Card className="glassmorphism border-primary/20">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center">
            <DollarSign className="w-4 h-4 mr-2 text-green-500" />
            Total P&L
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-green-500">${account.balance.toFixed(2)}</div>
          <div className="flex items-center text-sm text-muted-foreground mt-1">
            <TrendingUp className="w-4 h-4 mr-1" />
            +18.4% this month
          </div>
        </CardContent>
      </Card>

      <Card className="glassmorphism border-primary/20">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center">
            <Activity className="w-4 h-4 mr-2 text-primary" />
            Today's P&L
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className={`text-2xl font-bold ${account.equity >= 0 ? "text-green-500" : "text-red-500"}`}>
            ${account.equity.toFixed(2)}
          </div>
          <div className="flex items-center text-sm text-muted-foreground mt-1">
            {account.equity >= 0 ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
            Live session
          </div>
        </CardContent>
      </Card>

      <Card className="glassmorphism border-primary/20">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center">
            <Target className="w-4 h-4 mr-2 text-primary" />
            Win Rate
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{account.winRate ? account.winRate.toFixed(1) : "-"}%</div>
          <div className="flex items-center text-sm text-muted-foreground mt-1">
            <TrendingUp className="w-4 h-4 mr-1" />
            Above target
          </div>
        </CardContent>
      </Card>

      <Card className="glassmorphism border-primary/20">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center">
            <Activity className="w-4 h-4 mr-2 text-primary" />
            Total Trades
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{account.totalTrades || 0}</div>
          <div className="flex items-center text-sm text-muted-foreground mt-1">
            <Clock className="w-4 h-4 mr-1" />
            This month
          </div>
        </CardContent>
      </Card>

      <Card className="glassmorphism border-primary/20">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center">
            <TrendingUp className="w-4 h-4 mr-2 text-green-500" />
            Avg Win
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-green-500">${account.avgWin !== undefined ? account.avgWin.toFixed(2) : "0.00"}</div>
          <div className="flex items-center text-sm text-muted-foreground mt-1">
            <Target className="w-4 h-4 mr-1" />
            Per trade
          </div>
        </CardContent>
      </Card>

      <Card className="glassmorphism border-primary/20">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center">
            <TrendingDown className="w-4 h-4 mr-2 text-red-500" />
            Avg Loss
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-red-500">${account.avgLoss !== undefined ? account.avgLoss.toFixed(2) : "0.00"}</div>
          <div className="flex items-center text-sm text-muted-foreground mt-1">
            <Target className="w-4 h-4 mr-1" />
            Per trade
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
