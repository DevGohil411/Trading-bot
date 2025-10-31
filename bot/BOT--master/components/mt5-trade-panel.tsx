"use client";

import React, { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { TrendingUp, TrendingDown, Hash, Percent, Award } from "lucide-react";

type Trade = {
  ticket: number;
  symbol: string;
  volume: number;
  profit: number;
  type: number;
  time: string;
};

type TradesData = {
  open_trades: Trade[];
  closed_trades: Trade[];
};

function calcStats(closed_trades: Trade[]) {
  const totalPnL = closed_trades.reduce((sum, t) => sum + t.profit, 0);
  const winCount = closed_trades.filter(t => t.profit > 0).length;
  const totalTrades = closed_trades.length;
  const winRate = totalTrades > 0 ? (winCount / totalTrades) * 100 : 0;
  const avgWin = closed_trades.filter(t => t.profit > 0).reduce((sum, t) => sum + t.profit, 0) / (winCount || 1);
  const avgLoss = closed_trades.filter(t => t.profit < 0).reduce((sum, t) => sum + t.profit, 0) / (totalTrades - winCount || 1);
  return { totalPnL, winRate, totalTrades, avgWin, avgLoss };
}

function formatType(type: number) {
  if (type === 0) return "Buy";
  if (type === 1) return "Sell";
  return type;
}

function formatDate(dateStr: string) {
  if (!dateStr) return "";
  const d = new Date(dateStr);
  return d.toLocaleString();
}

export default function MT5TradePanel() {
  const [trades, setTrades] = useState<TradesData>({ open_trades: [], closed_trades: [] });
  const [loading, setLoading] = useState(false);

  const fetchTrades = async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/mt5/trades`);
      const data = await res.json();
      
      // Ensure data has the expected structure
      const safeTrades = {
        open_trades: Array.isArray(data?.open_trades) ? data.open_trades : [],
        closed_trades: Array.isArray(data?.closed_trades) ? data.closed_trades : []
      };
      
      setTrades(safeTrades);
    } catch (error) {
      console.error("Error fetching trades:", error);
      // Set empty arrays on error to prevent crashes
      setTrades({ open_trades: [], closed_trades: [] });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTrades();
    const interval = setInterval(fetchTrades, 15000);
    return () => clearInterval(interval);
  }, []);

  const stats = calcStats(trades.closed_trades);

  return (
    <Card className="glassmorphism border-primary/20 shadow-lg rounded-xl">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-xl font-bold">MT5 Trade History & Stats</CardTitle>
          <Button size="sm" onClick={fetchTrades} disabled={loading} className="bg-cyan-400 text-black hover:bg-cyan-500">
            {loading ? "Refreshing..." : "Refresh"}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
          <div className="flex flex-col items-center bg-zinc-900/60 rounded-lg p-3">
            <TrendingUp className="w-5 h-5 text-green-400 mb-1" />
            <div className="text-xs text-muted-foreground">Total PnL</div>
            <div className="font-bold text-green-400 text-lg">{stats.totalPnL.toFixed(2)}</div>
          </div>
          <div className="flex flex-col items-center bg-zinc-900/60 rounded-lg p-3">
            <Percent className="w-5 h-5 text-blue-400 mb-1" />
            <div className="text-xs text-muted-foreground">Win Rate</div>
            <div className="font-bold text-blue-400 text-lg">{stats.winRate.toFixed(1)}%</div>
          </div>
          <div className="flex flex-col items-center bg-zinc-900/60 rounded-lg p-3">
            <Hash className="w-5 h-5 text-yellow-400 mb-1" />
            <div className="text-xs text-muted-foreground">Total Trades</div>
            <div className="font-bold text-yellow-400 text-lg">{stats.totalTrades}</div>
          </div>
          <div className="flex flex-col items-center bg-zinc-900/60 rounded-lg p-3">
            <Award className="w-5 h-5 text-green-400 mb-1" />
            <div className="text-xs text-muted-foreground">Avg Win</div>
            <div className="font-bold text-green-400 text-lg">{isNaN(stats.avgWin) ? "0.00" : stats.avgWin.toFixed(2)}</div>
          </div>
          <div className="flex flex-col items-center bg-zinc-900/60 rounded-lg p-3">
            <TrendingDown className="w-5 h-5 text-red-400 mb-1" />
            <div className="text-xs text-muted-foreground">Avg Loss</div>
            <div className="font-bold text-red-400 text-lg">{isNaN(stats.avgLoss) ? "0.00" : stats.avgLoss.toFixed(2)}</div>
          </div>
        </div>

        {/* Open Trades Table */}
        <div className="mb-6">
          <h3 className="font-semibold mb-2 text-lg">Open Trades</h3>
          <div className="overflow-x-auto rounded-lg border border-zinc-800">
            <table className="min-w-full text-xs text-left">
              <thead className="bg-zinc-900 sticky top-0 z-10">
                <tr>
                  <th className="p-2">Ticket</th>
                  <th className="p-2">Symbol</th>
                  <th className="p-2">Volume</th>
                  <th className="p-2">Profit</th>
                  <th className="p-2">Type</th>
                  <th className="p-2">Time</th>
                </tr>
              </thead>
              <tbody>
                {trades.open_trades.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="text-center text-muted-foreground p-2">No open trades</td>
                  </tr>
                ) : (
                  trades.open_trades.map((trade, i) => (
                    <tr key={i} className={i % 2 === 0 ? "bg-zinc-900/40" : ""}>
                      <td className="p-2">{trade.ticket}</td>
                      <td className="p-2">{trade.symbol}</td>
                      <td className="p-2">{trade.volume}</td>
                      <td className={`p-2 font-bold ${trade.profit >= 0 ? "text-green-400" : "text-red-400"}`}>{trade.profit}</td>
                      <td className="p-2">{formatType(trade.type)}</td>
                      <td className="p-2">{formatDate(trade.time)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Closed Trades Table */}
        <div>
          <h3 className="font-semibold mb-2 text-lg">Closed Trades</h3>
          <div className="overflow-x-auto rounded-lg border border-zinc-800 max-h-[350px]">
            <table className="min-w-full text-xs text-left">
              <thead className="bg-zinc-900 sticky top-0 z-10">
                <tr>
                  <th className="p-2">Ticket</th>
                  <th className="p-2">Symbol</th>
                  <th className="p-2">Volume</th>
                  <th className="p-2">Profit</th>
                  <th className="p-2">Type</th>
                  <th className="p-2">Time</th>
                </tr>
              </thead>
              <tbody>
                {trades.closed_trades.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="text-center text-muted-foreground p-2">No closed trades</td>
                  </tr>
                ) : (
                  trades.closed_trades.map((trade, i) => (
                    <tr key={i} className={i % 2 === 0 ? "bg-zinc-900/40" : ""}>
                      <td className="p-2">{trade.ticket}</td>
                      <td className="p-2">{trade.symbol}</td>
                      <td className="p-2">{trade.volume}</td>
                      <td className={`p-2 font-bold ${trade.profit >= 0 ? "text-green-400" : "text-red-400"}`}>{trade.profit}</td>
                      <td className="p-2">{formatType(trade.type)}</td>
                      <td className="p-2">{formatDate(trade.time)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 