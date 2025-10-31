"use client";
import React, { useState, useEffect } from 'react';
import Header from "@/components/header"
import AdvancedAnalytics from "@/components/advanced-analytics"
import ProtectedRoute from "@/components/protected-route"
import { Badge } from "@/components/ui/badge"

const SYMBOLS = [
  'xauusd',
  'xagusd',
  'eurjpy',
  'gbpjpy',
  'usdjpy',
  'msft',
  'amd',
  'nvd',
  'us30',
];

const STRATEGIES = [
  'MMXM Strategy',
  'Judas Swing',
  'OTE Strategy',
  'AMD Cycle',
  'MSB Retest',
  'Order Block + CHOCH',
  'MMC Combo',
  'MMC Classic',
];

const fetchTradeStats = async (symbol: string, strategy: string) => {
  const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/analytics/trade-stats?symbol=${symbol}&strategy=${strategy}`);
  return res.json();
};

export default function AnalyticsPage() {
  const [symbol, setSymbol] = useState(SYMBOLS[0]);
  const [strategy, setStrategy] = useState(STRATEGIES[0]);
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchTradeStats(symbol, strategy)
      .then((data) => {
        if (data.success) setStats(data.data);
        else setError(data.error || 'No data');
        setLoading(false);
      })
      .catch((e) => {
        setError('API error');
        setLoading(false);
      });
  }, [symbol, strategy]);

  const buy = stats?.buy || { count: 0, win_rate: 0, premium: 0 };
  const sell = stats?.sell || { count: 0, win_rate: 0, premium: 0 };
  const totalProfit = (buy.premium || 0) + (sell.premium || 0);

  return (
    <ProtectedRoute>
      <div className="min-h-screen tech-dark-bg">
        <Header />
        <main className="container mx-auto px-4 py-16">
          <div className="text-center mb-12">
            <Badge className="mb-4 bg-primary/10 text-primary border-primary/20">Advanced Analytics</Badge>
            <h1 className="text-4xl font-bold mb-4">Performance Analytics</h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Deep insights into your trading performance with AI-powered analytics and forecasting
            </p>
          </div>
          <div className="flex flex-col md:flex-row gap-4 justify-center items-center mb-8">
            <select
              className="bg-[#18181c] text-white px-4 py-2 rounded shadow"
              value={symbol}
              onChange={e => setSymbol(e.target.value)}
            >
              {SYMBOLS.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
            <select
              className="bg-[#18181c] text-white px-4 py-2 rounded shadow"
              value={strategy}
              onChange={e => setStrategy(e.target.value)}
            >
              {STRATEGIES.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div className="max-w-2xl mx-auto bg-[#18181c] rounded-lg shadow p-6">
            {loading ? (
              <div className="text-center text-gray-400">Loading...</div>
            ) : error ? (
              <div className="text-center text-red-400">{error}</div>
            ) : (
              <table className="w-full text-center">
                <thead>
                  <tr className="border-b border-gray-700">
                    <th className="py-2">Symbol</th>
                    <th>Strategy</th>
                    <th>Buy Trades</th>
                    <th>Buy Win Rate</th>
                    <th>Buy Profit</th>
                    <th>Sell Trades</th>
                    <th>Sell Win Rate</th>
                    <th>Sell Profit</th>
                    <th>Total Profit</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="py-2 font-semibold">{symbol}</td>
                    <td>{strategy}</td>
                    <td>{buy.count}</td>
                    <td>{(buy.win_rate * 100).toFixed(1)}%</td>
                    <td className={buy.premium >= 0 ? 'text-green-400' : 'text-red-400'}>
                      ${buy.premium.toFixed(2)}
                    </td>
                    <td>{sell.count}</td>
                    <td>{(sell.win_rate * 100).toFixed(1)}%</td>
                    <td className={sell.premium >= 0 ? 'text-green-400' : 'text-red-400'}>
                      ${sell.premium.toFixed(2)}
                    </td>
                    <td className={totalProfit >= 0 ? 'text-green-400' : 'text-red-400'}>
                      ${totalProfit.toFixed(2)}
                    </td>
                  </tr>
                </tbody>
              </table>
            )}
          </div>
        </main>
      </div>
    </ProtectedRoute>
  )
}
