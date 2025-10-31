"use client";
import { useEffect, useState } from "react";

export function useMt5DashboardStats() {
  const [stats, setStats] = useState({
    balance: 0,
    totalTrades: 0,
    winRate: 0,
    avgWin: 0,
    avgLoss: 0,
    loading: true,
  });

  useEffect(() => {
    let isMounted = true;
    const fetchData = async () => {
      try {
        const [accRes, tradesRes] = await Promise.all([
          fetch("/api/mt5/accounts"),
          fetch("/api/mt5/trades"),
        ]);
        const accData = await accRes.json();
        const tradesData = await tradesRes.json();
        const account = accData && accData.length > 0 ? accData[0] : null;
        const closedTrades = tradesData.closed_trades || [];
        const wins = closedTrades.filter((t: any) => t.profit > 0);
        const losses = closedTrades.filter((t: any) => t.profit < 0);
        const totalTrades = closedTrades.length;
        const winRate = totalTrades > 0 ? (wins.length / totalTrades) * 100 : 0;
        const avgWin = wins.length > 0 ? wins.reduce((sum: number, t: any) => sum + t.profit, 0) / wins.length : 0;
        const avgLoss = losses.length > 0 ? losses.reduce((sum: number, t: any) => sum + t.profit, 0) / losses.length : 0;
        if (isMounted) {
          setStats({
            balance: account ? account.balance : 0,
            totalTrades,
            winRate,
            avgWin,
            avgLoss,
            loading: false,
          });
        }
      } catch {
        if (isMounted) setStats((s) => ({ ...s, loading: false }));
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return stats;
} 