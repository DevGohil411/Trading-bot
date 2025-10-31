"use client";
import { useEffect, useState } from "react";

export function useMt5AccountData() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;
    const fetchData = () => {
      fetch("/api/mt5/accounts")
        .then(res => res.json())
        .then(d => {
          if (isMounted) {
            // Ensure data is always an array with at least one account
            const safeData = Array.isArray(d) && d.length > 0 ? d : [{
              login: "0",
              server: "Demo",
              name: "Demo Account",
              balance: 0.0,
              equity: 0.0,
              margin: 0.0,
              freeMargin: 0.0,
              connected: false,
              winRate: 0,
              totalTrades: 0,
              avgWin: 0,
              avgLoss: 0
            }];
            setData(safeData);
            setLoading(false);
          }
        })
        .catch(error => {
          console.error("Error fetching MT5 account data:", error);
          if (isMounted) {
            // Provide fallback data on error
            setData([{
              login: "0",
              server: "Demo",
              name: "Demo Account (Offline)",
              balance: 0.0,
              equity: 0.0,
              margin: 0.0,
              freeMargin: 0.0,
              connected: false,
              winRate: 0,
              totalTrades: 0,
              avgWin: 0,
              avgLoss: 0
            }]);
            setLoading(false);
          }
        });
    };
    fetchData();
    const interval = setInterval(fetchData, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return { data, loading };
}