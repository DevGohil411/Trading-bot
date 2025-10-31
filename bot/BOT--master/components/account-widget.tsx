"use client";
import { useAccountData } from "@/hooks/use-account-data";

export default function AccountWidget() {
  const { data, loading } = useAccountData();

  if (loading) return <div>Loading...</div>;
  if (!data) return <div>No data</div>;

  return (
    <div>
      <h2>Account Info</h2>
      <div>Balance: {data.balance}</div>
      <div>Equity: {data.equity}</div>
      <h3>Open Trades:</h3>
      <ul>
        {(Array.isArray(data?.open_trades) ? data.open_trades : []).map((trade: any, i: number) => (
          <li key={i}>
            {trade.symbol} - {trade.volume} lots - Profit: {trade.profit}
          </li>
        ))}
      </ul>
    </div>
  );
} 