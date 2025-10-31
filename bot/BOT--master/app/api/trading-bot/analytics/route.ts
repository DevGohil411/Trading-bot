import type { NextRequest } from 'next/server';

const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL || 'http://localhost:8000';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol') || 'XAUUSD';
    const strategy = searchParams.get('strategy') || 'mmxm';

    const response = await fetch(
      `${FASTAPI_BASE_URL}/api/analytics/trade-stats?symbol=${encodeURIComponent(symbol)}&strategy=${encodeURIComponent(strategy)}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );

    if (!response.ok) {
      throw new Error(`FastAPI responded with status: ${response.status}`);
    }

    const data = await response.json();
    return Response.json(data);
  } catch (error) {
    console.error('Error fetching trade stats:', error);
    return Response.json(
      { error: 'Failed to fetch trade stats from trading server' },
      { status: 500 }
    );
  }
}
