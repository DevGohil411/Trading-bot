import type { NextRequest } from 'next/server';

const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL || 'http://localhost:8000';

export async function GET(request: NextRequest) {
  try {
    const response = await fetch(`${FASTAPI_BASE_URL}/api/activity-log`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`FastAPI responded with status: ${response.status}`);
    }

    const data = await response.json();
    return Response.json(data);
  } catch (error) {
    console.error('Error fetching activity log:', error);
    return Response.json(
      { error: 'Failed to fetch activity log from trading server' },
      { status: 500 }
    );
  }
}
