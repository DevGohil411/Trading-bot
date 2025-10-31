import type { NextRequest } from 'next/server';

const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL || 'http://localhost:8000';

export async function GET(request: NextRequest) {
  try {
    const response = await fetch(`${FASTAPI_BASE_URL}/api/bot/settings`, {
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
    console.error('Error fetching bot settings:', error);
    return Response.json(
      { error: 'Failed to fetch bot settings from trading server' },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const response = await fetch(`${FASTAPI_BASE_URL}/api/bot/settings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      throw new Error(`FastAPI responded with status: ${response.status}`);
    }

    const data = await response.json();
    return Response.json(data);
  } catch (error) {
    console.error('Error updating bot settings:', error);
    return Response.json(
      { error: 'Failed to update bot settings via trading server' },
      { status: 500 }
    );
  }
}
