import type { NextRequest } from 'next/server';

const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL || 'http://localhost:8000';

export async function GET(request: NextRequest) {
  try {
    const response = await fetch(`${FASTAPI_BASE_URL}/api/bot/status`, {
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
    console.error('Error fetching bot status:', error);
    return Response.json(
      { error: 'Failed to fetch bot status from trading server' },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action } = body;

    let endpoint = '';
    switch (action) {
      case 'start':
        endpoint = '/api/bot/start';
        break;
      case 'stop':
        endpoint = '/api/bot/stop';
        break;
      default:
        return Response.json({ error: 'Invalid action' }, { status: 400 });
    }

    const response = await fetch(`${FASTAPI_BASE_URL}${endpoint}`, {
      method: 'POST',
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
    console.error('Error controlling bot:', error);
    return Response.json(
      { error: 'Failed to control bot from trading server' },
      { status: 500 }
    );
  }
}
