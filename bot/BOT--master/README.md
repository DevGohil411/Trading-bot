# BOT--master (SniprX Dashboard)

A Next.js dashboard for managing and monitoring the SniprX trading bot, with optional Express-based auth/telegram backend.

### Features
- Next.js app with Tailwind UI
- Telegram notifications via `node-telegram-bot-api`
- API routes for Telegram bot webhook/notify
- Optional `server/` Express backend for Google OAuth and sessions

## Quickstart (Frontend)

### 1) Install dependencies
```bash
# From BOT--master/
npm install
```

### 2) Environment
Create `.env.local` with:
```ini
# Telegram
TELEGRAM_BOT_TOKEN=
TELEGRAM_BOT_USERNAME=

# Optional: Next public API to talk to Python FastAPI
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 3) Run
```bash
npm run dev
```
- App runs at `http://localhost:3000`
- It expects the Python API at `NEXT_PUBLIC_API_URL` for `/api/mt5/*` endpoints

## Optional Backend (Express)
Located in `server/`. Useful for Google OAuth and Telegram bot hosting.

### 1) Install
```bash
cd server
npm install
```

### 2) Environment
Copy `env.example` to `.env` and fill:
```ini
PORT=8000
SESSION_SECRET=your-super-secret-session-key-change-in-production
TELEGRAM_BOT_TOKEN=...
TELEGRAM_BOT_USERNAME=SniprXBot
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
```

### 3) Run
```bash
npm run dev
```

## Project structure (key files)
- `app/`: Next.js routes and pages
- `lib/telegram-bot.ts`: Telegram bot integration; uses `TELEGRAM_BOT_TOKEN`
- `pages/api/telegram/*`: webhook and notify API routes
- `server/server.js`: Express server (optional)

