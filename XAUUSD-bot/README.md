<<<<<<< HEAD
# XAUUSD-bot

A Python-based trading bot for MT5 with a FastAPI control server and Telegram notifications.

### Features
- Connects to MetaTrader 5 (MT5) and executes multi-strategy trades
- FastAPI server for bot control, MT5 login, and fetching trades
- Telegram bot notifications and basic webhook endpoints
- SQLite logging for trades, plus CSV and JSON logs

## Quickstart

- Python 3.10+ (Windows supported)
- MT5 terminal installed (defaults to `C:\\Program Files\\MetaTrader 5\\terminal64.exe`)

### 1) Install dependencies
```bash
# From XAUUSD-bot/
python -m venv venv
venv\\Scripts\\activate  # on Windows PowerShell
pip install -r requirements.txt
```

### 2) Environment variables
Copy `.env.example` to `.env.local` and fill in values:
```ini
NEXT_PUBLIC_API_URL=http://localhost:8000
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
TELEGRAM_BOT_USERNAME=YourBotUserName
```

- The FastAPI server reads `NEXT_PUBLIC_API_URL` for the frontend.
- Telegram credentials are used by the Next.js app and may be read by integrations.

### 3) Configure the bot
Edit `config.json` with your settings (redact secrets before sharing):
- `mt5.login`, `mt5.password`, `mt5.server`
- `symbols`, `timeframe`, `risk_settings`, `strategy_filters`
- `telegram.bot_token`, `telegram.chat_id` (optional if sending alerts from Python)
- `gemini.api_key` (if using LLM/ML integrations)

Example keys to check:
- `risk_settings.risk_per_trade`, `max_daily_loss`, `default_lot_size`
- `trading_sessions.enable_trading`
- `selected_strategies`

### 4) Initialize database schema (SQLite)
This project logs to `trades/trades.db`. Create the database with the unified schema:
```bash
# From XAUUSD-bot/
mkdir -p trades
python - <<'PY'
import sqlite3, os
os.makedirs('trades', exist_ok=True)
conn = sqlite3.connect('trades/trades.db')
conn.executescript(open('schema.sql','r',encoding='utf-8').read())
conn.commit(); conn.close()
print('SQLite trades.db initialized.')
PY
```

To dump the database later:
```bash
# Windows (powershell): ensure sqlite3 is installed or use Python dump
python - <<'PY'
import sqlite3, sys
conn = sqlite3.connect('trades/trades.db')
for line in conn.iterdump():
    sys.stdout.write(f"{line}\n")
conn.close()
PY
```

### 5) Run services
- Start API server (port 8000):
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```
- Start trading bot (in a separate shell):
```bash
python -m STOCKDATA
```

If you need to test MT5 login directly:
```bash
python test_mt5_login.py
```

## Project structure (key files)
- `api_server.py`: FastAPI app with endpoints for MT5 login, trades, and settings
- `STOCKDATA/main.py`: main trading loop; loads `config.json`, manages strategies, risk, and MT5
- `STOCKDATA/file.py`: trade placement, risk, and persistence helpers (creates SQLite `trades` table)
- `STOCKDATA/utils/trade_logger.py`: logs trade executions to CSV and SQLite
- `trades/`: SQLite db and CSV logs
- `logs/`: JSON logs for activity and daily stats

## Database schema
A unified table `trades` is provided to be compatible with both `file.py` and `utils/trade_logger.py` writers. See `schema.sql`.

Indexes are included on `timestamp`, `symbol`, `strategy`, and `trade_type` for faster queries.

## Notes
- MT5 path is configured in `STOCKDATA/main.py` as `MT5_TERMINAL_PATH`. Adjust if your installation is elsewhere.
- Do not commit secrets (MT5 password, Telegram token, Gemini key). Use `.env.local` and keep `config.json` sanitized before sharing.
- For production, consider moving from SQLite to a managed DB and adding robust auth for the API. 
=======
# trading_bot
>>>>>>> 09258285804d8b9120255b97deac3f2d5d748e6a
