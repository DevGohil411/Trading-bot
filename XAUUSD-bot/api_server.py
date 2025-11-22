import logging
import os
import threading
import subprocess
from fastapi import FastAPI, Request, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
try:
    import MetaTrader5 as mt5  # Available on Windows with terminal installed
except Exception:
    mt5 = None  # Allow API to run without MT5 in cloud deployments
from fastapi.responses import JSONResponse
import datetime
import sys
import json
import time
from STOCKDATA.utils.trade_logger import get_trade_stats
import shutil





app = FastAPI()

# Allow all origins for development (change in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_FILE = 'config.json'

def load_config():
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Config file not found or corrupted at {CONFIG_FILE}: {e}. Returning default config.")
        # Return a complete default config if file is missing or corrupted
        return {
            "symbols": ["XAUUSD", "XAGUSD", "GBPJPY", "USDJPY", "EURJPY", "US30", "AMD", "MSFT", "NVDA"],
            "timeframe": "TIMEFRAME_M15",
            "candle_count": 50,
            "candle_freshness_threshold_seconds": 900,
            "risk_settings": {
                "max_daily_loss": 500,
                "max_daily_profit": 1000,
                "risk_per_trade": 1.5,
                "max_daily_trades": 20,
                "default_lot_size": 0.05,
                "max_open_trades": 20,
                "allow_multiple_trades": True
            },
            "trading_sessions": {
                "enable_trading": True,
                "london": { "active": True, "start": "08:00", "end": "17:00" },
                "new_york": { "active": True, "start": "13:00", "end": "22:00" },
                "asian": { "active": False, "start": "00:00", "end": "09:00" },
                "custom": { "start": "08:00", "end": "17:00" }
            },
            "strategy_filters": {
                "killzone_filter": True,
                "news_filter": True,
                "volatility_filter": True,
                "trend_filter": True
            },
            "notifications": {
                "email_alerts": True,
                "telegram_alerts": True,
                "push_notifications": True,
                "trade_alerts": True
            },
            "advanced_settings": {
                "max_slippage": 3,
                "max_spread": 20,
                "auto_reconnect": True,
                "emergency_stop": True
            },
            "logging": {
                "log_dir": "logs",
                "log_level": "INFO"
            },
            "mt5": {
                "login": 5036996416,
                "password": "6kZs-oPr",
                "server": "MetaQuotes-Demo"
            }
        }
    except Exception as e:
        print(f"An unexpected error occurred while loading config file: {e}. Returning empty config.")
        return {}

# Helper function to save config.json (atomic write for safety)
def save_config(config_data):
    try:
        tmp_file = CONFIG_FILE + '.tmp'
        with open(tmp_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        shutil.move(tmp_file, CONFIG_FILE)
    except Exception as e:
        print(f"Error saving config file: {e}")

# Initial load of configuration when the API server starts
config = load_config()

# Global variable to hold the bot process
bot_process = None

# --- Activity Log ---
activity_log = []
activity_log_file = os.path.join("logs", "activity_log.json")
activity_log_lock = threading.Lock()

# Helper to add event to activity log
def log_event(event_type, title, details, tag):
    entry = {
        "type": event_type,
        "title": title,
        "details": details,
        "timestamp": datetime.datetime.now().isoformat(),
        "tag": tag
    }
    with activity_log_lock:
        activity_log.append(entry)
        # Keep only last 200 events
        if len(activity_log) > 200:
            activity_log[:] = activity_log[-200:]
        # Save to file
        try:
            with open(activity_log_file, "w") as f:
                json.dump(activity_log, f, indent=2)
        except Exception as e:
            print(f"Error writing activity log: {e}")

# Load activity log from file at startup
if os.path.exists(activity_log_file):
    try:
        with open(activity_log_file, "r") as f:
            activity_log = json.load(f)
    except Exception as e:
        print(f"Error loading activity log: {e}")

# In-memory storage for demo (replace with DB in production)
mt5_accounts = []

class MT5Account(BaseModel):
    login: str
    server: str
    name: str
    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    freeMargin: float = 0.0
    connected: bool = True

MT5_ACCOUNTS_FILE = 'mt5_accounts.json'

def save_mt5_accounts(accounts: List[MT5Account]):
    try:
        with open(MT5_ACCOUNTS_FILE, 'w') as f:
            json.dump([acc.dict() for acc in accounts], f, indent=2)
    except Exception as e:
        print(f"Error saving MT5 accounts file: {e}")

def load_mt5_accounts() -> List[MT5Account]:
    if not os.path.exists(MT5_ACCOUNTS_FILE):
        return []
    try:
        with open(MT5_ACCOUNTS_FILE, 'r') as f:
            accounts_data = json.load(f)
            return [MT5Account(**data) for data in accounts_data]
    except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
        print(f"MT5 accounts file not found or corrupted: {e}. Returning empty list.")
        return []

mt5_accounts = load_mt5_accounts()

class MT5LoginRequest(BaseModel):
    login: str
    password: str
    server: str
    name: str = ""

# Auth stubs to satisfy frontend checks when Node auth is not used
@app.get("/auth/status")
def auth_status():
    return {"authenticated": False, "user": None}

@app.get("/logout")
def logout():
    return {"success": True}

class GoogleUser(BaseModel):
    email: str
    name: str

@app.get("/api/account")
def get_account():
    # Replace this with your real trading bot data
    return {
        "balance": 1234.56,
        "equity": 1200.00,
        "open_trades": [
            {"symbol": "XAUUSD", "volume": 0.1, "profit": 12.5},
            {"symbol": "USDJPY", "volume": 0.2, "profit": -5.0}
        ]
    }

@app.post("/api/mt5/login")
def mt5_login(account: MT5LoginRequest):
    if mt5 is None:
        error_msg = "MetaTrader5 is not available in this deployment. This endpoint requires MT5 on Windows."
        print(f"ERROR: {error_msg}")
        return {"success": False, "error": error_msg}
    print(f"Attempting MT5 login for account: {account.login} on server: {account.server}")

    # First, disconnect if already connected
    print("Shutting down any existing MT5 connections...")
    mt5.shutdown()
    time.sleep(1)  # Give it a moment to disconnect

    # Attempt to connect with new credentials
    print("Initializing MT5 connection...")
    try:
        login_int = int(account.login)
    except (TypeError, ValueError):
        error_msg = f"Invalid login value: {account.login}"
        print(f"ERROR: {error_msg}")
        return {"success": False, "error": error_msg}

    if not mt5.initialize(login=login_int, password=account.password, server=account.server):
        error_msg = f"Failed to initialize MT5 terminal. Error: {mt5.last_error()}"
        print(f"ERROR: {error_msg}")
        log_event("error", "Login Failed", f"Failed to connect to MT5 with credentials: {account.login} on {account.server}", "error")
        return {"success": False, "error": error_msg}

    print("MT5 terminal initialized successfully. Attempting to login...")

    # Try to login with provided credentials
    authorized = mt5.login(login=login_int, password=account.password, server=account.server)
    if not authorized:
        error_msg = f"MT5 login failed. Error: {mt5.last_error()}"
        print(f"ERROR: {error_msg}")
        log_event("error", "MT5 Login Failed", f"Login failed for {account.login} on {account.server}", "error")
        mt5.shutdown()
        return {"success": False, "error": error_msg}

    print("MT5 login successful. Fetching account info...")

    # Fetch account info
    acc_info = mt5.account_info()
    if acc_info is None:
        error_msg = f"Failed to fetch account info. Error: {mt5.last_error()}"
        print(f"ERROR: {error_msg}")
        log_event("error", "MT5 Account Info Failed", f"Could not fetch account info for {account.login}", "error")
        mt5.shutdown()
        return {"success": False, "error": error_msg}

    print(f"Account info retrieved successfully. Login: {acc_info.login}, Balance: {acc_info.balance}")

    acc = MT5Account(
        login=str(acc_info.login),
        server=account.server,
        name=account.name or f"Account {str(acc_info.login)[-4:]}",
        balance=acc_info.balance,
        equity=acc_info.equity,
        margin=acc_info.margin,
        freeMargin=acc_info.margin_free,
        connected=True
    )

    # Save account (avoid duplicates)
    global mt5_accounts
    mt5_accounts = [a for a in mt5_accounts if a.login != acc.login or a.server != acc.server]
    mt5_accounts.append(acc)
    save_mt5_accounts(mt5_accounts)

    print(f"Account {acc.login} connected and saved successfully.")
    log_event("connection", "MT5 Connected", f"Account ****{str(acc_info.login)[-4:]} ({account.server})", "connection")

    # Don't shutdown here to keep the connection alive
    # mt5.shutdown()

    return {"success": True, "account": acc.dict()}

@app.get("/api/mt5/accounts", response_model=List[MT5Account])
def get_mt5_accounts():
    return mt5_accounts

@app.post("/api/mt5/logout")
def mt5_logout(account: MT5LoginRequest):
    global mt5_accounts
    initial_count = len(mt5_accounts)
    mt5_accounts = [acc for acc in mt5_accounts if acc.login != account.login or acc.server != account.server]
    if len(mt5_accounts) < initial_count:
        save_mt5_accounts(mt5_accounts)
        log_event("connection", "MT5 Disconnected", f"Account ****{str(account.login)[-4:]} ({account.server}) disconnected.", "connection")
        return {"success": True, "message": "Account disconnected successfully."}
    else:
        return JSONResponse(status_code=404, content={"success": False, "error": "Account not found."})

@app.get("/api/mt5/trades")
def get_mt5_trades():
    if mt5 is None:
        return JSONResponse(status_code=200, content={"success": False, "error": "MetaTrader5 not available in this deployment.", "open_trades": [], "closed_trades": []})
    config = load_config()
    mt5_config = config.get('mt5', {})
    if not mt5.initialize(login=mt5_config.get('login'), password=mt5_config.get('password'), server=mt5_config.get('server')):
        return JSONResponse(status_code=200, content={"success": False, "error": "MetaTrader 5 terminal not found or not running.", "open_trades": [], "closed_trades": []})
    try:
        # Get open positions
        open_trades = mt5.positions_get()
        open_trades_list = []
        if open_trades is not None:
            for pos in open_trades:
                open_trades_list.append({
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "profit": pos.profit,
                    "type": pos.type,
                    "time": datetime.datetime.fromtimestamp(pos.time).isoformat() if hasattr(pos, 'time') else None,
                    "comment": getattr(pos, 'comment', "")
                })
        # Get closed deals (history)
        utc_from = datetime.datetime.now() - datetime.timedelta(days=365)
        utc_to = datetime.datetime.now()
        history_deals = mt5.history_deals_get(utc_from, utc_to)
        closed_trades_list = []
        # Only include real closed trades (DEAL_TYPE_BUY/SELL and volume > 0)
        DEAL_TYPE_BUY = 0
        DEAL_TYPE_SELL = 1
        if history_deals is not None:
            for deal in history_deals:
                if deal.type in (DEAL_TYPE_BUY, DEAL_TYPE_SELL) and deal.volume > 0:
                    closed_trades_list.append({
                        "ticket": deal.ticket,
                        "symbol": deal.symbol,
                        "volume": deal.volume,
                        "price": deal.price,
                        "profit": float(deal.profit),
                        "type": deal.type,
                        "time": datetime.datetime.fromtimestamp(deal.time).isoformat() if hasattr(deal, 'time') else None,
                        "comment": getattr(deal, 'comment', "")
                    })
        mt5.shutdown()
        return {"success": True, "open_trades": open_trades_list, "closed_trades": closed_trades_list}
    except Exception as e:
        logging.error(f"Error fetching MT5 trades: {e}")
        return JSONResponse(status_code=200, content={"success": False, "error": f"Failed to fetch MT5 trades: {e}", "open_trades": [], "closed_trades": []})

@app.get("/api/bot/settings")
def get_bot_settings():
    config_data = load_config()
    return JSONResponse(content=config_data)

@app.post("/api/bot/settings")
async def update_bot_settings(settings_data: dict = Body(...)):
    """Update bot settings from UI, including strategy selection and killzone.
    Expected keys (any subset):
    - all_strategies: bool
    - selected_strategies: List[str]
    - killzone_map: dict
    - current_session: str
    Other keys will be merged into config.json as before.
    """
    try:
        # Persist UI runtime selections separately so the bot thread can poll them
        runtime_keys = ["all_strategies", "selected_strategies", "killzone_map", "current_session"]
        runtime_state_path = os.path.join("logs", "bot_runtime_settings.json")
        os.makedirs(os.path.dirname(runtime_state_path), exist_ok=True)
        try:
            with open(runtime_state_path, "w") as f:
                json.dump({k: settings_data.get(k) for k in runtime_keys if k in settings_data}, f, indent=2)
        except Exception:
            pass

        # Merge persistent config as before
        config.update(settings_data)
        save_config(config)
        log_event("config", "Bot Settings Updated", "Configuration settings updated successfully via UI.", "settings")
        return {"success": True, "message": "Settings updated successfully!"}
    except Exception as e:
        log_event("error", "Failed to Update Settings", f"Error updating configuration: {e}", "settings")
        return JSONResponse(status_code=500, content={"success": False, "error": f"Failed to update settings: {e}"})

def stream_output(pipe, prefix):
    """Reads from a pipe and prints lines with a prefix."""
    try:
        with pipe:
            for line in iter(pipe.readline, b''):
                print(f"[{prefix}] {line.decode().strip()}")
    except Exception as e:
        print(f"Error in stream_output thread: {e}")

def start_bot_process():
    """Starts the bot as a subprocess and streams its output."""
    global bot_process
    if bot_process and bot_process.poll() is None:
        print("Bot process is already running.")
        return

    print("Attempting to start the bot process...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    command = [sys.executable, "-m", "STOCKDATA"]
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        # Use CREATE_NEW_PROCESS_GROUP on Windows to ensure the bot process can be terminated cleanly
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        bot_process = subprocess.Popen(
            command,
            cwd=project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags
        )
        print(f"Bot process started with PID: {bot_process.pid}")

        # Start threads to stream stdout and stderr
        threading.Thread(target=stream_output, args=(bot_process.stdout, 'BOT-STDOUT'), daemon=True).start()
        threading.Thread(target=stream_output, args=(bot_process.stderr, 'BOT-STDERR'), daemon=True).start()

        log_event("bot_control", "Bot Started", "Trading bot process initiated.", "status")

    except Exception as e:
        print(f"Failed to start bot process: {e}")
        log_event("error", "Failed to Start Bot", f"Error starting bot process: {e}", "status")

@app.post("/api/bot/start")
def start_bot():
    """API endpoint to manually start the bot."""
    print("Received request to start bot...")
    start_bot_process()
    return {"success": True, "message": "Bot start initiated."}

@app.post("/api/bot/stop")
def stop_bot():
    global bot_process
    if bot_process is None or bot_process.poll() is not None:
        return JSONResponse(status_code=400, content={"success": False, "error": "Bot is not running."})
    try:
        # Terminate the process. On Windows, this will kill the process tree.
        # For cross-platform, process.terminate() or process.kill() might be needed.
        bot_process.terminate()
        # Optional: wait for process to terminate
        bot_process.wait(timeout=5) 
        bot_process = None # Reset global variable
        log_event("bot_control", "Bot Stopped", "Trading bot process terminated.", "status")
        return {"success": True, "message": "Bot stopped successfully!"}
    except Exception as e:
        log_event("error", "Failed to Stop Bot", f"Error stopping bot process: {e}", "status")
        return JSONResponse(status_code=500, content={"success": False, "error": f"Failed to stop bot: {e}"})

@app.get("/api/bot/status")
def bot_status():
    global bot_process
    running = bot_process is not None and bot_process.poll() is None
    return {"running": running}

@app.post("/api/user/google")
def save_google_user(user: GoogleUser):
    users_file = "users.json"
    users = []
    if os.path.exists(users_file):
        with open(users_file, "r") as f:
            users = json.load(f)
    for u in users:
        if u["email"] == user.email:
            u["name"] = user.name
            break
    else:
        users.append({"email": user.email, "name": user.name})
    with open(users_file, "w") as f:
        json.dump(users, f, indent=2)
    return {"success": True}

@app.get("/api/activity-log")
def get_activity_log():
    with activity_log_lock:
        return list(reversed(activity_log))  # newest first

@app.get("/api/analytics/trade-stats")
def api_trade_stats(symbol: str = Query(...), strategy: str = Query(...)):
    try:
        stats = get_trade_stats(symbol, strategy)
        return JSONResponse(content={"success": True, "data": stats})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)})

@app.get("/api/analytics/pnl-by-day")
def pnl_by_day(days: int = 14):
    """Return daily P/L totals for the last N days."""
    try:
        db_file = "trades/trades.db"
        import sqlite3, datetime as dt
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        since = (dt.datetime.utcnow() - dt.timedelta(days=days)).strftime("%Y-%m-%d")
        c.execute("""
            SELECT substr(entry_time,1,10) as d, COALESCE(SUM(CAST(profit as REAL)),0)
            FROM trades
            WHERE entry_time >= ?
            GROUP BY d
            ORDER BY d ASC
        """, (since,))
        rows = c.fetchall()
        conn.close()
        return {"success": True, "data": [{"date": d, "pnl": float(p)} for d, p in rows]}
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)})

@app.get("/api/analytics/pnl-by-weekday")
def pnl_by_weekday():
    """Return P/L aggregated by weekday (Mon..Sun)."""
    try:
        db_file = "trades/trades.db"
        import sqlite3
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        c.execute("""
            SELECT strftime('%w', entry_time) as wd, COALESCE(SUM(CAST(profit as REAL)),0) as pnl
            FROM trades
            GROUP BY wd
            ORDER BY wd
        """)
        rows = c.fetchall()
        conn.close()
        # map 0..6 to Mon..Sun style labels
        labels = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']
        return {"success": True, "data": [{"weekday": labels[int(wd)], "pnl": float(p)} for wd, p in rows]}
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)})

@app.on_event("startup")
async def startup_event():
    print("Server started. Bot will not auto-start. Control it from the web UI.")

# Optional: endpoint to update an account snapshot (balance/equity) from UI
@app.post("/api/mt5/update-account")
def update_account_snapshot(acc: MT5Account):
    global mt5_accounts
    mt5_accounts = [a for a in mt5_accounts if a.login != acc.login or a.server != acc.server]
    mt5_accounts.append(acc)
    save_mt5_accounts(mt5_accounts)
    log_event("connection", "MT5 Account Updated", f"Account ****{acc.login[-4:]} ({acc.server}) snapshot updated.", "connection")
    return {"success": True}

# To run: uvicorn api_server:app --reload 