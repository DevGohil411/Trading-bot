-- SQLite schema for XAUUSD-bot
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  -- Common identifiers
  ticket INTEGER UNIQUE,
  timestamp TEXT,              -- from trade_logger
  entry_time TEXT,             -- from file.py
  exit_time TEXT,

  -- Classification
  symbol TEXT,
  strategy TEXT,
  trade_type TEXT,             -- BUY/SELL

  -- Sizing and pricing
  lot_size REAL,
  requested_price REAL,
  executed_price REAL,
  stop_loss REAL,
  take_profit REAL,

  -- Accounting
  profit REAL,
  win_loss TEXT,
  risk_amount REAL,
  equity_before REAL,
  equity_after REAL,
  equity TEXT,                 -- kept for compatibility with trade_logger

  -- Meta
  slippage REAL,
  latency REAL,
  comment TEXT,
  order_result TEXT,
  order_comment TEXT
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
CREATE INDEX IF NOT EXISTS idx_trades_type ON trades(trade_type); 