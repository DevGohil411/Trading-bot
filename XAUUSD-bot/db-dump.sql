BEGIN TRANSACTION;
CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket INTEGER UNIQUE,
                symbol TEXT,
                strategy TEXT,
                trade_type TEXT,
                lot_size REAL,
                entry_time TEXT,
                stop_loss REAL,
                take_profit REAL,
                comment TEXT,
                risk_amount REAL,
                equity_before REAL,
                equity_after REAL,
                exit_time TEXT,
                profit REAL,
                win_loss TEXT,
                slippage REAL,
                latency REAL
            );
DELETE FROM "sqlite_sequence";
COMMIT;