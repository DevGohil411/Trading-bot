"""
Trade Statistics Logger Module
Provides functions to get trading statistics
"""

import sqlite3
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Setup logger for this module
logger = logging.getLogger('trade_logger')
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_trade_stats(symbol: str, strategy: str) -> Dict[str, Any]:
    """
    Get trading statistics for a specific symbol and strategy.
    
    Args:
        symbol: Trading symbol (e.g., 'XAUUSD')
        strategy: Strategy name
        
    Returns:
        Dictionary containing trade statistics
    """
    try:
        db_file = "trades/trades.db"
        
        if not os.path.exists(db_file):
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_profit": 0.0,
                "win_rate": 0.0,
                "avg_profit": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0
            }
        
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        
        # Get all trades for this symbol and strategy
        c.execute("""
            SELECT profit, entry_time, exit_time
            FROM trades
            WHERE symbol = ? AND strategy = ?
            ORDER BY entry_time DESC
        """, (symbol, strategy))
        
        trades = c.fetchall()
        conn.close()
        
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_profit": 0.0,
                "win_rate": 0.0,
                "avg_profit": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0
            }
        
        # Calculate statistics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if float(t[0]) > 0)
        losing_trades = sum(1 for t in trades if float(t[0]) < 0)
        
        total_profit = sum(float(t[0]) for t in trades)
        gross_profit = sum(float(t[0]) for t in trades if float(t[0]) > 0)
        gross_loss = abs(sum(float(t[0]) for t in trades if float(t[0]) < 0))
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        avg_profit = (gross_profit / winning_trades) if winning_trades > 0 else 0.0
        avg_loss = (gross_loss / losing_trades) if losing_trades > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "total_profit": round(total_profit, 2),
            "win_rate": round(win_rate, 2),
            "avg_profit": round(avg_profit, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2)
        }
        
    except Exception as e:
        print(f"Error getting trade stats: {e}")
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_profit": 0.0,
            "win_rate": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "error": str(e)
        }
