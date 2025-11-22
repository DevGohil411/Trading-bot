"""
STOCKDATA Package Main Entry Point
This file allows the package to be run as a module: python -m STOCKDATA
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import and run the actual main trading bot
try:
    from STOCKDATA.main import main
    
    if __name__ == "__main__":
        print("[*] Starting STOCKDATA Trading Bot from main.py...")
        main()
except ImportError as e:
    print(f"[ERROR] Failed to import main module: {e}")
    print("[INFO] Make sure all dependencies are installed")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
