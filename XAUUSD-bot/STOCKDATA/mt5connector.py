import MetaTrader5 as mt5

# ==== Fill your credentials here ====
account = 5036996416                         # 🔑 Your MT5 account number (int)
# ====================================

# Initialize MetaTrader 5
# Manual connection logic (for illustration):
mt5.initialize()
if mt5.initialize():
    print(" MT5 connected successfully!")
else:
    print(" MT5 connection failed:", mt5.last_error())

account_info = mt5.account_info()
if account_info is not None:
    print(" Logged in as:", account_info.login)
    print(" Balance:", account_info.balance)
else:
    print(" MT5 connection failed or not logged in:", mt5.last_error())
    print(" Login failed, check your credentials!")

# Don't forget to shutdown when done
# Removed mt5.shutdown() call. Shutdown handled by main bot or manual session only.
