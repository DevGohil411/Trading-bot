
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tradingeconomics_calendar import fetch_te_calendar
from news_fetcher import fetch_market_news

print(fetch_te_calendar()[:5])
print(fetch_market_news(source='newsapi')[:5])
