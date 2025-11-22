import logging
import requests
import pandas as pd

def fetch_market_news(source='newsapi'):
    """
    Fetch market news from NewsAPI or Yahoo Finance (via yfinance).
    Returns a list of dicts: date, title, description, url
    """
    if source == 'newsapi':
        # You must set your NewsAPI key here
        API_KEY = '6f24a501f01a468898385ffb407340d3'
        if not API_KEY:
            logging.warning('No NewsAPI key set in news_fetcher.py')
            return []
        url = f'https://newsapi.org/v2/everything?q=market&language=en&sortBy=publishedAt&pageSize=20&apiKey={API_KEY}'
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get('articles', [])
            news = []
            for item in articles:
                news.append({
                    'date': item.get('publishedAt', ''),
                    'title': item.get('title', ''),
                    'description': item.get('description', ''),
                    'url': item.get('url', ''),
                    'source': item.get('source', {}).get('name', '')
                })
            return news
        except Exception as e:
            logging.warning(f"NewsAPI fetch failed: {e}")
            return []
    elif source == 'yahoo':
        try:
            import yfinance as yf
            ticker = yf.Ticker('SPY')
            news_raw = ticker.news
            news = []
            for item in news_raw:
                news.append({
                    'date': item.get('providerPublishTime', ''),
                    'title': item.get('title', ''),
                    'description': item.get('summary', ''),
                    'url': item.get('link', ''),
                    'source': item.get('publisher', '')
                })
            return news
        except Exception as e:
            logging.warning(f"Yahoo Finance news fetch failed: {e}")
            return []
    else:
        logging.warning(f"Unknown news source: {source}")
        return []
