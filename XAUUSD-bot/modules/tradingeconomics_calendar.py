import requests
import pandas as pd
import logging

def fetch_te_calendar():
    """
    Fetch economic calendar from TradingEconomics public API.
    Returns a list of dicts with keys: date, time, country, event, actual, forecast, previous, impact
    """
    url = 'https://api.tradingeconomics.com/calendar?c=guest:guest'
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        events = []
        for item in data:
            # Parse date and time
            dt = item.get('Date', '')
            date, time = '', ''
            if dt:
                try:
                    dt_obj = pd.to_datetime(dt)
                    date = dt_obj.strftime('%Y-%m-%d')
                    time = dt_obj.strftime('%H:%M')
                except Exception:
                    date, time = dt, ''
            events.append({
                'date': date,
                'time': time,
                'country': item.get('Country', ''),
                'event': item.get('Event', ''),
                'actual': item.get('Actual', ''),
                'forecast': item.get('Forecast', ''),
                'previous': item.get('Previous', ''),
                'impact': item.get('Importance', '')
            })
        return events
    except Exception as e:
        logging.warning(f"TradingEconomics calendar fetch failed: {e}")
        return []
