import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_crypto_data(coin):
    # Get historical data for the last 30 days
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart/range"
    params = {
        "vs_currency": "usd",
        "from": int(start_time.timestamp()),
        "to": int(end_time.timestamp())
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        print(response.json())
        return response.json()
    else:
        raise Exception("Error fetching data from CoinGecko API")

def process_data(data):
    # Convert timestamp and price data to DataFrame
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df['price']  # Return only the price series

def get_prediction_data(coin, days=30):
    """
    Fetch and process cryptocurrency data
    Args:
        coin (str): cryptocurrency id
        days (int): number of days of historical data to fetch
    Returns:
        pd.Series: processed price data
    """
    try:
        data = fetch_crypto_data(coin)
        df = process_data(data)
        if df.empty:
            raise ValueError("No data available")
        return df
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        raise