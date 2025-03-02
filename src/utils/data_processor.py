import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def preprocess_data(data):
    # Convert price data to numpy array
    if isinstance(data, pd.Series):
        prices = data.values.reshape(-1, 1)
    else:
        prices = np.array([price for price in data]).reshape(-1, 1)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(prices)
    
    return normalized_data, scaler

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)

def validate_input(coin, days):
    if not isinstance(coin, str):
        raise ValueError("Coin must be a string")
    if not isinstance(days, int) or days < 1 or days > 100:  # Updated maximum days
        raise ValueError("Days must be an integer between 1 and 100")  # Updated error message