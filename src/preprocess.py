import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Feature engineering
    df['Return'] = df['Close'].pct_change()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df = df.dropna()

    return df

def compute_rsi(series, window):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_sequences(df, sequence_length):
    scaler = StandardScaler()
    data = scaler.fit_transform(df.values)

    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length, 3])  # Close price

    return np.array(X), np.array(y), scaler
