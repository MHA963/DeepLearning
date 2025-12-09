# data_loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_sequence_data(csv_path, lookback=48, features=None, label_col="label", test_size=0.2):
    """
    Loads time series CSV data, creates sequences, normalizes features, and splits into train/test.
    
    Args:
        csv_path (str): Path to CSV file.
        lookback (int): Number of timesteps per sequence.
        features (list): List of feature column names.
        label_col (str): Name of the target column.
        test_size (float): Fraction of data for test split.
        
    Returns:
        X (np.ndarray): All sequences.
        y (np.ndarray): All labels.
        X_train, X_test, y_train, y_test: Train/test splits normalized.
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    
    if features is None:
        features = ["open", "high", "low", "close", "volume"]
    
    data = df[features].values
    labels = df[label_col].values
    
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(labels[i+lookback])
        
    X = np.array(X)
    y = np.array(y)
    
    # Train/-test split
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Normalizes features
    scaler = MinMaxScaler()
    X_train_shape = X_train.shape
    X_train = scaler.fit_transform(X_train.reshape(-1, len(features))).reshape(X_train_shape)
    X_test = scaler.transform(X_test.reshape(-1, len(features))).reshape(X_test.shape)
    
    return X, y, X_train, X_test, y_train, y_test
