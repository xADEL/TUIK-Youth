"""
Data Preprocessing Module
Handles loading, cleaning, and feature engineering for TÜİK unemployment data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str = "data/tuik_youth_unemployment.csv") -> pd.DataFrame:
    """Load and parse the unemployment dataset."""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate features from the date column.
    
    Creates:
        - year, month: basic date components
        - month_sin, month_cos: cyclical encoding to capture seasonality
        - time_index: sequential counter to capture long-term trend
    """
    df = df.copy()
    
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Cyclical encoding helps the model understand that December and January are close
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    df['time_index'] = range(len(df))
    
    return df


def prepare_data_for_regression(df: pd.DataFrame, test_size: float = 0.2):
    """
    Split and scale data for regression models.
    Uses temporal split to avoid data leakage in time series.
    
    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_columns
    """
    feature_columns = ['year', 'month', 'month_sin', 'month_cos', 'time_index']
    
    X = df[feature_columns].values
    y = df['unemployment_rate'].values
    
    # Temporal split: train on earlier data, test on later data
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns


def prepare_data_for_arima(df: pd.DataFrame, test_size: float = 0.2):
    """
    Prepare data for ARIMA-style time series models.
    
    Returns:
        train_series, test_series, full_series
    """
    series = df.set_index('date')['unemployment_rate']
    
    split_idx = int(len(series) * (1 - test_size))
    train_series = series[:split_idx]
    test_series = series[split_idx:]
    
    return train_series, test_series, series


def create_future_features(year: int, month: int, last_time_index: int) -> np.ndarray:
    """Build a feature array for predicting a future date."""
    time_index = last_time_index + ((year - 2025) * 12 + month - 12)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    return np.array([[year, month, month_sin, month_cos, time_index]])


if __name__ == "__main__":
    # Quick test of the preprocessing pipeline
    df = load_data()
    print(f"Loaded {len(df)} records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    df = create_features(df)
    print(f"\nFeatures created: {list(df.columns)}")
    
    X_train, X_test, y_train, y_test, scaler, features = prepare_data_for_regression(df)
    print(f"\nRegression split: Train={len(X_train)}, Test={len(X_test)}")
    
    train_s, test_s, full_s = prepare_data_for_arima(df)
    print(f"ARIMA split: Train={len(train_s)}, Test={len(test_s)}")
