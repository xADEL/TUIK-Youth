"""
Baseline Model
A simple Linear Regression baseline for comparing against more complex models.
"""

import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


class BaselineForecaster:
    """
    Simple regression baseline for unemployment forecasting.
    Provides a benchmark to compare against the main supervised models.
    """
    
    def __init__(self, model_type: str = "linear"):
        """
        Initialize with either linear regression or random forest.
        
        Args:
            model_type: "linear" or "random_forest"
        """
        self.model_type = model_type
        if model_type == "linear":
            self.model = LinearRegression()
        else:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.scaler = None
        self.feature_columns = None
        self.last_time_index = None
        
    def fit(self, X_train, y_train, scaler, feature_columns, last_time_index):
        """
        Train the baseline model.
        
        Args:
            X_train: Scaled training features
            y_train: Target values
            scaler: Fitted StandardScaler
            feature_columns: Feature names
            last_time_index: For calculating future time indices
        """
        self.model.fit(X_train, y_train)
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.last_time_index = last_time_index
        
        print(f"{self.model_type.title()} model fitted successfully")
        
        return self
    
    def predict(self, X) -> np.ndarray:
        """Predict on scaled features."""
        return self.model.predict(X)
    
    def predict_date(self, year: int, month: int) -> float:
        """
        Predict unemployment for a specific future date.
        
        Args:
            year: Target year
            month: Target month (1-12)
            
        Returns:
            Predicted rate as a percentage
        """
        # Calculate how far into the future
        months_from_end = (year - 2025) * 12 + (month - 12)
        time_index = self.last_time_index + months_from_end
        
        # Build feature vector
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        features = np.array([[year, month, month_sin, month_cos, time_index]])
        features_scaled = self.scaler.transform(features)
        
        return self.model.predict(features_scaled)[0]
    
    def save(self, filepath: str = "outputs/baseline_model.pkl"):
        """Save the model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_type': self.model_type,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'last_time_index': self.last_time_index
            }, f)
        print(f"Baseline model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str = "outputs/baseline_model.pkl"):
        """Load a saved model."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        forecaster = cls(model_type=data['model_type'])
        forecaster.model = data['model']
        forecaster.scaler = data['scaler']
        forecaster.feature_columns = data['feature_columns']
        forecaster.last_time_index = data['last_time_index']
        
        return forecaster


if __name__ == "__main__":
    from data_preprocessing import load_data, create_features, prepare_data_for_regression
    
    # Load and prepare data
    df = load_data()
    df = create_features(df)
    X_train, X_test, y_train, y_test, scaler, features = prepare_data_for_regression(df)
    
    # Train the baseline
    lr_forecaster = BaselineForecaster(model_type="linear")
    lr_forecaster.fit(X_train, y_train, scaler, features, df['time_index'].iloc[-1])
    
    # Test predictions
    y_pred = lr_forecaster.predict(X_test)
    print(f"\nTest predictions: {y_pred}")
    
    # Future prediction
    pred = lr_forecaster.predict_date(2026, 6)
    print(f"\n2026-06 Prediction: {pred:.2f}")
    
    lr_forecaster.save()
