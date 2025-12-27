"""
Supervised Learning Models
Implements Linear Regression, Random Forest, and Gradient Boosting for unemployment forecasting.
"""

import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class SupervisedForecaster:
    """
    Unified wrapper for training and using sklearn regression models.
    Handles model training, prediction, persistence, and feature importance.
    """
    
    MODELS = {
        "linear": LinearRegression,
        "random_forest": lambda: RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5),
        "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3)
    }
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize the forecaster with the specified model type.
        
        Args:
            model_type: One of "linear", "random_forest", or "gradient_boosting"
        """
        self.model_type = model_type
        
        if model_type == "linear":
            self.model = LinearRegression()
        else:
            self.model = self.MODELS[model_type]()
        
        self.scaler = None
        self.feature_columns = None
        self.last_time_index = None
        self.is_fitted = False
        
    def fit(self, X_train, y_train, scaler, feature_columns, last_time_index):
        """
        Train the model on the provided data.
        
        Args:
            X_train: Scaled training features
            y_train: Target values
            scaler: The fitted StandardScaler (needed for predicting new data)
            feature_columns: List of feature names
            last_time_index: Last time index in training data (for future predictions)
        """
        self.model.fit(X_train, y_train)
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.last_time_index = last_time_index
        self.is_fitted = True
        
        print(f"{self.model_type.replace('_', ' ').title()} model fitted successfully")
        
        return self
    
    def predict(self, X) -> np.ndarray:
        """Generate predictions for the given feature matrix."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def predict_date(self, year: int, month: int) -> float:
        """
        Predict unemployment for a specific future date.
        
        Args:
            year: Target year (e.g., 2026)
            month: Target month (1-12)
            
        Returns:
            Predicted unemployment rate as a percentage
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Figure out how many months into the future this is
        months_from_end = (year - 2025) * 12 + (month - 12)
        time_index = self.last_time_index + months_from_end
        
        # Build the feature vector
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        features = np.array([[year, month, month_sin, month_cos, time_index]])
        features_scaled = self.scaler.transform(features)
        
        return self.model.predict(features_scaled)[0]
    
    def get_feature_importance(self):
        """
        Return feature importances for tree-based models.
        Returns None for linear regression.
        """
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_columns, self.model.feature_importances_))
        return None
    
    def save(self, filepath: str = None):
        """Serialize the model to disk."""
        if filepath is None:
            filepath = f"outputs/{self.model_type}_model.pkl"
            
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_type': self.model_type,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'last_time_index': self.last_time_index
            }, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load a previously saved model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        forecaster = cls(model_type=data['model_type'])
        forecaster.model = data['model']
        forecaster.scaler = data['scaler']
        forecaster.feature_columns = data['feature_columns']
        forecaster.last_time_index = data['last_time_index']
        forecaster.is_fitted = True
        
        return forecaster


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from src.data_preprocessing import load_data, create_features, prepare_data_for_regression
    
    # Load and prepare data
    df = load_data()
    df = create_features(df)
    X_train, X_test, y_train, y_test, scaler, features = prepare_data_for_regression(df)
    
    print("="*50)
    print("TRAINING ALL MODELS")
    print("="*50)
    
    # Train and evaluate each model type
    for model_type in ["linear", "random_forest", "gradient_boosting"]:
        print(f"\n--- {model_type.upper()} ---")
        forecaster = SupervisedForecaster(model_type=model_type)
        forecaster.fit(X_train, y_train, scaler, features, df['time_index'].iloc[-1])
        
        # Evaluate on test set
        y_pred = forecaster.predict(X_test)
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # Example future prediction
        pred = forecaster.predict_date(2026, 6)
        print(f"2026-06 Prediction: {pred:.2f}%")
        
        # Show feature importance for tree models
        importance = forecaster.get_feature_importance()
        if importance:
            print(f"Feature Importance: {importance}")
        
        forecaster.save()
