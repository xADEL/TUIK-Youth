"""
ARIMA Model
Time series forecasting using ARIMA from statsmodels.
"""

import warnings
import pickle
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class ARIMAForecaster:
    """
    ARIMA-based forecaster for unemployment time series.
    Handles model selection, fitting, and forecasting with confidence intervals.
    """
    
    def __init__(self, order=(1, 1, 1)):
        """
        Initialize with ARIMA order.
        
        Args:
            order: Tuple of (p, d, q) where:
                   p = autoregressive order
                   d = differencing order
                   q = moving average order
        """
        self.order = order
        self.model = None
        self.model_fit = None
        self.training_data = None
        
    def check_stationarity(self, series: pd.Series) -> dict:
        """
        Run the Augmented Dickey-Fuller test.
        A p-value < 0.05 suggests the series is stationary.
        """
        result = adfuller(series.dropna())
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05
        }
    
    def find_best_order(self, series: pd.Series, max_p=3, max_d=2, max_q=3) -> tuple:
        """
        Search for the best ARIMA order using AIC.
        Tests all combinations up to the specified maximums.
        """
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fit = model.fit()
                        if fit.aic < best_aic:
                            best_aic = fit.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        print(f"Best ARIMA order: {best_order} (AIC: {best_aic:.2f})")
        self.order = best_order
        return best_order
    
    def fit(self, series: pd.Series, auto_order: bool = True):
        """
        Fit the ARIMA model to the time series.
        
        Args:
            series: Time series with datetime index
            auto_order: If True, automatically search for best (p,d,q)
        """
        self.training_data = series
        
        if auto_order:
            self.find_best_order(series)
        
        self.model = ARIMA(series, order=self.order)
        self.model_fit = self.model.fit()
        
        print(f"ARIMA{self.order} fitted successfully")
        print(f"AIC: {self.model_fit.aic:.2f}, BIC: {self.model_fit.bic:.2f}")
        
        return self
    
    def predict(self, steps: int = 1) -> pd.Series:
        """Forecast the next `steps` periods."""
        if self.model_fit is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model_fit.forecast(steps=steps)
    
    def predict_with_confidence(self, steps: int = 1, alpha: float = 0.05):
        """
        Forecast with confidence intervals.
        
        Returns:
            (mean_forecast, confidence_interval_dataframe)
        """
        if self.model_fit is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        forecast = self.model_fit.get_forecast(steps=steps)
        mean = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=alpha)
        
        return mean, conf_int
    
    def predict_in_sample(self) -> pd.Series:
        """Get fitted values for the training period."""
        if self.model_fit is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model_fit.fittedvalues
    
    def forecast_date(self, target_year: int, target_month: int) -> tuple:
        """
        Forecast for a specific future date.
        
        Returns:
            (prediction, lower_bound, upper_bound)
        """
        if self.training_data is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        last_date = self.training_data.index[-1]
        target_date = pd.Timestamp(year=target_year, month=target_month, day=1)
        
        # How many months until the target date?
        months_diff = (target_year - last_date.year) * 12 + (target_month - last_date.month)
        
        if months_diff <= 0:
            raise ValueError(f"Target date must be after {last_date.strftime('%Y-%m')}")
        
        mean, conf_int = self.predict_with_confidence(steps=months_diff)
        
        prediction = mean.iloc[-1]
        lower = conf_int.iloc[-1, 0]
        upper = conf_int.iloc[-1, 1]
        
        return prediction, lower, upper
    
    def save(self, filepath: str = "outputs/arima_model.pkl"):
        """Serialize the fitted model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'order': self.order,
                'model_fit': self.model_fit,
                'training_data': self.training_data
            }, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str = "outputs/arima_model.pkl"):
        """Load a previously saved model."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        forecaster = cls(order=data['order'])
        forecaster.model_fit = data['model_fit']
        forecaster.training_data = data['training_data']
        
        return forecaster


if __name__ == "__main__":
    from data_preprocessing import load_data, create_features, prepare_data_for_arima
    
    # Load and prepare data
    df = load_data()
    df = create_features(df)
    train_series, test_series, full_series = prepare_data_for_arima(df)
    
    # Fit ARIMA
    forecaster = ARIMAForecaster()
    forecaster.fit(train_series, auto_order=True)
    
    # Check stationarity
    stat_result = forecaster.check_stationarity(train_series)
    print(f"\nStationarity test: {stat_result}")
    
    # Forecast the test period
    predictions = forecaster.predict(steps=len(test_series))
    print(f"\nTest predictions:\n{predictions}")
    
    # Forecast a specific future date
    pred, lower, upper = forecaster.forecast_date(2026, 6)
    print(f"\n2026-06 Forecast: {pred:.2f} (95% CI: {lower:.2f} - {upper:.2f})")
    
    forecaster.save()
