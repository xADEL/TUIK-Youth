"""
TÜİK Youth Unemployment Forecasting
Gradio web application for predicting Turkish youth unemployment rates.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# --- Data Loading ---

def load_data(mode="required"):
    """Load the TÜİK unemployment dataset based on mode."""
    filepath = "data/tuik_youth_unemployment_full.csv" if mode == "extended" else "data/tuik_youth_unemployment.csv"
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def create_features(df):
    """Engineer time-based features for the ML models."""
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Sequential index to capture the overall trend
    df['time_idx'] = np.arange(len(df))
    
    # Encode month as sine/cosine to capture seasonal patterns
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df


# --- Forecasting Model ---

class ProperForecaster:
    """
    Wrapper around sklearn regression models for unemployment forecasting.
    Handles feature scaling, training, and prediction for future dates.
    """
    
    FEATURE_COLS = ['year', 'month', 'month_sin', 'month_cos', 'time_idx']
    
    def __init__(self, model_type="gradient_boosting"):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = None
        self.last_time_idx = None
        self.train_metrics = {}
        
        # Initialize the appropriate sklearn model
        if model_type == "linear":
            self.model = LinearRegression()
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
    
    def fit(self, df):
        """Train the model on the provided dataframe."""
        X = df[self.FEATURE_COLS].values
        y = df['unemployment_rate'].values
        
        # Normalize features for better model performance
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        # Remember the last time index for future predictions
        self.last_time_idx = df['time_idx'].iloc[-1]
        
        # Compute training metrics for display
        y_pred = self.model.predict(X_scaled)
        self.train_metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        
        return self
    
    def predict(self, year, month, time_idx):
        """Predict unemployment rate for a given year/month."""
        year = int(year)
        month = int(month)
        
        # Build the feature vector for this prediction
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        features = np.array([[year, month, month_sin, month_cos, time_idx]])
        features_scaled = self.scaler.transform(features)
        
        return self.model.predict(features_scaled)[0]


# --- Model Storage ---

MODELS = {"required": {}, "extended": {}}
DATA_CACHE = {"required": None, "extended": None}


def train_all_models(mode="required"):
    """Train all three models for the specified dataset mode."""
    global MODELS, DATA_CACHE
    
    if MODELS[mode]:
        return
    
    print(f"\n{'='*50}")
    print(f"Training {mode.upper()} models")
    print(f"{'='*50}")
    
    df = load_data(mode)
    df = create_features(df)
    DATA_CACHE[mode] = df
    
    print(f"Data: {len(df)} records from {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
    
    for model_type in ["linear", "random_forest", "gradient_boosting"]:
        model = ProperForecaster(model_type=model_type)
        model.fit(df)
        MODELS[mode][model_type] = model
        print(f"\n{model_type}:")
        print(f"  RMSE: {model.train_metrics['rmse']:.4f}")
        print(f"  MAE:  {model.train_metrics['mae']:.4f}")
        print(f"  R²:   {model.train_metrics['r2']:.4f}")
    
    print(f"\n{'='*50}\n")


def get_model(model_type, mode):
    """Retrieve a trained model, training first if necessary."""
    train_all_models(mode)
    return MODELS[mode][model_type]


def get_data(mode):
    """Retrieve the cached dataset for a mode."""
    train_all_models(mode)
    return DATA_CACHE[mode]


# --- Prediction Logic ---

def make_prediction(year, month_name, model_name, mode):
    """Generate a prediction and visualization for the user's input."""
    
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    
    model_key_map = {
        "Linear Regression": "linear",
        "Random Forest": "random_forest",
        "Gradient Boosting": "gradient_boosting"
    }
    
    year = int(year)
    month = month_map[month_name]
    model_key = model_key_map[model_name]
    
    model = get_model(model_key, mode)
    df = get_data(mode)
    
    # Calculate how far into the future we're predicting
    last_date = df['date'].max()
    months_ahead = (year - last_date.year) * 12 + (month - last_date.month)
    future_time_idx = model.last_time_idx + months_ahead
    
    prediction = model.predict(year, month, future_time_idx)
    fig = create_forecast_plot(df, year, month, prediction, model_name, mode, months_ahead)
    
    # Confidence interval widens for predictions further in the future
    rmse = model.train_metrics['rmse']
    error_margin = rmse * (1 + 0.02 * max(0, months_ahead))
    lower = prediction - 1.96 * error_margin
    upper = prediction + 1.96 * error_margin
    
    result = f"""
## Prediction Result

| Parameter | Value |
|-----------|-------|
| **Predicted Rate** | **{prediction:.2f}%** |
| 95% Confidence | {lower:.2f}% - {upper:.2f}% |
| Target Date | {month_name} {year} |
| Months Ahead | {months_ahead} |

### Model Performance (Training)
| Metric | Value |
|--------|-------|
| RMSE | {model.train_metrics['rmse']:.4f} |
| MAE | {model.train_metrics['mae']:.4f} |
| R² | {model.train_metrics['r2']:.4f} |

---
*Model: {model_name} | Data: {df['date'].min().year}-{df['date'].max().year} ({len(df)} records)*
"""
    
    return result, fig


# --- Visualization ---

def create_forecast_plot(df, pred_year, pred_month, prediction, model_name, mode, months_ahead):
    """Plot historical data with the forecasted point."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['date'], df['unemployment_rate'], 'o-', 
            linewidth=2, markersize=3, color='#2E86AB', label='Historical Data')
    
    pred_date = pd.Timestamp(year=pred_year, month=pred_month, day=15)
    ax.scatter([pred_date], [prediction], s=200, color='#E63946', 
               marker='*', zorder=5, edgecolors='white', linewidths=1,
               label=f'Prediction: {prediction:.1f}%')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Unemployment Rate (%)', fontsize=12)
    ax.set_title(f'Youth Unemployment Forecast\n{model_name} | {pred_month}/{pred_year} ({months_ahead} months ahead)', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def create_trend_plot(mode):
    """Plot the unemployment trend with a linear fit."""
    df = get_data(mode)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['date'], df['unemployment_rate'], 'o-', 
            linewidth=2, markersize=4, color='#2E86AB', label='Actual')
    
    # Fit and display a simple linear trend
    z = np.polyfit(df['time_idx'], df['unemployment_rate'], 1)
    trend_line = np.polyval(z, df['time_idx'])
    ax.plot(df['date'], trend_line, '--', color='#E63946', 
            linewidth=2, label=f'Linear Trend ({z[0]:.3f}/month)')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Unemployment Rate (%)', fontsize=12)
    ax.set_title(f'Youth Unemployment Trend ({df["date"].min().year}-{df["date"].max().year})', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


# --- Gradio Interface ---

def create_app():
    """Build and return the Gradio application."""
    
    # Pre-train models on startup so users don't wait
    train_all_models("required")
    train_all_models("extended")
    
    with gr.Blocks(
        title="TÜİK Youth Unemployment Forecaster",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="red")
    ) as app:
        
        gr.Markdown("""
        # TÜİK Youth Unemployment Forecaster
        Predict youth unemployment rates (ages 15-24) using supervised learning models.
        """)
        
        with gr.Tabs():
            # Required dataset tab (2023-2025)
            with gr.TabItem("📊 2023-2025 (Required)"):
                gr.Markdown("> **Project Requirement:** 34 monthly records from 2023-2025")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        year1 = gr.Dropdown([2026, 2027, 2028, 2029, 2030], value=2026, label="Year")
                        month1 = gr.Dropdown(
                            ["January", "February", "March", "April", "May", "June",
                             "July", "August", "September", "October", "November", "December"],
                            value="June", label="Month"
                        )
                        model1 = gr.Dropdown(
                            ["Linear Regression", "Random Forest", "Gradient Boosting"],
                            value="Gradient Boosting", label="Model"
                        )
                        btn1 = gr.Button("🔮 Predict", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        out1_text = gr.Markdown()
                        out1_plot = gr.Plot()
                
                btn1.click(
                    fn=lambda y, m, model: make_prediction(y, m, model, "required"),
                    inputs=[year1, month1, model1],
                    outputs=[out1_text, out1_plot]
                )
                
                gr.Markdown("### 📈 Trend Analysis")
                trend1_plot = gr.Plot()
                trend1_btn = gr.Button("Show Trend")
                trend1_btn.click(fn=lambda: create_trend_plot("required"), outputs=trend1_plot)
            
            # Extended dataset tab (2005-2025)
            with gr.TabItem("📊 2005-2025 (Extended)"):
                gr.Markdown("> **Extended Analysis:** 250 monthly records from 2005-2025 (20 years)")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        year2 = gr.Dropdown([2026, 2027, 2028, 2029, 2030], value=2026, label="Year")
                        month2 = gr.Dropdown(
                            ["January", "February", "March", "April", "May", "June",
                             "July", "August", "September", "October", "November", "December"],
                            value="June", label="Month"
                        )
                        model2 = gr.Dropdown(
                            ["Linear Regression", "Random Forest", "Gradient Boosting"],
                            value="Gradient Boosting", label="Model"
                        )
                        btn2 = gr.Button("🔮 Predict", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        out2_text = gr.Markdown()
                        out2_plot = gr.Plot()
                
                btn2.click(
                    fn=lambda y, m, model: make_prediction(y, m, model, "extended"),
                    inputs=[year2, month2, model2],
                    outputs=[out2_text, out2_plot]
                )
                
                gr.Markdown("### 📈 Trend Analysis")
                trend2_plot = gr.Plot()
                trend2_btn = gr.Button("Show Trend")
                trend2_btn.click(fn=lambda: create_trend_plot("extended"), outputs=trend2_plot)
        
        gr.Markdown("""
        ---
        **Models:** LinearRegression, RandomForestRegressor (100 trees), GradientBoostingRegressor (100 trees)
        
        **Features:** Year, Month, Cyclical Month Encoding (sin/cos), Time Index
        
        *Course: Introduction to Artificial Intelligence | Data: TÜİK (Turkish Statistical Institute)*
        """)
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.launch()
