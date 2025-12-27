---
title: Youth Unemployment Forecasting
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# TÜİK Youth Unemployment Forecasting

Predict Turkish youth unemployment rates (ages 15-24) using machine learning models trained on real TÜİK data.

**Live Demo:** [https://huggingface.co/spaces/xAdeell/tuik-youth-unemployment](https://huggingface.co/spaces/xAdeell/tuik-youth-unemployment)

## Features

- **Three ML Models:** Linear Regression, Random Forest, Gradient Boosting
- **Two Datasets:** 2023-2025 (34 records) and 2005-2025 (250 records)
- **Interactive Predictions:** Select any future date (2026-2030)
- **Confidence Intervals:** 95% CI based on model RMSE
- **Trend Visualization:** View historical trends and forecasts

## Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/xADEL/TUIK-Youth.git
cd TUIK-Youth

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
python app.py
```

The app will start at `http://localhost:7860`

## Project Structure

```
TUIK-Youth/
├── app.py                 # Gradio web application
├── requirements.txt       # Python dependencies
├── data/
│   ├── tuik_youth_unemployment.csv       # 2023-2025 data (34 records)
│   └── tuik_youth_unemployment_full.csv  # 2005-2025 data (250 records)
├── src/
│   ├── data_preprocessing.py   # Data loading and feature engineering
│   ├── supervised_models.py    # ML model implementations
│   ├── evaluate.py             # Model evaluation and visualization
│   ├── baseline_model.py       # Baseline model for comparison
│   └── arima_model.py          # ARIMA time series model
└── report/
    ├── REPORT.md           # Project report (Markdown)
    └── *.png               # Visualization images
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| gradio | ≥4.0.0 | Web interface |
| pandas | ≥1.5.0 | Data manipulation |
| numpy | ≥1.23.0 | Numerical operations |
| scikit-learn | ≥1.2.0 | ML models |
| matplotlib | ≥3.6.0 | Visualization |
| statsmodels | ≥0.14.0 | ARIMA model |

## Models

| Model | Description | Hyperparameters |
|-------|-------------|-----------------|
| Linear Regression | Simple linear model | Default |
| Random Forest | Ensemble of 100 trees | max_depth=10 |
| Gradient Boosting | Sequential boosting | max_depth=5, lr=0.1 |

## Features Used

- **year**: Numeric year (2005-2025)
- **month**: Month (1-12)
- **month_sin/cos**: Cyclical encoding for seasonality
- **time_idx**: Sequential index for trend

## Data Source

TÜİK (Turkish Statistical Institute) - Seasonally Adjusted Youth Unemployment Rate (Ages 15-24)

## Usage Example

```python
from src.data_preprocessing import load_data, create_features
from src.supervised_models import SupervisedForecaster

# Load and prepare data
df = load_data()
df = create_features(df)

# Train model
model = SupervisedForecaster(model_type="gradient_boosting")
model.fit(X_train, y_train, scaler, features, last_time_index)

# Predict June 2026
prediction = model.predict_date(2026, 6)
print(f"Prediction: {prediction:.2f}%")
```

## Authors

- Yiğit Abuş (20214043035)
- Adel Aljamous (20212022023)
- Berkant Günel (20212022069)
- Leen Alwahsh (20212022281)
- Hashem Alshuwaikh (20212022206)

**Course:** ESOF325 - Introduction to Artificial Intelligence  
**Instructor:** Dr. Savaş Ünsal  
**Date:** December 2025
