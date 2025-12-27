"""
Model Evaluation and Visualization
Compares model performance and generates the required plots for the report.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def calculate_metrics(y_true, y_pred) -> dict:
    """Compute MAE, RMSE, and R² for the predictions."""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R²': r2_score(y_true, y_pred)
    }


def print_metrics_comparison(metrics_dict: dict):
    """Print a formatted comparison table and identify the best model."""
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    print(f"{'Model':<25} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
    print("-"*60)
    
    best_rmse = float('inf')
    best_model = None
    
    for model_name, metrics in metrics_dict.items():
        print(f"{model_name:<25} {metrics['MAE']:<10.4f} {metrics['RMSE']:<10.4f} {metrics['R²']:<10.4f}")
        if metrics['RMSE'] < best_rmse:
            best_rmse = metrics['RMSE']
            best_model = model_name
    
    print("="*60)
    print(f"✓ Best Model: {best_model} (lowest RMSE: {best_rmse:.4f})")
    
    return best_model


def plot_real_vs_predicted(dates, y_true, predictions_dict,
                           save_path: str = "outputs/visualizations/real_vs_predicted.png"):
    """
    Show actual vs predicted values for all models on the same chart.
    Helps visualize how well each model tracks the real data.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    ax.plot(dates, y_true, 'o-', label='Actual', linewidth=2.5, markersize=8, color=colors[0])
    
    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        ax.plot(dates, y_pred, '--', label=model_name, linewidth=2, 
                markersize=5, color=colors[(idx+1) % len(colors)], alpha=0.8)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Unemployment Rate (%)', fontsize=12)
    ax.set_title('Youth Unemployment: Actual vs Predicted (Test Set)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def plot_residuals(y_true, predictions_dict,
                   save_path: str = "outputs/visualizations/residuals.png"):
    """
    Plot residuals for each model to check for patterns or bias.
    Ideally, residuals should be randomly scattered around zero.
    """
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    colors = ['#A23B72', '#F18F01', '#C73E1D']
    
    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        residuals = y_true - y_pred
        axes[idx].scatter(y_pred, residuals, alpha=0.7, color=colors[idx % len(colors)], s=60)
        axes[idx].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[idx].set_xlabel('Predicted', fontsize=11)
        axes[idx].set_ylabel('Residuals', fontsize=11)
        axes[idx].set_title(f'{model_name} Residuals', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def plot_trend_line(df: pd.DataFrame,
                    save_path: str = "outputs/visualizations/trend_line.png"):
    """
    Visualize the overall unemployment trend with a fitted linear line.
    Shows whether unemployment is generally rising or falling.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['date'], df['unemployment_rate'], 'o-', linewidth=2.5, 
            markersize=6, color='#2E86AB')
    
    # Fit a simple linear trend
    z = np.polyfit(range(len(df)), df['unemployment_rate'], 1)
    p = np.poly1d(z)
    ax.plot(df['date'], p(range(len(df))), '--', color='#A23B72', 
            linewidth=2, label=f'Trend (slope: {z[0]:.3f})')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Unemployment Rate (%)', fontsize=12)
    ax.set_title('Youth Unemployment Rate Trend (2023-2025)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def plot_seasonality(df: pd.DataFrame,
                     save_path: str = "outputs/visualizations/seasonality.png"):
    """
    Show monthly patterns in unemployment.
    Helps identify if certain months consistently have higher/lower rates.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Boxplot showing distribution for each month
    monthly_data = [df[df['month'] == m]['unemployment_rate'].values for m in range(1, 13)]
    bp = axes[0].boxplot(monthly_data, patch_artist=True)
    
    colors = plt.cm.RdYlBu(np.linspace(0.2, 0.8, 12))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[0].set_xlabel('Month', fontsize=11)
    axes[0].set_ylabel('Unemployment Rate (%)', fontsize=11)
    axes[0].set_title('Monthly Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Bar chart of monthly averages
    monthly_avg = df.groupby('month')['unemployment_rate'].mean()
    axes[1].bar(monthly_avg.index, monthly_avg.values, color='#A23B72', alpha=0.7)
    axes[1].set_xlabel('Month', fontsize=11)
    axes[1].set_ylabel('Average Unemployment Rate (%)', fontsize=11)
    axes[1].set_title('Monthly Seasonality Pattern', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def plot_model_comparison_bar(metrics_dict: dict,
                              save_path: str = "outputs/visualizations/model_comparison.png"):
    """
    Create a grouped bar chart comparing all models across all metrics.
    Makes it easy to see which model performs best at a glance.
    """
    models = list(metrics_dict.keys())
    metrics = ['MAE', 'RMSE', 'R²']
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for i, metric in enumerate(metrics):
        values = [metrics_dict[m][metric] for m in models]
        bars = ax.bar(x + i*width, values, width, label=metric, color=colors[i])
        
        # Add value labels on top of each bar
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from src.data_preprocessing import load_data, create_features, prepare_data_for_regression
    from src.supervised_models import SupervisedForecaster
    
    # Load and prepare data
    df = load_data()
    df = create_features(df)
    X_train, X_test, y_train, y_test, scaler, features = prepare_data_for_regression(df)
    
    # Get test dates for plotting
    split_idx = int(len(df) * 0.8)
    test_dates = df['date'].iloc[split_idx:].values
    
    # Train all models and collect their predictions
    predictions = {}
    metrics_all = {}
    
    for model_type in ["linear", "random_forest", "gradient_boosting"]:
        model = SupervisedForecaster(model_type=model_type)
        model.fit(X_train, y_train, scaler, features, df['time_index'].iloc[-1])
        
        y_pred = model.predict(X_test)
        predictions[model_type.replace('_', ' ').title()] = y_pred
        metrics_all[model_type.replace('_', ' ').title()] = calculate_metrics(y_test, y_pred)
        
        model.save()
    
    # Print comparison and generate all charts
    best_model = print_metrics_comparison(metrics_all)
    
    print("\nGenerating visualizations...")
    plot_trend_line(df)
    plot_seasonality(df)
    plot_real_vs_predicted(test_dates, y_test, predictions)
    plot_residuals(y_test, predictions)
    plot_model_comparison_bar(metrics_all)
    
    print("\n✓ Evaluation complete! All visualizations saved to outputs/visualizations/")
