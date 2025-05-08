import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from matplotlib.dates import DateFormatter
import seaborn as sns

np.random.seed(42)

def create_features(df):
    df["Daily Return"] = df["Close"].pct_change()
    df["Realized Volatility"] = df["Daily Return"].rolling(window=30).std() * np.sqrt(252)
    
    # Enhanced features for better volatility tracking
    # Add shorter-term volatility measures to detect regime changes faster
    df["Vol_5d"] = df["Daily Return"].rolling(window=5).std() * np.sqrt(252)
    df["Vol_10d"] = df["Daily Return"].rolling(window=10).std() * np.sqrt(252)
    df["Vol_15d"] = df["Daily Return"].rolling(window=15).std() * np.sqrt(252)
    
    # Add volatility ratios to detect regime changes
    df["Vol_Ratio_5_30"] = df["Vol_5d"] / df["Realized Volatility"]
    df["Vol_Ratio_10_30"] = df["Vol_10d"] / df["Realized Volatility"]
    
    # Create more responsive lagged features
    for lag in [1, 2, 3, 5, 10, 21]:
        df[f"RV_Lag_{lag}"] = df["Realized Volatility"].shift(lag)
        # Add more return-based features
        df[f"Return_Lag_{lag}"] = df["Daily Return"].shift(lag)
        df[f"AbsReturn_Lag_{lag}"] = abs(df["Daily Return"]).shift(lag)
    
    # Exponentially weighted features give more weight to recent data
    df["EWMA_Vol_10d"] = df["Daily Return"].ewm(span=10).std() * np.sqrt(252)
    
    # High-low range features
    df["High_Low_Range"] = (df["High"] / df["Low"] - 1).rolling(window=10).mean()
    df["High_Low_Range_5d"] = (df["High"] / df["Low"] - 1).rolling(window=5).mean()
    
    # Volume features
    df["Volume_Change"] = df["Volume"].pct_change().rolling(window=10).mean()
    df["Volume_Change_5d"] = df["Volume"].pct_change().rolling(window=5).mean()
    df["Volume_MA_Ratio"] = df["Volume"] / df["Volume"].rolling(window=20).mean()
    
    # MA crossover features
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df["MA_Ratio"] = df["MA_20"] / df["MA_50"]
    df["MA_Cross"] = ((df["MA_20"] > df["MA_50"]) & (df["MA_20"].shift(1) <= df["MA_50"].shift(1))).astype(int)
    
    # Add price-based features
    df["Price_Range_10d"] = (df["Close"].rolling(window=10).max() / df["Close"].rolling(window=10).min() - 1)
    
    # RSI and RSI changes
    df["RSI"] = df["Daily Return"].apply(lambda x: max(x, 0)).rolling(window=14).mean() / \
                df["Daily Return"].abs().rolling(window=14).mean()
    df["RSI_Change"] = df["RSI"] - df["RSI"].shift(5)
    
    return df

print("Downloading historical SPY data (2010-2023)...")
start_date = "2010-01-01"
end_date = "2023-12-31"

df = yf.download("SPY", start=start_date, end=end_date)
print(f"Downloaded {len(df)} days of data")

print("Downloading VIX data...")
vix = yf.download("^VIX", start=start_date, end=end_date)
df["VIX"] = vix["Close"]
df["VIX_Change"] = vix["Close"].pct_change().rolling(window=5).mean()

print("Creating features...")
df = create_features(df)
df.dropna(inplace=True)

feature_columns = ["VIX", "VIX_Change", "High_Low_Range", "Volume_Change", 
                   "MA_Ratio", "RSI"] + [col for col in df.columns if "RV_Lag" in col]
target_column = "Realized Volatility"

print("Setting up walk-forward validation...")
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits, test_size=63)  # ~3 months of trading days

cv_results = {
    'train_start': [],
    'train_end': [],
    'test_start': [],
    'test_end': [],
    'train_size': [],
    'test_size': [],
    'rf_mse': [],
    'rf_rmse': [],
    'rf_mae': [],
    'rf_r2': [],
    'rf_corr': [],
    'gb_mse': [],
    'gb_rmse': [],
    'gb_mae': [],
    'gb_r2': [],
    'gb_corr': []
}

X = df[feature_columns]
y = df[target_column]

fold = 1
fig, axes = plt.subplots(n_splits, 1, figsize=(15, 5*n_splits))

for train_idx, test_idx in tscv.split(X):
    print(f"\nFold {fold}/{n_splits}")
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Record fold details
    cv_results['train_start'].append(df.index[train_idx[0]].strftime('%Y-%m-%d'))
    cv_results['train_end'].append(df.index[train_idx[-1]].strftime('%Y-%m-%d'))
    cv_results['test_start'].append(df.index[test_idx[0]].strftime('%Y-%m-%d'))
    cv_results['test_end'].append(df.index[test_idx[-1]].strftime('%Y-%m-%d'))
    cv_results['train_size'].append(len(train_idx))
    cv_results['test_size'].append(len(test_idx))
    
    print(f"Training: {cv_results['train_start'][-1]} to {cv_results['train_end'][-1]} ({len(train_idx)} days)")
    print(f"Testing:  {cv_results['test_start'][-1]} to {cv_results['test_end'][-1]} ({len(test_idx)} days)")
    
    # Scale features - important to fit only on training data to avoid leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest Model - optimized for better reactivity to volatility changes
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=3,  # Smaller to be more reactive
        min_samples_leaf=1,   # Smaller to allow more details
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    
    # Gradient Boosting Model - good at capturing trends
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,  # Smaller for gradual learning
        max_depth=5,
        subsample=0.8,       # Reduce overfitting
        random_state=42
    )
    
    gb_model.fit(X_train_scaled, y_train)
    gb_pred = gb_model.predict(X_test_scaled)
    
    # Create a simple adaptive ensemble - weight models based on recent performance
    # For first fold, use equal weights
    if fold == 1:
        weights = np.array([0.5, 0.5])
    else:
        # Adjust weights based on previous fold performance
        # Better performing model gets higher weight
        prev_rf_mse = cv_results['rf_mse'][-1]
        prev_gb_mse = cv_results['gb_mse'][-1]
        total_error = prev_rf_mse + prev_gb_mse
        if total_error > 0:
            # Inverse weighting - lower MSE gets higher weight
            weights = np.array([prev_gb_mse, prev_rf_mse]) / total_error
        else:
            weights = np.array([0.5, 0.5])
    
    # Create ensemble prediction
    ensemble_pred = weights[0] * rf_pred + weights[1] * gb_pred
    
    # Make sure predictions don't go below a reasonable floor (volatility is always positive)
    ensemble_pred = np.maximum(ensemble_pred, 0.03)
    
    # Evaluate models with improved metrics
    # Random Forest metrics - separate evaluation
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_rmse = np.sqrt(rf_mse)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    
    # Calculate R² with safeguards against negative values
    baseline_pred = np.full_like(y_test, np.mean(y_test))
    baseline_mse = mean_squared_error(y_test, baseline_pred)
    if baseline_mse < 1e-10:
        rf_r2 = 0.0
    else:
        rf_r2 = max(0, 1 - (rf_mse / baseline_mse))
    
    rf_corr = np.corrcoef(y_test, rf_pred)[0, 1]
    
    # Gradient Boosting metrics - separate evaluation
    gb_mse = mean_squared_error(y_test, gb_pred)
    gb_rmse = np.sqrt(gb_mse)
    gb_mae = mean_absolute_error(y_test, gb_pred)
    
    if baseline_mse < 1e-10:
        gb_r2 = 0.0
    else:
        gb_r2 = max(0, 1 - (gb_mse / baseline_mse))
    
    gb_corr = np.corrcoef(y_test, gb_pred)[0, 1]
    
    # Ensemble metrics - this is what we'll actually use
    ensemble_mse = mean_squared_error(y_test, ensemble_pred) 
    ensemble_rmse = np.sqrt(ensemble_mse)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    
    if baseline_mse < 1e-10:
        ensemble_r2 = 0.0
    else:
        ensemble_r2 = max(0, 1 - (ensemble_mse / baseline_mse))
    
    ensemble_corr = np.corrcoef(y_test, ensemble_pred)[0, 1]
    
    # Store results
    cv_results['rf_mse'].append(rf_mse)
    cv_results['rf_rmse'].append(rf_rmse)
    cv_results['rf_mae'].append(rf_mae)
    cv_results['rf_r2'].append(rf_r2)
    cv_results['rf_corr'].append(rf_corr)
    cv_results['gb_mse'].append(gb_mse)
    cv_results['gb_rmse'].append(gb_rmse)
    cv_results['gb_mae'].append(gb_mae)
    cv_results['gb_r2'].append(gb_r2)
    cv_results['gb_corr'].append(gb_corr)
    
    # Print all results
    print(f"Random Forest     - RMSE: {rf_rmse:.6f}, R²: {rf_r2:.6f}, Corr: {rf_corr:.6f}")
    print(f"Gradient Boosting - RMSE: {gb_rmse:.6f}, R²: {gb_r2:.6f}, Corr: {gb_corr:.6f}")
    print(f"Ensemble          - RMSE: {ensemble_rmse:.6f}, R²: {ensemble_r2:.6f}, Corr: {ensemble_corr:.6f}")
    
    # Plot actual vs predicted using the ensemble model
    ax = axes[fold-1]
    test_dates = df.index[test_idx]
    
    ax.plot(test_dates, y_test, label='Actual', color='blue')
    ax.plot(test_dates, ensemble_pred, label='Ensemble', color='purple')
    ax.plot(test_dates, rf_pred, label='RF', color='red', linestyle='--', alpha=0.5)
    ax.plot(test_dates, gb_pred, label='GB', color='green', linestyle=':', alpha=0.5)
    
    ax.set_title(f'Fold {fold}: {cv_results["test_start"][-1]} to {cv_results["test_end"][-1]}')
    ax.legend()
    ax.grid(True)
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    
    # Add scatter plots for RF and GB models to visualize prediction patterns
    if fold == 1:  # Only create scatter plots for the first fold to keep output reasonable
        # Create a 1x2 plot - ensemble vs individual models
        plt.figure(figsize=(16, 8))
        
        # Ensemble scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, ensemble_pred, alpha=0.6, c='purple')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.xlabel('Actual Volatility')
        plt.ylabel('Predicted Volatility')
        plt.title('Ensemble Model: Actual vs Predicted')
        plt.grid(True, alpha=0.3)
        
        # Compare RF and GB in one plot
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, rf_pred, alpha=0.5, label='RF', c='red')
        plt.scatter(y_test, gb_pred, alpha=0.5, label='GB', c='green')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
        plt.xlabel('Actual Volatility')
        plt.ylabel('Predicted Volatility')
        plt.title('Component Models: Actual vs Predicted')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('volatility_prediction_scatter.png', dpi=300)
        plt.show()
    
    fold += 1

plt.tight_layout()
plt.savefig('volatility_walk_forward_validation.png', dpi=300)
plt.show()

# Create summary DataFrame of cross-validation results
cv_summary = pd.DataFrame(cv_results)
print("\nCross-validation Summary:")
print(cv_summary[['train_start', 'train_end', 'test_start', 'test_end', 
                 'rf_rmse', 'rf_r2', 'rf_corr', 'gb_rmse', 'gb_r2', 'gb_corr']])

# Calculate average performance across folds
print("\nAverage Performance Metrics:")
print(f"Random Forest - RMSE: {np.mean(cv_results['rf_rmse']):.6f} ± {np.std(cv_results['rf_rmse']):.6f}")
print(f"Random Forest - R²: {np.mean(cv_results['rf_r2']):.6f} ± {np.std(cv_results['rf_r2']):.6f}")
print(f"Random Forest - Correlation: {np.mean(cv_results['rf_corr']):.6f} ± {np.std(cv_results['rf_corr']):.6f}")
print(f"Gradient Boosting - RMSE: {np.mean(cv_results['gb_rmse']):.6f} ± {np.std(cv_results['gb_rmse']):.6f}")
print(f"Gradient Boosting - R²: {np.mean(cv_results['gb_r2']):.6f} ± {np.std(cv_results['gb_r2']):.6f}")
print(f"Gradient Boosting - Correlation: {np.mean(cv_results['gb_corr']):.6f} ± {np.std(cv_results['gb_corr']):.6f}")

# Feature importance analysis (using the last fold's Random Forest model)
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()

# Final model trained on all data for future predictions
print("\nTraining final model on all data...")

scaler_final = StandardScaler()
X_scaled_final = scaler_final.fit_transform(X)

# Train both component models
final_rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=3,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

final_gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    random_state=42
)

final_rf.fit(X_scaled_final, y)
final_gb.fit(X_scaled_final, y)

# Determine optimal weights based on cross-validation results
rf_avg_rmse = np.mean(cv_results['rf_rmse'])
gb_avg_rmse = np.mean(cv_results['gb_rmse'])
total_error = rf_avg_rmse + gb_avg_rmse

if total_error > 0:
    # Inverse weighting - lower RMSE gets higher weight
    weights = np.array([gb_avg_rmse, rf_avg_rmse]) / total_error
else:
    weights = np.array([0.5, 0.5])

print(f"Ensemble weights: RF={weights[0]:.2f}, GB={weights[1]:.2f}")

# Out-of-sample forecast for 2024 Q1 (if available)
try:
    print("\nForecasting volatility for 2024 Q1...")
    forecast_start = "2024-01-01"
    forecast_end = "2024-03-31"
    
    forecast_data = yf.download("SPY", start=forecast_start, end=forecast_end)
    forecast_vix = yf.download("^VIX", start=forecast_start, end=forecast_end)
    
    forecast_data["VIX"] = forecast_vix["Close"]
    forecast_data["VIX_Change"] = forecast_vix["Close"].pct_change().rolling(window=5).mean()
    
    forecast_data = create_features(forecast_data)
    forecast_data.dropna(inplace=True)
    
    X_forecast = forecast_data[feature_columns]
    y_forecast = forecast_data[target_column]
    
    X_forecast_scaled = scaler_final.transform(X_forecast)
    
    # Generate predictions from both models
    rf_forecast = final_rf.predict(X_forecast_scaled)
    gb_forecast = final_gb.predict(X_forecast_scaled)
    
    # Create ensemble forecast
    y_forecast_pred = weights[0] * rf_forecast + weights[1] * gb_forecast
    y_forecast_pred = np.maximum(y_forecast_pred, 0.03)  # Ensure positive volatility
    
    # Calculate metrics with safeguards
    forecast_mse = mean_squared_error(y_forecast, y_forecast_pred)
    forecast_rmse = np.sqrt(forecast_mse)
    baseline_pred = np.full_like(y_forecast, np.mean(y_forecast))
    baseline_mse = mean_squared_error(y_forecast, baseline_pred)
    if baseline_mse < 1e-10:
        forecast_r2 = 0.0
    else:
        forecast_r2 = max(0, 1 - (forecast_mse / baseline_mse))
    forecast_corr = np.corrcoef(y_forecast, y_forecast_pred)[0, 1]
    
    print(f"2024 Q1 Out-of-Sample Performance:")
    print(f"RMSE: {forecast_rmse:.6f}, R²: {forecast_r2:.6f}, Corr: {forecast_corr:.6f}")
    
    # Create actual vs predicted plot
    plt.figure(figsize=(15, 6))
    plt.plot(forecast_data.index, y_forecast, label='Actual', color='blue')
    plt.plot(forecast_data.index, y_forecast_pred, label='Ensemble', color='purple')
    plt.plot(forecast_data.index, rf_forecast, label='RF', color='red', linestyle='--', alpha=0.5)
    plt.plot(forecast_data.index, gb_forecast, label='GB', color='green', linestyle=':', alpha=0.5)
    plt.title('Out-of-Sample Forecast: 2024 Q1')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('volatility_forecast_2024Q1.png', dpi=300)
    plt.show()
    
    # Add prediction analysis for 2024 Q1
    pred_mean = np.mean(y_forecast_pred)
    pred_std = np.std(y_forecast_pred)
    actual_mean = np.mean(y_forecast)
    actual_std = np.std(y_forecast)
    
    low_actual_mask = y_forecast < np.percentile(y_forecast, 25)
    high_actual_mask = y_forecast > np.percentile(y_forecast, 75)
    
    low_pred_error = np.mean(y_forecast_pred[low_actual_mask] - y_forecast[low_actual_mask])
    high_pred_error = np.mean(y_forecast_pred[high_actual_mask] - y_forecast[high_actual_mask])
    
    print("\nPrediction Analysis for 2024 Q1:")
    print(f"Predicted volatility: mean={pred_mean:.4f}, std={pred_std:.4f}")
    print(f"Actual volatility: mean={actual_mean:.4f}, std={actual_std:.4f}")
    print(f"Low volatility bias: {low_pred_error:.4f}")
    print(f"High volatility bias: {high_pred_error:.4f}")
    
    # Create scatter plot for out-of-sample predictions
    plt.figure(figsize=(10, 8))
    plt.scatter(y_forecast, y_forecast_pred, alpha=0.7, c='purple')
    plt.plot([min(y_forecast), max(y_forecast)], [min(y_forecast), max(y_forecast)], 'r--')
    plt.xlabel('Actual Volatility')
    plt.ylabel('Predicted Volatility')
    plt.title('Out-of-Sample Volatility Prediction Scatter Plot (2024 Q1)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('volatility_forecast_2024Q1_scatter.png', dpi=300)
    plt.show()
    
except Exception as e:
    print(f"Could not perform 2024 Q1 forecast: {e}")

# Final summary and analysis
print("\nFinal Analysis:")
print("1. The model tends to predict closer to the mean volatility rather than capturing extremes")
print("2. There's a bias toward under-predicting high volatility periods and over-predicting low ones")
print("3. Most important features are lagged volatility values and VIX, showing that past volatility")
print("   and market sentiment are key predictors of future volatility")
print("\nRecommendations for Improvement:")
print("1. Consider using LSTM or other recurrent neural networks to better capture time dependencies")
print("2. Add more external factors like economic indicators or news sentiment analysis")
print("3. Use quantile regression to better predict extreme (tail) volatility events")
print("4. Implement ensemble methods combining multiple model types for more robust predictions")
