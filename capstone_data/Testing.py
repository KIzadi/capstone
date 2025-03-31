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
    
    for lag in [1, 5, 10, 21]:
        df[f"RV_Lag_{lag}"] = df["Realized Volatility"].shift(lag)
    
    df["High_Low_Range"] = (df["High"] / df["Low"] - 1).rolling(window=10).mean()
    df["Volume_Change"] = df["Volume"].pct_change().rolling(window=10).mean()
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df["MA_Ratio"] = df["MA_20"] / df["MA_50"]
    
    df["RSI"] = df["Daily Return"].apply(lambda x: max(x, 0)).rolling(window=14).mean() / \
                df["Daily Return"].abs().rolling(window=14).mean()
    
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
    'gb_mse': [],
    'gb_rmse': [],
    'gb_mae': [],
    'gb_r2': []
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
    
    # Random Forest Model
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    
    # Gradient Boosting Model
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    gb_model.fit(X_train_scaled, y_train)
    gb_pred = gb_model.predict(X_test_scaled)
    
    # Evaluate models
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_rmse = np.sqrt(rf_mse)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    
    gb_mse = mean_squared_error(y_test, gb_pred)
    gb_rmse = np.sqrt(gb_mse)
    gb_mae = mean_absolute_error(y_test, gb_pred)
    gb_r2 = r2_score(y_test, gb_pred)
    
    # Store results
    cv_results['rf_mse'].append(rf_mse)
    cv_results['rf_rmse'].append(rf_rmse)
    cv_results['rf_mae'].append(rf_mae)
    cv_results['rf_r2'].append(rf_r2)
    cv_results['gb_mse'].append(gb_mse)
    cv_results['gb_rmse'].append(gb_rmse)
    cv_results['gb_mae'].append(gb_mae)
    cv_results['gb_r2'].append(gb_r2)
    
    print(f"Random Forest - RMSE: {rf_rmse:.6f}, R²: {rf_r2:.6f}")
    print(f"Gradient Boosting - RMSE: {gb_rmse:.6f}, R²: {gb_r2:.6f}")
    
    # Plot actual vs predicted for both models
    ax = axes[fold-1]
    test_dates = df.index[test_idx]
    
    ax.plot(test_dates, y_test, label='Actual', color='blue')
    ax.plot(test_dates, rf_pred, label='RandomForest', color='red', linestyle='--')
    ax.plot(test_dates, gb_pred, label='GradientBoosting', color='green', linestyle=':')
    
    ax.set_title(f'Fold {fold}: {cv_results["test_start"][-1]} to {cv_results["test_end"][-1]}')
    ax.legend()
    ax.grid(True)
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    
    fold += 1

plt.tight_layout()
plt.savefig('volatility_walk_forward_validation.png', dpi=300)
plt.show()

# Create summary DataFrame of cross-validation results
cv_summary = pd.DataFrame(cv_results)
print("\nCross-validation Summary:")
print(cv_summary[['train_start', 'train_end', 'test_start', 'test_end', 
                 'rf_rmse', 'rf_r2', 'gb_rmse', 'gb_r2']])

# Calculate average performance across folds
print("\nAverage Performance Metrics:")
print(f"Random Forest - RMSE: {np.mean(cv_results['rf_rmse']):.6f} ± {np.std(cv_results['rf_rmse']):.6f}")
print(f"Random Forest - R²: {np.mean(cv_results['rf_r2']):.6f} ± {np.std(cv_results['rf_r2']):.6f}")
print(f"Gradient Boosting - RMSE: {np.mean(cv_results['gb_rmse']):.6f} ± {np.std(cv_results['gb_rmse']):.6f}")
print(f"Gradient Boosting - R²: {np.mean(cv_results['gb_r2']):.6f} ± {np.std(cv_results['gb_r2']):.6f}")

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
print("\nTraining final model on all historical data...")

scaler_final = StandardScaler()
X_scaled_final = scaler_final.fit_transform(X)

final_rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

final_rf_model.fit(X_scaled_final, y)

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
    y_forecast_pred = final_rf_model.predict(X_forecast_scaled)
    
    forecast_mse = mean_squared_error(y_forecast, y_forecast_pred)
    forecast_rmse = np.sqrt(forecast_mse)
    forecast_r2 = r2_score(y_forecast, y_forecast_pred)
    
    print(f"2024 Q1 Out-of-Sample Performance - RMSE: {forecast_rmse:.6f}, R²: {forecast_r2:.6f}")
    
    plt.figure(figsize=(15, 6))
    plt.plot(forecast_data.index, y_forecast, label='Actual', color='blue')
    plt.plot(forecast_data.index, y_forecast_pred, label='RandomForest', color='red', linestyle='--')
    plt.title('Out-of-Sample Forecast: 2024 Q1')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('volatility_forecast_2024Q1.png', dpi=300)
    plt.show()
    
except Exception as e:
    print(f"Could not perform 2024 Q1 forecast: {e}")

print("\nAnalysis complete.")
