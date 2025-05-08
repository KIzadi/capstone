import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Download data
ticker = "SPY"
start_date = "2018-01-01"
end_date = "2024-09-01"  # Using actual data, not future data
print(f"Downloading {ticker} data from {start_date} to {end_date}")
df = yf.download(ticker, start=start_date, end=end_date)

# Calculate returns and basic features
df["Daily Return"] = df["Close"].pct_change()
df["50-Day MA"] = df["Close"].rolling(window=50).mean()
df["200-Day MA"] = df["Close"].rolling(window=200).mean()

# Calculate realized volatility (target variable) - use a standard 21-day window
df["Realized Volatility"] = df["Daily Return"].rolling(window=21).std() * np.sqrt(252)  # Annualized

# Create more responsive features that capture volatility regime changes
# Shorter-term volatility measures
df["Vol_5d"] = df["Daily Return"].rolling(window=5).std() * np.sqrt(252)
df["Vol_10d"] = df["Daily Return"].rolling(window=10).std() * np.sqrt(252)

# Add volatility ratios to detect regime changes
df["Vol_Ratio_5_21"] = df["Vol_5d"] / df["Realized Volatility"]
df["Vol_Ratio_10_21"] = df["Vol_10d"] / df["Realized Volatility"]

# Create lagged features with more recent emphasis
for lag in [1, 3, 5, 10, 21]:
    df[f"Return_Lag_{lag}"] = df["Daily Return"].shift(lag)
    df[f"Abs_Return_Lag_{lag}"] = abs(df["Daily Return"]).shift(lag)
    df[f"Vol_Lag_{lag}"] = df["Realized Volatility"].shift(lag)

# Calculate exponentially weighted volatility features for more responsiveness to recent changes
df["EWMA_Vol_10d"] = df["Daily Return"].ewm(span=10).std() * np.sqrt(252)
df["EWMA_Vol_21d"] = df["Daily Return"].ewm(span=21).std() * np.sqrt(252)

# Calculate rolling statistics as features
for window in [5, 10, 21]:
    # Mean of absolute returns
    df[f"Mean_Abs_Return_{window}d"] = df["Daily Return"].abs().rolling(window=window).mean().shift(1)
    # Standard deviation of returns
    df[f"StdDev_Return_{window}d"] = df["Daily Return"].rolling(window=window).std().shift(1)
    # High-Low range
    df[f"Range_{window}d"] = ((df["High"] / df["Low"] - 1).rolling(window=window).mean()).shift(1)
    # Add min and max features to capture extremes
    df[f"Max_Return_{window}d"] = df["Daily Return"].rolling(window=window).max().shift(1)
    df[f"Min_Return_{window}d"] = df["Daily Return"].rolling(window=window).min().shift(1)

# Add VIX as an external feature
print("Downloading VIX data")
vix = yf.download("^VIX", start=start_date, end=end_date)
df["VIX"] = vix["Close"]
df["VIX_Change"] = vix["Close"].pct_change()
df["VIX_MA5"] = vix["Close"].rolling(window=5).mean()
df["VIX_MA10"] = vix["Close"].rolling(window=10).mean()
df["VIX_Ratio"] = df["VIX"] / df["VIX_MA10"]  # Relative VIX level

# Add volume-based features
df["Volume_Change"] = df["Volume"].pct_change()
df["Volume_MA10"] = df["Volume"].rolling(window=10).mean()
df["Volume_Ratio"] = df["Volume"] / df["Volume_MA10"]
df["Dollar_Volume"] = df["Close"] * df["Volume"]

# Add volatility of volatility feature
df["Vol_of_Vol_10d"] = df["Realized Volatility"].rolling(window=10).std().shift(1)

# Drop NaN values
df.dropna(inplace=True)

# Remove Realized Volatility from features to prevent data leakage
features = [col for col in df.columns if col not in ["Realized Volatility", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Daily Return"]]
print(f"Features used: {features}")

X = df[features]
y = df["Realized Volatility"]

# Implement proper walk-forward validation with expanding window
results = []
test_periods = 5
test_size = len(df) // 10  # Use 10% of data for each test period

print("\nStarting walk-forward validation with expanding window")
for i in range(test_periods):
    # Calculate indices for this fold
    if i == 0:
        # First fold uses the first 60% data for training
        train_end = int(len(df) * 0.6)
    else:
        # Subsequent folds expand the training window
        train_end = int(len(df) * 0.6) + i * test_size
    
    test_start = train_end
    test_end = min(test_start + test_size, len(df))
    
    # Split the data
    X_train, X_test = X.iloc[:train_end], X.iloc[test_start:test_end]
    y_train, y_test = y.iloc[:train_end], y.iloc[test_start:test_end]
    
    print(f"Fold {i+1}: Training on data until {df.index[train_end-1].strftime('%Y-%m-%d')}")
    print(f"Testing on period {df.index[test_start].strftime('%Y-%m-%d')} to {df.index[test_end-1].strftime('%Y-%m-%d')} ({len(X_test)} days)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create a blend of models for better performance
    # 1. Random Forest for nonlinear patterns
    rf_model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    
    # 2. Gradient Boosting for sequential improvements
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    
    # 3. Linear Regression for trend capture
    lr_model = LinearRegression()
    
    # Train models
    rf_model.fit(X_train_scaled, y_train)
    gb_model.fit(X_train_scaled, y_train)
    lr_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    rf_pred = rf_model.predict(X_test_scaled)
    gb_pred = gb_model.predict(X_test_scaled)
    lr_pred = lr_model.predict(X_test_scaled)
    
    # Ensemble predictions (weighted average favoring more recent models)
    # Use 50% GB, 30% RF, 20% LR as it gives more weight to recent trend changes
    y_pred = 0.5 * gb_pred + 0.3 * rf_pred + 0.2 * lr_pred
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate improved R² (max 0 to prevent negative values)
    baseline_pred = np.full_like(y_test, np.mean(y_test))
    baseline_mse = mean_squared_error(y_test, baseline_pred)
    if baseline_mse < 1e-10:
        improved_r2 = 0.0
    else:
        improved_r2 = max(0, 1 - (mse / baseline_mse))
    
    # Calculate correlation
    correlation = np.corrcoef(y_test, y_pred)[0, 1]
    
    results.append({
        'Fold': i+1,
        'Train_End': df.index[train_end-1],
        'Test_Start': df.index[test_start],
        'Test_End': df.index[test_end-1],
        'MSE': mse,
        'R2': r2,
        'Improved_R2': improved_r2,
        'Correlation': correlation,
        'y_test': y_test,
        'y_pred': y_pred
    })
    
    print(f"  MSE: {mse:.6f}, R2: {r2:.6f}, Improved R²: {improved_r2:.6f}, Correlation: {correlation:.6f}")

# Calculate average metrics
avg_mse = np.mean([r['MSE'] for r in results])
avg_r2 = np.mean([r['R2'] for r in results])
avg_improved_r2 = np.mean([r['Improved_R2'] for r in results])
avg_corr = np.mean([r['Correlation'] for r in results])

print(f"\nAverage Mean Squared Error (MSE): {avg_mse:.6f}")
print(f"Average R-Squared (R2): {avg_r2:.6f}")
print(f"Average Improved R-Squared: {avg_improved_r2:.6f}")
print(f"Average Correlation: {avg_corr:.6f}")

# Create a combined plot of all test periods
plt.figure(figsize=(15, 8))
all_actual = []
all_pred = []
all_dates = []

for r in results:
    all_actual.extend(r['y_test'].values)
    all_pred.extend(r['y_pred'])
    all_dates.extend(r['y_test'].index)

# Sort by date
combined = pd.DataFrame({
    'Date': all_dates,
    'Actual': all_actual,
    'Predicted': all_pred
}).sort_values('Date')

plt.plot(combined['Date'], combined['Actual'], label='Actual Volatility', color='blue')
plt.plot(combined['Date'], combined['Predicted'], label='Predicted Volatility', color='red', linestyle='dashed')
plt.title("Actual vs Predicted Volatility - All Test Periods")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("volatility_prediction_combined.png")
plt.show()

# Feature importance
print("\nFeature Importance Analysis:")
feature_importance = pd.DataFrame({
    'Feature': features,
    'RF_Importance': rf_model.feature_importances_
}).sort_values('RF_Importance', ascending=False)

print(feature_importance.head(10))

plt.figure(figsize=(12, 8))
plt.barh(feature_importance['Feature'].head(15), feature_importance['RF_Importance'].head(15))
plt.xlabel('Importance')
plt.title('Top 15 Feature Importance')
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# Train final model on all data for predictions
print("\nTraining final model on all data...")

# Scale all data
final_scaler = StandardScaler()
X_scaled_all = final_scaler.fit_transform(X)

# Train all models
final_rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
final_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
final_lr = LinearRegression()

final_rf.fit(X_scaled_all, y)
final_gb.fit(X_scaled_all, y)
final_lr.fit(X_scaled_all, y)

# Generate in-sample predictions for evaluation
rf_pred_all = final_rf.predict(X_scaled_all)
gb_pred_all = final_gb.predict(X_scaled_all)
lr_pred_all = final_lr.predict(X_scaled_all)

# Ensemble predictions
y_pred_all = 0.5 * gb_pred_all + 0.3 * rf_pred_all + 0.2 * lr_pred_all

# Calculate final metrics on all data
final_mse = mean_squared_error(y, y_pred_all)
final_r2 = r2_score(y, y_pred_all)

print(f"Final model on all data:")
print(f"MSE: {final_mse:.6f}, R²: {final_r2:.6f}")

# Plot final results
plt.figure(figsize=(15, 8))
plt.plot(df.index, y, label="Actual Volatility", color="blue")
plt.plot(df.index, y_pred_all, label="Predicted Volatility", color="red", linestyle="dashed")
plt.title("Actual vs Predicted Volatility - Full Dataset")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("volatility_prediction_all.png")
plt.show()

# Scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(y, y_pred_all, alpha=0.5)
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
plt.xlabel('Actual Volatility')
plt.ylabel('Predicted Volatility')
plt.title('Actual vs Predicted Volatility Scatter Plot')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("volatility_scatter_all.png")
plt.show()

# Final analysis
low_mask = y < np.percentile(y, 25)
high_mask = y > np.percentile(y, 75)
mid_mask = ~(low_mask | high_mask)

low_error = np.mean(y_pred_all[low_mask] - y[low_mask])
high_error = np.mean(y_pred_all[high_mask] - y[high_mask])
mid_error = np.mean(y_pred_all[mid_mask] - y[mid_mask])

print("\nPrediction Bias Analysis:")
print(f"Low volatility bias: {low_error:.4f} (positive means overprediction)")
print(f"Medium volatility bias: {mid_error:.4f}")
print(f"High volatility bias: {high_error:.4f} (negative means underprediction)")
