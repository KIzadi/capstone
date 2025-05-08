import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Download historical data
ticker = "SPY"
start_date = "2018-01-01"
end_date = "2024-09-01"  # Using actual data, not future data
print(f"Downloading {ticker} data from {start_date} to {end_date}")
df = yf.download(ticker, start=start_date, end=end_date)

# Calculate daily returns
df['daily_return'] = df['Close'].pct_change() * 100  # in percentage
df = df.dropna()

# Calculate realized volatility (for comparison with GARCH forecasts)
df['realized_vol'] = df['daily_return'].rolling(window=21).std() * np.sqrt(252/100)  # Annualized and scaled

# Implement walk-forward validation
results = []
tscv = TimeSeriesSplit(n_splits=5)
window_size = 252  # Approximately 1 year of trading days

print("\nStarting walk-forward validation")
for train_idx, test_idx in tscv:
    # Ensure we have enough data for GARCH estimation
    if len(train_idx) < window_size:
        continue
        
    # Get training and testing sets
    train_returns = df['daily_return'].iloc[train_idx]
    test_returns = df['daily_return'].iloc[test_idx]
    test_volatility = df['realized_vol'].iloc[test_idx]
    
    # Fit GARCH model
    model = arch_model(train_returns, vol='Garch', p=1, q=1, dist='normal')
    model_fit = model.fit(disp='off')
    
    # Make one-step-ahead forecasts for the test period
    forecasts = []
    for i in range(len(test_idx)):
        # Update the model with new data
        if i > 0:
            update_returns = train_returns.append(test_returns.iloc[:i])
            model_update = arch_model(update_returns, vol='Garch', p=1, q=1, dist='normal')
            model_fit = model_update.fit(disp='off', update=True)
        
        # Get forecast for the next day
        forecast = model_fit.forecast(horizon=1)
        conditional_vol = np.sqrt(forecast.variance.iloc[-1, 0]) * np.sqrt(252/100)  # Annualize
        forecasts.append(conditional_vol)
    
    # Convert forecasts to numpy array
    forecasts = np.array(forecasts)
    
    # Calculate metrics
    mse = mean_squared_error(test_volatility, forecasts)
    r2 = r2_score(test_volatility, forecasts)
    
    results.append({
        'MSE': mse,
        'R2': r2,
        'Test Start': df.index[test_idx[0]],
        'Test End': df.index[test_idx[-1]],
        'actual': test_volatility,
        'forecast': forecasts
    })
    
    print(f"Fold: Test period {df.index[test_idx[0]]} to {df.index[test_idx[-1]]}")
    print(f"  MSE: {mse:.6f}, R2: {r2:.6f}")

# Calculate average metrics
avg_mse = np.mean([r['MSE'] for r in results])
avg_r2 = np.mean([r['R2'] for r in results])

print(f"\nAverage Mean Squared Error (MSE): {avg_mse:.6f}")
print(f"Average R-Squared (R2): {avg_r2:.6f}")

# Train final model on all data
final_model = arch_model(df['daily_return'], vol='Garch', p=1, q=1, dist='normal')
final_model_fit = final_model.fit(disp='off')

# Generate in-sample forecasts
forecast = final_model_fit.forecast(horizon=1)
df['garch_vol'] = np.nan
df.loc[df.index[1:], 'garch_vol'] = np.sqrt(forecast.variance.iloc[:-1, 0].values) * np.sqrt(252/100)  # Annualize

# Calculate final metrics
valid_idx = df.dropna().index
final_mse = mean_squared_error(df.loc[valid_idx, 'realized_vol'], df.loc[valid_idx, 'garch_vol'])
final_r2 = r2_score(df.loc[valid_idx, 'realized_vol'], df.loc[valid_idx, 'garch_vol'])

print(f"\nFinal GARCH model on all data:")
print(f"MSE: {final_mse:.6f}, R2: {final_r2:.6f}")
print(f"GARCH Parameters: {final_model_fit.params}")

# Plot results
plt.figure(figsize=(15, 8))
plt.plot(df.index, df['realized_vol'], label='Realized Volatility', color='blue')
plt.plot(df.index, df['garch_vol'], label='GARCH Predicted Volatility', color='red', linestyle='dashed')
plt.title('Actual vs GARCH Predicted Volatility (SPY)')
plt.legend()
plt.tight_layout()
plt.savefig("garch_prediction_results.png")
plt.show()

# Generate a 30-day ahead forecast
forecast_horizon = 30
future_forecast = final_model_fit.forecast(horizon=forecast_horizon)
future_volatility = np.sqrt(future_forecast.variance.iloc[-1].values) * np.sqrt(252/100)  # Annualize

# Create forecast dates (next 30 trading days)
last_date = df.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')

# Plot future forecast
plt.figure(figsize=(15, 8))
plt.plot(df.index[-90:], df['realized_vol'][-90:], label='Historical Realized Volatility', color='blue')
plt.plot(forecast_dates, future_volatility, label='GARCH 30-Day Forecast', color='green', marker='o')
plt.title('GARCH Volatility Forecast - Next 30 Trading Days')
plt.legend()
plt.tight_layout()
plt.savefig("garch_future_forecast.png")
plt.show()

print("\nGARCH 30-Day Volatility Forecast:")
forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Forecasted Volatility': future_volatility
})
print(forecast_df.head(10))
