import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

unseen_start_date = "2024-09-01"
unseen_end_date = "2025-01-01"
train_start_date = "2018-01-01"
train_end_date = "2024-08-31"
df_train = yf.download("SPY", start=train_start_date, end=train_end_date)

df_train["Daily Return"] = df_train["Close"].pct_change()

df_train["Realized Volatility"] = df_train["Daily Return"].rolling(window=30).std()

for lag in [1, 5, 10]:
    df_train[f"RV_Lag_{lag}"] = df_train["Realized Volatility"].shift(lag)

vix_train = yf.download("^VIX", start=train_start_date, end=train_end_date)
df_train["VIX"] = vix_train["Close"]

df_train.dropna(inplace=True)

features_train = ["Realized Volatility", "VIX"] + [col for col in df_train.columns if "RV_Lag" in col]
X_train = df_train[features_train]
y_train = df_train["Realized Volatility"]

model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
df_unseen = yf.download("SPY", start=unseen_start_date, end=unseen_end_date)

# Compute daily returns
df_unseen["Daily Return"] = df_unseen["Close"].pct_change()
df_unseen["Realized Volatility"] = df_unseen["Daily Return"].rolling(window=30).std()

for lag in [1, 5, 10]:
    df_unseen[f"RV_Lag_{lag}"] = df_unseen["Realized Volatility"].shift(lag)

vix_unseen = yf.download("^VIX", start=unseen_start_date, end=unseen_end_date)
df_unseen["VIX"] = vix_unseen["Close"]

df_unseen.dropna(inplace=True)
features_unseen = ["Realized Volatility", "VIX"] + [col for col in df_unseen.columns if "RV_Lag" in col]
X_unseen = df_unseen[features_unseen]
y_unseen_pred = model.predict(X_unseen)

mse_unseen = mean_squared_error(df_unseen["Realized Volatility"], y_unseen_pred)
r2_unseen = r2_score(df_unseen["Realized Volatility"], y_unseen_pred)

print(f"Mean Squared Error (MSE) - Unseen Data: {mse_unseen:.6f}")
print(f"R-Squared (R2) - Unseen Data: {r2_unseen:.6f}")

results_unseen = pd.DataFrame({"Actual": df_unseen["Realized Volatility"], "Predicted": y_unseen_pred}, index=df_unseen.index)
print(results_unseen.head())
plt.figure(figsize=(12, 6))
plt.plot(df_unseen.index, df_unseen["Realized Volatility"], label="Actual Volatility", color="blue")
plt.plot(df_unseen.index, y_unseen_pred, label="Predicted Volatility", color="red", linestyle="dashed")
plt.title("Actual vs Predicted Volatility (SPY) with unseen data")
plt.legend()
plt.show()
