import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

ticker = "SPY"
start_date = "2018-01-01"
end_date = "2025-01-01"
df = yf.download(ticker, start=start_date, end=end_date)

df["Daily Return"] = df["Close"].pct_change()

df["50-Day MA"] = df["Close"].rolling(window=50).mean()
df["200-Day MA"] = df["Close"].rolling(window=200).mean()

df["Realized Volatility"] = df["Daily Return"].rolling(window=30).std()

for lag in [1, 5, 10]:
    df[f"RV_Lag_{lag}"] = df["Realized Volatility"].shift(lag)

vix = yf.download("^VIX", start=start_date, end=end_date)
df["VIX"] = vix["Close"]

df.dropna(inplace=True)

features = ["Realized Volatility", "VIX"] + [col for col in df.columns if "RV_Lag" in col]
X = df[features]
y = df["Realized Volatility"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE) - Extended: {mse:.6f}")
print(f"R-Squared (R2) - Extended: {r2:.6f}")

results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred}, index=y_test.index)
print(results.head())

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label="Actual Volatility", color="blue")
plt.plot(y_test.index, y_pred, label="Predicted Volatility", color="red", linestyle="dashed")
plt.title("Actual vs Predicted Volatility (SPY) - Extended to 2025")
plt.legend()
plt.show()
