import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from arch import arch_model
from sklearn.metrics import mean_squared_error, r2_score

ticker = "SPY"
train_start_date = "2018-01-01"
train_end_date = "2024-08-31"
unseen_start_date = "2024-09-01"
unseen_end_date = "2025-01-01"

df_train = yf.download(ticker, start=train_start_date, end=train_end_date)

df_train["Daily Return"] = df_train["Close"].pct_change() * 100  
df_train.dropna(inplace=True)

returns = df_train["Daily Return"]

garch_model = arch_model(returns, vol="Garch", p=1, q=1, rescale=False) 
garch_fit = garch_model.fit(disp="off")

df_unseen = yf.download(ticker, start=unseen_start_date, end=unseen_end_date)

df_unseen["Daily Return"] = df_unseen["Close"].pct_change() * 100 
df_unseen["Realized Volatility"] = df_unseen["Daily Return"].rolling(window=30).std()

df_unseen.dropna(inplace=True)

garch_forecast = garch_fit.forecast(start=returns.index[-1], horizon=len(df_unseen), reindex=True)

print(f"Length of GARCH forecast variance: {len(garch_forecast.variance.dropna().values)}")

if len(garch_forecast.variance.dropna().values) > 0:
    y_garch_pred = np.sqrt(garch_forecast.variance.dropna().values)
else:
    print("Error: GARCH model did not generate predictions. Adjust forecasting parameters.")
    y_garch_pred = np.array([]) 

print(f"GARCH Predictions Shape: {len(y_garch_pred)}")
print(f"Unseen Data Shape: {len(df_unseen)}")

if len(y_garch_pred) > 0:
    df_unseen_garch = df_unseen.iloc[:len(y_garch_pred)].copy()
    df_unseen_garch["GARCH Predicted Volatility"] = y_garch_pred

    df_unseen_garch = df_unseen_garch.dropna(subset=["Realized Volatility", "GARCH Predicted Volatility"])
    if len(df_unseen_garch) > 0:
        mse_garch = mean_squared_error(df_unseen_garch["Realized Volatility"], df_unseen_garch["GARCH Predicted Volatility"])
        r2_garch = r2_score(df_unseen_garch["Realized Volatility"], df_unseen_garch["GARCH Predicted Volatility"])
        print(f"Mean Squared Error (MSE) - GARCH Model: {mse_garch:.6f}")
        print(f"R-Squared (R2) - GARCH Model: {r2_garch:.6f}")

        results_garch = pd.DataFrame({"Actual": df_unseen_garch["Realized Volatility"], 
                                      "GARCH Predicted": df_unseen_garch["GARCH Predicted Volatility"]}, 
                                      index=df_unseen_garch.index)
        print(results_garch.head())

        plt.figure(figsize=(12, 6))
        plt.plot(df_unseen.index, df_unseen["Realized Volatility"], label="Actual Volatility", color="blue")
        plt.plot(df_unseen_garch.index, df_unseen_garch["GARCH Predicted Volatility"], label="GARCH Predicted Volatility", color="green", linestyle="dotted")
        plt.title("GARCH Model: Actual vs Predicted Volatility (SPY)")
        plt.legend()
        plt.show()
    else:
        print("Skipping MSE calculation and plotting due to empty GARCH predictions.")
else:
    print("Skipping entire evaluation as GARCH model did not generate predictions.")
