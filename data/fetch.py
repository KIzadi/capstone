import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

tickers = ["AAPL", "TSLA", "NVDA"]
start_date = "2018-01-01"
end_date = "2025-01-01"

stock_data = {ticker: yf.download(ticker, start=start_date, end=end_date) for ticker in tickers}

for ticker, df in stock_data.items():
    df.to_csv(f"{ticker}_stock_data.csv")

for ticker, df in stock_data.items():
    print(f"\n{ticker} Data Preview:")
    print(df.head())

for ticker, df in stock_data.items():
    df["Daily Return"] = df["Close"].pct_change()
    df["50-Day MA"] = df["Close"].rolling(window=50).mean()
    df["200-Day MA"] = df["Close"].rolling(window=200).mean()
    df["Rolling Volatility"] = df["Daily Return"].rolling(window=30).std()
    df.fillna(method="ffill", inplace=True)

plt.figure(figsize=(12, 6))
for ticker in tickers:
    plt.plot(stock_data[ticker]["Close"], label=f"{ticker} Close Price")
plt.title("Stock Closing Prices (2018-2025)")
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
for ticker in tickers:
    plt.plot(stock_data[ticker]["Daily Return"], label=f"{ticker} Daily Return", alpha=0.7)
plt.title("Stock Daily Returns (2018-2025)")
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
for ticker in tickers:
    plt.plot(stock_data[ticker]["Rolling Volatility"], label=f"{ticker} 30-Day Rolling Volatility", alpha=0.7)
plt.title("Stock Rolling Volatility (30-Day)")
plt.legend()
plt.show()

vix = yf.download("^VIX", start=start_date, end=end_date)
plt.figure(figsize=(12, 6))
for ticker in tickers:
    plt.plot(stock_data[ticker]["Close"], label=f"{ticker} Close Price")
plt.plot(vix["Close"], label="VIX", linestyle="dashed", alpha=0.7)
plt.title("Stock Prices vs VIX (2018-2025)")
plt.legend()
plt.show()
