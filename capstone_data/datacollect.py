import yfinance as yf
import pandas as pd

spy = yf.download("SPY", start="2010-01-01", end="2025-03-10")
spy.to_csv("spy_stock_data.csv")