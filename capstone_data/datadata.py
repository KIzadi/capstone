import yfinance as yf
import pandas as pd

ticker = "SPY"
start_date = "2000-01-01" 
end_date = "2024-12-31" 

df = yf.download(ticker, start=start_date, end=end_date)

df.to_csv("spy_stock_data1.csv")

print("SPY data downloaded successfully!")
print(df.head())
