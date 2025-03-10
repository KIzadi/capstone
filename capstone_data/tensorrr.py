import pandas as pd

df = pd.read_csv("spy_stock_data.csv", skiprows=2, parse_dates=["Date"])

df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

df.set_index("Date", inplace=True)

print(df.head())
