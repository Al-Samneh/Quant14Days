import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

df_wti = yf.download("CL=F", start="2000-01-01", end="2025-01-01", multi_level_index=False)
print("WTI Crude Oil Prices")
df_wti.head()

df_brent = yf.download("BZ=F", start="2000-01-01", end="2025-01-01", multi_level_index=False)
print("Brent Crude Oil Prices")
df_brent.head()

# We want to add log returns to the data

df_wti["Log Returns"] = (np.log(df_wti["Close"] / df_wti["Close"].shift(1)))
df_brent["Log Returns"] = (np.log(df_brent["Close"] / df_brent["Close"].shift(1)))

df_wti = df_wti[1:]
df_brent = df_brent[1:]

print(df_wti['Log Returns'])
print("---------------------------------")
print(df_brent['Log Returns'])

def classify_returns(df):
    df['Class'] = np.where(df['Log Returns'] > 0, 1, 0)
    return df

classify_returns(df_wti)
classify_returns(df_brent)


print(df_wti.head())
print(df_brent.head())

df_brent['momentum_5_days'] = df_wti['Close'].rolling(window=5).mean()
df_wti['momentum_5_days'] = df_wti['Close'].rolling(window=5).mean()

df_brent['momentum_20_days'] = df_wti['Close'].rolling(window=20).mean()
df_wti['momentum_20_days'] = df_wti['Close'].rolling(window=20).mean()

df_brent['momentum_50_days'] = df_wti['Close'].rolling(window=50).mean()
df_wti['momentum_50_days'] = df_wti['Close'].rolling(window=50).mean()

df_brent['rolling_volatility_5_days'] = df_wti['Log Returns'].rolling(window=5).std()
df_wti['rolling_volatility_5_days'] = df_wti['Log Returns'].rolling(window=5).std()

df_brent['rolling_volatility_20_days'] = df_wti['Log Returns'].rolling(window=20).std()
df_wti['rolling_volatility_20_days'] = df_wti['Log Returns'].rolling(window=20).std()

df_brent['rolling_volatility_50_days'] = df_wti['Log Returns'].rolling(window=50).std()
df_wti['rolling_volatility_50_days'] = df_wti['Log Returns'].rolling(window=50).std()

df_brent


