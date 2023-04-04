import requests
import json
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Collect real-world data using a Web API
api_key = 'HN5ZKE3677Q1PMJM'
symbol = 'CS'
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={api_key}'
response = requests.get(url)
data = json.loads(response.text)

# Step 2: Data preparation
prices = []
for date, values in data['Time Series (Daily)'].items():
    prices.append([date, float(values['4. close'])])
prices.reverse()  # Reverse the order so that the oldest dates come first

# Step 3: Use of Python built-in data structures and pandas data frames
df = pd.DataFrame(prices, columns=['date', 'close'])

# Step 4: Use of conditional statements, loop control statements, and loops
for i, row in df.iterrows():
    # Example of a conditional statement
    if row['close'] < 10:
        print(
            f"Credit Suisse stock was trading below CHF 10 on {row['date']}.")

# Step 5: Use of procedural programming


def plot_data(df):
    plt.plot(df['date'], df['close'])
    plt.title('Credit Suisse Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (CHF)')
    plt.show()


# Step 6: Use of tables and visualization for data exploration. Plot the closing price of Credit Suisse over time
plot_data(df)

# Step 7: Integration of statistical analysis. Calculate the daily returns of Credit Suisse
df['return'] = df['close'].pct_change()

# Calculate the correlation between Credit Suisse and the S&P 500 index
sp500_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=SPY&apikey={api_key}'
sp500_response = requests.get(sp500_url)
sp500_data = json.loads(sp500_response.text)
sp500_prices = []
for date, values in sp500_data['Time Series (Daily)'].items():
    sp500_prices.append([date, float(values['4. close'])])
sp500_prices.reverse()
sp500_df = pd.DataFrame(sp500_prices, columns=['date', 'sp500_close'])
merged_df = pd.merge(df, sp500_df, on='date')
corr = merged_df['return'].corr(merged_df['sp500_close'])
print(f"The correlation between Credit Suisse and the S&P 500 is {corr:.2f}.")
