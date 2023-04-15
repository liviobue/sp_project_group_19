from flask import Flask, render_template
import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Step 1: API key and endpoint
api_key = 'HN5ZKE3677Q1PMJM'
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=TSLA&apikey={api_key}'

# Step 2: Retrieve data from API and convert to DataFrame
response = requests.get(url)
data = json.loads(response.text)
prices = []
for date, values in data['Time Series (Daily)'].items():
    prices.append([date, float(values['4. close'])])
prices.reverse()
df = pd.DataFrame(prices, columns=['date', 'close'])

# Step 3: Calculate daily return
df['return'] = df['close'].pct_change()

# Step 4: Calculate correlation with S&P 500
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
print(f"The correlation between Tesla and the S&P 500 is {corr:.2f}.")

# Step 5: Flask app
@app.route('/')
def index():
# generate plot
    img = BytesIO()
    plt.figure(figsize=(13, 6))  # set the figure size to 12 inches (width) by 6 inches (height)
    plt.plot(df['date'], df['close'])
    plt.title('Tesla Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Closing Price ($)')
    plt.xticks(df['date'][::10]) # set x ticks to show every 10th date
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # render template
    return render_template('index.html', corr=corr, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
