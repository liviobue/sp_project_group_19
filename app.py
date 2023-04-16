from flask import Flask, render_template, request, make_response
import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
import time
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import requests

app = Flask(__name__)

# Step 1: API key and endpoint
api_key = 'HN5ZKE3677Q1PMJM'

def get_stock_data(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={api_key}'

    # Step 2: Retrieve data from API and convert to DataFrame
    response = requests.get(url)
    data = json.loads(response.text)

    # Define connection
    db_user = 'root'
    db_password = 'test123'
    db_host = 'db'
    db_port = '3306'
    db_name = 'tesla_stock_price'
    db_connection = f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

    # Connect to the database
    engine = create_engine(db_connection, echo=True)
    connection = engine.connect()

    # Create table if it doesn't exist
    table_name = f"{symbol}_daily_prices"
    create_table_query = text(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            date DATE,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            volume INT,
            dividend FLOAT,
            split_coeff FLOAT
        )
    """)
    connection.execute(create_table_query)

    # Insert data into MySQL database
    for date, values in data['Time Series (Daily)'].items():
        insert_query = text(f"""
            INSERT INTO {table_name} (date, open, high, low, close, volume)
            VALUES (:date, :open, :high, :low, :close, :volume)
        """)
        connection.execute(insert_query, {'date': date, 'open': values['1. open'], 'high': values['2. high'],
                                          'low': values['3. low'], 'close': values['4. close'], 'volume': values['6. volume']})
    # Commit changes
    connection.commit()

    print("Successfully inserted data")

    # Execute the query
    query = f'SELECT * FROM {table_name}'
    df = pd.read_sql_query(query, engine)

    # Step 3: Calculate daily return
    df['return'] = df['close'].pct_change().apply(
        lambda x: "{:.3f} %".format(x*100))
    return (df)


def calc_corr_sp500(df):
    # Step 4: Calculate correlation with S&P 500
    sp500_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=SPY&apikey={api_key}'
    sp500_response = requests.get(sp500_url)
    sp500_data = json.loads(sp500_response.text)
    time.sleep(3)
    sp500_prices = []
    for date, values in sp500_data['Time Series (Daily)'].items():
        sp500_prices.append([date, float(values['4. close'])])
    sp500_prices.reverse()
    sp500_df = pd.DataFrame(sp500_prices, columns=['date', 'sp500_close'])
    merged_df = pd.merge(df, sp500_df, on='date')
    corr = merged_df['return'].corr(merged_df['sp500_close'])
    print(f"The correlation between Tesla and the S&P 500 is {corr:.2f}.")
    return corr

def get_stock_name(symbol):
    url = f"https://finance.yahoo.com/quote/{symbol}"
    response = requests.get(url)
    start = response.text.find('<title>') + len('<title>')
    end = response.text.find('(', start)
    return response.text[start:end].strip()

# Step 5: Flask app


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/stock_info', methods=['POST'])
def stock_info():
    symbol = request.form['symbol'].upper()
    try:
        df = get_stock_data(symbol)
        corr = calc_corr_sp500(df)
        stock_name = get_stock_name(symbol)

        # generate plot
        img = BytesIO()
        # set the figure size to 12 inches (width) by 6 inches (height)
        plt.figure(figsize=(13, 6))
        plt.plot(df['date'], df['close'])
        plt.title(f"{stock_name} Stock Price")
        plt.xlabel('Date')
        plt.ylabel('Closing Price ($)')
        plt.xticks(df['date'][::10])  # set x ticks to show every 10th date
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        # render template
        return render_template('stock_info.html', corr=corr, plot_url=plot_url, table=df, pd=pd, stock_name=stock_name)
    except Exception as e:
        resp = make_response(render_template(
            'error.html', symbol=symbol, error=str(e)))
        # set no_cache to True to prevent caching of the error page
        resp.cache_control.no_cache = True
        return resp

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
