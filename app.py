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
import folium
from datetime import date
import pdfkit
import yfinance as yf
from scipy.stats import normaltest
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import mpld3
from mpld3 import plugins
import urllib.parse
import matplotlib.dates as mdates

app = Flask(__name__)

# Step 1: API key and endpoint
api_key = 'HN5ZKE3677Q1PMJM'

# Define connection
db_user = 'root'
db_password = 'test123'
db_host = 'db'
db_port = '3306'
db_name = 'tesla_stock_price'
db_connection = f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'


def get_stock_data(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={api_key}'
    # Step 2: Retrieve data from API and convert to DataFrame
    response = requests.get(url)
    data = json.loads(response.text)
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


def get_stock_data_for_page(start_date, end_date, symbol):
    table_name = f"{symbol}_daily_prices"
    engine = create_engine(db_connection, echo=True)
    connection = engine.connect()
    # Execute the query
    query = f"SELECT * FROM {table_name} WHERE date BETWEEN '{start_date}' AND '{end_date}'"
    df = pd.read_sql_query(query, engine)
    df.drop_duplicates(subset=['date'], inplace=True)
    # Step 3: Calculate daily return
    df['return'] = df['close'].sort_index(ascending=False).pct_change().apply(
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
    print(merged_df)
    return corr


def get_stock_name(symbol):
    url = f"https://finance.yahoo.com/quote/{symbol}"
    response = requests.get(url)
    start = response.text.find('<title>') + len('<title>')
    end = response.text.find('(', start)
    return response.text[start:end].strip()


def createMap(symbol):
    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}'
    # Step 2: Retrieve data from API and get address
    response = requests.get(url)
    data = response.json()
    address = data['Address']
    encoded_address = urllib.parse.quote(address)
    time.sleep(3)
    url = "https://nominatim.openstreetmap.org/search?q={}&format=json".format(
        encoded_address)
    response = requests.get(url).json()
    if response:
        lat = response[0]["lat"]
        lng = response[0]["lon"]
        # Step 4: Plot location on map
        m = folium.Map(location=[lat, lng], zoom_start=10)
        folium.Marker([lat, lng]).add_to(m)
        return m._repr_html_()
    else:
        return "No infomation found"

def stock_chart(df, start_date, end_date, stock_name):
    img = BytesIO()
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(df['date'], df['close'], color='blue', label='Closing Price')
    ax.set_title(f"{stock_name} Stock Price between {start_date} and {end_date}")
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price ($)')
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # automatically adjust x-axis ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # format x-axis tick labels
    plt.xticks(rotation=45)  # rotate x-axis tick labels for better visibility
    plt.tight_layout()  # adjust subplot layout for better spacing
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url


def export_pdf(corr, plots, df, pd, stock_name, map_html, autocorr_test_plot,
                              granger_causality_test_result, stationarity_test_result, normality_test_result):
    # Get the rendered HTML
    rendered_html = render_template('stock_info.html', plots=plots, autocorr_test_plot=autocorr_test_plot, granger_causality_test_result=granger_causality_test_result,
                                    stationarity_test_result=stationarity_test_result, corr=corr, table=df, pd=pd, stock_name=stock_name, map_html=map_html, normality_test_result=normality_test_result)

    # Create PDF from rendered HTML
    pdf = pdfkit.from_string(rendered_html, False)

    # Return the response
    return pdf


def normality_test(df):
    stat, p = normaltest(df['close'])
    print(f'Statistics={stat:.3f}, p={p:.3f}')
    alpha = 0.05
    if p > alpha:
        return ('Stock Data looks Gaussian (fail to reject H0)')
    else:
        return ('Stock Data does not look Gaussian (reject H0)')


def stationarity_test(df):
    result = adfuller(df['close'])
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    if result[1] > 0.05:
        return ('Fail to reject the null hypothesis (H0), data has a unit root and is non-stationary')
    else:
        return ('Reject the null hypothesis (H0), data does not have a unit root and is stationary')


def granger_causality_test(df):
    maxlag = 7
    results = []
    try:
        for lag in range(1, maxlag+1):
            granger_test = grangercausalitytests(
                df[['close', 'volume']], maxlag=lag, verbose=False)
            p_values = [round(granger_test[i+1][0]['ssr_ftest'][1], 4)
                        for i in range(lag)]
            results.append({'max_lag': lag, 'p_values': p_values})
    except Exception as e:
        return f"An error occurred: {e}"
    return results


def autocorr_test(df):
    img = BytesIO()
    lag = 20
    try:
        fig, ax = plt.subplots(figsize=(13, 6))
        plot_acf(df['close'], lags=lag, ax=ax)
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        return plot_url
    except ValueError:
        return "Error: dataframe too short for desired lag"


def moving_average_chart(df):
    img = BytesIO()
    column = 'close'
    window_size = 20
    rolling_mean = df[column].rolling(window=window_size).mean()
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(df.index, df[column], label='Actual Price')
    ax.plot(rolling_mean.index, rolling_mean,
            label=f'{window_size}-day Moving Average')
    ax.legend(loc='upper left')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f'{column} Moving Average Chart')
    plt.xticks(rotation=45)
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url


def heatmap(df):
    # remove percentage sign from column 'change'
    df['return'] = df['return'].str.replace('%', '').astype(float)

    img = BytesIO()
    corr_method = 'pearson'
    cmap = 'coolwarm'
    corr_matrix = df.drop(columns=['date']).corr(method=corr_method)
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.heatmap(corr_matrix, cmap=cmap, annot=True, annot_kws={'size': 12})
    ax.set_title('Correlation Heatmap')
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url


def create_subplot(stock_plot, autocorr_test_plot, moving_average_chart_plot, heatmap_plot):
    # create a figure with four subplots
    fig, axs = plt.subplots(2, 2, figsize=(11, 6))
    axs[0, 0].imshow(plt.imread(BytesIO(base64.b64decode(stock_plot))))
    axs[0, 1].imshow(plt.imread(BytesIO(base64.b64decode(autocorr_test_plot))))
    axs[1, 0].imshow(plt.imread(
        BytesIO(base64.b64decode(moving_average_chart_plot))))
    axs[1, 1].imshow(plt.imread(BytesIO(base64.b64decode(heatmap_plot))))
    # remove the x and y axis labels and ticks from each subplot
    for ax in axs.flat:
        ax.set(xticks=[], yticks=[], xlabel='', ylabel='')
    # add titles to each subplot
    axs[0, 0].set_title('Stock Chart')
    axs[0, 1].set_title('Autocorrelation Test Plot')
    axs[1, 0].set_title('Moving Average Chart')
    axs[1, 1].set_title('Correlation Heatmap')
    # adjust the space between subplots and add padding
    # create an interactive plot with zoom and pan functionality
    plugins.connect(fig, plugins.MousePosition(fontsize=14))
    # encode the plot as a base64 string and return it
    return mpld3.fig_to_html(fig)

# Step 5: Flask app


@app.route('/')
def index():
    today = date.today()
    return render_template('index.html', date=date)


@app.route('/stock_info', methods=['POST'])
def stock_info():
    symbol = request.form['symbol'].upper()
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    try:
        stock_name = get_stock_name(symbol)
        get_stock_data(symbol)
        df = get_stock_data_for_page(start_date, end_date, symbol)
        if df.empty:
            resp = make_response(render_template(
                'error.html', symbol=symbol, error="No stock information for this time period"))
            resp.cache_control.no_cache = True
            return resp
        corr = calc_corr_sp500(df)
        map_html = createMap(symbol)
        normality_test_result = normality_test(df)
        stationarity_test_result = stationarity_test(df)
        granger_causality_test_result = granger_causality_test(df)
        granger_causality_test_result = json.dumps(granger_causality_test_result, indent=4)

        stock_plot = stock_chart(df, start_date, end_date, stock_name)
        autocorr_test_plot = autocorr_test(df)
        moving_average_chart_plot = moving_average_chart(df)
        heatmap_plot = heatmap(df)

        plots = create_subplot(
            stock_plot, autocorr_test_plot, moving_average_chart_plot, heatmap_plot)

        # Create PDF
        pdf_data = export_pdf(corr, plots, df, pd, stock_name, map_html, autocorr_test_plot,
                              granger_causality_test_result, stationarity_test_result, normality_test_result)
        pdf_data_base64 = base64.b64encode(pdf_data).decode('utf-8')
        # render template
        return render_template('stock_info.html', plots=plots, autocorr_test_plot=autocorr_test_plot, granger_causality_test_result=granger_causality_test_result, stationarity_test_result=stationarity_test_result, corr=corr, table=df, pd=pd, stock_name=stock_name, map_html=map_html, normality_test_result=normality_test_result, pdf_data=pdf_data_base64)
    except Exception as e:
        resp = make_response(render_template(
            'error.html', symbol=symbol, error=str(e)))
        resp.cache_control.no_cache = True
        return resp


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
