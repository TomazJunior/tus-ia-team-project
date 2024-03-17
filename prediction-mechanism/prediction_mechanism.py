import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import LabelEncoder

initial_df = None

def load_data(file_path):
    global initial_df
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    df.rename(columns={'Close': 'Adj Close'}, inplace=True)
    initial_df = df
    start_date = '2021-04-05 11:45:00'
    end_date = '2021-04-12 18:15:00'
    end_date_inital = '2021-04-12 18:55:00'
    initial_df = df.loc[(df.index >= start_date) & (df.index <= end_date_inital)]
    df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    df.drop(['Open', 'High', 'Low', 'Volume'], axis=1, inplace=True)
    df.to_csv('data/FinalFigures.csv')
    return df

def visualize(df):
    plt.figure(figsize=(12, 8))
    plt.plot(df.index, df['Adj Close'])
    plt.title('Bitcoin Price')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.show()

def test_stationarity(df):
    adf_result = adfuller(df['Adj Close'].dropna())
    kpss_result = kpss(df['Adj Close'], regression='c', nlags='auto')
    return adf_result, kpss_result

def difference_series(df):
    df['Adj Close'] = df['Adj Close'].fillna(method='bfill') 
    diff = df[['Adj Close']].diff().dropna() 
    return diff

def auto_arima_model(orig_df, exog=None):
    orig_df['Adj Close'] = orig_df['Adj Close'].apply(lambda x: max(x, 0.0001))

    log_prices = np.log(orig_df['Adj Close'])
    if exog is not None:
        model = pm.auto_arima(log_prices, exogenous=exog, start_p=1, start_q=1,
                              test='adf',
                              max_p=3, max_q=3, m=1,
                              d=None, seasonal=False,
                              trace=True, error_action='ignore',
                              suppress_warnings=True, stepwise=True)
    else:
        model = pm.auto_arima(log_prices, start_p=1, start_q=1,
                              test='adf',
                              max_p=3, max_q=3, m=1,
                              d=None, seasonal=False,
                              trace=True, error_action='ignore',
                              suppress_warnings=True, stepwise=True)

    order = model.order
    differenced_series = log_prices.diff(order[1]).dropna()
    residuals = model.arima_res_.resid
    return order, differenced_series, residuals

def evaluate_forecast(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

def model(df_index, namePlot, p_d_q, isARIMAX=False, exog_df = None):  
    time_series = np.log(df_index['Adj Close'])
    if not isARIMAX:
        model = ARIMA(time_series, order=p_d_q)
        fitted = model.fit()
        fc = fitted.get_forecast(steps=7, dynamic=True)
    else:
        exog = df_index['encoded_sentiment']
        model = SARIMAX(time_series, exog=exog, order=p_d_q)
        fitted = model.fit()
        fc = fitted.get_forecast(steps=7, exog=exog_df, dynamic=True)
    
    fc_summary = fc.summary_frame(alpha=0.05)
    fc_mean = fc_summary['mean']
    fc_lower = fc_summary['mean_ci_lower']
    fc_upper = fc_summary['mean_ci_upper']
    
    plt.figure(figsize=(12, 10), dpi=200)
    plt.plot(df_index.index[-10:], df_index['Adj Close'][-10:], label='BTC Price', marker='o')
    start_datetime = datetime.datetime(2021, 4, 12, 18, 20, 0)
    fc_value = np.exp(fc_mean)
    mae, mse, rmse = evaluate_forecast(initial_df['Adj Close'][-7:], fc_value)
    print(f'Forecast {namePlot} Evaluation:')
    print('Mean Absolute Error (MAE):', mae)
    print('Mean Squared Error (MSE):', mse)
    print('Root Mean Squared Error (RMSE):', rmse)
    past_10_days = [datetime.datetime(2021, 4, 12, 17, 30, 0) + datetime.timedelta(minutes=5 * x) for x in range(10)]
    future_7_days = [start_datetime + datetime.timedelta(minutes=5 * x) for x in range(7)]
    plt.plot(future_7_days, fc_value, label='Forecast', linewidth=1.5, marker='o', color='#00E4A4')
    plt.fill_between(future_7_days, np.exp(fc_lower), np.exp(fc_upper), color='#ffb61c', alpha=.8, label='95% Confidence')
    # for i, txt in enumerate(np.exp(fc_mean)):
    #     plt.text(future_7_days[i], txt + 140, f'{round(txt, 2)}', ha='left', va='top', rotation=35, fontsize=8)
    plt.title(f'Forecasted Price {namePlot}: 5 Minute intervals April 2021')
    plt.xticks((past_10_days+ future_7_days),[dt.strftime('%H:%M') for dt in (past_10_days+ future_7_days)], rotation=45, ha='right', fontsize=10)
    plt.legend(loc='upper right', fontsize=8)
    plt.savefig(f'visualizations/{namePlot}')
    plt.show()


def main():
    file_path = 'data/btc_usd_5m_bitstamp_18-08-2011_27-04-2021.csv'
    df = load_data(file_path)
    orig_df = df.copy()
    visualize(df)

    adf_result, kpss_result = test_stationarity(df)
    print('ADF Statistic:', adf_result[0])
    print('ADF p-value:', adf_result[1])
    print('KPSS Statistic:', kpss_result[0])
    print('KPSS p-value:', kpss_result[1])

    if adf_result[1] > 0.05:
        print('Data is not stationary. Performing differencing...')
        df = difference_series(df)
        adf_result, kpss_result = test_stationarity(df)
        print('ADF Statistic after differencing:', adf_result[0])
        print('ADF p-value after differencing:', adf_result[1])
        print('KPSS Statistic after differencing:', kpss_result[0])
        print('KPSS p-value after differencing:', kpss_result[1])

    exog = None 
    
    order, differenced_series, residuals = auto_arima_model(df, exog=exog)
    print('ARIMA Order:', order)
    model(orig_df, 'Without Sentiment', order)
    merged_df = pd.read_csv('data/price_with_sentiment_merged.csv')
    label_encoder = LabelEncoder()
    merged_df['encoded_sentiment'] = label_encoder.fit_transform(merged_df['sentiment_of_the_day'])
    order_with_exo, differenced_series, residuals = auto_arima_model(merged_df[:-7], exog='encoded_sentiment')
    merged_df['Date'] = pd.to_datetime(merged_df['Timestamp'], unit='ms')
    merged_df.set_index('Date', inplace=True)
    start_date = '2021-04-05 11:45:00'
    end_date = '2021-04-12 18:15:00'
    merged_df = merged_df.loc[(merged_df.index >= start_date) & (merged_df.index <= end_date)]
    exog_df = merged_df['encoded_sentiment'][-7:]
    model(merged_df, 'With Sentiment', order_with_exo, True, exog_df=exog_df)

if __name__ == "__main__":
    main()
