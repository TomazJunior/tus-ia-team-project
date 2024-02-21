import yfinance as yf
import random
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
import datetime

def setUpTheData():
    df = pd.DataFrame(yf.download('BTC-USD'))
    df = df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
    sample_sentiment = []
    for i in range(len(df['Adj Close'])):
        #initially setting up sample sentiment values until real data available
        sample_sentiment.append(random.choice([.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0]))
    df['Sentiment'] = sample_sentiment
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date') 
    df = df.loc['2022-01-01':'2022-12-31']
    df.to_csv('data/bitcoin_2022_dataset.csv')

#Make the time series stationary
df = pd.read_csv('data/bitcoin_2022_dataset.csv')
original_df = df.copy()

df['Adj Close'] = np.log(df['Adj Close'])
def visualise(df):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    ax.plot(df['Date'], df['Adj Close'])
    plt.show()

#Test to see if the data is stationary, need p_value less than 0.05 here or kpss
def adt(df):
    result = adfuller(df['Adj Close'].dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    return result[0], result [1]

#Similar test for stationary data, need p_value less than 0.05 here or adt
def kpss_testing(df):
    result_kpss = kpss(df['Adj Close'], regression='c', nlags='auto')
    print('KPSS Statistic:', result_kpss[0])
    print('p-value:', result_kpss[1])
    print('Critical Values:', result_kpss[3])
    return result_kpss[0], result_kpss[1]

adt_result, adt_p_value = adt(df)
kpss_result, kpss_p_value = kpss_testing(df)

#Finding the optimal number of differencing
def find_difference_value(p_value, df):
    if p_value > .05:
        difference_value = 0
        while p_value > 0.05:
            print('\n p value too large, trying differencing \n')
            df['Adj Close'] = df['Adj Close'].diff()
            df.dropna(inplace=True)
            difference_value += 1 
            statistic, p_value = adt(df)
        
        print(f'Success. Significant values achieved after {difference_value} differencing')
    return df

# Find order of differencing 
df = find_difference_value(adt_p_value, df)
# run kpss on differenced data 
kpss_testing(df)

#visualise the constant mean and variance
visualise(df)

#Now that it is stationary, we must find the AR and MA values
# Plot ACF
plot_acf(df['Adj Close'].dropna(), lags=20)
plt.title('Autocorrelation Function (ACF)')
plt.show()
# Plot PACF
plot_pacf(df['Adj Close'].dropna(), lags=20)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

def auto_arima(orig_df):
    orig_df = np.log(orig_df['Adj Close'])
    model = pm.auto_arima(orig_df,
                          start_p=10,
                          start_q=10,
                          test='adf',
                          max_p=10, 
                          max_q=10, 
                          m=1,
                          d=None,           
                          seasonal=False,   
                          D=0, 
                          trace=True,
                          error_action='ignore',  
                          suppress_warnings=True,
                         stepwise = True)
    # difference df by d found by auto arima
    differenced_by_auto_arima = orig_df.diff(model.order[1])
    return model.order, differenced_by_auto_arima, model.resid()

auto_p_d_q, differenced_by_auto_arima, fitted_residuals = auto_arima(original_df)

def model(df, p_d_q):    
    time_series = np.log(df)
    model = ARIMA(time_series, order = p_d_q)
    fitted = model.fit()
    fc = fitted.get_forecast(7) 
    fc = (fc.summary_frame(alpha=0.05))
    fc_mean = fc['mean']
    fc_lower = fc['mean_ci_lower']
    fc_upper = fc['mean_ci_upper'] 
    plt.figure(figsize=(12,8), dpi=100)
    plt.plot(original_df['Date'][-50:],original_df['Adj Close'][-50:], label='BTC Price')
    future_7_days =  [str(datetime.datetime(2023, 1, 1, 0, 0, 0) + datetime.timedelta(days=x)) for x in range(7)]
    plt.plot(future_7_days, np.exp(fc_mean), label='mean_forecast', linewidth = 1.5)
    plt.fill_between(future_7_days, np.exp(fc_lower),np.exp(fc_upper), color='b', alpha=.1, label = '95% Confidence')
    plt.title('7 Day Forecast')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()

model(original_df['Adj Close'], (10,1,10))

