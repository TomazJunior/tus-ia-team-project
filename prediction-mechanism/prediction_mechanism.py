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
from sklearn.preprocessing import LabelEncoder
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error

def setUpTheData():
    df = pd.DataFrame(yf.download('BTC-USD', start='2021-01-11', end='2022-01-11'))
    df = df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
    print(df.info())
    # df['Date'] = pd.to_datetime(df['date'])
    # df = df.set_index('Date') 
    # df = df.loc['2021-01-01':'2021-12-31']
    df.to_csv('data/bitcoin_2021_dataset.csv')

setUpTheData()
#Make the time series stationary
df = pd.read_csv('data/bitcoin_2021_dataset.csv')
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

def evaluate_model(forecast, name):
    actual_figures_for_time_period = [42737.08, 44009.50,42596.13,43098.80,43288.90, 43203.21,42252.79]
    mae = mean_absolute_error(actual_figures_for_time_period, forecast)
    mse = mean_squared_error(actual_figures_for_time_period, forecast)
    rmse = np.sqrt(mse)
    
    print(f'{name} Model Evaluation:')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')

def model(df, namePlot, p_d_q,isARIMAX=False):    
    time_series = np.log(df['Adj Close'])
    
    if not isARIMAX:
        model = ARIMA(time_series, order=p_d_q)
        fitted = model.fit()
        fc = fitted.get_forecast(steps=7)
    else:
        exog = df['encoded_sentiment']
        model = SARIMAX(time_series, exog=exog, order=p_d_q)
        fitted = model.fit()

        #Encoding the exogenous variable
        exog_df_2022 = pd.read_csv('data/sentiment_summary_2022.csv')
        exog_df_2022['Date'] = pd.to_datetime(exog_df_2022['date'], format = '%Y-%m-%d')
        label_encoder = LabelEncoder()
        exog_df_2022['encoded_sentiment'] = label_encoder.fit_transform(exog_df_2022['sentiment_of_the_day'])
        exog_forecast_data = exog_df_2022[:7]
        fc = fitted.get_forecast(steps=7, exog=exog_forecast_data['encoded_sentiment'])
    
    fc_summary = fc.summary_frame(alpha=0.05)
    fc_mean = fc_summary['mean']
    fc_lower = fc_summary['mean_ci_lower']
    fc_upper = fc_summary['mean_ci_upper']
    
    plt.figure(figsize=(24, 24), dpi=100)
    plt.plot(df['Date'][-10:], df['Adj Close'][-10:], label='BTC Price', marker='o')
    future_7_days = [str(datetime.datetime(2022, 1, 11) + datetime.timedelta(days=x)).split()[0] for x in range(7)]
    plt.plot(future_7_days, np.exp(fc_mean), label='mean_forecast', linewidth=1.5, marker='o')
    plt.fill_between(future_7_days, np.exp(fc_lower), np.exp(fc_upper), color='g', alpha=.1, label='95% Confidence')
    for i, txt in enumerate(np.exp(fc_mean)):
        plt.text(future_7_days[i], txt, f'{round(txt, 2)}', ha='right', va='bottom')
    plt.title('Forecasted Price: 11 January 2022 - 17 January 2022')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.legend(loc='upper left', fontsize=8)
    plt.savefig(f'visualizations/{namePlot}')
    plt.show()
    print(np.exp(fc_mean))
    evaluate_model(np.exp(fc_mean), namePlot)

model(original_df, 'forecast_without_sentiment', (10,1,10))

#Merge the sentiment data to the bitcoin price data
df_sentiment_2021 = pd.read_csv('data/sentiment_summary_2021.csv')
original_df['Date'] = pd.to_datetime(original_df['Date'], format = '%Y-%m-%d')
df_sentiment_2021['Date'] = pd.to_datetime(df_sentiment_2021['date'], format = '%Y-%m-%d')

merged_df = pd.merge(original_df, df_sentiment_2021, on='Date', how='left')
merged_df['Negative'] = merged_df['Negative'].fillna(method='ffill')
merged_df['Nuetral'] = merged_df['Nuetral'].fillna(method='ffill')
merged_df['Positive'] = merged_df['Positive'].fillna(method='ffill')
merged_df['sentiment_of_the_day'] = merged_df['sentiment_of_the_day'].fillna(method='ffill')


merged_df.to_csv('data/price_with_sentiment_merged.csv', index=False)

#Encoding the exogenous variable
label_encoder = LabelEncoder()
merged_df['Date']  = merged_df['Date'].astype('str')
merged_df['encoded_sentiment'] = label_encoder.fit_transform(merged_df['sentiment_of_the_day'])

fitted_model = model(merged_df,'forecast_with_sentiment', (10,1,10), True)