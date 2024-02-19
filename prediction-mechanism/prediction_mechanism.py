import yfinance as yf
import random
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def setUpTheData():
    df = pd.DataFrame(yf.download('BTC-USD'))
    df = df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
    sample_sentiment = []
    for i in range(len(df['Adj Close'])):
        #initially setting up sample sentiment values until real data available
        sample_sentiment.append(random.choice([.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0]))
    df['Sentiment'] = sample_sentiment
    df.to_csv('data/pre_prediction_data.csv')

df = pd.read_csv('data/pre_prediction_data.csv')
df = df[2300:]

#differencing to achieve a stationary time series
df['Adj Close_diff'] = df['Adj Close'].diff()
result = adfuller(df['Adj Close_diff'].dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])

#Stationarity is achieved here as the p value has decreased to 7.819338330590412e-17 which is lower than the significance 0.05 value
#Stationarity is important because many time series models assume that the statistical properties of the series remain constant over 
# time. By differencing, you can remove trends or seasonality in the data, making it more amenable to modeling.

#The below two plots will allow us to see at which lag period there is a positive or negative correlation with the current
#observation and an observation at a particular lag.

# Plot ACF
plot_acf(df['Adj Close_diff'].dropna(), lags=40)
plt.title('Autocorrelation Function (ACF)')
plt.show()
# Plot PACF
plot_pacf(df['Adj Close_diff'].dropna(), lags=40)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()


train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]
# Example: SARIMAX model with order=(p, d, q)
model = SARIMAX(train['Adj Close_diff'], order=(9, 1, 9),seasonal_order=(0,0,0,0), exog=train['Sentiment'], trend='c')
result = model.fit()

# Check the summary and evaluate the model as needed
print(result.summary())
#TRAINING
#train_size = int(len(df) * 0.8)
#train, test = df[:train_size], df[train_size:]

#arimax_model = SARIMAX(train['Adj Close'], order=(p, d, q),seasonal_order=(0,0,0,0), exog=train['Sentiment'], trend='c').fit()

