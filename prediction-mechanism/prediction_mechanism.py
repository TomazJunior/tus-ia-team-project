#This file will use the daily sentiment value
#as the exogeneous variable in the ARIMAX model
#I will add the code for this

data = {
    'time': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05'],
    'bitcoin_adj_close': [30000, 31000, 32000, 33000, 34000],
    'sentiment': [0.2, 0.5, -0.1, 0.3, -0.2]
}

df = ''