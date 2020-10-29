import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from scipy import stats
import plotly.express as px
import plotly.figure_factory as ff
# import plotly.graph_objects as go
from helper_functions import normalize, show_plot, interactive_plot


''' Stock Data Analysis and Visualisations
    Stock data imported from csv file:
        AAPL: Apple
        BA: Boeing
        T: AT&T
        MGM: MGM Resorts
        AMZN: Amazon
        IBM: IBM
        TSLA: Tesla Motors
        GOOG: Google
        sp500: S&P 500 (measure 500 largest companies)
            https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
'''
'''Task #2: Loading Data and Basic Analysis'''
print('Task #2')
# Read stocks and returns csv into a dataframe
stocks_df = pd.read_csv('stock.csv')
stocks_norm_df = pd.read_csv('stocks_norm.csv')
returns_df = pd.read_csv('returns.csv')

# Sort stocks by date
stocks_df.sort_values(by='Date', inplace=True)

# Total number of stocks
print(f'Total number of stocks in dataset: {len(stocks_df.columns[1:])}')

# Stocks in dataset
included_stocks = ''
for stock in stocks_df.columns[1:]:
    included_stocks += stock + ' '
print(f'Stocks in dataset: {included_stocks}')

# challenge
# 1: average price of S&P500
sp500_mean_price = round(stocks_df['sp500'].mean(), 2)
print(f'Average price of S&P500: ${sp500_mean_price}')

# 2: Which stock has the min dispersion from mean in dollar value (std)
stocks_stds = stocks_df.std()
min_std_index = stocks_stds.argmin() + 1
min_std_stock = stocks_df.columns[min_std_index]
min_std_value = round(stocks_stds.min(), 2)
print(f'Min dispersion: {stocks_df.columns[min_std_index]} ${min_std_value}')

# 3: What is the max price form AMZN over time period?
amzn_max_price = max(stocks_df['AMZN'])
print(f'Max AMZN price: ${amzn_max_price}')

# Extra: Get a statistical summary of a dataset
print(round(stocks_df.describe(), 2))
print()


'''Task #3: Matplotlib Data Analysis'''
print('Task #3')
# First step of data analysis is to find null values
print(stocks_df.isnull().sum())  # this is clean data so no nulls in it

# Get dataframe info
print(stocks_df.info())

# Show stocks data in plot
# show_plot(stocks_df, 'Stock price data over time')

# # Challenge: Display normalized stock data
# normalize and save data to a new csv for easier loading
# stocks_norm_df = normalize(stocks_df)
# stocks_norm_df.to_csv('stocks_norm.csv', index=False)
# show_plot(stocks_norm_df, 'Normalized stock prices over time')


'''Task #4: Interactive Data Analysis'''
print('Task #4')
# interactive plotting with plotly.express (px), plotly.graph_objects (go)
# # Challenge 1: Call interactive plot function on normalized stocks
# interactive_plot(stocks_norm_df, 'Normalized stock prices over time')

# # Challenge 2: Calculate loss of 100 shares in S&P 500 from feb 19 to mar 23
# sp500_initial = float(stocks_df[stocks_df['Date'] == '2020-02-19']['sp500'])
# sp500_final = float(stocks_df[stocks_df['Date'] == '2020-03-23']['sp500'])
# sp500_last = float(stocks_df.iloc[-1]['sp500'])
# lost_money = round(100 * (sp500_final - sp500_initial), 2)
# last_money = round(100 * (sp500_last - sp500_initial), 2)
# print(f'dValue 100 shares S&P500 Feb 19 to Mar 23: ${lost_money}')
# print(f'dValue 100 shares S&P500 Feb 19 to Last: ${last_money}')
print()


'''Task #5/#6: Calculate Individual Stock Daily Returns'''
print('Task #5')
# # Calculate daily return of all stock prices
# Use these commands to generate and save returns csv
# returns_df = get_daily_returns(stocks_df)
# returns_df.to_csv('./returns.csv', index=False)

print(returns_df.describe())
print()
# show_plot(daily_returns_df, 'Static plot of daily returns')
# interactive_plot(daily_returns_df, 'Interactive plot of daily returns')


'''Task #7: Calculate correlations between daily returns'''
print('Task #7')
# Calculate correlations, drop date from this
# cm = returns_df.drop(columns=['Date']).corr()
# plt.figure(figsize=(10, 10))
# sns.heatmap(cm, annot=True)
# plt.show()

# Challenge 1: Top two stocks pos corr with sp500
# IBM, GOOG
# Challenge 2: Correlation between AMZN and BA
# 0.27, very little correlation
# Challenge 3: Correlation between MGM and BA
# 0.55, decently correlated
print()


'''Task #8: Plot Histogram for daily returns'''
print('Task #8')
# # Static plot of histograms for daily returns
# returns_df.hist(figsize=(10, 10), bins=40)
# plt.show()

# Interactive plot
data = []
for stock in returns_df.columns[1:]:
    data.append(returns_df[stock].values)

# Plotly figure factory module (ff) has wrapper functions to create unique
# charts and interactive subplots
fig = ff.create_distplot(data, returns_df.columns[1:])
fig.show()
