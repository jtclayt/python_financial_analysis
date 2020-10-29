'''
    matplotlib is a data visualisation library for python
    can do line charts, scatter plots, pie charts, histograms, 3d,
    and box plots
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

stock_df = pd.read_csv('stocks.csv')
returns_df = pd.read_csv('daily_returns.csv')


'''Line plots'''
# stock_df.plot(x='Date', y='AAPL', label='Apple Stock Prices', linewidth=3)
# # Change y label
# plt.ylabel('Price ($)')
# # change legend location, default upper left
# plt.legend(loc='upper center')
# # Add a title and change its font size
# plt.title('Line Chart for Apple stock price', size=20)

# # Challenge
# stock_df.plot(
#     x='Date', y='sp500', label='S&P 500 Price', linewidth=3, color='r')
# plt.ylabel('Price ($)')
# plt.title('S&P 500 Price over time')
# plt.show()


'''Scatter plot'''
# # Apple vs sp500 scatter plot
# x_axis = returns_df['AAPL']
# y_axis = returns_df['sp500']
# plt.scatter(x=x_axis, y=y_axis)
# plt.xlabel('Apple Daily Returns')
# plt.ylabel('S&P 500 Daily Returns')
# plt.title('S&P 500 vs Apple Daily Returns', size=18)

# # challenge
# x_axis = returns_df['GOOG']
# y_axis = returns_df['sp500']
# plt.scatter(x=x_axis, y=y_axis, color='g')
# plt.xlabel('Google Daily Returns')
# plt.ylabel('S&P 500 Daily Returns')
# plt.title('S&P 500 vs Google Daily Returns', size=18)
# plt.show()


'''Pie Chart'''
# # simple pie chart example
# values = [20, 55, 5, 17, 4]
# colors = ['g', 'r', 'y', 'b', 'm']
# labels = ['AAPL', 'GOOG', 'T', 'TSLA', 'AMZN']
# # add emphasis to certain section of chart
# explode = [0, 0.2, 0, 0, 0.1]
# # create the figure
# plt.figure(figsize=(7, 7))
# # add pie chart with data to figure
# plt.pie(values, colors=colors, labels=labels, explode=explode)
# # display labels in the legend

# # challenge
# values = [1, 1, 1, 1, 1]
# colors = ['g', 'r', 'y', 'b', 'm']
# labels = ['AAPL', 'GOOG', 'T', 'TSLA', 'AMZN']
# explode = [0, 0.1, 0, 0, 0.1]
# plt.figure(figsize=(7, 7))
# plt.pie(values, colors=colors, labels=labels, explode=explode)
# plt.legend(loc='best', labels=labels)
# plt.title('My portfolio', size=18)
# plt.show()


'''Histograms'''
# # example/challenge
# mu_goog = returns_df['GOOG'].mean()
# sigma_goog = returns_df['GOOG'].std()
# mu_goog_r = round(mu_goog, 2)
# sigma_goog_r = round(sigma_goog, 2)
# NUM_BINS = 30
# TITLE_STR = f'GOOG Returns: mu={mu_goog_r} sigma={sigma_goog_r}'

# plt.figure(figsize=(7, 5))
# plt.hist(returns_df['GOOG'], NUM_BINS)
# plt.grid()
# plt.xlabel('Daily Return')
# plt.ylabel('Count')
# plt.title(TITLE_STR, size=18)
# plt.show()


'''Multiple plots on one figure'''
# # example and challenge
# y_values = ['AAPL', 'sp500', 'GOOG']
# labels = ['Apple', 'S&P 500', 'Google']
# colors = ['r', 'm', 'b']
# stock_df.plot(x='Date', y=y_values, label=labels, color=colors, linewidth=3)
# plt.ylabel('Price ($)')
# plt.title('Stock Prices over time', size=18)
# plt.legend(loc='upper center')
# plt.show()


'''Subplots'''
# # example and challenge
# plt.figure(figsize=(10, 10))
# # command to start a subplot on figure, plot is 2 rows 2 columns
# plt.subplot(2, 2, 1)
# plt.plot(stock_df['AAPL'], 'r--')
# plt.grid()
# plt.title('Apple stock price over time', size=15)
# plt.ylabel('Price ($)')
# # move to next subplot
# plt.subplot(2, 2, 2)
# plt.plot(stock_df['GOOG'], 'g-.')
# plt.grid()
# plt.title('Google stock price over time', size=15)
# plt.ylabel('Price ($)')
# plt.subplot(2, 2, 3)
# plt.plot(stock_df['sp500'], 'b.')
# plt.grid()
# plt.title('S&P 500 stock price over time', size=15)
# plt.ylabel('Price ($)')
# plt.show()


'''3d plots'''
# mpl_toolkits.mplot3d toolkit that extend matplotlib functions
# https://matplotlib.org/mpl_toolkits/index.html
# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111, projection='3d')
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# y = [5, 6, 2, 3, 13, 4, 1, 2, 4, 8]
# z = [2, 3, 3, 5, 7, 9, 11, 9, 10, 7]
# ax.scatter(x, y, z, color='r', marker='o')
# # set labels for the axes
# ax.set_xlabel('X label')
# ax.set_ylabel('Y label')
# ax.set_zlabel('Z label')
# plt.show()

# # challenge
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(
#     returns_df['AAPL'],
#     returns_df['GOOG'],
#     returns_df['sp500'],
#     color='g',
#     marker='*'
# )
# plt.title('S&P 500 vs Apple vs Google Daily Returns', size=18)
# ax.set_xlabel('Apple Daily Return')
# ax.set_ylabel('Google Daily Return')
# ax.set_zlabel('S&P 500 Daily Returns')
# plt.show()


'''Box Plots'''
np.random.seed(20)
# np.random.normal creates a normal dist with args:
#   mean, std dev, and num points
data = [
    np.random.normal(100, 20, 2000),
    np.random.normal(100, 30, 2000),
    np.random.normal(100, 20, 2000),
    np.random.normal(100, 5, 2000),
    np.random.normal(100, 10, 500)
]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
boxplot = ax.boxplot(data)
plt.show()
