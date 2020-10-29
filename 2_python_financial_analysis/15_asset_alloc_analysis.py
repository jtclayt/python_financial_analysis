''' Portfolio Asset Allocation and Statistical Analysis
    Section covers analysing individual stocks:
        - Returns
        - Risk
        - Sharpe ratio
    Covers how to distribute funds and balance risk

    Types of Assets:
        - Equities (Stocks):
            Represents ownership of a certain percentage of a company
            Traded on stock exchanges and online brokerages
            Generally liquid, easily bought and sold

        - Fixed Income Securities (Bonds):
            Issued by gov/corp represent a loan made by investor
            Bonds pay investor a fixed interest rate
            Used to raise funds/capital
            Less risk > less interest, greater risk > greater interest
            Bond coupons represent semi-annual interest payments

        - Exchange Traded Funds (ETF):
            ETFS are groups of securities or track an index (like sp500)
            Traded on exchanges like any other stock
            Can include stocks, bonds, and commodities
            Low management fees and risk diversification
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import plotly.express as px
# import plotly.figure_factory as ff
from helper_functions import portfolio_alloc, interactive_plot, normalize


'''Task #2: Loading Data and Basic Analysis'''
print('Task #2')
# Read stocks and returns csv into a dataframe
stocks_df = pd.read_csv('stock.csv')
stocks_norm_df = pd.read_csv('stocks_norm.csv')
returns_df = pd.read_csv('returns.csv')

# Sort stocks by date
stocks_df.sort_values(by='Date', inplace=True)

# # Plotly interactive plot of normalized stock prices
# interactive_plot(stocks_norm_df, 'Normalized stock prices')
print()

''' Task #3:
    Asset Allocation:
        - Investment strategy to balance risk tolerance, maximize returns
            over a certain timespan
        - Many asset classes to consider:
            Equities
            Fixed Income Securities
            Cash and equivalents
            Exchange Traded Funds
            Real Estate
            Commodities

    Conventional wisdom for allocation:
        100 - age = % stock allocation (roughly)
'''

''' Task #4:
    Perform random asset allocation and calculate portfolio daily return
'''
print('Task #4')
# # Assuming we have $1M to invest
# # Starting with random weights of the 9 securities
# # Set np seed
# # np.random.seed(101)
# # Get random weights
# weights = np.array(np.random.random(9))
# # normalize the random weights
# weights /= np.sum(weights)
# stocks = stocks_df.columns[1:].values
# portfolio_df = stocks_norm_df.copy()

# # change normalized value to total allocated value
# for i, stock in enumerate(stocks):
#     portfolio_df[stock] *= weights[i] * 1e6

# # Add an additional column of daily sum of portfolio, asix is refering to
# # summing along the row instead of the column
# portfolio_df['Daily Value ($)'] = portfolio_df[
#     portfolio_df != 'Date'].sum(axis=1)

# # Create a daily return column
# portfolio_df['Daily Return (%)'] = 0.0000
# for i in range(1, len(portfolio_df)):
#     curr_value = portfolio_df['Daily Value ($)'][i]
#     last_value = portfolio_df['Daily Value ($)'][i-1]
#     daily_return = (curr_value / last_value - 1) * 100
#     portfolio_df['Daily Return (%)'][i] = daily_return

# print(portfolio_df)
print()


''' Task #5: Portfolio allocation: daily return/worth calculation\
    Created functions for getting daily value and returns
'''
print('Task #5')
np.random.seed(101)
weights = np.array(np.random.random(len(stocks_df.columns[1:])))
weights /= np.sum(weights)

portfolio_df = portfolio_alloc(stocks_df, weights)
print(portfolio_df)
print()

'''Task #6: Portfolio Data Visualisation'''
print('Task #6')
# # Plot daily returns over time
# fig = px.line(
#     x=portfolio_df['Date'],
#     y=portfolio_df['Daily Return (%)'],
#     title='Portfolio Daily % Return'
# )
# fig.show()

# # Plot daily value of each stock holding
# interactive_plot(
#     portfolio_df.drop(['Daily Value ($)', 'Daily Return (%)'], axis=1),
#     title='Stock allocation values'
# )

# # Print a histogram of daily returns
# fig = px.histogram(portfolio_df, x='Daily Return (%)')
# fig.show()

# # Challenge: Print plot of overall daily worth vs time
# fig = px.line(
#     x=portfolio_df['Date'],
#     y=portfolio_df['Daily Value ($)'],
#     title='Portfolio Total Value over Time'
# )
# fig.show()
print()


''' Task #7: Porfolio Statistical Metrics
    - Daily return (p => closing price, t => some time)
        (p[t] - p[t-1]) / p[t-1] * 100%
    - Cummaltive Return (p => closing price, t => some time, 0 => initial)
        (p[t] - p[0]) / p[0] * 100%
    - Average Daily Return
        Average of the daily returns over some time period
        Standard deviation about this mean represents risk
    - Sharpe Ratio
        Rp => return of investment
        Rf => risk free return (US gov bond)
        Sp => std dev of daily return)
        (Rp - Rf) / Sp
        Used to calculate return compared to its risk
'''


'''Task #8: Calculate Cummulative Return, Average Daily Return, Sharpe ratio'''
print('Task #8')
curr_value = portfolio_df['Daily Value ($)'].values[-1]
initial_value = portfolio_df['Daily Value ($)'].values[0]
cummulative_return = round((curr_value / initial_value - 1) * 100, 2)
print(f'Cummaltive Return of Portfolio: {cummulative_return}%')

return_mean = round(portfolio_df['Daily Return (%)'].mean(), 2)
return_std = round(portfolio_df['Daily Return (%)'].std(), 2)
print(f'Mean of daily return: {return_mean}%')
print(f'Standard deviation of daily return: {return_std}%')

# daily return of 10yr us treasury bond
risk_free_return = 0.73 / (10 * 365)
sharpe_ratio_exact = (return_mean - risk_free_return) / return_std
sharpe_ratio_est = return_mean / return_std
# Calculate annual sharpe ratio, 252 trading days a year
annual_sharpe_exact = sharpe_ratio_exact * np.sqrt(252)
annual_sharpe_est = sharpe_ratio_est * np.sqrt(252)
print(f'Exact annual Sharpe ratio: {annual_sharpe_exact}')
print(f'Approximate annual Sharpe ratio: {annual_sharpe_est}')
