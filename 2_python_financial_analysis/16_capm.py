''' Capital Asset Pricing Model (CAPM)
    - Model that describes relationship between expected return and risk of
        securities
    - Indicates expected return is equal to risk free return plus risk premium

    Risk free rate (Rf):
        - Asset that has a return with no standard deviation
        - Ex: 10 year US Treasury Bond

    Market Portfolio return (Rm):
        - Market portfolio includes all securities in the market
        - Ex: S&P500

    Beta:
        - Slope of line regression (market return vs stock return)
        - Measure of volatility or systemic risk of security compared to market
        - Beta = 1: price activity is correlated to market
        - Beta < 1 (defensive): Security is less volatile then market
        - Beta > 1 (aggressive): Security is more volatile then market

    Alpha:
        - Excess return over market

    CAPM formula:
        Ri => Expected return
        Rf => Risk free return
        Bi => Beta between stock and market
        Rm => Market return
        (Rm - Rf) => Risk premium, incentive for investing
        Ri = Rf + Bi(Rm - Rf)
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
# from scipy import stats
import plotly.express as px
# import plotly.figure_factory as ff
# import plotly.graph_objects as go
from helper_functions import portfolio_alloc, interactive_plot, normalize, \
    calc_capm, analyze_capm, analyze_portfolio


'''Task #2: Loading Data and Basic Analysis'''
print('Task #2')
# Read stocks and returns csv into a dataframe
stocks_df = pd.read_csv('stock.csv')
stocks_norm_df = pd.read_csv('stocks_norm.csv')
returns_df = pd.read_csv('returns.csv')
print()


'''Task #4: Calculate Beta for a single stock'''
print('Task #5')
# Fit a ploynomial of degree one (line) to the scatter plot
# Beta for AAPL will be the slope of this line
# Alpha is the excess return over market
aapl_beta, aapl_alpha = np.polyfit(
    returns_df['sp500'], returns_df['AAPL'], deg=1)
print(f'AAPL: Beta={aapl_beta}, Alpha={aapl_alpha}')
# returns_df.plot(x='sp500', y='AAPL', kind='scatter')
# sp500_aapl_reg = beta * returns_df['sp500'] + alpha
# plt.plot(returns_df['sp500'], sp500_aapl_reg, color='r', linewidth=3)
# plt.show()

# Challenge TSLA
tsla_beta, tsla_alpha = np.polyfit(
    returns_df['sp500'], returns_df['TSLA'], deg=1)
print(f'TSLA: Beta={tsla_beta}, Alpha={tsla_alpha}')
returns_df.plot(x='sp500', y='TSLA', kind='scatter')
# sp500_aapl_reg = tsla_beta * returns_df['sp500'] + tsla_alpha
# plt.plot(returns_df['sp500'], sp500_aapl_reg, color='r', linewidth=3)
# plt.show()
print()


'''Task #5: Apply CAPM formula for a single stock'''
# Calculating the annual return of sp500, 252 trading days annually
print('Task #7')
sp500_annual_return = round(returns_df['sp500'].mean() * 252, 2)
print(f'S&P500 average daily return: {sp500_annual_return}%')
aapl_expected_return = calc_capm(aapl_beta, sp500_annual_return)
aapl_actual_return = round(returns_df['AAPL'].mean() * 252, 2)
print(f'AAPL expected return: {aapl_expected_return}%')
print(f'AAPL actual return: {aapl_actual_return}%')
print()

# Challenge, repeat for AT&T
t_beta, t_alpha = np.polyfit(returns_df['sp500'], returns_df['T'], 1)
print(f'S&P500 average daily return: {sp500_annual_return}%')
t_expected_return = calc_capm(t_beta, sp500_annual_return)
t_actual_return = round(returns_df['T'].mean() * 252, 2)
print(f'T expected return: {t_expected_return}%')
print(f'T actual return: {t_actual_return}%')
print()


'''Task #6: Calculate all CAPM for given stocks'''
print('Task #6')
capm_data = analyze_capm(returns_df, sp500_annual_return)
print()


'''Task #7: Use portfolio weight and calculate return'''
print('Task #7')
weights = np.random.random(8)
weights /= np.sum(weights)
print('Using random weights for portfolio')
print(weights)
analyze_portfolio(weights, capm_data)
