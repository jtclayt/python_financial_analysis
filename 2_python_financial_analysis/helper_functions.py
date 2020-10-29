'''A set of helper functions for stock analysis'''
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np


def normalize(df):
    '''Normalize prices of a given data set'''
    new_df = df.copy()
    for symbol in new_df.columns[1:]:
        for i in range(len(new_df[symbol])):
            new_df[symbol] /= new_df[symbol][0]
    return new_df


def get_daily_returns(df):
    '''Gets the daily returns for a dataframe of stock prices'''
    returns_df = df.copy()
    for symbol in df.columns[1:]:
        returns_df[symbol][0] = 0
        for i in range(1, len(df[symbol])):
            returns_df[symbol][i] = (df[symbol][i] / df[symbol][i-1] - 1) * 100
    return returns_df


def show_plot(df, title):
    '''Function takes in a df and title and displays a plot of data'''
    df.plot(x='Date', figsize=(15, 7), linewidth=3)
    plt.title(title, size=18)
    plt.grid()
    plt.show()


def interactive_plot(df, title):
    '''Function for creating a plotly interactive plot'''
    fig = px.line(title=title)
    for symbol in df.columns[1:]:
        fig.add_scatter(x=df['Date'], y=df[symbol], name=symbol)
    fig.show()


def portfolio_alloc(df, weights, amount=1e6):
    '''Get the values of stocks in a portfolio according to given weights'''
    portfolio_df = normalize(df)
    for i, stock in enumerate(df.columns[1:]):
        portfolio_df[stock] *= weights[i] * amount
    append_daily_value(portfolio_df)
    append_daily_returns(portfolio_df)
    return portfolio_df


def append_daily_value(df):
    '''Add a column for daily portfolio value'''
    df['Daily Value ($)'] = df[df != 'Date'].sum(axis=1)


def append_daily_returns(df):
    '''Add a column for daily portfolio returns'''
    df['Daily Return (%)'] = 0.0000
    for i in range(1, len(df)):
        curr_value = df['Daily Value ($)'][i]
        last_value = df['Daily Value ($)'][i-1]
        daily_return = (curr_value / last_value - 1) * 100
        df['Daily Return (%)'][i] = daily_return


def calc_capm(beta, rm, rf=0):
    '''Calculate a single stock capm'''
    return round(rf + beta * (rm - rf), 2)


def analyze_capm(df, rm):
    '''Analyze all stocks in a dataframe, compare to sp500'''
    data = {}
    for stock in df.columns[1:]:
        if stock != 'sp500':
            beta, alpha = np.polyfit(df['sp500'], df[stock], 1)
            er = calc_capm(beta, rm)
            ar = round(df[stock].mean() * 252, 2)
            data[stock] = {
                'beta': beta,
                'alpha': alpha,
                'er': er,
                'ar': ar
            }
            print(f'{stock}: Beta={round(beta, 2)}, Alpha={round(alpha, 3)}')
            print(f'{stock} expected return: {er}%')
            print(f'{stock} actual return: {ar}%')
            print()

    return data


def analyze_portfolio(weights, data):
    portfolio_er = 0
    portfolio_ar = 0
    keys = list(data.keys())
    for i in range(len(keys)):
        portfolio_er += weights[i] * data[keys[i]]['er']
        portfolio_ar += weights[i] * data[keys[i]]['ar']
    print(f'Portfolio expected return: {round(portfolio_er, 2)}%')
    print(f'Portfolio actual return: {round(portfolio_ar, 2)}%')
