import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
# from tensorflow import keras
from helper_functions import normalize, interactive_plot, show_data_plot
from helper_functions import individual_stock, trading_window

'''Task #2'''
print('Task #2: Import Data')
# Import data
stock_price_df = pd.read_csv('stock.csv')
stock_price_norm_df = pd.read_csv('stock_norm.csv')
stock_vol_df = pd.read_csv('stock_volume.csv')

# Sort data frames
stock_price_df = stock_price_df.sort_values(by='Date')
stock_vol_df = stock_vol_df.sort_values(by='Date')

# Check for null values
# print(stock_price_df.isnull().sum())
# print(stock_vol_df.isnull().sum())

# Statistical summary of data frames
# print(stock_price_df.describe())
# print(stock_vol_df.describe())

print()

'''Task #3: Basic Data Analysis'''
print('Task #3')
# stock_price_norm_df = normalize(stock_price_df)  # Normalize stock prices
# interactive_plot(stock_price_norm_df, 'Normalized Stock Prices')
# interactive_plot(stock_vol_df, 'Stock Volumes')

print()


'''Task #4: Prepare data before training AI/ML agent'''
print('Task #4')
# Data will be divided into 75% foir training and 25% for testing
# Training set: used to train model
# Testing set: Used to test trained model. Make sure testing data set has never
#   been seen by the trained model before
# Get price and volume of one stock into a df
aapl_price_vol_df = individual_stock(stock_price_df, stock_vol_df, 'AAPL')
# append price shifted back a day to a new target column, cut last row
aapl_pvt_df = trading_window(aapl_price_vol_df)  # pvt -> price, vol, target
# print(aapl_pvt_df)

# scale the data
sc = MinMaxScaler(feature_range=(0, 1))
aapl_pvt_scaled_df = sc.fit_transform(aapl_pvt_df.drop(columns=['Date']))
# print(aapl_pvt_scaled_df)

# separate inputs (X) and outputs (y) for model
X = aapl_pvt_scaled_df[:, :2]  # get all rows and first two cols
y = aapl_pvt_scaled_df[:, -1:]
# print(X)
# print(y)

# Split the data
split = int(0.65 * len(X))  # Find number of rows to use for training/testing
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]

# show_data_plot(X_train, 'Training Data')
# show_data_plot(X_test, 'Testing Data')
print()


''' Task #5: Simple Linear Regression Theory/Intuition
    Try to predict Y based on X
    Simple: Only examines relationship between 2 variables
    Linear: Dependant and independent vars have linear relationship

    Find: Y = mX + b

    Use: Least sum of squares
    Residual: error between model (line) and given data
    di = yi[estimated] - yi[actual]
    Minimum sum of least squares: min(sum(di^2))
    find best line thorugh our data
'''


''' Task #6: Regularization and Ridge Regression
    Ridge Regression: Introduce alpha term for penalty to generalize model
    min(sum(di^2) + a^2)
'''


''' Task #7: Build and train a ridge linear regression model '''
print('Task: #7')
# Create and train Ridge Linear regression model
regression_model = Ridge()
regression_model.fit(X_train, y_train)

# Test model and calculate accuracy
lr_accuracy = regression_model.score(X_test, y_test)
print(f'Ridge regression score: {lr_accuracy}')

# Make predicition

