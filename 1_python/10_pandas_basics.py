'''
    Pandas is a data manipulation and analysis tool built on Numpy
    Pandas uses a data structure known as DataFrame, like using excel in python
    DataFrames store rows and columns of tabuilar data
    a Series is a single column of a DataFrame
'''

import pandas as pd


'''Pandas basics'''
# list1 = ['AAPL', 'AMZN', 'T']
# label = ['stock#1', 'stock#2', 'stock#3']
# print(list1, label)

# one_d_series = pd.Series(data=list1, index=label)
# print(one_d_series)

# # creating a single data frame with multiple rows
# bank_client_df = pd.DataFrame({
#     'Bank Client ID': [111, 222, 333, 444],
#     'Bank Client Name': ['Chanel', 'Steve', 'Mitch', 'Ryan'],
#     'Net Worth ($)': [3500, 29000, 10000, 2000],
#     'Years with bank': [3, 4, 9, 5]
# })
# print(bank_client_df)
# print(bank_client_df.head(2))  # .head(n) gets first n rows
# print(bank_client_df.tail(1))  # same as head but last rows

# Challenge
# portfolio_df = pd.DataFrame({
#     'Symbol': ['MSFT', 'HD', 'LUMN'],
#     'Shares': [30, 10, 50],
#     'Price ($)': [212, 280, 10]
# })
# print(portfolio_df)
# # sum portfolio value
# portfolio_value = sum(portfolio_df['Shares'] * portfolio_df['Price ($)'])
# print(f'Portfolio value: ${portfolio_value}')


'''HTML data, applying functions, and sorting'''
# # Importing data to a pandas dataframe from a csv
# bank_df = pd.read_csv('./bank_client_information.csv')
# print(bank_df)
# # Save dataframe to csv, without index numbers
# bank_df.to_csv('sample_pandas_output.csv', index=False)

# # Importing data from an html page
# house_prices_df = pd.read_html(
#     'https://www.livingin-canada.com/house-prices-canada.html')
# print(house_prices_df)

# # Challenge
# retirement_df = pd.read_html('https://www.ssa.gov/oact/progdata/nra.html')
# print(retirement_df[0])


'''DataFrame operations'''
# bank_client_df = pd.DataFrame({
#     'Bank Client ID': [111, 222, 333, 444],
#     'Bank Client Name': ['Chanel', 'Steve', 'Mitch', 'Ryan'],
#     'Net Worth ($)': [3500, 29000, 10000, 2000],
#     'Years with bank': [3, 4, 9, 5]
# })

# # Select rows that satisfy a condition
# loyal_clients_df = bank_client_df[bank_client_df['Years with bank'] > 4]
# print(loyal_clients_df)
# steve = bank_client_df[bank_client_df['Bank Client Name'] == 'Steve']
# print(steve)

# # delete column from data fram
# del bank_client_df['Bank Client ID']
# print(bank_client_df)

# # challenge
# high_value_clients_df = bank_client_df[
#   bank_client_df['Net Worth ($)'] > 5000]
# net_worth_high_value = high_value_clients_df['Net Worth ($)'].sum()
# print(high_value_clients_df)
# print(net_worth_high_value)


'''Pandas with functions'''
# bank_client_df = pd.DataFrame({
#     'Bank Client ID': [111, 222, 333, 444],
#     'Bank Client Name': ['Chanel', 'Steve', 'Mitch', 'Ryan'],
#     'Net Worth ($)': [3500, 29000, 10000, 2000],
#     'Years with bank': [3, 4, 9, 5]
# })


# def networth_update(balance):
#     '''Function for increasing networth of clients'''
#     return balance * 1.1


# # apply a function to a column of a dataframe, example saves back to df
# bank_client_df['Net Worth ($)'] = bank_client_df[
#     'Net Worth ($)'].apply(networth_update)
# print(bank_client_df)

# # apply len function to a column
# print(bank_client_df['Bank Client Name'].apply(len))

# # Challenge


# def increase_price(price):
#     return 2 * price + 100


# bank_client_df['Net Worth ($)'] = bank_client_df[
#     'Net Worth ($)'].apply(increase_price)
# print(bank_client_df)
# print(f"Total networth: ${bank_client_df['Net Worth ($)'].sum()}")


'''Ordering and sorting'''
# bank_client_df = pd.DataFrame({
#     'Bank Client ID': [111, 222, 333, 444],
#     'Bank Client Name': ['Chanel', 'Steve', 'Mitch', 'Ryan'],
#     'Net Worth ($)': [3500, 29000, 10000, 2000],
#     'Years with bank': [3, 4, 9, 5]
# })

# # Sorting a data frame, need to have inplace True for it to alter df
# bank_client_df.sort_values(by='Years with bank', inplace=True)
# print(bank_client_df)

# # challenge
# # To sort in descending order add ascending=False
# bank_client_df.sort_values(by='Net Worth ($)', ascending=False, inplace=True)
# print(bank_client_df)


'''Merging, joining, and concatenation'''
# Concatenation example
df1 = pd.DataFrame({
    'A': ['A1', 'A2'],
    'B': ['B1', 'B2'],
    'C': ['C1', 'C2']
})

df2 = pd.DataFrame({
    'A': ['A3', 'A4'],
    'B': ['B3', 'B4'],
    'C': ['C3', 'C4']
})

df3 = pd.DataFrame({
    'A': ['A5', 'A6'],
    'B': ['B5', 'B6'],
    'C': ['C5', 'C6']
})

# Ignore index to reset indexes
big_df = pd.concat([df1, df2, df3], ignore_index=True)
print(big_df)

# Creating dataframe from a dictionary
raw_data_1 = {
    'Bank Client ID': ['1', '2', '3', '4', '5'],
    'First Name': ['Nancy', 'Alex', 'Shep', 'Max', 'Allen'],
    'Last Name': ['Rob', 'Ali', 'George', 'Mitch', 'Steve']
}
bank_df1 = pd.DataFrame(raw_data_1)
print(bank_df1)

raw_data_2 = {
    'Bank Client ID': ['6', '7', '8', '9', '10'],
    'First Name': ['Bill', 'Dina', 'Sarah', 'Heather', 'Holly'],
    'Last Name': ['Chris', 'Moe', 'Smith', 'Bob', 'Fife']
}
bank_df2 = pd.DataFrame(raw_data_2)
print(bank_df2)

raw_salary_data = {
    'Bank Client ID': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'Annual Salary': [
        25000, 32000, 44000, 15000, 55000, 78000, 99000, 150000, 27000, 91000
    ]
}
bank_salary_df = pd.DataFrame(raw_salary_data)
print(bank_salary_df)

# concat all clients to one dataframe
bank_all_df = pd.concat([bank_df1, bank_df2], ignore_index=True)
print(bank_all_df)

merged_bank_info_df = pd.merge(
    bank_salary_df, bank_all_df, on='Bank Client ID')
merged_bank_info_df = merged_bank_info_df.append({
    'Bank Client ID': '11',
    'Annual Salary': 99000,
    'First Name': 'Justin',
    'Last Name': 'Clayton'
}, ignore_index=True)
print(merged_bank_info_df)
