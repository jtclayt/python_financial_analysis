'''
    Seaborn is a data visualisation library that builds on matplotlib
    Offers enhanced features over matplotlib
    https://seaborn.pydata.org/examples/index.html
'''
import pandas as pd  # data manipulation
import numpy as np  # data statistics, numerical analysis
import matplotlib.pyplot as plt  # data visualisation
import seaborn as sns  # Statistical data visualisation
from sklearn.datasets import load_breast_cancer

'''Seaborn basics'''
# np.c_ class object translates slice objects to concat along second axis
# example
x1 = np.array([1, 2, 3])
x2 = np.array([4, 5, 6])
z = np.c_[x1, x2]  # creates new array pairing elements at same index
# print(z)

# Get data set, load into dataframe
cancer = load_breast_cancer()
cancer_df = pd.DataFrame(
    np.c_[cancer['data'], cancer['target']],
    columns=np.append(cancer['feature_names'], ['target'])
)

# # Plot scatter plot between mean area and mean smoothness
# sns.scatterplot(
#     x='mean area', y='mean smoothness', hue='target', data=cancer_df
# )
# # Count plot of data set, ie how many of each 0 and 1 target in df
# sns.countplot(cancer_df['target'])

# # challenge
# sns.scatterplot(
#     x='mean radius', y='mean area', hue='target', data=cancer_df
# )
# plt.show()


'''Seaborn pairplot, distplot, and heatmaps'''
# pairplot will plot given vars against eachother, can be intesive process
# sns.pairplot(
#     cancer_df,
#     hue='target',
#     vars=[
#         'mean radius', 'mean texture', 'mean area', 'mean perimeter',
#         'mean smoothness'
#     ]
# )

# plt.figure(figsize=(20, 10))
# # heatmap will plot all the given variables with some color dist of values
# sns.heatmap(cancer_df.corr(), annot=True)

# # distplot combines histogram with kernel density estimate (KDE)
# # KDE is used to plot probability density of a continuous variable
# # distplots being depricated use displot instead
# sns.displot(cancer_df['mean radius'], kde=True)

# challenge
mean_radius_target0 = cancer_df[cancer_df['target'] == 0]['mean radius']
mean_radius_target1 = cancer_df[cancer_df['target'] == 1]['mean radius']
fig = plt.figure(figsize=(10, 5))
# can use ax=axes to set plots on subplots
# axes = fig.subplots(1, 2)
sns.distplot(mean_radius_target0, label='Target 0', bins=25, color='red')
sns.distplot(mean_radius_target1, label='Target 1', bins=25, color='green')
plt.grid()
plt.legend(loc='best')
plt.show()
