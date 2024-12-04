# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:48:17 2024

@author: bryan
"""
%pip install scikit-learn
%pip install numpy
%pip install pandas
%pip install nba_api
%pip install matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# %%
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
# %% 

notyet = pd.read_csv('finalized_dataset.csv')


rows1 = notyet.iloc[::2].reset_index(drop=True)  # First of each pair (e.g., "L" rows)
rows2 = notyet.iloc[1::2].reset_index(drop=True)  # Second of each pair (e.g., "W" rows)

# Rename columns for rows1 and rows2
rows1 = rows1.add_suffix('1')
rows2 = rows2.add_suffix('2')

combinedshuffled_df = pd.concat([rows1, rows2], axis=1)
combinedshuffled_df = combinedshuffled_df.drop(columns=['Unnamed: 01', 'Unnamed: 02'])
combinedshuffled_df['WL1'] = combinedshuffled_df['WL1'].map({'L': 0, 'W': 1})
combinedshuffled_df['WL2'] = combinedshuffled_df['WL2'].map({'L': 0, 'W': 1})
combinedshuffled_df = combinedshuffled_df.drop(columns=['Game_ID1', 'Game_ID2'])
combinedshuffled_df = combinedshuffled_df.dropna()


group1_cols = combinedshuffled_df.columns[0:10]  # Columns 0-9
group2_cols = combinedshuffled_df.columns[10:20]  # Columns 10-19

# Step 2: Generate a random mask to decide which rows to swap
swap_mask = np.random.choice([True, False], size=len(combinedshuffled_df))

# Step 3: Swap values between Group 1 and Group 2 for the selected rows
for i in range(len(combinedshuffled_df)):
    if swap_mask[i]:
        # Swap Group 1 and Group 2 values for row i
        combinedshuffled_df.loc[i, group1_cols], combinedshuffled_df.loc[i, group2_cols] = combinedshuffled_df.loc[i, group2_cols].values, combinedshuffled_df.loc[i, group1_cols].values
combinedshuffled_df = combinedshuffled_df.drop(columns=['WL2'])
# Print the resulting DataFrame to verify the swap
print(combinedshuffled_df.head())

# %% EDA begin!

# univariate analysis
for col in combinedshuffled_df.columns:
    sns.histplot(combinedshuffled_df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
#%% 
team1_win = combinedshuffled_df[combinedshuffled_df['WL1'] == 1]
team1_loss = combinedshuffled_df[combinedshuffled_df['WL1'] == 0]

for col in combinedshuffled_df.columns:
    t_stat, p_val = ttest_ind(team1_win[col], team1_loss[col])
    if(p_val > 0.05):
        print(f"{col}: p-value={p_val}")

# interesting, ORB but just for team 2 is giving wierd answers..
# %%
corr_matrix = combinedshuffled_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

