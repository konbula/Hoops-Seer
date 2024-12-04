# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 02:35:10 2024

@author: bryan
"""

#%% import

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#%%
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

#%% bring in dataset

notyet = pd.read_csv('finalized_dataset.csv')

# %% make the dataset be half by combining the row

notyet.head()

# %%

rows1 = notyet.iloc[::2].reset_index(drop=True)  # First of each pair (e.g., "L" rows)
rows2 = notyet.iloc[1::2].reset_index(drop=True)  # Second of each pair (e.g., "W" rows)

# Rename columns for rows1 and rows2
rows1 = rows1.add_suffix('1')
rows2 = rows2.add_suffix('2')

# Combine the two DataFrames column-wise
combined_df = pd.concat([rows1, rows2], axis=1)
combined_df = combined_df.drop(columns=['Unnamed: 01', 'Unnamed: 02'])
combined_df['WL1'] = combined_df['WL1'].map({'L': 0, 'W': 1})
combined_df = combined_df.drop(columns=['WL2', 'Game_ID1', 'Game_ID2'])
combined_df = combined_df.dropna()
print(combined_df.shape)
combined_df.head()
# %%

# %%

# print(combined_df.columns)
# 
# print(combined_df.shape)
# print(combined_df[combined_df.isnull().any(axis=1)])

# %%



# Drop irrelevant columns


# %%

# Define features (X) and target (y)
X = combined_df.drop(columns=['WL1'])
y = combined_df['WL1']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

# %% run the logistic again and cross validation

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Split temporary dataset into validation (15%) and test (15%) datasets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

model = LogisticRegression(penalty='l2', C=1.0, random_state=42)
# %%
# Train the model on the training set
model.fit(X_train, y_train)

# Evaluate on the validation set
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Validation Accuracy: {val_accuracy:.2f}")

# %%

# Evaluate on the test set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Test Accuracy: {test_accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

























# %% The mixed approach
# Combine the two DataFrames column-wise
combinedshuffled_df = pd.concat([rows1, rows2], axis=1)
combinedshuffled_df = combinedshuffled_df.drop(columns=['Unnamed: 01', 'Unnamed: 02'])
combinedshuffled_df['WL1'] = combinedshuffled_df['WL1'].map({'L': 0, 'W': 1})
combinedshuffled_df['WL2'] = combinedshuffled_df['WL2'].map({'L': 0, 'W': 1})
combinedshuffled_df = combinedshuffled_df.drop(columns=['Game_ID1', 'Game_ID2'])
combinedshuffled_df = combinedshuffled_df.dropna()

# %% test
print(combinedshuffled_df.columns)
combinedshuffled_df.head()
# %%

group1_cols = combinedshuffled_df.columns[0:10]  # Columns 0-9
group2_cols = combinedshuffled_df.columns[10:20]  # Columns 10-19

# Step 2: Generate a random mask to decide which rows to swap
swap_mask = np.random.choice([True, False], size=len(combinedshuffled_df))

# Step 3: Swap values between Group 1 and Group 2 for the selected rows
for i in range(len(combinedshuffled_df)):
    if swap_mask[i]:
        # Swap Group 1 and Group 2 values for row i
        combinedshuffled_df.loc[i, group1_cols], combinedshuffled_df.loc[i, group2_cols] = combinedshuffled_df.loc[i, group2_cols].values, combinedshuffled_df.loc[i, group1_cols].values

# Print the resulting DataFrame to verify the swap
print(combinedshuffled_df.head())
combinedshuffled_df = combinedshuffled_df.drop(columns=['WL2'])
# %%

print(combinedshuffled_df.columns)


























# %% testing on randomized data
# logistic regression

# Assuming combinedshuffled_df is your DataFrame
X = combinedshuffled_df.drop(columns=['WL1'])
y = combinedshuffled_df['WL1']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

# %% Run the logistic regression again with cross-validation

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Split temporary dataset into validation (15%) and test (15%) datasets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

model = LogisticRegression(penalty='l2', C=1.0, random_state=42)

# %% Train the model on the training set
model.fit(X_train, y_train)

# Evaluate on the validation set
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Validation Accuracy: {val_accuracy:.2f}")

# %% Evaluate on the test set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Test Accuracy: {test_accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# %%

# Perform grid search with proper solver and penalty combinations
param_grid = [
    {'penalty':['l1','l2','elasticnet','none'],
    'C' : np.logspace(-4,4,20),
    'solver': ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter'  : [100,1000,2500,5000]
}
]

grid_search = GridSearchCV(model,param_grid = param_grid, cv = 3, verbose=True,n_jobs=-1)

thing = grid_search.fit(X_train, y_train)

# Get the best hyperparameters from the grid search
best_params = thing.best_estimator_

print(f"Best hyperparameters: {best_params}")

# Evaluate the model with the best hyperparameters on the test set
best_model = grid_search.best_estimator_
print(f'Accuracy - : {best_model.score(X_test,y_test):.3f}')



















#%% trying random forest

# Assuming combinedshuffled_df is your DataFrame
X = combinedshuffled_df.drop(columns=['WL1'])
y = combinedshuffled_df['WL1']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

# %% Run the Random Forest again with cross-validation

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Split temporary dataset into validation (15%) and test (15%) datasets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Create a new RandomForestClassifier model (you can adjust hyperparameters as needed)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# %% Train the model on the training set
model.fit(X_train, y_train)

# Evaluate on the validation set
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Validation Accuracy: {val_accuracy:.2f}")

# %% Evaluate on the test set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Test Accuracy: {test_accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# %% hyperparamter testing

X = combinedshuffled_df.drop(columns=['WL1'])
y = combinedshuffled_df['WL1']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)

# Fit the model with hyperparameter tuning
grid_search.fit(X_train, y_train)

# Get the best parameters from GridSearchCV
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Train the model using the best parameters
best_rf_model = grid_search.best_estimator_

# Predict and evaluate accuracy on the test set
y_pred = best_rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Optionally, run with validation split as well
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Split temporary dataset into validation (15%) and test (15%) datasets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Train the model on the training set
best_rf_model.fit(X_train, y_train)

# Evaluate on the validation set
y_val_pred = best_rf_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Evaluate on the test set
y_test_pred = best_rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Test Accuracy: {test_accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))


















# %% trying XGboost
# %% testing on randomized data

# Assuming combinedshuffled_df is your DataFrame
X = combinedshuffled_df.drop(columns=['WL1'])
y = combinedshuffled_df['WL1']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Import XGBoost classifier
from xgboost import XGBClassifier

# Train the XGBoost model
model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

# %% Run the XGBoost again with cross-validation

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Split temporary dataset into validation (15%) and test (15%) datasets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Create a new XGBoost model (you can adjust hyperparameters as needed)
model = XGBClassifier(n_estimators=100, random_state=42)

# %% Train the model on the training set
model.fit(X_train, y_train)

# Evaluate on the validation set
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Validation Accuracy: {val_accuracy:.2f}")

# %% Evaluate on the test set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Test Accuracy: {test_accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))






# %% hyperparameter tuning

# %% testing on randomized data

# Assuming combinedshuffled_df is your DataFrame
X = combinedshuffled_df.drop(columns=['WL1'])
y = combinedshuffled_df['WL1']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],  # Number of trees (boosting rounds)
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Step size
    'max_depth': [3, 6, 10],  # Maximum depth of the tree
    'min_child_weight': [1, 3, 5],  # Minimum sum of instance weight needed in a child
    'subsample': [0.7, 0.8, 0.9],  # Fraction of training samples used
    'colsample_bytree': [0.7, 0.8, 0.9],  # Fraction of features used for each tree
    'gamma': [0, 0.1, 0.3],  # Minimum loss reduction for further partitioning
}

# Initialize the XGBoost model
model = XGBClassifier(random_state=42)

# Set up GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)

# Fit the model with the grid search
grid_search.fit(X_train, y_train)

# Best hyperparameters
print("Best Hyperparameters: ", grid_search.best_params_)

# Evaluate on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.2f}")
# Fitting 5 folds for each of 2916 candidates, totalling 14580 fits
# Best Hyperparameters:  {'colsample_bytree': 0.7, 'gamma': 0.3, 'learning_rate': 0.01, 'max_depth': 10, 'min_child_weight': 5, 'n_estimators': 150, 'subsample': 0.9}
# Test Accuracy: 0.62



# %%
import seaborn as sns


































# %% attempt of tyring logistic regression after dropping ORB, DRB, 3par, and FTR, and maybe assists?

combinedshuffledtrimmed_df = combinedshuffled_df.drop(columns=['orb_percentage_uw1', 'orb_percentage_uw2', 'drb_percentage_uw1', 'drb_percentage_uw2', '3par_uw1', '3par_uw2', 'ftr_uw2', 'ftr_uw1'])

# %%
X = combinedshuffledtrimmed_df.drop(columns=['WL1'])
y = combinedshuffledtrimmed_df['WL1']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

# %% Run the logistic regression again with cross-validation

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Split temporary dataset into validation (15%) and test (15%) datasets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

model = LogisticRegression(penalty='l2', C=1.0, random_state=42)

# %% Train the model on the training set
model.fit(X_train, y_train)

# Evaluate on the validation set
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Validation Accuracy: {val_accuracy:.2f}")

# %% Evaluate on the test set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Test Accuracy: {test_accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

















#%% trying random forest

# Assuming combinedshuffled_df is your DataFrame
X = combinedshuffledtrimmed_df.drop(columns=['WL1'])
y = combinedshuffledtrimmed_df['WL1']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

# %% Run the Random Forest again with cross-validation

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Split temporary dataset into validation (15%) and test (15%) datasets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Create a new RandomForestClassifier model (you can adjust hyperparameters as needed)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# %% Train the model on the training set
model.fit(X_train, y_train)

# Evaluate on the validation set
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Validation Accuracy: {val_accuracy:.2f}")

# %% Evaluate on the test set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Test Accuracy: {test_accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))




















# %% trying XGboost
# %% testing on randomized data

# Assuming combinedshuffled_df is your DataFrame
X = combinedshuffledtrimmed_df.drop(columns=['WL1'])
y = combinedshuffledtrimmed_df['WL1']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Import XGBoost classifier
from xgboost import XGBClassifier

# Train the XGBoost model
model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

# %% Run the XGBoost again with cross-validation

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Split temporary dataset into validation (15%) and test (15%) datasets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Create a new XGBoost model (you can adjust hyperparameters as needed)
model = XGBClassifier(n_estimators=100, random_state=42)

# %% Train the model on the training set
model.fit(X_train, y_train)

# Evaluate on the validation set
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Validation Accuracy: {val_accuracy:.2f}")

# %% Evaluate on the test set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Test Accuracy: {test_accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))







# %% run and save models
import joblib
# logist reg

X = combinedshuffled_df.drop(columns=['WL1'])
y = combinedshuffled_df['WL1']
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# Predict and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")

# Run the logistic regression again with cross-validation

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Split temporary dataset into validation (15%) and test (15%) datasets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

model = LogisticRegression(C=3792.690190732246, max_iter=1000, random_state=42, solver='saga')

# Train the model on the training set
model.fit(X_train, y_train)

# Evaluate on the validation set
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)



# Evaluate on the test set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Validation Accuracy: {val_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))
joblib.dump(model, 'logistic_regression_model.pkl')




# %%

X = combinedshuffled_df.drop(columns=['WL1'])
y = combinedshuffled_df['WL1']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest model with the specified hyperparameters
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    min_samples_split=10,
    min_samples_leaf=4,
    bootstrap=True,
    random_state=42
)

# Train the model on the training set
model.fit(X_train, y_train)

# Predict and evaluate accuracy on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {accuracy:.2f}")

# Split data for validation and testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Re-train the model on the new training set
model.fit(X_train, y_train)

# Evaluate on the validation set
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Set Accuracy: {val_accuracy:.2f}")

# Evaluate on the test set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Set Accuracy: {test_accuracy:.2f}")
joblib.dump(model, 'rf.pkl')
