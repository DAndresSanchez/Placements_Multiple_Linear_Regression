# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:40:08 2020

Work placements salary prediction based on grades and education.
Use of Multiple Linear Regression. Comparison with Ridge and Lasso.

@author: David Andres
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# Import data from csv file
filename = r".\data\Placement_Data_Full_Class.csv"
df = pd.read_csv(filename)

# Initial EDA
print(df.head(10))
print(df.shape)
print(df.dtypes)
print(df.describe())
print(df.isna().sum())

# Data cleaning and pre-processing
# Drop individuals not currently working
data = df.dropna(subset=['salary'])
# Drop secondary education and non-relevant information
data.drop(columns=['sl_no', 'ssc_b', 'hsc_b', 'hsc_s', 'status'], inplace=True)

# final EDA
print(data.head(10))
print(data.shape)
print(data.dtypes)
print(data.describe())
print(data.isna().sum())

# Reset index of final data
data.reset_index(inplace=True, drop=True)

# Get dummy variables for categorical data
data = pd.get_dummies(data, drop_first=True)

# Remove outliers
z_scores = stats.zscore(data)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 5).all(axis=1)
data = data[filtered_entries]

# Split of data into train and test
X = data.drop(columns=['salary'])
y = data.salary
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Visualisation of relevant numeric columns
sns.pairplot(data, vars=['degree_p', 'etest_p', 'mba_p', 'salary'])

# Salary box-plot
plt.boxplot(data.salary)
plt.show()

# Linear regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred_reg = regressor.predict(X_test)

print('Linear Regressor:')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_reg))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_reg))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_reg)))
print('Error relative to mean:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred_reg)) / y.mean() * 100, 2),
      '%')
print('Score: ', regressor.score(X_test, y_test))

comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_reg})
comparison.plot(kind='bar', figsize=(10, 8))
plt.title('Linear regression')
plt.xlabel('Person index')
plt.ylabel('Salary')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])

# Cross validation
cv_results = cross_val_score(regressor, X, y, cv=5)
print(cv_results)
np.mean(cv_results)

# Linear regression with MinMaxScaler
steps = [('scaler', MinMaxScaler()),
         ('regressor', LinearRegression())]

pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
y_pred_pip = pipeline.predict(X_test)

print('Linear Regressor with MinMaxScaler:')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_pip))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_pip))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_pip)))
print('Error relative to mean:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred_pip)) / y.mean() * 100, 2),
      '%')
print('Score: ', pipeline.score(X_test, y_test))

comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_pip})
comparison.plot(kind='bar', figsize=(10, 8))
plt.title('Linear regression with MinMaxScaler')
plt.xlabel('Person index')
plt.ylabel('Salary')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

cv_results = cross_val_score(pipeline, X, y, cv=5)
print(cv_results)
np.mean(cv_results)

# Ridge
ridge = Ridge(alpha=0.1, normalize=True)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge.score(X_test, y_test)

# Lasso
lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso.score(X_test, y_test)

# Lasso for feature selection
names = X.columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation=90)
_ = plt.ylabel('Coefficients')
_ = plt.grid(linestyle='-', linewidth=0.5)
plt.show()

comparison = pd.DataFrame({'Feature': names, 'Lasso Coefficient': lasso_coef})
comparison.plot(kind='bar', figsize=(10, 8))
plt.title('Lasso for feature selection')
plt.xlabel('Feature')
plt.ylabel('Coefficients')
plt.xticks(range(len(names)), names, rotation=90)
plt.grid(linestyle='-', linewidth=0.5)
plt.show()

# Summary of selected features and discarded features
non_selected_feat = names[abs(lasso_coef) == 0]
selected_feat = names[abs(lasso_coef) != 0]

print('total features: {}'.format(len(names)))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {} - {}'.format(len(non_selected_feat), non_selected_feat[0]))
