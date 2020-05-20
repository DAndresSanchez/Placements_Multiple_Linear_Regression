# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:40:08 2020

@author: user
"""
#%% Libraries

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from scipy import stats

#%% Import data

# import data from csv file
filename = r"C:\Users\user\Documents\Google Drive\Business\Python\Job_MultRegression\Placement_Data_Full_Class.csv"
df = pd.read_csv(filename)

# initial EDA
print(df.head(10))
print(df.shape)
print(df.dtypes)
print(df.describe())
print(df.isna().sum())


#%% Data cleaning and preprocessing

# drop individuals not currently working
data = df.dropna(subset=['salary'])
# drop secondary education and non-relevant information
#data.drop(columns=['sl_no', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s', 'status'], inplace=True)
data.drop(columns=['sl_no', 'ssc_b', 'hsc_b', 'hsc_s', 'status'], inplace=True)



# final EDA
print(data.head(10))
print(data.shape)
print(data.dtypes)
print(data.describe())
print(data.isna().sum())

# reset index of final data
data.reset_index(inplace=True, drop = True)

# get dummy variables for categorical data
data = pd.get_dummies(data, drop_first=True)

# remove outliers
z_scores = stats.zscore(data)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 5).all(axis=1)
data = data[filtered_entries]


# split of data into train and test
X = data.drop(columns=['salary'])
y = data.salary

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%%  Graphs
## graphical representation of relevant numeric columns
#sns.pairplot(data, vars=['degree_p','etest_p','mba_p','salary'])
#
## salary boxplot
#plt.boxplot(data.salary)
#plt.show()


#%% Linear regression

regressor = LinearRegression()  
regressor.fit(X_train, y_train)
y_pred_reg = regressor.predict(X_test)

print('Linear Regressor:')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_reg))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_reg))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_reg)))
print('Error relative to mean:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred_reg))/y.mean()*100, 2), '%')

comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_reg})

comparison.plot(kind='bar',figsize=(10,8))
plt.title('Linear regression')
plt.xlabel('Person index')
plt.ylabel('Salary')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()



#%% Linear regression with MinMaxScaler
steps = [('scaler', MinMaxScaler()),
         ('regressor', LinearRegression())]

pipeline = Pipeline(steps)

pipeline.fit(X_train, y_train)
y_pred_pip = pipeline.predict(X_test)

print('Linear Regressor with MinMaxScaler:')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_pip))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_pip))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_pip)))
print('Error relative to mean:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred_pip))/y.mean()*100, 2), '%')

comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_pip})

comparison.plot(kind='bar',figsize=(10,8))
plt.title('Linear regression with MinMaxScaler')
plt.xlabel('Person index')
plt.ylabel('Salary')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  






