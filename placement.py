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
data.drop(columns=['sl_no', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s', 'status'], inplace=True)

# final EDA
print(data.head(10))
print(data.shape)
print(data.dtypes)
print(data.describe())
print(data.isna().sum())

# reset index of final data
data.reset_index(inplace=True, drop = True)

# graphical representation of relevant numeric columns
#sns.pairplot(data, vars=['degree_p','etest_p','mba_p','salary'])

# get dummy variables for categorical data
data = pd.get_dummies(data, drop_first=True)

#%% Linear regression



# split of data into train and test
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.salary, test_size=0.2, random_state=0)



regressor = LinearRegression()  
regressor.fit(X_train, y_train)













