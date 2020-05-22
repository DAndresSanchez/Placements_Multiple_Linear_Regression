# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:19:15 2020

@author: user
"""


#%% Libraries

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

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

# drop secondary education and non-relevant information
data = df.drop(columns=['sl_no', 'ssc_b', 'hsc_b', 'hsc_s', 'salary'])

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

## remove outliers
#z_scores = stats.zscore(data)
#abs_z_scores = np.abs(z_scores)
#filtered_entries = (abs_z_scores < 5).all(axis=1)
#data = data[filtered_entries]

# split of data into train and test
X = data.drop(columns=['status_Placed'])
y = data.status_Placed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%%  Logistic Regression model

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# evaluation of the model
print('Accuracy: {}%'.format(round((y_test == y_pred).sum()/len(y_test)*100,2)))


y_pred_prob = logreg.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show();

roc_auc_score(y_test, y_pred_prob)



