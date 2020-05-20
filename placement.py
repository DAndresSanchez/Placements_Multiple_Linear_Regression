# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:40:08 2020

@author: user
"""
#%% Libraries

import pandas as pd


#%% Import data

filename = r"C:\Users\user\Documents\Google Drive\Business\Python\Job_MultRegression\Placement_Data_Full_Class.csv"
data = pd.read_csv(filename)

print(data.head(10))
print(data.shape)
print(data.dtypes)
print(data.describe())
print(data.isnull().count())
print(data.isna().count())