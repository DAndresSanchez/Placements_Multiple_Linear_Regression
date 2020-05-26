# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:19:15 2020

Work placements prediction based on education.
Comparison of Random Forest Classifier, Logistic Regression and KNeighbors Classifier.

@author: David Andres
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

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
# Drop secondary education and non-relevant information
data = df.drop(columns=['sl_no', 'ssc_b', 'hsc_b', 'hsc_s', 'salary'])

# Final EDA
print(data.head(10))
print(data.shape)
print(data.dtypes)
print(data.describe())
print(data.isna().sum())

# Reset index of final data
data.reset_index(inplace=True, drop=True)

# Get dummy variables for categorical data
data = pd.get_dummies(data, drop_first=True)

# Split of data into train and test
X = data.drop(columns=['status_Placed'])
X = scale(X)
y = data.status_Placed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# Evaluation of the model
print('Accuracy: {}%'.format(round((y_test == y_pred).sum() / len(y_test) * 100, 2)))

# Plot ROC curve
y_pred_prob = logreg.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()

# Get ROC_AUC score
roc_auc_score(y_test, y_pred_prob)

# Cross validation
cv_scores = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')
print(cv_scores)

# Hyperparameter tuning for Logistic Regression
dual = [True, False]
max_iter = [100, 110, 120, 130, 140]
param_grid = dict(dual=dual, max_iter=max_iter)

# use GridSearchCV with LogisticRegression and the chosen hyperparameters
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
logreg_cv.fit(X, y)

# Summary of results
print("Best: %f using %s" % (logreg_cv.best_score_, logreg_cv.best_params_))

# Hyperparameter tuning for K-Neighbors Classifier
n_neigh = range(2, 10, 2)
param_grid = dict(n_neighbors=n_neigh)

# Use GridSearchCV with KNeightbors and the chosen hyperparameters
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X, y)

# Summary of results
print("Best: %f using %s" % (knn_cv.best_score_, knn_cv.best_params_))

# K-Neighbors Classifier

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Evaluation of the model
print('Accuracy: {}%'.format(round((y_test == y_pred).sum() / len(y_test) * 100, 2)))

# Plot ROC curve
y_pred_prob = knn.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='KNeighbors Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNeighbors ROC Curve')
plt.show()

# Get ROC_AUC score
roc_auc_score(y_test, y_pred_prob)

# Cross validation
cv_scores = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')
print(cv_scores)

# Hyperparameter tuning for Random Forests
n_estim = range(20, 200, 20)
param_grid = dict(n_estimators=n_estim)

# Use GridSearchCV with KNeightbors and the chosen hyperparameters
clf = RandomForestClassifier()
clf_cv = GridSearchCV(clf, param_grid, cv=5)
clf_cv.fit(X, y)

# Summary of results
print("Best: %f using %s" % (clf_cv.best_score_, clf_cv.best_params_))

# Random Forests Classifier
clf = RandomForestClassifier(n_estimators=60)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation of the model
print('Accuracy: {}%'.format(round((y_test == y_pred).sum() / len(y_test) * 100, 2)))

# Plot ROC curve
y_pred_prob = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Random Forests Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forests ROC Curve')
plt.show()

# Get ROC_AUC score
roc_auc_score(y_test, y_pred_prob)

# Cross validation
cv_scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
print(cv_scores)
