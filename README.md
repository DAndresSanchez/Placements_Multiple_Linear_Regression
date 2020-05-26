# Work Placements analysis through Multiple Linear Regression

## Data source
https://www.kaggle.com/benroshan/factors-affecting-campus-placement

## Description
### Salary prediction based on grades and education
- File: salary_linear_regression.py 
- Analysis of the data to predict salary from education.
- Use of Multiple Linear Regression. 
- Comparison with Ridge and Lasso.

### Work placements prediction based on education
- File: placement_classifier.py 
- Analysis of the data to predict the occurrence of a work placement.
- Use and comparison of Random Forest Classifier, Logistic Regression and KNeighbors Classifier.

## Revised skills
- Data import from .csv files
- Data cleaning with pandas
- Visualisation in Matplotlib 
- Supervised learning: classifiers with scikit-learn

## Results

### Salary prediction based on grades and education

- Salary prediction vs. Actual salary

![Salary comparison](.\images\salary_prediction.png)

- Feature selection with Lasso

![Feature selection](.\images\feature_selection.png)





### Work placements prediction based on education
ROC curves for:

- Logistic Regression

![Logistic Regression](.\images\ROC_LogReg.png)

- K-Nearest Neighbors

![KNNeighbors](.\images\ROC_knn.png)

- Random Forests

![Random Forests](.\images\ROC_RandomForests.png)