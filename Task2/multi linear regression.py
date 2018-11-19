# Multiple Linear Regression

# Importing the libraries

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_boston

# Importing the dataset
boston = load_boston()
dataset = pd.DataFrame(boston.data)
#dataset.columns = ['a','b','TradeDate','TradeTime','CumPnL','DailyCumPnL','RealisedPnL','UnRealisedPnL',
#'CCYCCY','CCYCCYPnLDaily','Position','CandleOpen','Price']

X = dataset.iloc[:, :-1].values
A = dataset.iloc[:, :-1]
y = dataset.iloc[:, 12].values
B = dataset.iloc[:, 12]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 75)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
# to get the accuracy of the dataset
y_pred = regressor.predict(X_test)
from sklearn.metrics import r2_score 
print("------------------------")
print(r2_score(y_test,y_pred))

# Building the optimal model using Backward Elimination
# define the significane of column
#import statsmodels.formula.api as sm
#X = np.append(arr = np.ones((506, 1)).astype(int), values = X, axis = 1) # to give a bias y = mx+c where c will have 
#X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11]]
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#print(regressor_OLS.summary())
#X_opt = X[:, [0, 1, 3, 4, 5]]
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#print(regressor_OLS.summary())
#X_opt = X[:, [0, 3, 4, 5]]
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#print(regressor_OLS.summary())
#X_opt = X[:, [0, 3, 5]]
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#print(regressor_OLS.summary())
#X_opt = X[:, [0, 3]]
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#print(regressor_OLS.summary())