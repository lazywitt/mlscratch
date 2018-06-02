''' These two functions perform linear regression for a dataset
one with single variable other with multiple.
Dataset was imported from sklearn samples'''
#basic linear regression
from sklearn import linear_model
from sklearn import datasets
import numpy as np
import pandas as pd
import statsmodels.api as sm
def linear_reg():
    data = datasets.load_boston() 
    df = pd.DataFrame(data.data, columns=data.feature_names)
    target = pd.DataFrame(data.target, columns=["MEDV"])
    X = df
    y = target["MEDV"]
    lm = linear_model.LinearRegression()
    model = lm.fit(X,y)
    predictions = lm.predict(X)
    print(predictions)[0:5]
    print(lm.score(X,y))
#linear regression in multiple variables
def mul_linear_reg():
    data = datasets.load_boston()  
    df = pd.DataFrame(data.data, columns=data.feature_names)
    target = pd.DataFrame(data.target, columns=["MEDV"])
    X = df[["RM", "LSTAT"]]
    y = target["MEDV"]
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
linear_reg()
mul_linear_reg()    
