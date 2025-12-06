import pandas as pd
import numpy as np
from rmse import root_mean_square_error
from r_squared import r2
from sklearn.linear_model import Ridge

def Ridge_regression_train(x, y,lamb):
    #adding Bias:
    x = x.copy()
    y = y.copy()
    x = np.c_[np.ones((len(x),1)),x]
    #Calculating the coefficients:
    I = np.identity(x.shape[1])
    I[0,0] = 0 #not affect the B0
    B = np.linalg.inv((x.T @ x) + lamb*I) @ x.T @y
    return B, x@B

def predicted_ridge(x,b):
    x = x.copy()
    x = np.c_[np.ones((len(x),1)),x]
    return x@b


def ridge_model_sklean(x_train,y_train,x_test,y_test,lamb):
    x_test = x_test.iloc[:,:-1].copy()
    ridge_model = Ridge(alpha=lamb)
    ridge_model.fit(x_train,y_train)
    y_pred = ridge_model.predict(x_test)
    RMSE = root_mean_square_error(y_pred,y_test)
    R2=r2(y_pred,y_test)
    return RMSE, R2