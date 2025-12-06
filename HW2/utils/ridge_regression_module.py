import pandas as pd
import numpy as np
from utils.rmse import root_mean_square_error
from utils.r_squared import r2
from sklearn.linear_model import Ridge

class Ridge_regression:
    def __init__(self,x_train,y_train,x_test,y_test,lamb=np.arange(0,80,5),folds=10):
        from utils.crossvalidation import cross_validation_ridge
        self.b_lambda = cross_validation_ridge(x_train,y_train, lamb,folds).lamb.iloc[0]
        self.ridge_bethas = self.Ridge_regression_train(x_train,y_train,self.b_lambda)
        self.ridge_model = Ridge(alpha=self.b_lambda)
        self.ridge_model.fit(x_train,y_train)
        self.r2_scratch = self.rmse_r2_model_scratch(y_test,x_test)[1]
        self.r2_builtin = self.rmse_r2_model_sklean(y_test,x_test)[1]

    @staticmethod
    def Ridge_regression_train(x, y,lamb):
        #adding Bias:
        x = x.copy()
        y = y.copy()
        x = np.c_[np.ones((len(x),1)),x]
        #Calculating the coefficients:
        I = np.identity(x.shape[1])
        I[0,0] = 0 #not affect the B0
        B = np.linalg.inv((x.T @ x) + lamb*I) @ x.T @y
        return B
    
    @staticmethod
    def predicted_ridge_static(x,b):
        x = x.copy()
        if (x.shape[1] > 8):
            x = x.drop(columns=x.columns[-1])
        x = np.c_[np.ones((len(x),1)),x]
        return x@b
    

    def predicted_ridge(self,x,*args):
        if len(args) > 0:
            b = args[0]
        else:
            b = self.ridge_bethas
        return self.predicted_ridge_static(x,b)
    


    def rmse_r2_model_sklean(self,y_test,x_test):
        if (x_test.shape[1] > 8):
            x_test = x_test.drop(columns=x_test.columns[-1])
        else:
            x_test = x_test.copy()
        y_pred = self.ridge_model.predict(x_test)
        RMSE = root_mean_square_error(y_pred,y_test)
        R2=r2(y_pred,y_test)
        return RMSE, R2

    def rmse_r2_model_scratch(self,y_test,x_test):
        y_pred = self.predicted_ridge(x_test)
        RMSE = root_mean_square_error(y_pred,y_test)
        R2=r2(y_pred,y_test)
        return RMSE, R2
    
if __name__ == "__main__":
    print("alo")