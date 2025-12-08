import numpy as np
import pandas as pd
from utils.rmse import root_mean_square_error
from utils.ridge_regression_module import Ridge_regression
from utils.r_squared import r2

def cross_validation_ridge(x_train:pd.DataFrame,y_train:pd.DataFrame,lamb,folds=10):
    np.random.seed(42)
    seq = np.arange(x_train.shape[0])
    np.random.shuffle(seq)
    x_train_random = x_train.copy().iloc[seq]
    y_train_random = y_train.copy().iloc[seq]
    x_train_subsets = np.split(x_train_random,10)
    y_train_subsets = np.split(y_train_random,10)
    results = []
    for k in lamb:
        RMSE_results = []
        r2_results = []
        for i in range(folds):
            temp_x = np.concatenate(x_train_subsets[:i] + x_train_subsets[i+1:])
            temp_y = np.concatenate(y_train_subsets[:i] + y_train_subsets[i+1:])
            b= Ridge_regression.Ridge_regression_train(temp_x, temp_y,k)
            y_predicted = Ridge_regression.predicted_ridge_static(x_train_subsets[i],b)
            RMSE = root_mean_square_error(y_predicted, y_train_subsets[i]); RMSE_results.append(RMSE)
            R2 = r2(y_predicted, y_train_subsets[i]); r2_results.append(R2)
        mean_rmse = np.mean(RMSE_results); mean_r2 = np.mean(r2_results)
        results.append((mean_rmse,mean_r2,k,b))
    final = sorted(results, key=lambda x: x[1],reverse=True)
    final_df = pd.DataFrame(final,columns=["Mean_RMSE","Mean_r2","lamb","betha"])

    return final_df

if __name__ == "__main__":
    print("hello")