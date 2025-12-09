import numpy as np
import pandas as pd

from utils.rmse import root_mean_square_error
from utils.ridge_regression_module import Ridge_regression
from utils.r_squared import r2
from utils.pls_regression_module import train_pls, predict_pls

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


def cross_validation_pls(x_train : pd.DataFrame, y_train : pd.DataFrame, max_components = 9, folds = 10):
    np.random.seed(42)
    seq = np.arange(x_train.shape[0])
    np.random.shuffle(seq)

    x_train_random = x_train.copy().iloc[seq].reset_index(drop=True)
    y_train_random = y_train.copy().iloc[seq].reset_index(drop=True)

    indices_folds = np.array_split(x_train_random.index, folds)
    x_train_subsets = [x_train_random.loc[idx] for idx in indices_folds]
    y_train_subsets = [y_train_random.loc[idx] for idx in indices_folds]

    results = []

    for k in range(1, max_components + 1):
        RMSE_results = []
        r2_results = []

        for i in range(folds):
            x_val = x_train_subsets[i].copy()
            y_val = y_train_subsets[i].copy()

            x_train_folds = x_train_subsets[:i] + x_train_subsets[i+1:]
            y_train_folds = y_train_subsets[:i] + y_train_subsets[i+1:]

            x_treino = pd.concat(x_train_folds)
            y_treino = pd.concat(y_train_folds)

            mean_x = x_treino.mean()
            std_x = x_treino.std()
            mean_y = y_treino.mean()
            std_y = y_treino.std()

            std_x[std_x == 0] = 1
            std_y[std_y == 0] = 1

            x_treino_zs = (x_treino - mean_x) / std_x
            y_treino_zs = (y_treino - mean_y) / std_y

            x_val_zs = (x_val - mean_x) / std_x

            B_pls = train_pls(x_treino_zs.values, y_treino_zs.values, n_components=k)

            y_pred_zs = predict_pls(x_val_zs.values, B_pls)
            y_predicted = (y_pred_zs * std_y.values) + mean_y.values

            target = y_val.values.flatten()
            prediction = y_predicted.flatten()

            rmse_val = root_mean_square_error(prediction, target)
            r2_val = r2(prediction, target)
            
            RMSE_results.append(rmse_val)
            r2_results.append(r2_val)

        mean_rmse = np.mean(RMSE_results)
        mean_r2 = np.mean(r2_results)
        
        results.append((mean_rmse, mean_r2, k))

    final = sorted(results, key=lambda x: x[0], reverse=False)
    final_df = pd.DataFrame(final, columns=["Mean_RMSE", "Mean_R2", "n_components"])
    
    return final_df

if __name__ == "__main__":
    print("hello")
