import numpy as np
import pandas as pd
from typing import Tuple, Callable 
from utils.rmse import root_mean_square_error
from utils.r_squared import r2 

def cross_validation(df: pd.DataFrame, target: str, model: object, number_folds: int) -> Tuple[np.float64, np.float64, np.float64, np.float64]:
    """
    Performs k fold cross validation for a model.

    args:
        df (pd.dataframe): the dataset on which cross validation will be performed.
        target (str): the target variable of the ML problem.
        model (Callable): the model to be evaluated.
        number_folds (int): the number of folds the dataset will be divided.

    returns:
        [(np.ndarray),(np.ndarray), (np.ndarray), (np.ndarray)]: The average root mean squared error, r_squared and the standard deviation of these values. 
    """

    df_len = len(df)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    fold_len = int(np.round(df_len/number_folds))

    folds = []

    for i in range(number_folds): 
        fold = df_shuffled.iloc[(fold_len*i):(fold_len*(i+1))]
        x = fold.drop(target, axis=1).values
        y = fold[target].values.reshape(-1,1)
        folds.append((x, y))

    mean_rmse = []
    mean_r2 = []

    for fold in range(number_folds):
        train_fold_x = np.vstack([f[0] for i, f in enumerate(folds) if i != fold])
        train_fold_y = np.vstack([f[1] for i, f in enumerate(folds) if i != fold])

        test_fold_x, test_fold_y = folds[fold]

        model.fit(train_fold_x, train_fold_y)

        model_predictions = model.predict(test_fold_x) 

        model_rmse = root_mean_square_error(model_predictions, test_fold_y)
        model_r2 = r2(model_predictions, test_fold_y)

        mean_rmse.append(model_rmse)
        mean_r2.append(model_r2)
        
    return np.mean(mean_rmse), np.std(mean_rmse), np.mean(mean_r2), np.std(mean_r2)
