import numpy as np
import pandas as pd
from typing import Tuple, Callable 
from utils.scores import scores

def cross_validation(df: pd.DataFrame, target: str, model: object, number_folds: int) -> Tuple[np.float64, np.float64, np.float64]:
    """
    Performs k fold cross validation for a model.

    args:
        df (pd.dataframe): the dataset on which cross validation will be performed.
        target (str): the target variable of the ML problem.
        model (Callable): the model to be evaluated.
        number_folds (int): the number of folds the dataset will be divided.

    returns:
        [(np.ndarray),(np.float64), (np.ndarray)]: The average accuracy, precision and recall of these values. 
    """

    df_len = len(df)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    fold_len = int(np.round(df_len/number_folds))

    folds = []

    for i in range(number_folds): 
        fold = df_shuffled.iloc[(fold_len*i):(fold_len*(i+1))]
        x = fold.drop(target, axis=1).values
        y = fold[target].values
        folds.append((x, y))

    mean_accuracy = []
    mean_precision = []
    mean_recall = []

    for fold in range(number_folds):
        train_fold_x = np.concatenate([f[0] for i, f in enumerate(folds) if i != fold])
        train_fold_y = np.concatenate([f[1] for i, f in enumerate(folds) if i != fold])

        test_fold_x, test_fold_y = folds[fold]

        model.fit(train_fold_x, train_fold_y)

        model_predictions = model.predict(test_fold_x)

        accuracy, precision, recall = scores(model_predictions, test_fold_y)

        mean_accuracy.append(accuracy)
        mean_precision.append(precision)
        mean_recall.append(recall)
        
    return np.mean(mean_accuracy), np.mean(mean_precision), np.mean(mean_recall) 

