import numpy as np
import pandas as pd
from typing import Tuple

def train_test_split(df: pd.DataFrame, target_variable: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates training and testing splits of a dataframe.

    Args:
        df (Pandas.DataFrame): The DataFrame to be split.
        target_variable (String): The target variable of the ML problem.

    Returns: 
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The training sets and testing sets.
    """

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_train_rows = int(len(df_shuffled)*0.8)

    train_set = df_shuffled.iloc[:n_train_rows]
    test_set = df_shuffled.iloc[n_train_rows:]

    x_train_set_df = train_set.drop(target_variable, axis=1)
    x_test_set_df = test_set.drop(target_variable, axis=1)

    y_train_set_df = train_set[target_variable]
    y_test_set_df = test_set[target_variable]

    x_train_set = np.array(x_train_set_df)
    x_test_set = np.array(x_test_set_df)
    y_train_set = np.array(y_train_set_df)
    y_test_set = np.array(y_test_set_df)

    return x_train_set, x_test_set, y_train_set, y_test_set




