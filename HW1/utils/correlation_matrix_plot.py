# Importing relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
def correlation_matrix_plot(df : pd.core.frame.DataFrame, target: str) -> Tuple[pd.core.frame.DataFrame, sns.PairGrid]:
    """
    Shows a matrix of scatter plots between all the pairs of predictors with indications of class label 

    Args:
        df (pd.core.frame.DataFrame): The DataFrame on which we will operate
        target (String): The column of the DataFrame that is the target variable

    Returns:
        (pd.core.frame.DataFrame): The correlation matrix between all pairs of predictors.
        (sns.PairGrid): The scatter plot matrix of all pairs of predictos
    """
    # We can use pandas.corr() to get a DataFrame that is also a correlation matrix between all pairs of numerical predictors
    numerical_df = df.select_dtypes(include=np.number)
    corr_matrix = numerical_df.corr()

    # We can plot the scatter matrix with sns.pairplot
    grid = sns.pairplot(df, hue=target, diag_kind="hist")

    return corr_matrix, grid




