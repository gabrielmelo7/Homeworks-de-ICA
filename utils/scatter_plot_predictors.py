# Importing relevant libraries
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def scatter_plot_predictors(df : pd.core.frame.DataFrame, target: str, number_of_plots: int) -> None:
    """
    Shows a matrix of scatter plots between the most related variables of the DataFrame

    Args:
        df (pd.core.frame.DataFrame): The DataFrame on which we will operate
        target (String): The column of the DataFrame that is the target variable
        number_of_plots (int): How many scatter plots are going to be shown

    Returns:
        None
    """
    
    # Given that we do not know how many attributes the DataFrame has it is better to only display the plots of a given set of variables. 
    # We decide which of them are going to be shown using correlation coefficient and then select the givn number of attributes
    corr_matrix = df.corr()
    corr_list = corr_matrix[target].sort_values(ascending=False)
    attributes = corr_list.index[:number_of_plots].tolist()

    # Then finally we can do the scatter_matrix 
    scatter_matrix(df[attributes], figsize=(12,8))
    plt.savefig("scatter_matrix.png")




