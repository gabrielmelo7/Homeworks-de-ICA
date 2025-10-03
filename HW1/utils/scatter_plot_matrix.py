# Importing relevant libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def scatter_plot_matrix(df : pd.core.frame.DataFrame, target: str) -> pd.core.frame.DataFrame:
    """
    Shows a matrix of scatter plots between all the pairs of predictors with indications of class label 

    Args:
        df (pd.core.frame.DataFrame): The DataFrame on which we will operate
        target (String): The column of the DataFrame that is the target variable

    Returns:
        (pd.core.frame.DataFrame): The correlation matrix between all pairs of predictors.
    """
    # We can use pandas.corr() to get a DataFrame that is also a correlation matrix between all pairs of predictors
    corr_matrix = df.corr()

    # We can plot the scatter matrix with sns.pairplot
    sns.pairplot(df, hue="target", diag_kind="hist")
    plt.savefig("scatter_matrix.png")

    return corr_matrix




