import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

def columns_plot(df: pd.core.frame.DataFrame): #-> matplotlib.figure.Figure:
    """
    Plots the histograms and box-plots of all the variables in the DataFrame

    Args:
        df (pd.core.frame.DataFrame): The DataFrame that contains the data to be ploted

    Returns:
        (matplotlib.figure.Figure): The histograms and box-plots we obtained
    """
    # We begin by selecting columns with numerical data only. With the number of columns we can determine
    # The shape of the figure to be ploted
    treated_df = df.select_dtypes(include=np.number)
    df_columns = treated_df.shape[1]
    rows = int(df_columns / 3)
    
    # With that done we can create the figures
    fig_hist, axs_hist = plt.subplots(rows, rows, figsize=(12,8))
    fig_box, axs_box= plt.subplots(rows, rows, figsize=(12,8))

    
    # Now we iterate on each axe and plot the histogram and the violin plot for that column
    iterator = 0
    for i in range(rows):
        for j in range(rows):
            sns.histplot(data=df, x=df.columns[iterator], ax=axs_hist[i,j], kde=True)
            sns.boxplot(data=df, x = df.columns[iterator], ax=axs_box[i,j])
            iterator += 1

    # Now we can add the titles and finish the function
    fig_hist.suptitle("Histogramas dos preditores")
    fig_box.suptitle("Box-plots dos preditores")
    fig_hist.tight_layout()
    fig_box.tight_layout()

    return fig_hist, fig_box 

def unconditional_stats(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """
    Calculates the mean, standard deviation and skewness of every numerical column in a data frame

    Args:
        df (pd.core.frame.DataFrame): The DataFrame on which we will calculate the values

    Returns:
        (pd.core.frame.DataFrame): A DataFrame which contains the calculated values
    """

    #   We begin by selecting columns with numerical data only. Then we create a dictionary on which
    # we will store the results of: mean, standard deviation and skewness
    df_numerical = df.select_dtypes(include=np.number)
    results = {}

    # For each column we calculate the metrics mentioned before
    for column in df_numerical.columns:
        results[column] = {
                "mean": np.mean(df_numerical[column]),
                "std": np.std(df_numerical[column]),
                "skewness": skew(df_numerical[column])
                }

    # Now we return a DataFrame which contains the metrics as rows and the predictors as columns
    return pd.DataFrame(results)

# Getting the names from the given arguments
data_path = sys.argv[1] 
csv_name = sys.argv[2]
hist_name = sys.argv[3]
box_name = sys.argv[4]

# First we read the data
air_data = pd.read_csv(data_path)

# Now we can apply the functions and obtain our results.
histograms, boxplots  = columns_plot(air_data)
metrics_df = unconditional_stats(air_data)
metrics_df.to_csv(os.path.join("./results/class_unconditional", csv_name))
histograms.savefig(os.path.join("./results/class_unconditional", hist_name))
boxplots.savefig(os.path.join("./results/class_unconditional", box_name))

