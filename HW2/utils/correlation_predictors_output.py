import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def corr_pred_out(transformed_data:pd.DataFrame):
    """
    This function is responsable for showing the results correlation between SO2 - outcome - and the other predictors
    
    INPUT:
    transformed_data: dataframe containing yeojohnson transformation of original data

    OUTPUT:
    figure containing heatmap of predictors and our outcome - SO2 :fig

    dataframe containing the values of correspondency
    """


    corr_matrix = transformed_data.iloc[:,:-1].corr()
    fig, axs = plt.subplots(1,1,figsize=(4,8))
    sns.heatmap(corr_matrix[['SO2']].sort_values(by='SO2', ascending=False),
                annot=True,
                fmt='.2f',
                cmap='coolwarm')
    plt.tight_layout()
    plt.title("Correlation between predictors and outcome")
    plt.subplots_adjust(top=0.8)
    return fig, corr_matrix[['SO2']]