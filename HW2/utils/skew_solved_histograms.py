import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def skewness_res_plot(transformed_data:pd.DataFrame):
    """
    This function is responsable for showing the results of aplication of yeojohnson´s skewness formula, plotting features´ histograms
    
    INPUT:
    transformed_data: dataframe containing yeojohnson transformation of original data

    OUTPUT:
    figure containing all the histograms
    """
    size = len(transformed_data.columns[:-1])
    c = np.sqrt(size)
    if c - int(c) !=0:
        c = int(c)
        l = c+1
    else:
        c = int(c)
        l = int(c)
    fig, axs = plt.subplots(l,c,figsize=(12,8))
    for index,column in enumerate(transformed_data.columns[:-1]):
        sns.histplot(transformed_data,x=column,ax=axs[index//3,index%3])
    plt.tight_layout(h_pad=0.5)
    fig.suptitle("skew-solved histograms")
    fig.subplots_adjust(top=0.9)
    return fig
