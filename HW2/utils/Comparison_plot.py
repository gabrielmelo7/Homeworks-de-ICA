import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.ridge_regression_module import Ridge_regression as rr


def comparison_predicted_real(b,y_real,dataset,**kwargs):
    """
    Input:
    y_real -> Real values of the same x
    dataset -> dataset used for calculating the Predicted values with or without the Air Quality parameter

    Output:
    fig -> figure containing the comparison plot
    """
    dataset= dataset.copy()

    classification = dataset.iloc[:,-1]

    map_values = {
        'Hazardous': 4,
        'Poor': 3,
        'Moderate': 2,
        'Good': 1
        }
    dataset["Air Quality"] = dataset["Air Quality"].map(map_values)

    y_pred = rr.predicted_ridge_static(dataset,b)

    default={
        'classification':True
    }

    df = pd.DataFrame({
        'Real':y_real,
        "Predicted":y_pred,
        "Air Quality":classification #picks up only the labels for Air Quality
    })

    config={**default,**kwargs}
    
    fig, axs= plt.subplots(1,1,figsize=(12,8))
    if config["classification"]:
        sns.scatterplot(df, x='Predicted',y='Real',hue='Air Quality',s=60,alpha=0.6)
    else:
        sns.scatterplot(df, x='Predicted',y='Real',s=60,alpha=0.6)
    sns.regplot(df,x='Predicted',y="Real",scatter=False,label="Model´s Regression",color='red',line_kws={'linewidth':2})
    axs.legend(loc='upper left')
    axs.set_ylabel("Real Y")
    axs.set_xlabel("Predicted Y")
    fig.suptitle("Predicted x Real")
    fig.tight_layout()
    return axs


def comparison_scratch_buitin(r2_scratch, r2_builtin): 
    """
    This function aims to show the bar plot showing the differences between r2´s


    Input:
    r2_scratch -> the value of R2 for from scratch implementation
    r2_builtin -> the value of R2 for built-in implementation
    Output:
    fig -> figure containing the comparison plot
    """
    fig, axs = plt.subplots(1,1,figsize=(12,8))
    df = pd.DataFrame({
        "ridge_model" : ['Scratch Ridge','Built-in Ridge'],
        "R2_score": [r2_scratch,r2_builtin]

    })
    sns.barplot(df,x='ridge_model',y="R2_score",ax=axs)
    axs.bar_label(axs.containers[0],fmt="%.4f")
    axs.set_xlabel("Ridge implementations")
    axs.set_ylabel("Value for R^2")
    fig.suptitle("Comparison from Scratch and Built-in")
    fig.tight_layout()
    return axs
    

def skewness_res_plot(transformed_data:pd.DataFrame):
    """
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


def correlation_plot(transformed_data:pd.DataFrame):
    map_values = {
    'Hazardous': 4,
    'Poor': 3,
    'Moderate': 2,
    'Good': 1
    }
    transformed_data = transformed_data.copy()
    transformed_data["Air Quality"] = transformed_data["Air Quality"].map(map_values)
    corr_matrix = transformed_data.corr()
    fig, axs = plt.subplots(1,1,figsize=(8,4))
    sns.heatmap(corr_matrix[['SO2']].sort_values(by='SO2', ascending=False).T,
                annot=True,
                fmt='.2f',
                cmap='coolwarm')
    fig.tight_layout(rect=(0,0,1,1))
    fig.subplots_adjust(top=0.9)
    return fig