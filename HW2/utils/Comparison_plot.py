import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Ridge_regression import predicted_ridge


def comparison_predicted_real(b,y_real,dataset,**kwargs):
    """
    Input:
    y_real -> Real values of the same x
    dataset -> dataset used for calculating the Predicted values with or without the Air Quality parameter

    Output:
    fig -> figure containing the comparison plot
    """
    dataset= dataset.copy()

    if dataset.shape[1] > 8:
        classification = dataset.iloc[:,-1]
        dataset.drop(columns=dataset.columns[-1])

    y_pred = predicted_ridge(dataset.iloc[:,:-1],b)

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
    