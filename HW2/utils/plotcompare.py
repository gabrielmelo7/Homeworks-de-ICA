import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def plotcompare(y_real: np.ndarray, y_predict: np.ndarray, classes: np.ndarray) -> plt.Figure:
    """
    Plots a comparison between the predictions of models and the true values.

    Args:
        y_real (np.ndarray): The real target values of the dataset.
        y_predict (np.ndarray): The predicted values of the model. 
        classes (np.ndarray): The classes of the instances.
    
    Returns: 
        (plt.Figure): A comparative plot of the predictions.
    """

    fig, ax = plt.subplots(figsize=(10,6))

    sns.scatterplot(x=y_real, y=y_predict, hue=classes, palette="tab10", ax=ax) 

    min_val = min(y_real.min(), y_predict.min())
    max_val = max(y_real.max(), y_predict.max()) 

    sns.lineplot(x=[min_val, max_val], y=[min_val, max_val],
            linestyle='--',
            color='black',
            label="Previsão perfeita",
            ax=ax
            )
    plt.title("Comparação entre valores reais e previsões do modelo")
    plt.xlabel("Valores reais")
    plt.ylabel("Valores previstos")
    plt.legend()
    
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)


    return fig
