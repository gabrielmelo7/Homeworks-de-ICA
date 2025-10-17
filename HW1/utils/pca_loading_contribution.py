import matplotlib.pyplot as plt
import numpy as np

def pca_loading_contribution(eigenvectors: np.array,pc:int,features):
    fig, axs = plt.subplots(figsize=(12,8))
    vector = eigenvectors[pc-1]
    sum = np.sum(np.abs(vector))
    formated_values = [f'{np.abs(value/sum) *100:.1f}%' for value in vector]
    fig.suptitle(f"PC{pc} Features Contribution")
    bars = axs.barh(features, np.abs(vector/sum),edgecolor='black')
    axs.set_ylabel('Features')
    axs.set_xlabel('Contribution')
    axs.bar_label(bars,labels=formated_values,padding=3,fmt='%.2f%%')
    fig.tight_layout()
    return fig