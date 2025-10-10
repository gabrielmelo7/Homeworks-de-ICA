import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import mpl_axes_aligner
def pca_biplot(df_scores:pd.DataFrame, eigenvectors:np.array,feature_labels, **kwargs):
    default = {
        'plotlike':'both'
    }
    config = {**default, **kwargs}
    fig, axs = plt.subplots(figsize=(12,8))
    if config['plotlike'] == 'both' or config['plotlike'] == 'scores':
        sns.scatterplot(df_scores, x='PC1', y='PC2', hue='Air Quality',palette='Pastel1',s=80,alpha=1)
    #Define the x and y of our vectors:
    if config['plotlike'] == 'loadings' or config['plotlike'] == 'both':
        PC1 = eigenvectors[0]
        PC2 = eigenvectors[1]
        #Colormap Definition:
        C = np.sqrt(PC1**2 + PC2**2)
        ax2 = axs.twinx().twiny()
        zeros = np.zeros(PC1.shape)
        ax2.quiver(zeros,zeros,PC1,PC2,C, cmap='Dark2',scale_units='xy',scale=1,angles='xy')
        ax2.set_xlim(-1.1,1.1)
        ax2.set_ylim(-1.1,1.1)
        #Features:
        

        for i in range(len(eigenvectors[0])):
            ax2.text(PC1[i]*1.1, PC2[i]*1.15, s=f'{feature_labels[i]}', fontsize=10, weight='800')

        mpl_axes_aligner.align.xaxes(axs, 0, ax2, 0, 0.5)
        #align y = 0 of ax and ax2 with the center of figure
        mpl_axes_aligner.align.yaxes(axs, 0, ax2, 0, 0.5)
    return fig