import numpy as np
import matplotlib.pyplot as plt

def pca_scree_plot(eigenvalues:float, **kwargs):
    default = {
        'figsize':(12,8),
        'barplot_color':'#24a0ff', #barplot
        'markersize':5, #plot
        'marker': 'o',  #plot
        'markerfacecolor':'red',    #plot
        'markeredgecolor':'red',    #plot
        'plot_color':'Black',   #Barplot
        'weight':'600', #text
        'edgecolor':'black' #Barplot
    }

    config = {**default, **kwargs}

    fig,axs = plt.subplots(figsize=config['figsize'])
    size = len(eigenvalues)
    X = np.arange(1,size+1)

    sum = np.sum(np.abs(eigenvalues))
    axs.bar(X,eigenvalues,color=config['barplot_color'],edgecolor=config['edgecolor'])
    axs.plot(X,eigenvalues,color=config['plot_color'],markersize=config['markersize'],marker=config['marker'],markerfacecolor=config['markerfacecolor'],markeredgecolor=config['markeredgecolor'],label='Contribution')
    axs.set_xlabel("Components in Descending Order",fontsize=12)
    axs.set_ylabel("Eigenvalue",fontsize=12)
    for index, eigenvalue in enumerate(eigenvalues):
        axs.text(index+1,eigenvalue+0.1,
                 f'{eigenvalue/sum * 100 :.2f}%',weight=config['weight'])
    plt.grid(color='gray',alpha=0.6)
    axs.set_title('Principal Components Contribuition')
    axs.set_xticks(X)
    axs.legend()
    return fig
    