import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def pca_calculation(df:pd.DataFrame,plot:bool)->tuple[list:np.array,list:np.array,plt.figure]:
    """
    With the Covariance Matrix, calculates the eigenvalues and eigenvectors using numpy,
    giving the PC1 and PC2 and the eigenvalues of each.

    Args:
    df: standardized dataFrame from the original, keep the classification.
    plot: True or False, define if plots or not the PCA 2D visualization

    Return:
    (Eigenvalues, Eigenvectors) ordered in descending order, Eigenvalues[0] will be the highest and
    Eigenvalues[1] the second heighest. Eigenvectors index corresponds to the Eigenvalues index.

    """
    #Copy without label:
    df_copy = df.iloc[:,:-1]

    covariance_matrix = df_copy.corr()

    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    # Now we need to know wich one is the highest and the second highest:

    index_PC1 = np.argmax(eigenvalues)
    copy = eigenvalues.copy()
    copy[index_PC1] = -np.inf
    index_PC2 = np.argmax(copy) 

    #Defining the vectors by the index of the highest´s eigenvalues:
    PC1 = eigenvectors[index_PC1]
    PC2 = eigenvectors[index_PC2]
    #defining Eigenvalues for PC´s
    PC1_EIGEN = eigenvalues[index_PC1]
    PC2_EIGEN = eigenvalues[index_PC2]
    #Calculating the values of the original df in the pca´s plan
    score_PC1 = PC1.dot(df_copy.T)
    score_PC2 = PC2.dot(df_copy.T)

    df_pc = pd.DataFrame({'PC1':score_PC1, 'PC2':score_PC2}); df_pc['Air Quality'] = df['Air Quality']

    fig,axs = plt.subplots(figsize=(12,8),dpi=300)

    """
    Plotting configurations:
    """
    MAX_SIZE = 1.0
    zeros = np.zeros(len(PC1))
    scale_factor = MAX_SIZE/np.max(PC1)
    C = np.sqrt(PC1[:]**2 + PC2[:]**2) #Colormaping function
    PC1_RESIZED = PC1*scale_factor
    PC2_RESIZED = PC2*scale_factor
    sns.scatterplot(df_pc,x='PC1', y='PC2', hue='Air Quality', palette='muted', s=80, alpha=0.8)
    axs.quiver(zeros,zeros,PC1_RESIZED,PC2_RESIZED,C, scale=1, cmap='inferno',scale_units='xy',headwidth=5,width=0.0025)
    for i in range(len(PC1)):
        axs.text(PC1[i]*scale_factor,PC2[i]*scale_factor,f'{list[i][:15]}',fontsize=8,ha='center',va='center',color='white',
             bbox=dict(boxstyle='round,pad=0.1',
                       facecolor='black',
                       alpha=0.5))
    
    if plot:
        plt.show()
    
    return ([PC1_EIGEN,PC2_EIGEN],[PC1,PC2],fig)