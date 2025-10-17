import numpy as np
import pandas as pd

def pca_calculation(df:pd.DataFrame,save:bool = 0)->tuple[list:np.array,list:np.array,pd.DataFrame]:
    """
    With the Covariance Matrix, calculates the eigenvalues, eigenvectors and the scores using numpy

    Args:
    df: standardized dataFrame from the original, keep the classification.
    save: True or False, define True to save the csv with the PCA scores

    Return:
    (Eigenvalues, Eigenvectors, Data_frame_with_scores). Returns the Eigenvalues and vectors in descending order.

    """
    #Copy without label:
    df_copy = df.iloc[:,:-1]

    covariance_matrix = df_copy.corr()

    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    # Now we need to know wich one is the highest and the second highest:

    indexes = np.flip(np.argsort(eigenvalues)) #Gets descending order
    eigenvalues = eigenvalues[indexes]
    eigenvectors = eigenvectors[:,indexes]

    #Defining the vectors by the index of the highest´s eigenvalues:
    PC1 = eigenvectors[:,0]
    PC2 = eigenvectors[:,1]
    #Calculating the values of the original df in the pca´s plan
    score_PC1 = PC1.dot(df_copy.T)
    score_PC2 = PC2.dot(df_copy.T)

    df_pc = pd.DataFrame({'PC1':score_PC1, 'PC2':score_PC2}); df_pc['Air Quality'] = df['Air Quality']
    if save:
        df_pc.to_csv('PCA_Score.csv',index=False)
    
    return eigenvalues, eigenvectors, df_pc