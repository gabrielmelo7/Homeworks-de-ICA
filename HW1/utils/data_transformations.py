import numpy as np
import pandas as pd
from scipy import stats

def standard_zscore(data : pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the data in a DataFrame by subtracting the mean and dividing by the standard deviation (Z-score).

    Args:
        data (pd.DataFrame): The input DataFrame with numerical data.

    Returns:
        pd.DataFrame: A new DataFrame with the scaled data, preserving original index and columns.
    """

    return (data - data.mean()) / data.std()

def boxcox_transform(data : pd.DataFrame) -> pd.DataFrame:
    """
    Applies the Box-Cox transformation to each column of the DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with the transformed data.
    """

    transformed_data = data.copy()
    for col in transformed_data.columns:
        # Returns the column after the transformation and the lambda that best fits
        transformed_data[col], _ = stats.boxcox(transformed_data[col])
    return transformed_data

def yeojohnson_transform(data : pd.DataFrame) -> pd.DataFrame:
    """
    Applies the Yeo-Johnson transformation to each column of the DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with the transformed data.
    """

    transformed_data = data.copy()
    for col in transformed_data.columns:
        # Returns the column after the transformation and the lambda that best fits
        transformed_data[col], _ = stats.yeojohnson(transformed_data[col])
    return transformed_data

def spatial_sign_transform(data : pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the data using the Euclidean norm for each column.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with the normalized data.
    """

    return data / np.linalg.norm(data.values, axis=0)
