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

def boxcox_transform(data : pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Applies the Box-Cox transformation to each column of a DataFrame, handling non-positive values through a shift.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        Tuple(pd.DataFrame, np.ndarray, np.ndarray):
            - The new DataFrame with the transformed data.
            - An array with the lambda values found for each column.
            - An array with the offset (shift) values applied to each column.
    """

    number_columns = len(data.columns)
    shift = np.zeros(number_columns)
    lambda_values = np.zeros(number_columns)

    transformed_data = data.copy()
    for i, col in enumerate(transformed_data.columns):
        
        min_value = transformed_data[col].min()
        if min_value <= 0:
            shift[i] = abs(min_value) + 1 
        
        shifted_data = transformed_data[col] + shift[i]

        # Returns the column after the transformation and the lambda that best fits
        transformed_data[col], lambda_values[i]  = stats.boxcox(shifted_data)

    return transformed_data, lambda_values, shift

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
