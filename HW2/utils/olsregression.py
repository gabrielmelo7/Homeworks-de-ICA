import numpy as np

def ols_regression(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculates the weights for linear regression given an input and desired output using ordinary least squares.

    Args:
        x (np.ndarray): The given input values.
        y (np.ndarray): The given output values.

    Returns:
        (np.ndarray): The weights of the linear regression model.

    """
    ones = np.ones((x.shape[0], 1))
    x_aug = np.hstack([ones, x])

    x_transpose = x_aug .T
    x_term = x_transpose @ x_aug 
    try:
        x_term_inv = np.linalg.inv(x_term)
    except LinAlgError as lae:
        x_term_inv = np.linalg.pinv(x_term)
    
    b = x_term_inv @ x_transpose @ y


    return b

