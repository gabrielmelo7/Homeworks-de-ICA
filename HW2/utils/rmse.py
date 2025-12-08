import numpy as np

def root_mean_square_error(prediction: np.ndarray, target: np.ndarray) -> np.float64:
    """
    Calculates the mean squared error of the prediction of a model.

    Args:
        prediction (np.ndarray): The predicted values of the model.
        target (np.ndarray): The target values.

    Returns:
        (np.float64): The rmse of the model.

    """
    mat_diff = target - prediction
    mse = np.mean(mat_diff**2) 

    return np.sqrt(mse)
