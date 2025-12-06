import numpy as np

def r2(prediction: (np.ndarray), target: (np.ndarray)) -> np.float64:
    """
    Calculates the r squared of a model.

    Args:
        prediction (np.ndarray): The predicted values of the model.
        target (np.ndarray): The target values.

    Returns:
        (np.float64): The r_squared of the model.

    """
    mat_res_var = target - prediction
    ss_res = np.sum(mat_res_var**2)
    mat_tot_var = target - np.mean(target)
    ss_tot = np.sum(mat_tot_var**2)

    ss_term = ss_res/ss_tot

    return 1 - ss_term


