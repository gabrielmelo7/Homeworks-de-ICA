import numpy as np
class ols_regression:

    def __init__(self):
        self.b = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Calculates the weights for linear regression given an input and desired output using ordinary least squares.

        Args:
            x (np.ndarray): The given input values.
            y (np.ndarray): The given output values.

        """
        ones = np.ones((x.shape[0], 1))
        x_aug = np.hstack([ones, x])

        x_transpose = x_aug.T
        x_term = x_transpose @ x_aug 
        try:
            x_term_inv = np.linalg.inv(x_term)
        except LinAlgError as lae:
            x_term_inv = np.linalg.pinv(x_term)
        
        self.b = x_term_inv @ x_transpose @ y

    def predict(self, x:np.ndarray) -> np.ndarray:
        """
        Calculates the prediction of the model to a given input.

        Args:
            x (np.ndarray): The inputs.

        Returns:
            (np.ndarray): The predictions of the model
        """
        if (self.b == None).any():
            print("The model has not been trained yet.")
        else:
            ones = np.ones((x.shape[0], 1))
            x_aug = np.hstack([ones, x])
            ols_predictions = x_aug @ self.b 
            return ols_predictions




