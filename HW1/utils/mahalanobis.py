import numpy as np


def mahalanobis_distances(data):
    """Calculates the Mahalanobis distance for each observation.

    This function computes the Mahalanobis distance of each observation (row)
    in the input data array from the multivariate mean of the dataset.

    Args:
        data (numpy.ndarray): A 2D array of shape (n_samples, n_features)
            where n_samples is the number of observations (e.g., time-series
            points) and n_features is the number of variables (e.g., sensors).

    Returns:
        distances: A 1D array of shape (n_samples,) containing the
            calculated Mahalanobis distance for each corresponding
            observation in the input array `x`.

        cov: The covariance matrix of the data.
    """
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)

    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov)  # Usa pseudo-inversa como fallback

    distances = []

    for i in data:
        diff = i - mean
        mahalanobis_score = diff.T @ inv @ diff
        distances.append(mahalanobis_score)

    print(f"Mean row: {mean}")
    return distances, cov
