import numpy as np
from statistics import mode

class knn:
    def __init__(self, k):
        self.k = k 

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train-x)**2, axis=1))

            k_indices = np.argsort(distances)[:self.k]
            k_neighbors_labels = [self.y_train[i] for i in k_indices]

            common = mode(k_neighbors_labels)
            predictions.append(common)

        return np.array(predictions)

