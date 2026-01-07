import numpy as np

def confusion_matrix(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    
    classes = np.unique(np.concatenate((target, prediction)))
    num_classes = len(classes)

    matrix = np.zeros((num_classes, num_classes))

    map_vals = {}
    for index, val in zip(range(num_classes), classes):
        map_vals[val] = index

    for index in range(len(prediction)):
        targ_val = target[index]
        pred_val = prediction[index]
        matrix[map_vals[targ_val], map_vals[pred_val]] += 1

    return matrix
