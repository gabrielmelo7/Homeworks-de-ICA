import numpy as np
from utils.confusion_matrix import confusion_matrix
from typing import Tuple

def scores(prediction: np.ndarray, target: np.ndarray) -> [np.float64, np.float64, np.float64]:
    """
    Calculates the accuracy, precision and recall of a classification model with macro averaging.

    Args:
        prediction (np.ndarray): The predicted values of the model.
        target (np.ndarray): The target values.
                                                                
    Returns:
         [np.float64, np.float64, np.float64]: The accuracy, precision and recall of the model.
 
    """

    conf_mat = confusion_matrix(prediction,target)

    classes = np.unique(np.concatenate((target, prediction)))
    num_classes = len(classes)

    true_positives = np.zeros(num_classes)
    false_positives = np.zeros(num_classes)
    false_negatives = np.zeros(num_classes)

    rows, columns = conf_mat.shape

    for row in range(rows):
        true_positives[row] += conf_mat[row, row]
        for column in range(columns):
            if row != column:
                false_negatives[row] += conf_mat[row, column]
                false_positives[column] += conf_mat[row,column]

    precision_by_class = true_positives / (true_positives + false_positives)
    recall_by_class = true_positives / (true_positives + false_negatives)

    accuracy = true_positives.sum() / conf_mat.sum()
    precision = sum(precision_by_class) / len(precision_by_class) 
    recall = sum(recall_by_class) / len(recall_by_class) 

    return accuracy, precision, recall
