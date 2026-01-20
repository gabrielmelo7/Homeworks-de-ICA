import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from utils.knn import knn
from utils.confusion_matrix import confusion_matrix
from utils.train_test_split import splitter
from utils.scores import scores 
from utils.crossvalKNN import cross_validation 

data_df = pd.read_csv("data/data_yeojohnson_zscore.csv") 

k_numbers = np.array([1,3,5,7,9])

for k in k_numbers:
    KNN = knn(k)
    accuracy, precision, recall = cross_validation(data_df, "Air Quality", KNN, 5)
    print(f"Accuracy for k = {k}: {accuracy}")
    print(f"Precision for k = {k}: {precision}")
    print(f"Recall for k = {k}: {recall}\n")

train_y, train_X, test_y, test_X = splitter(data_df)

train_X = np.array(train_X)
train_y = np.array(train_y)
test_X = np.array(test_X)
test_y = np.array(test_y)

test_y.shape

KNN = knn(5)

KNN.fit(train_X, train_y)

predictions = KNN.predict(test_X)

confusion_matrix = confusion_matrix(predictions, test_y)

labels = ["Hazardous", "Poor", "Moderate", "Good"]

plt.figure()
sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=".0f",
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        cbar=False
        )

plt.xlabel("Previsto")
plt.ylabel("Real")
plt.tight_layout()
print(confusion_matrix)
plt.savefig(os.path.join("./images", r"confusion_matrix_knn.png"))

