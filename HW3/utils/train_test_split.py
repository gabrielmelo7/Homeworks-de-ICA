import numpy as np
import pandas as pd

"""
Divides the dataset in  80% training, 20% testing
"""

def splitter(dataset):
    dataset.rename(columns={'Proximity_to_Industrial_Areas':'Prox_Industrial','Population_Density':'Pop_Density'},inplace=True)
    train, test = np.split(dataset.sample(frac=1, random_state=42), [int(.8*len(dataset))])
    label_mapping = {"Hazardous": 1, "Poor": 2,"Moderate":3,"Good":4}
    train_y = train.iloc[:,-1].map(label_mapping)
    test_y = test.iloc[:,-1].map(label_mapping)
    train_x = train.iloc[:,:-1]
    test_x = test.iloc[:,:-1]
    return train_y,train_x, test_y, test_x

if __name__ == "__main__":
    print("Testing splitter:\n100 samples")

    np.random.seed(42)
    x_test = np.random.randint(0,1000,100)

    x_train, x_test = splitter(pd.DataFrame(x_test))
    
    print("Shapes, it must be (80,1) - (20,1):")
    print("shapes:",x_train.shape, x_test.shape)

