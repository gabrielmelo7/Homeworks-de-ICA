import pandas as pd
import numpy as np
from utils.ridge_regression_module import Ridge_regression



transformed_data = pd.read_csv("./data/data_yeojohnson_zscore.csv")
#splitting the dataset:
train, test = np.split(transformed_data.sample(frac=1, random_state=42), [int(.8*len(transformed_data))])

# Splitting into training, validation and test:
#adopting 80% - training
# 10% - validation
# 10% - test

x_train = train.drop(columns=["SO2",'Air Quality']); x_test = test.drop(columns=["SO2"])

y_train = train.SO2; y_test = test.SO2

ridge = Ridge_regression(x_train,y_train,x_test,y_test)

print(f'Best lambda: {ridge.b_lambda}')
print(f'bethas for best lambda: {ridge.ridge_bethas}')
print(f'r2_builtin: {ridge.r2_builtin}')
print(f'r2_scratch: {ridge.r2_scratch}')
print(f'20 primeiras predições: {ridge.predicted_ridge(x_test)[0:20]}')




